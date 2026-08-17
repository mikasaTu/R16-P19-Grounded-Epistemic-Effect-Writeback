"""Bounded hot ASCEL ledger with cold hash-chained audit summaries."""

from __future__ import annotations

from collections import OrderedDict, deque
import hashlib
import json
from typing import Deque, Dict

from .phase5_audit_store import AuditStore
from .phase5_compaction import append_bounded, bounded_size, insert_lru
from .phase5_ledger_reference import ReferenceLedger
from .phase5_types import EvidenceReceipt, LedgerDecision


class BoundedASCELLedger(ReferenceLedger):
    def __init__(self, active_capacity: int = 4, receipts_per_source: int = 4, recent_effects: int = 4, keep_audit_rows: bool = False) -> None:
        super().__init__()
        self.active_capacity = int(active_capacity)
        self.receipts_per_source = int(receipts_per_source)
        self.recent_effects_capacity = int(recent_effects)
        self.active = OrderedDict()
        self.receipts = OrderedDict()
        self.hot_receipts: Dict[str, Deque[EvidenceReceipt]] = {}
        self.recent_effects: "OrderedDict[str, bool]" = OrderedDict()
        self.audit = AuditStore(keep_rows=keep_audit_rows)

    def request(self, effect_id: str, attempt_id: str, command_id: str, epoch: int) -> None:
        if effect_id not in self.active and len(self.active) >= self.active_capacity:
            evicted_effect = next(iter(self.active))
            for key in [key for key in self.hot_receipts if key[0] == evicted_effect]:
                del self.hot_receipts[key]
        for key in [key for key in self.hot_receipts if key[0] == effect_id]:
            del self.hot_receipts[key]
        insert_lru(self.active, effect_id, (attempt_id, command_id), self.active_capacity)
        self.credit[effect_id] = False
        self.revoked_at.setdefault(effect_id, -1)
        self.audit.append({"kind": "REQUEST", "effect_id": effect_id, "attempt_id": attempt_id, "command_id": command_id, "epoch": epoch, "accepted": True})

    def revoke(self, effect_id: str, epoch: int) -> None:
        super().revoke(effect_id, epoch)
        insert_lru(self.recent_effects, effect_id, False, self.recent_effects_capacity)
        self.audit.append({"kind": "REVOKE", "effect_id": effect_id, "epoch": epoch, "accepted": True})

    def ingest(self, receipt: EvidenceReceipt, epoch: int) -> LedgerDecision:
        receipt_digest = hashlib.sha256(json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt_status = self.audit.receipt_status(receipt.receipt_id, receipt_digest)
        if receipt_status != "new":
            reason = "duplicate" if receipt_status == "duplicate" else "event_id_collision"
            decision = LedgerDecision(self.truth.get(receipt.effect_id, False), self.credit.get(receipt.effect_id, False), False, reason)
        else:
            # Reference semantics without retaining the global receipt dictionary.
            active = self.active.get(receipt.effect_id)
            if not receipt.valid_at(epoch):
                decision = LedgerDecision(self.truth.get(receipt.effect_id, False), False, False, "expired")
            elif receipt.observed_at <= self.revoked_at.get(receipt.effect_id, -1):
                decision = LedgerDecision(self.truth.get(receipt.effect_id, False), False, False, "pre_revocation")
            elif receipt.source_version < self.source_versions.get(receipt.source_id, -1):
                decision = LedgerDecision(self.truth.get(receipt.effect_id, False), False, False, "stale_source_version")
            elif not receipt.truth_value:
                self.source_versions[receipt.source_id] = receipt.source_version
                self.revoke(receipt.effect_id, epoch)
                decision = LedgerDecision(False, False, True, "contradiction")
            else:
                self.source_versions[receipt.source_id] = receipt.source_version
                same_attempt = active == (receipt.attempt_id, receipt.command_id)
                if receipt.realization_kind == "external":
                    self.truth[receipt.effect_id] = True
                    self.credit[receipt.effect_id] = False
                    insert_lru(self.recent_effects, receipt.effect_id, True, self.recent_effects_capacity)
                    decision = LedgerDecision(True, False, True, "accepted_external")
                elif not same_attempt:
                    decision = LedgerDecision(self.truth.get(receipt.effect_id, False), False, False, "stale_or_cross_attempt")
                else:
                    candidates = [
                        item for bucket in self.hot_receipts.values() for item in bucket
                        if item.truth_value
                        and item.realization_kind == "attempt"
                        and (item.attempt_id, item.command_id) == active
                        and item.effect_id == receipt.effect_id
                        and item.observed_at > self.revoked_at.get(receipt.effect_id, -1)
                        and item.valid_at(epoch)
                    ] + [receipt]
                    groups = {item.independent_key()[0] for item in candidates}
                    captures = {item.independent_key()[1] for item in candidates}
                    frames = {item.independent_key()[2] for item in candidates}
                    verified = len(groups) >= 2 and len(captures) >= 2 and len(frames) >= 2
                    if verified:
                        self.truth[receipt.effect_id] = True
                        self.credit[receipt.effect_id] = bool(receipt.active_attempt_credit)
                        insert_lru(self.recent_effects, receipt.effect_id, True, self.recent_effects_capacity)
                        decision = LedgerDecision(True, self.credit[receipt.effect_id], True, "accepted_current_verified")
                    else:
                        decision = LedgerDecision(self.truth.get(receipt.effect_id, False), False, True, "accepted_pending_independence")
            if decision.accepted and receipt.truth_value and receipt.realization_kind == "attempt" and self.active.get(receipt.effect_id) == (receipt.attempt_id, receipt.command_id):
                bucket_key = (receipt.effect_id, receipt.attempt_id, receipt.source_id)
                append_bounded(self.hot_receipts, bucket_key, receipt, self.receipts_per_source)
        self.audit.append({"kind": "RECEIPT", "receipt_id": receipt.receipt_id, "effect_id": receipt.effect_id, "epoch": epoch, "accepted": decision.accepted, "reason": decision.reason})
        return decision

    def hot_snapshot(self) -> dict:
        return {
            "active": dict(self.active),
            "truth": dict(self.truth),
            "credit": dict(self.credit),
            "revoked_at": dict(self.revoked_at),
            "recent_effects": dict(self.recent_effects),
            "receipt_counts": {"|".join(key): len(value) for key, value in self.hot_receipts.items()},
            "audit": self.audit.compact_summary(),
        }

    def hot_memory_bytes(self) -> int:
        return bounded_size({"active": self.active, "truth": self.truth, "credit": self.credit, "revoked_at": self.revoked_at, "recent_effects": self.recent_effects, "hot_receipts": self.hot_receipts, "source_versions": self.source_versions})

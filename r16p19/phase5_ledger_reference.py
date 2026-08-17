"""Unbounded executable reference semantics for Phase-5 receipts."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, Optional

from .phase5_types import EvidenceReceipt, LedgerDecision


class ReferenceLedger:
    def __init__(self) -> None:
        self.active: Dict[str, tuple[str, str]] = {}
        self.receipts: "OrderedDict[str, EvidenceReceipt]" = OrderedDict()
        self.truth: Dict[str, bool] = {}
        self.credit: Dict[str, bool] = {}
        self.source_versions: Dict[str, int] = {}
        self.revoked_at: Dict[str, int] = {}
        self.positive_index = defaultdict(list)

    def request(self, effect_id: str, attempt_id: str, command_id: str, epoch: int) -> None:
        self.active[effect_id] = (attempt_id, command_id)
        self.credit[effect_id] = False
        self.revoked_at.setdefault(effect_id, -1)

    def revoke(self, effect_id: str, epoch: int) -> None:
        self.truth[effect_id] = False
        self.credit[effect_id] = False
        self.revoked_at[effect_id] = epoch

    def ingest(self, receipt: EvidenceReceipt, epoch: int) -> LedgerDecision:
        if receipt.receipt_id in self.receipts:
            previous = self.receipts[receipt.receipt_id]
            reason = "duplicate" if previous.to_dict() == receipt.to_dict() else "event_id_collision"
            return LedgerDecision(self.truth.get(receipt.effect_id, False), self.credit.get(receipt.effect_id, False), False, reason)
        self.receipts[receipt.receipt_id] = receipt
        active = self.active.get(receipt.effect_id)
        if not receipt.valid_at(epoch):
            return LedgerDecision(self.truth.get(receipt.effect_id, False), False, False, "expired")
        if receipt.observed_at <= self.revoked_at.get(receipt.effect_id, -1):
            return LedgerDecision(self.truth.get(receipt.effect_id, False), False, False, "pre_revocation")
        previous_version = self.source_versions.get(receipt.source_id, -1)
        if receipt.source_version < previous_version:
            return LedgerDecision(self.truth.get(receipt.effect_id, False), False, False, "stale_source_version")
        self.source_versions[receipt.source_id] = receipt.source_version
        if not receipt.truth_value:
            self.revoke(receipt.effect_id, epoch)
            return LedgerDecision(False, False, True, "contradiction")
        same_attempt = active == (receipt.attempt_id, receipt.command_id)
        if receipt.realization_kind == "external":
            self.truth[receipt.effect_id] = True
            self.credit[receipt.effect_id] = False
            return LedgerDecision(True, False, True, "accepted_external")
        if not same_attempt:
            return LedgerDecision(self.truth.get(receipt.effect_id, False), False, False, "stale_or_cross_attempt")
        index_key = (receipt.effect_id, receipt.attempt_id, receipt.command_id)
        candidates = [
            item for item in self.positive_index[index_key]
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
            self.positive_index[index_key].append(receipt)
            return LedgerDecision(True, self.credit[receipt.effect_id], True, "accepted_current_verified")
        self.positive_index[index_key].append(receipt)
        return LedgerDecision(self.truth.get(receipt.effect_id, False), False, True, "accepted_pending_independence")

    def visible_state(self) -> dict:
        return {"active": dict(self.active), "truth": dict(self.truth), "credit": dict(self.credit), "revoked_at": dict(self.revoked_at)}

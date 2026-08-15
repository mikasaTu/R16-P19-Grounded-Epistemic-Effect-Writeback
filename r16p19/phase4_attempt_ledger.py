"""Attempt-scoped evidence ledger and fact/attribution separation."""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Counter as CounterType
from typing import Dict, Iterable, List, Optional, Tuple

from .phase4_types import (
    AttemptRecord,
    AttemptStatus,
    EffectFactRecord,
    FactState,
    LedgerEvent,
    ProofValidity,
    RealizationProof,
    deterministic_id,
)
from .types import Decision


class AttemptLedgerError(RuntimeError):
    pass


class AttemptScopedLedger:
    """Append-only attempt history with current-scope verification.

    ``attempt_scope=False`` and the two other toggles are used only for frozen
    comparison/ablation arms.  The M4 defaults are the strict ASCEL behavior.
    """

    def __init__(
        self,
        effects: Iterable[str],
        attempt_scope: bool = True,
        attribution_split: bool = True,
        pre_realization_revocation: bool = True,
    ) -> None:
        self.effects = tuple(effects)
        self.attempt_scope = bool(attempt_scope)
        self.attribution_split = bool(attribution_split)
        self.pre_realization_revocation = bool(pre_realization_revocation)
        self.attempts: "OrderedDict[str, AttemptRecord]" = OrderedDict()
        self.facts: Dict[str, EffectFactRecord] = {
            effect: EffectFactRecord(effect_id=effect) for effect in self.effects
        }
        self.proofs: "OrderedDict[str, RealizationProof]" = OrderedDict()
        self.active_attempts: Dict[str, Optional[str]] = {
            effect: None for effect in self.effects
        }
        self.generations: CounterType[str] = Counter()
        self.latest_revocation_epoch: Dict[str, int] = {
            effect: -1 for effect in self.effects
        }
        self.events: "OrderedDict[str, LedgerEvent]" = OrderedDict()
        self.audit_rows: List[dict] = []
        self.counters: CounterType[str] = Counter()
        self.pooled_evidence: Dict[str, Dict[str, Tuple[str, str]]] = {
            effect: {} for effect in self.effects
        }

    def _append_event(self, event: LedgerEvent) -> bool:
        if event.effect_id not in self.facts:
            raise AttemptLedgerError("unknown effect %s" % event.effect_id)
        if event.event_id in self.events:
            if self.events[event.event_id].canonical_bytes() != event.canonical_bytes():
                raise AttemptLedgerError("event ID reused with different bytes")
            self.counters["duplicate_events"] += 1
            return False
        self.events[event.event_id] = event
        return True

    def _active(self, effect_id: str) -> Optional[AttemptRecord]:
        attempt_id = self.active_attempts.get(effect_id)
        return self.attempts.get(attempt_id) if attempt_id is not None else None

    def request(self, event: LedgerEvent) -> AttemptRecord:
        if not self._append_event(event):
            active = self._active(event.effect_id)
            if active is None:
                raise AttemptLedgerError("duplicate request has no active attempt")
            return active
        previous = self._active(event.effect_id)
        if previous is not None and previous.status == AttemptStatus.ACTIVE:
            previous.status = AttemptStatus.SUPERSEDED
            previous.end_epoch = event.epoch
            self.counters["superseded_attempts"] += 1
        self.generations[event.effect_id] += 1
        generation = int(self.generations[event.effect_id])
        attempt_id = deterministic_id(
            "attempt", event.effect_id, generation, event.event_id
        )
        attempt = AttemptRecord(
            attempt_id=attempt_id,
            effect_id=event.effect_id,
            generation=generation,
            command_event_id=None,
            start_epoch=event.epoch,
            end_epoch=None,
            status=AttemptStatus.ACTIVE,
        )
        self.attempts[attempt_id] = attempt
        self.active_attempts[event.effect_id] = attempt_id
        fact = self.facts[event.effect_id]
        fact.active_attempt_id = attempt_id
        reset_current_scope = self.attempt_scope or fact.fact_state in (
            FactState.UNKNOWN,
            FactState.REVOKED,
        )
        if fact.fact_state != FactState.REALIZED and reset_current_scope:
            fact.fact_state = FactState.UNKNOWN
            fact.fact_epoch = max(fact.fact_epoch, event.epoch)
            fact.verified_fact_observations = []
            fact.physical_truth_confidence = 0.0
        if self.attempt_scope:
            self.pooled_evidence[event.effect_id] = {}
        self.audit_rows.append(
            {
                "event_id": event.event_id,
                "kind": "REQUEST",
                "effect_id": event.effect_id,
                "attempt_id": attempt_id,
                "accepted": True,
            }
        )
        return attempt

    def command(self, event: LedgerEvent) -> bool:
        if not self._append_event(event):
            return False
        active = self._active(event.effect_id)
        accepted = active is not None and active.status == AttemptStatus.ACTIVE
        if accepted and self.attempt_scope and event.attempt_id != active.attempt_id:
            accepted = False
        if accepted:
            active.command_event_id = event.event_id
            self.counters["commands"] += 1
        else:
            self.counters["rejected_commands"] += 1
        self.audit_rows.append(
            {
                "event_id": event.event_id,
                "kind": "COMMAND",
                "effect_id": event.effect_id,
                "attempt_id": event.attempt_id,
                "accepted": accepted,
            }
        )
        return accepted

    def _scope_match(self, event: LedgerEvent, active: Optional[AttemptRecord]) -> bool:
        if active is None or active.status != AttemptStatus.ACTIVE:
            return False
        if not self.attempt_scope:
            return True
        return (
            event.attempt_id == active.attempt_id
            and event.command_event_id == active.command_event_id
            and event.epoch > self.latest_revocation_epoch[event.effect_id]
        )

    def positive(self, event: LedgerEvent) -> bool:
        if not self._append_event(event):
            return False
        if event.evidence_id is None or event.sensor_id is None or event.evidence_digest is None:
            raise AttemptLedgerError("positive evidence is missing receipt fields")
        fact = self.facts[event.effect_id]
        active = self._active(event.effect_id)
        current_epoch = event.epoch >= fact.fact_epoch
        scope_match = self._scope_match(event, active)
        accepted = bool(current_epoch and scope_match)
        if not accepted:
            if active is not None and event.attempt_id != active.attempt_id:
                self.counters["stale_evidence_rejected"] += 1
            if active is not None and event.command_event_id != active.command_event_id:
                self.counters["superseded_command_evidence_rejected"] += 1
            self.audit_rows.append(
                {
                    "event_id": event.event_id,
                    "kind": "POSITIVE",
                    "effect_id": event.effect_id,
                    "attempt_id": event.attempt_id,
                    "accepted": False,
                    "retained_for_audit": True,
                }
            )
            return False
        evidence_bucket = self.pooled_evidence[event.effect_id]
        if event.evidence_digest in evidence_bucket:
            self.counters["duplicate_evidence_rejected"] += 1
            return False
        evidence_bucket[event.evidence_digest] = (event.sensor_id, event.evidence_id)
        assert active is not None
        if event.evidence_id not in active.evidence_ids:
            active.evidence_ids.append(event.evidence_id)
        if event.evidence_digest not in active.evidence_digests:
            active.evidence_digests.append(event.evidence_digest)
        active.evidence_sensors[event.evidence_digest] = event.sensor_id
        fact.fact_state = FactState.OBSERVED
        fact.fact_epoch = event.epoch
        fact.verified_fact_observations = [
            evidence_id for _, evidence_id in evidence_bucket.values()
        ]
        sensors = {sensor for sensor, _ in evidence_bucket.values()}
        if len(sensors) >= 2:
            fact.fact_state = FactState.VERIFIED
            fact.physical_truth_confidence = 1.0
            self.counters["verifications"] += 1
        self.audit_rows.append(
            {
                "event_id": event.event_id,
                "kind": "POSITIVE",
                "effect_id": event.effect_id,
                "attempt_id": event.attempt_id,
                "accepted": True,
                "sensor_count": len(sensors),
                "fact_state": fact.fact_state.value,
            }
        )
        return True

    def witness(self, event: LedgerEvent) -> Optional[RealizationProof]:
        if not self._append_event(event):
            return None
        fact = self.facts[event.effect_id]
        active = self._active(event.effect_id)
        scope_match = self._scope_match(event, active)
        witness_current = bool(scope_match and fact.fact_state == FactState.VERIFIED)
        if not witness_current:
            self.counters["witnesses_rejected"] += 1
            if active is not None and event.attempt_id != active.attempt_id:
                self.counters["stale_witness_rejected"] += 1
            if active is not None and event.command_event_id != active.command_event_id:
                self.counters["superseded_witness_rejected"] += 1
            if event.epoch <= self.latest_revocation_epoch[event.effect_id]:
                self.counters["post_revocation_witness_rejected"] += 1
            self.audit_rows.append(
                {
                    "event_id": event.event_id,
                    "kind": "WITNESS",
                    "effect_id": event.effect_id,
                    "attempt_id": event.attempt_id,
                    "accepted": False,
                }
            )
            return None
        assert active is not None
        proof_id = deterministic_id(
            "proof", event.effect_id, event.epoch, event.event_id
        )
        proof = RealizationProof(
            proof_id=proof_id,
            effect_id=event.effect_id,
            fact_epoch=event.epoch,
            attributed_attempt_id=active.attempt_id,
            evidence_ids=list(active.evidence_ids),
            witness_id=event.event_id,
            support_clause_id=None,
        )
        self.proofs[proof_id] = proof
        fact.fact_state = FactState.REALIZED
        fact.fact_epoch = event.epoch
        fact.physical_truth_confidence = 1.0
        fact.realization_proof_ids.append(proof_id)
        active.status = AttemptStatus.SUCCEEDED
        active.end_epoch = event.epoch
        active.attribution_witness_ids.append(event.event_id)
        self.counters["attributed_realizations"] += 1
        self.audit_rows.append(
            {
                "event_id": event.event_id,
                "kind": "WITNESS",
                "effect_id": event.effect_id,
                "attempt_id": event.attempt_id,
                "accepted": True,
                "proof_id": proof_id,
                "attributed": True,
            }
        )
        return proof

    def external_realization(self, event: LedgerEvent) -> RealizationProof:
        if not self._append_event(event):
            for proof_id in reversed(self.facts[event.effect_id].realization_proof_ids):
                return self.proofs[proof_id]
            raise AttemptLedgerError("duplicate external event has no proof")
        fact = self.facts[event.effect_id]
        active = self._active(event.effect_id)
        attributed_attempt = None
        if not self.attribution_split and active is not None:
            attributed_attempt = active.attempt_id
        evidence_ids = []
        raw_evidence = event.payload.get("evidence_ids", [])
        if isinstance(raw_evidence, list):
            evidence_ids = [str(value) for value in raw_evidence]
        proof_id = deterministic_id(
            "proof", event.effect_id, event.epoch, event.event_id, "external"
        )
        proof = RealizationProof(
            proof_id=proof_id,
            effect_id=event.effect_id,
            fact_epoch=event.epoch,
            attributed_attempt_id=attributed_attempt,
            evidence_ids=evidence_ids,
            witness_id=event.event_id,
            support_clause_id=None,
        )
        self.proofs[proof_id] = proof
        fact.fact_state = FactState.REALIZED
        fact.fact_epoch = event.epoch
        fact.physical_truth_confidence = 1.0
        fact.verified_fact_observations = list(evidence_ids)
        fact.realization_proof_ids.append(proof_id)
        if attributed_attempt is not None and active is not None:
            active.status = AttemptStatus.SUCCEEDED
            active.end_epoch = event.epoch
            active.attribution_witness_ids.append(event.event_id)
            self.counters["false_external_attributions"] += 1
        else:
            self.counters["external_truth_without_credit"] += 1
        self.audit_rows.append(
            {
                "event_id": event.event_id,
                "kind": "EXTERNAL_REALIZATION",
                "effect_id": event.effect_id,
                "attempt_id": attributed_attempt,
                "accepted": True,
                "proof_id": proof_id,
                "attributed": attributed_attempt is not None,
            }
        )
        return proof

    def negative_or_contradiction(self, event: LedgerEvent) -> List[str]:
        if not self._append_event(event):
            return []
        fact = self.facts[event.effect_id]
        active = self._active(event.effect_id)
        state_before = fact.fact_state
        should_revoke = state_before == FactState.REALIZED or (
            self.pre_realization_revocation
            and state_before in (FactState.OBSERVED, FactState.VERIFIED)
        )
        invalidated = []
        if should_revoke:
            self.latest_revocation_epoch[event.effect_id] = max(
                self.latest_revocation_epoch[event.effect_id], event.epoch
            )
            for proof_id in fact.realization_proof_ids:
                proof = self.proofs[proof_id]
                if proof.validity_status == ProofValidity.VALID:
                    proof.validity_status = ProofValidity.REVOKED
                    invalidated.append(proof_id)
            fact.fact_state = FactState.REVOKED
            fact.fact_epoch = event.epoch
            fact.invalidation_event_id = event.event_id
            fact.physical_truth_confidence = 0.0
            fact.verified_fact_observations = []
            self.pooled_evidence[event.effect_id] = {}
            if active is not None and active.status in (
                AttemptStatus.ACTIVE,
                AttemptStatus.SUCCEEDED,
            ):
                active.status = AttemptStatus.INVALIDATED
                active.end_epoch = event.epoch
                active.evidence_ids = []
                active.evidence_digests = []
                active.evidence_sensors = {}
            self.counters["revocations"] += 1
        else:
            self.counters["ignored_pre_realization_contradictions"] += 1
        self.audit_rows.append(
            {
                "event_id": event.event_id,
                "kind": "CONTRADICTION",
                "effect_id": event.effect_id,
                "state_before": state_before.value,
                "revoked": should_revoke,
                "invalidated_proof_ids": list(invalidated),
            }
        )
        return invalidated

    def invalidate_effect_from_support(
        self, effect_id: str, proof_ids: Iterable[str], event_id: str, epoch: int
    ) -> None:
        fact = self.facts[effect_id]
        changed = False
        for proof_id in proof_ids:
            proof = self.proofs.get(proof_id)
            if proof is not None and proof.validity_status == ProofValidity.VALID:
                proof.validity_status = ProofValidity.INVALID
                changed = True
        if changed or fact.fact_state == FactState.REALIZED:
            fact.fact_state = FactState.REVOKED
            fact.fact_epoch = max(fact.fact_epoch, epoch)
            fact.invalidation_event_id = event_id
            fact.physical_truth_confidence = 0.0

    def effect_fact_verified(self, effect_id: str) -> bool:
        fact = self.facts[effect_id]
        if fact.fact_state != FactState.REALIZED:
            return False
        return any(
            self.proofs[proof_id].validity_status == ProofValidity.VALID
            for proof_id in fact.realization_proof_ids
        )

    def attempt_attributed_success(self, effect_id: str) -> bool:
        active_id = self.active_attempts.get(effect_id)
        if active_id is None:
            return False
        return any(
            self.proofs[proof_id].validity_status == ProofValidity.VALID
            and self.proofs[proof_id].attributed_attempt_id == active_id
            for proof_id in self.facts[effect_id].realization_proof_ids
        )

    def decide(self, effect_id: str) -> Decision:
        if self.effect_fact_verified(effect_id):
            return Decision.ADVANCE_TO_NEXT_SUBTASK
        fact = self.facts[effect_id]
        active = self._active(effect_id)
        if fact.fact_state == FactState.REVOKED:
            if active is not None and active.status == AttemptStatus.INVALIDATED:
                return Decision.ROLLBACK_OR_REPLAN
            return Decision.RETRY_CURRENT_EFFECT
        if active is None or active.status in (
            AttemptStatus.FAILED,
            AttemptStatus.SUPERSEDED,
            AttemptStatus.INVALIDATED,
        ):
            return Decision.RETRY_CURRENT_EFFECT
        return Decision.REOBSERVE

    def summary(self) -> dict:
        return {
            "attempt_scope": self.attempt_scope,
            "attribution_split": self.attribution_split,
            "pre_realization_revocation": self.pre_realization_revocation,
            "attempts": [value.to_dict() for value in self.attempts.values()],
            "facts": {key: value.to_dict() for key, value in self.facts.items()},
            "proofs": [value.to_dict() for value in self.proofs.values()],
            "active_attempts": dict(self.active_attempts),
            "latest_revocation_epoch": dict(self.latest_revocation_epoch),
            "event_count": len(self.events),
            "ledger_size": len(self.events) + len(self.attempts) + len(self.proofs),
            "counters": dict(self.counters),
            "audit_rows": list(self.audit_rows),
        }

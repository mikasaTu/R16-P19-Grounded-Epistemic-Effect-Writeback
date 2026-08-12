"""Epistemic effect memory arms and append-only provenance ledger."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

from .config import RESIDENT_MEMORY_SLOTS, TASKS
from .ontology import dependents, prerequisites_by_task
from .types import (
    Decision,
    EffectRecord,
    EpistemicState,
    Event,
    EventType,
)


class ProvenanceError(RuntimeError):
    pass


class ProvenanceLedger:
    """Append-only ledger plus bounded resident cache.

    Parent references always target the append-only ledger, so cache eviction
    cannot create dangling provenance even when the parent is no longer
    resident.
    """

    def __init__(self, capacity: int = RESIDENT_MEMORY_SLOTS):
        if capacity != RESIDENT_MEMORY_SLOTS:
            raise ValueError("Phase-1 capacity is frozen at 32")
        self.capacity = capacity
        self.events: "OrderedDict[str, Event]" = OrderedDict()
        self.resident: Deque[str] = deque()
        self.max_resident_seen = 0
        self.duplicate_event_count = 0

    def append(self, event: Event) -> bool:
        event.validate()
        if event.event_id in self.events:
            existing = self.events[event.event_id]
            if existing.canonical_bytes() != event.canonical_bytes():
                raise ProvenanceError("event ID reused with different bytes")
            self.duplicate_event_count += 1
            return False
        missing = [parent for parent in event.parent_ids if parent not in self.events]
        if missing:
            raise ProvenanceError("dangling parent(s): %s" % ",".join(missing))
        self.events[event.event_id] = event
        self.resident.append(event.event_id)
        while len(self.resident) > self.capacity:
            self.resident.popleft()
        self.max_resident_seen = max(self.max_resident_seen, len(self.resident))
        return True

    def dangling_parent_count(self) -> int:
        known = set(self.events)
        return sum(
            1
            for event in self.events.values()
            for parent in event.parent_ids
            if parent not in known
        )


@dataclass(frozen=True)
class ArmCapabilities:
    typed: bool
    verification: bool
    contradiction_detection: bool
    contradiction_recovery: bool
    command_is_progress: bool
    monolithic: bool
    oracle: bool


CAPABILITIES: Dict[str, ArmCapabilities] = {
    "B1": ArmCapabilities(False, False, False, False, False, False, False),
    "B2": ArmCapabilities(False, False, False, False, True, False, False),
    "B3": ArmCapabilities(False, False, False, False, False, True, False),
    "B4": ArmCapabilities(True, False, True, False, False, False, False),
    "B5": ArmCapabilities(True, True, True, False, False, False, False),
    "B6": ArmCapabilities(True, True, True, True, False, False, False),
    "B7": ArmCapabilities(True, True, True, True, False, False, True),
}


class MemoryArm:
    def __init__(self, arm: str, task_key: str, ontology: dict):
        if arm not in CAPABILITIES:
            raise ValueError("unknown arm %s" % arm)
        self.arm = arm
        self.task_key = task_key
        self.capabilities = CAPABILITIES[arm]
        self.effects = list(TASKS[task_key].effects)
        self.records = {effect: EffectRecord(effect) for effect in self.effects}
        self.prerequisites = prerequisites_by_task(ontology, task_key)
        self.dependents = dependents(self.effects, self.prerequisites)
        self.ledger = ProvenanceLedger()
        self.alias_rejections = 0
        self.alias_acceptances = 0
        self.transition_violations: List[str] = []
        self.contradiction_events = 0
        self.recovery_events = 0
        self.decision_counts = {decision.value: 0 for decision in Decision}

    def _record_receipt(self, record: EffectRecord, event: Event) -> bool:
        receipt = event.receipt
        if receipt is None:
            raise ValueError("physical evidence event lacks a receipt")
        digest_seen = receipt.frame_digest in record.evidence_digests
        if digest_seen:
            self.alias_rejections += 1
            return False
        record.evidence_digests[receipt.frame_digest] = receipt.sensor_identity
        record.evidence_ids.append(receipt.evidence_id)
        return True

    def _has_independent_verification(self, record: EffectRecord) -> bool:
        return (
            len(record.evidence_digests) >= 2
            and len(set(record.evidence_digests.values())) >= 2
        )

    def _block_dependents(self, effect_id: str) -> List[str]:
        blocked: Set[str] = set()
        frontier = list(self.dependents.get(effect_id, []))
        while frontier:
            candidate = frontier.pop()
            if candidate in blocked:
                continue
            blocked.add(candidate)
            frontier.extend(self.dependents.get(candidate, []))
        for dependent in blocked:
            dependent_record = self.records[dependent]
            if dependent_record.state == EpistemicState.REALIZED:
                dependent_record.state = EpistemicState.INVALIDATED_REALIZATION
                if self.capabilities.contradiction_recovery:
                    dependent_record.recovery_route = [
                        "ROLLBACK_OR_REPLAN",
                        "RETRY_CURRENT_EFFECT",
                    ]
        return sorted(blocked)

    def _process_oracle(self, event: Event) -> None:
        if event.effect_id is None:
            return
        record = self.records[event.effect_id]
        truth = event.payload.get("physical_truth")
        if truth is True:
            record.state = EpistemicState.REALIZED
        elif event.event_type in (
            EventType.OBSERVE_NEGATIVE,
            EventType.TIMEOUT,
        ):
            record.state = EpistemicState.STALLED
            record.recovery_route = ["RETRY_CURRENT_EFFECT"]
        elif event.event_type == EventType.CONTRADICTION:
            record.state = EpistemicState.INVALIDATED_REALIZATION
            record.recovery_route = ["ROLLBACK_OR_REPLAN", "RETRY_CURRENT_EFFECT"]
            record.blocked_dependents = self._block_dependents(event.effect_id)

    def _process_baseline(self, event: Event) -> None:
        if event.effect_id is None:
            return
        record = self.records[event.effect_id]
        if self.arm == "B1":
            if event.event_type in (
                EventType.OBSERVE_POSITIVE,
                EventType.VERIFY_POSITIVE,
                EventType.REALIZATION_WITNESS,
            ):
                if event.receipt is not None:
                    self._record_receipt(record, event)
                record.state = EpistemicState.REALIZED
            elif event.event_type == EventType.TIMEOUT:
                record.state = EpistemicState.STALLED
            return
        if self.arm == "B2":
            if event.event_type == EventType.COMMAND:
                record.command_event_id = event.event_id
                record.state = EpistemicState.REALIZED
            return
        if self.arm == "B3":
            if event.event_type in (
                EventType.REQUEST,
                EventType.IMAGINE,
                EventType.COMMAND,
                EventType.OBSERVE_POSITIVE,
                EventType.VERIFY_POSITIVE,
                EventType.REALIZATION_WITNESS,
            ):
                if event.receipt is not None:
                    self._record_receipt(record, event)
                record.state = EpistemicState.REALIZED
            elif (
                event.event_type == EventType.TIMEOUT
                and record.state != EpistemicState.REALIZED
            ):
                record.state = EpistemicState.STALLED

    def _process_typed(self, event: Event) -> None:
        if event.effect_id is None:
            return
        record = self.records[event.effect_id]
        before = record.state
        kind = event.event_type
        if kind == EventType.REQUEST:
            if before in (
                None,
                EpistemicState.STALLED,
                EpistemicState.INVALIDATED_REALIZATION,
            ):
                record.state = EpistemicState.REQUESTED
                if before == EpistemicState.INVALIDATED_REALIZATION:
                    record.recovery_route = []
                    record.invalidated_by = None
        elif kind == EventType.IMAGINE:
            if before in (None, EpistemicState.REQUESTED, EpistemicState.IMAGINED):
                record.state = EpistemicState.IMAGINED
        elif kind == EventType.COMMAND:
            record.command_event_id = event.event_id
            if before is None:
                record.state = EpistemicState.REQUESTED
        elif kind in (EventType.OBSERVE_POSITIVE, EventType.VERIFY_POSITIVE):
            accepted = self._record_receipt(record, event)
            if accepted and record.state not in (
                EpistemicState.REALIZED,
                EpistemicState.INVALIDATED_REALIZATION,
            ):
                record.state = EpistemicState.OBSERVED
            if (
                kind == EventType.VERIFY_POSITIVE
                and self.capabilities.verification
                and self._has_independent_verification(record)
                and record.state != EpistemicState.INVALIDATED_REALIZATION
            ):
                record.state = EpistemicState.VERIFIED
            if (
                not accepted
                and before != EpistemicState.VERIFIED
                and record.state == EpistemicState.VERIFIED
            ):
                self.alias_acceptances += 1
        elif kind == EventType.REALIZATION_WITNESS:
            causally_linked = (
                record.command_event_id is not None
                and record.command_event_id in event.parent_ids
            )
            verification_ready = (
                record.state == EpistemicState.VERIFIED
                if self.capabilities.verification
                else record.state in (EpistemicState.OBSERVED, EpistemicState.VERIFIED)
            )
            if causally_linked and verification_ready:
                record.state = EpistemicState.REALIZED
        elif kind in (EventType.OBSERVE_NEGATIVE, EventType.TIMEOUT):
            if record.state != EpistemicState.REALIZED:
                record.state = EpistemicState.STALLED
                if self.capabilities.contradiction_recovery:
                    record.recovery_route = ["RETRY_CURRENT_EFFECT", "REOBSERVE"]
        elif kind == EventType.CONTRADICTION and self.capabilities.contradiction_detection:
            self.contradiction_events += 1
            if record.state == EpistemicState.REALIZED:
                record.state = EpistemicState.INVALIDATED_REALIZATION
                record.invalidated_by = event.event_id
                record.blocked_dependents = self._block_dependents(event.effect_id)
                if self.capabilities.contradiction_recovery:
                    record.recovery_route = [
                        "ROLLBACK_OR_REPLAN",
                        "RETRY_CURRENT_EFFECT",
                    ]
                    self.recovery_events += 1
            elif record.state == EpistemicState.INVALIDATED_REALIZATION:
                record.duplicate_contradictions += 1

        if kind == EventType.COMMAND and record.state == EpistemicState.REALIZED:
            self.transition_violations.append("command_directly_realized")
        if kind == EventType.IMAGINE and record.state in (
            EpistemicState.VERIFIED,
            EpistemicState.REALIZED,
        ):
            self.transition_violations.append("imagined_directly_verified_or_realized")

    def process(self, event: Event, current_effect: Optional[str]) -> Decision:
        is_new = self.ledger.append(event)
        if is_new:
            if self.capabilities.oracle:
                self._process_oracle(event)
            elif not self.capabilities.typed:
                self._process_baseline(event)
            else:
                self._process_typed(event)
        decision = self.decide(current_effect, event)
        self.decision_counts[decision.value] += 1
        return decision

    def decide(self, current_effect: Optional[str], event: Event) -> Decision:
        if self.capabilities.oracle:
            requested = event.payload.get("oracle_decision")
            if requested is not None:
                return Decision(requested)
        if current_effect is None:
            return Decision.SAFE_STOP
        record = self.records[current_effect]
        state = record.state
        if state == EpistemicState.INVALIDATED_REALIZATION:
            if self.capabilities.contradiction_recovery and record.recovery_route:
                return Decision.ROLLBACK_OR_REPLAN
            return Decision.SAFE_STOP
        if state == EpistemicState.STALLED:
            if self.arm in ("B4", "B6", "B7") and (
                self.arm != "B6" or record.recovery_route
            ):
                return Decision.RETRY_CURRENT_EFFECT
            return Decision.SAFE_STOP
        if state == EpistemicState.REALIZED:
            return Decision.ADVANCE_TO_NEXT_SUBTASK
        if event.event_type == EventType.CONTRADICTION:
            return Decision.REOBSERVE
        return Decision.REOBSERVE

    def semantic_fingerprint(self, effect_id: str) -> Tuple[object, ...]:
        record = self.records[effect_id]
        return (
            record.state,
            tuple(record.recovery_route),
            tuple(record.blocked_dependents),
            record.invalidated_by,
        )

    def current_summary(self) -> dict:
        return {
            "arm": self.arm,
            "task_key": self.task_key,
            "effects": {key: value.to_dict() for key, value in self.records.items()},
            "resident_slot_count": len(self.ledger.resident),
            "resident_slot_count_max": self.ledger.max_resident_seen,
            "ledger_event_count": len(self.ledger.events),
            "dangling_parent_count": self.ledger.dangling_parent_count(),
            "duplicate_event_count": self.ledger.duplicate_event_count,
            "alias_rejections": self.alias_rejections,
            "alias_acceptances": self.alias_acceptances,
            "transition_violations": list(self.transition_violations),
            "decision_counts": dict(self.decision_counts),
        }


def make_arms(task_key: str, ontology: dict, arms: Iterable[str]) -> Dict[str, MemoryArm]:
    return {arm: MemoryArm(arm, task_key, ontology) for arm in arms}

"""Frozen strong comparison arms and wrappers for Phase-3."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from enum import Enum
from typing import Dict, Mapping, MutableMapping, Set

from .memory import MemoryArm, ProvenanceLedger
from .types import Decision, Event, EventType


MAIN_ARMS = (
    "B2_COMMAND_PROGRESS",
    "B3_MONOLITHIC",
    "POSTCHECK_RECOVERY",
    "PERSISTENCE_RECOVERY",
    "TYPED_MATCHED_RECOVERY",
    "B6_FULL",
)
STRONG_ARMS = (
    "POSTCHECK_RECOVERY",
    "PERSISTENCE_RECOVERY",
    "TYPED_MATCHED_RECOVERY",
    "B6_FULL",
)
ABLATION_ARMS = ("B6_NO_PROVENANCE", "B6_NO_INVALIDATION")


class BinaryState(str, Enum):
    UNKNOWN = "UNKNOWN"
    TRUE = "TRUE"
    FALSE = "FALSE"


class ExistingMemoryAdapter:
    def __init__(self, public_name: str, internal_arm: str, task_key: str, ontology: dict):
        self.public_name = public_name
        self.memory = MemoryArm(internal_arm, task_key, ontology)

    def process(self, event: Event, current_effect: str | None) -> Decision:
        return self.memory.process(event, current_effect)

    def decide(self, current_effect: str | None, event: Event) -> Decision:
        return self.memory.decide(current_effect, event)

    def current_summary(self) -> dict:
        value = self.memory.current_summary()
        value["arm"] = self.public_name
        value["internal_frozen_arm"] = self.memory.arm
        return value


class StrongRecoveryBaseline:
    """POSTCHECK, persistence, or typed-matched strong baseline."""

    def __init__(self, arm: str, task_key: str, effects: tuple[str, ...], persistence_k: int):
        if arm not in (
            "POSTCHECK_RECOVERY",
            "PERSISTENCE_RECOVERY",
            "TYPED_MATCHED_RECOVERY",
        ):
            raise ValueError("unknown strong baseline %s" % arm)
        if persistence_k not in (2, 4, 8):
            raise ValueError("persistence K is outside the frozen candidates")
        self.arm = arm
        self.task_key = task_key
        self.effects = tuple(effects)
        self.persistence_k = int(persistence_k)
        self.ledger = ProvenanceLedger()
        self.state: MutableMapping[str, str] = {
            effect: BinaryState.UNKNOWN.value for effect in effects
        }
        self.positive_streak: Counter[str] = Counter()
        self.last_positive_decision_index: MutableMapping[str, int | None] = {
            effect: None for effect in effects
        }
        self.sensor_digests: MutableMapping[str, Dict[str, str]] = {
            effect: {} for effect in effects
        }
        self.seen_frame_digests: MutableMapping[str, Set[str]] = {
            effect: set() for effect in effects
        }
        self.last_event_type: MutableMapping[str, str | None] = {
            effect: None for effect in effects
        }
        self.decision_counts: Counter[str] = Counter()
        self.alias_rejections = 0
        self.transition_violations = []
        self.contradiction_events = 0
        self.recovery_events = 0

    def _record_positive(self, event: Event) -> bool:
        assert event.effect_id is not None
        receipt = event.receipt
        if receipt is None:
            return False
        if receipt.frame_digest in self.seen_frame_digests[event.effect_id]:
            self.alias_rejections += 1
            return False
        self.seen_frame_digests[event.effect_id].add(receipt.frame_digest)
        self.sensor_digests[event.effect_id][receipt.frame_digest] = (
            receipt.sensor_identity
        )
        return True

    def _process_postcheck(self, event: Event) -> None:
        effect = event.effect_id
        if effect is None:
            return
        if event.event_type in (
            EventType.OBSERVE_POSITIVE,
            EventType.VERIFY_POSITIVE,
            EventType.REALIZATION_WITNESS,
        ):
            self._record_positive(event)
            self.state[effect] = BinaryState.TRUE.value
        elif event.event_type in (
            EventType.OBSERVE_NEGATIVE,
            EventType.CONTRADICTION,
            EventType.TIMEOUT,
        ):
            self.state[effect] = BinaryState.FALSE.value

    def _process_persistence(self, event: Event) -> None:
        effect = event.effect_id
        if effect is None:
            return
        if event.event_type in (
            EventType.OBSERVE_POSITIVE,
            EventType.VERIFY_POSITIVE,
            EventType.REALIZATION_WITNESS,
        ):
            accepted = self._record_positive(event)
            decision_index = event.payload.get("decision_index")
            if decision_index is None:
                # Phase-3 broker receipts always carry the tick.  Retain a
                # deterministic fallback for direct unit tests.
                decision_index = event.event_index
            if (
                accepted
                and self.last_positive_decision_index[effect] != int(decision_index)
            ):
                self.positive_streak[effect] += 1
                self.last_positive_decision_index[effect] = int(decision_index)
            if self.positive_streak[effect] >= self.persistence_k:
                self.state[effect] = BinaryState.TRUE.value
        elif event.event_type in (
            EventType.OBSERVE_NEGATIVE,
            EventType.CONTRADICTION,
            EventType.TIMEOUT,
        ):
            self.positive_streak[effect] = 0
            self.last_positive_decision_index[effect] = None
            self.state[effect] = BinaryState.FALSE.value

    def _process_typed(self, event: Event) -> None:
        effect = event.effect_id
        if effect is None:
            return
        state = self.state[effect]
        kind = event.event_type
        if kind == EventType.REQUEST:
            if state in (
                BinaryState.UNKNOWN.value,
                BinaryState.FALSE.value,
                "INVALIDATED_REALIZATION",
            ):
                self.state[effect] = "REQUESTED"
                self.sensor_digests[effect].clear()
                self.seen_frame_digests[effect].clear()
        elif kind == EventType.IMAGINE and state != "REALIZED":
            self.state[effect] = "IMAGINED"
        elif kind in (EventType.OBSERVE_POSITIVE, EventType.VERIFY_POSITIVE):
            self._record_positive(event)
            self.state[effect] = "OBSERVED"
            sensors = set(self.sensor_digests[effect].values())
            if len(sensors) >= 2:
                self.state[effect] = "VERIFIED"
        elif kind == EventType.REALIZATION_WITNESS:
            self._record_positive(event)
            if self.state[effect] == "VERIFIED":
                self.state[effect] = "REALIZED"
        elif kind in (EventType.OBSERVE_NEGATIVE, EventType.TIMEOUT):
            if self.state[effect] != "REALIZED":
                self.state[effect] = BinaryState.FALSE.value
        elif kind == EventType.CONTRADICTION:
            self.contradiction_events += 1
            self.state[effect] = BinaryState.FALSE.value
            self.recovery_events += 1

    def process(self, event: Event, current_effect: str | None) -> Decision:
        is_new = self.ledger.append(event)
        if is_new:
            if event.effect_id is not None:
                self.last_event_type[event.effect_id] = event.event_type.value
            if event.event_type == EventType.CONTRADICTION:
                self.contradiction_events += int(self.arm != "TYPED_MATCHED_RECOVERY")
            if self.arm == "POSTCHECK_RECOVERY":
                self._process_postcheck(event)
            elif self.arm == "PERSISTENCE_RECOVERY":
                self._process_persistence(event)
            else:
                self._process_typed(event)
        decision = self.decide(current_effect, event)
        self.decision_counts[decision.value] += 1
        return decision

    def decide(self, current_effect: str | None, event: Event) -> Decision:
        if current_effect is None:
            return Decision.SAFE_STOP
        state = self.state[current_effect]
        if state in (BinaryState.TRUE.value, "REALIZED"):
            return Decision.ADVANCE_TO_NEXT_SUBTASK
        if state == BinaryState.FALSE.value:
            if self.last_event_type[current_effect] == EventType.CONTRADICTION.value:
                return Decision.ROLLBACK_OR_REPLAN
            return Decision.RETRY_CURRENT_EFFECT
        return Decision.REOBSERVE

    def current_summary(self) -> dict:
        return {
            "arm": self.arm,
            "task_key": self.task_key,
            "states": dict(self.state),
            "positive_streak": dict(self.positive_streak),
            "persistence_k": self.persistence_k if self.arm == "PERSISTENCE_RECOVERY" else None,
            "resident_slot_count": len(self.ledger.resident),
            "resident_slot_count_max": self.ledger.max_resident_seen,
            "ledger_event_count": len(self.ledger.events),
            "dangling_parent_count": self.ledger.dangling_parent_count(),
            "duplicate_event_count": self.ledger.duplicate_event_count,
            "alias_rejections": self.alias_rejections,
            "alias_acceptances": 0,
            "transition_violations": list(self.transition_violations),
            "contradiction_events": self.contradiction_events,
            "recovery_events": self.recovery_events,
            "decision_counts": dict(self.decision_counts),
        }


class B6NoProvenanceAdapter(ExistingMemoryAdapter):
    """Wrapper that makes an unlinked realization witness command-linked."""

    def process(self, event: Event, current_effect: str | None) -> Decision:
        transformed = event
        if (
            event.event_type == EventType.REALIZATION_WITNESS
            and event.effect_id is not None
        ):
            command_id = self.memory.records[event.effect_id].command_event_id
            if command_id is not None and command_id not in event.parent_ids:
                transformed = replace(
                    event, parent_ids=tuple(event.parent_ids) + (command_id,)
                )
        return self.memory.process(transformed, current_effect)


class B6NoInvalidationAdapter(ExistingMemoryAdapter):
    """Wrapper that prevents contradiction from invalidating realization."""

    def process(self, event: Event, current_effect: str | None) -> Decision:
        transformed = event
        if event.event_type == EventType.CONTRADICTION:
            transformed = replace(
                event,
                event_type=EventType.IRRELEVANT,
                effect_id=None,
                receipt=None,
                parent_ids=(),
                payload={"ablation": "invalidation_disabled"},
            )
        return self.memory.process(transformed, current_effect)


def make_phase3_arm(
    arm: str,
    task_key: str,
    ontology: dict,
    effects: tuple[str, ...],
    persistence_k: int,
):
    if arm == "B2_COMMAND_PROGRESS":
        return ExistingMemoryAdapter(arm, "B2", task_key, ontology)
    if arm == "B3_MONOLITHIC":
        return ExistingMemoryAdapter(arm, "B3", task_key, ontology)
    if arm == "B6_FULL":
        return ExistingMemoryAdapter(arm, "B6", task_key, ontology)
    if arm == "B6_NO_PROVENANCE":
        return B6NoProvenanceAdapter(arm, "B6", task_key, ontology)
    if arm == "B6_NO_INVALIDATION":
        return B6NoInvalidationAdapter(arm, "B6", task_key, ontology)
    return StrongRecoveryBaseline(arm, task_key, effects, persistence_k)

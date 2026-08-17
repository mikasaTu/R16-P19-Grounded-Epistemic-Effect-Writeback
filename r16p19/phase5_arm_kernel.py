"""Frozen arm semantics over a common attempt-scoped event stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .phase4_arms import make_phase4_arm
from .phase4_event_broker import Phase4EventBroker
from .phase4_types import LedgerEvent, LedgerEventKind
from .types import Decision


ARMS = ("M0_TYPED_MATCHED", "M1_VERSIONED_POSTCHECK", "M2_B6_FROZEN", "M3_ASCEL_CORE", "M4_ASCEL_FULL")
CONDITIONS = ("C0_CLEAN", "A1_NOOP_RETRY_STALE", "A2_CROSS_ATTEMPT_MIX", "A3_CONTRADICTION_LATE_WITNESS", "A4_POST_REALIZATION_REVERSAL", "A5_EXTERNAL_REALIZATION", "V1_SINGLE_VIEW_FALSE_POSITIVE")


class VersionedPostcheck:
    """Frozen K=2, TTL-bounded postcheck without attempt isolation."""

    def __init__(self, ttl: int = 8) -> None:
        self.ttl = int(ttl)
        self.positive_epochs: List[int] = []
        self.source_ids: List[str] = []
        self.realized = False
        self.credited = False
        self.active_attempt = None

    def process(self, event: LedgerEvent) -> Decision:
        if event.kind == LedgerEventKind.REQUEST:
            self.active_attempt = event.attempt_id
        elif event.kind == LedgerEventKind.POSITIVE:
            self.positive_epochs = [value for value in self.positive_epochs if event.epoch - value <= self.ttl]
            self.positive_epochs.append(event.epoch)
            self.source_ids.append(str(event.sensor_id))
        elif event.kind == LedgerEventKind.WITNESS and len(self.positive_epochs) >= 2:
            self.realized = True
            self.credited = True
        elif event.kind == LedgerEventKind.EXTERNAL_REALIZATION:
            self.realized = True
            self.credited = True
        elif event.kind in (LedgerEventKind.NEGATIVE, LedgerEventKind.CONTRADICTION, LedgerEventKind.SUPPORT_INVALIDATION):
            self.realized = False
            self.credited = False
            self.positive_epochs = []
        return Decision.ADVANCE_TO_NEXT_SUBTASK if self.realized else Decision.REOBSERVE

    def effect_fact_verified(self, effect_id: str) -> bool:
        return self.realized

    def attempt_attributed_success(self, effect_id: str) -> bool:
        return self.credited


def _make(name: str, unit_id: str):
    if name == "M1_VERSIONED_POSTCHECK":
        return VersionedPostcheck()
    mapping = {
        "M0_TYPED_MATCHED": "M0_TYPED_MATCHED",
        "M2_B6_FROZEN": "M1_B6_ORIGINAL",
        "M3_ASCEL_CORE": "M2_ATTEMPT_ONLY",
        "M4_ASCEL_FULL": "M4_ASCEL_FULL",
        "NO_ATTEMPT_SCOPE": "NO_ATTEMPT_SCOPE",
        "NO_PRE_REALIZATION_REVOCATION": "NO_PRE_REALIZATION_REVOCATION",
        "NO_TRUTH_CREDIT_SPLIT": "NO_ATTRIBUTION_SPLIT",
        "NO_SUPPORT_GRAPH": "NO_SUPPORT_VALIDITY",
    }
    return make_phase4_arm(mapping[name], "PHASE5_LIBERO", ("TASK_GOAL",), {"TASK_GOAL": ()}, {"TASK_GOAL": []}, unit_id)


def event_sequence(condition: str, unit_id: str, seed: int) -> List[LedgerEvent]:
    broker = Phase4EventBroker(unit_id, seed)
    request1, command1 = broker.prefix("TASK_GOAL")
    if condition == "C0_CLEAN":
        broker.positive("TASK_GOAL", "base_view", request1.attempt_id, command1.event_id, True)
        broker.positive("TASK_GOAL", "wrist_view", request1.attempt_id, command1.event_id, True)
        broker.witness("TASK_GOAL", request1.attempt_id, command1.event_id, True)
    elif condition == "A1_NOOP_RETRY_STALE":
        broker.positive("TASK_GOAL", "base_view", request1.attempt_id, command1.event_id, False)
        broker.positive("TASK_GOAL", "wrist_view", request1.attempt_id, command1.event_id, False)
        broker.prefix("TASK_GOAL")
        broker.witness("TASK_GOAL", request1.attempt_id, command1.event_id, False)
    elif condition == "A2_CROSS_ATTEMPT_MIX":
        broker.positive("TASK_GOAL", "base_view", request1.attempt_id, command1.event_id, False)
        request2, command2 = broker.prefix("TASK_GOAL")
        broker.positive("TASK_GOAL", "wrist_view", request2.attempt_id, command2.event_id, False, force_verification=True)
        broker.witness("TASK_GOAL", request2.attempt_id, command2.event_id, False)
    elif condition == "A3_CONTRADICTION_LATE_WITNESS":
        broker.positive("TASK_GOAL", "base_view", request1.attempt_id, command1.event_id, True)
        broker.positive("TASK_GOAL", "wrist_view", request1.attempt_id, command1.event_id, True)
        broker.contradiction("TASK_GOAL", False)
        broker.witness("TASK_GOAL", request1.attempt_id, command1.event_id, False)
    elif condition == "A4_POST_REALIZATION_REVERSAL":
        broker.positive("TASK_GOAL", "base_view", request1.attempt_id, command1.event_id, True)
        broker.positive("TASK_GOAL", "wrist_view", request1.attempt_id, command1.event_id, True)
        broker.witness("TASK_GOAL", request1.attempt_id, command1.event_id, True)
        broker.contradiction("TASK_GOAL", False)
        broker.witness("TASK_GOAL", request1.attempt_id, command1.event_id, False)
    elif condition == "A5_EXTERNAL_REALIZATION":
        broker.external_realization("TASK_GOAL")
    elif condition == "V1_SINGLE_VIEW_FALSE_POSITIVE":
        broker.positive("TASK_GOAL", "base_view", request1.attempt_id, command1.event_id, False)
        broker.positive("TASK_GOAL", "base_view", request1.attempt_id, command1.event_id, False, force_verification=True)
        broker.witness("TASK_GOAL", request1.attempt_id, command1.event_id, False)
    else:
        raise ValueError(condition)
    return broker.records


def evaluate_arm(name: str, condition: str, unit_id: str, seed: int) -> dict:
    arm = _make(name, unit_id)
    decisions = []
    for event in event_sequence(condition, unit_id, seed):
        decisions.append(arm.process(event).value)
    verified = bool(arm.effect_fact_verified("TASK_GOAL"))
    credited = bool(arm.attempt_attributed_success("TASK_GOAL"))
    physical_truth_at_decision = condition in ("C0_CLEAN", "A5_EXTERNAL_REALIZATION")
    false_advance = verified and not physical_truth_at_decision
    return {
        "arm": name,
        "condition": condition,
        "verified": verified,
        "credited": credited,
        "physical_truth_at_decision": physical_truth_at_decision,
        "false_grounded_advance": false_advance,
        "final_decision": decisions[-1],
        "event_count": len(decisions),
    }

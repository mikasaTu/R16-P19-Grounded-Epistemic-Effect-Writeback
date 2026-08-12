"""Preregistered actor-free trace construction and seven-arm evaluation."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import write_json, write_jsonl
from .config import EXPERIMENT_SOURCE, TASKS
from .memory import MemoryArm
from .ontology import load_ontology
from .simulator import DemoLabels, deterministic_target_effect, frame_digest
from .types import (
    Decision,
    EpistemicState,
    Event,
    EventType,
    EvidenceReceipt,
    canonical_stream_sha256,
)


CONDITIONS = tuple("C%d" % value for value in range(8))
ARMS = tuple("B%d" % value for value in range(1, 8))
NON_ORACLE_ARMS = ARMS[:-1]
PHYSICAL_EVENT_TYPES = {
    EventType.OBSERVE_POSITIVE,
    EventType.VERIFY_POSITIVE,
    EventType.REALIZATION_WITNESS,
    EventType.OBSERVE_NEGATIVE,
    EventType.CONTRADICTION,
}
EXPECTED_DECISION = {
    "C0": Decision.ADVANCE_TO_NEXT_SUBTASK,
    "C1": Decision.RETRY_CURRENT_EFFECT,
    "C2": Decision.ADVANCE_TO_NEXT_SUBTASK,
    "C3": Decision.ROLLBACK_OR_REPLAN,
    "C4": Decision.REOBSERVE,
    "C5": Decision.ADVANCE_TO_NEXT_SUBTASK,
    "C6": Decision.ADVANCE_TO_NEXT_SUBTASK,
    "C7": Decision.RETRY_CURRENT_EFFECT,
}
FINAL_TRUTH = {
    "C0": True,
    "C1": False,
    "C2": True,
    "C3": False,
    "C4": False,
    "C5": True,
    "C6": True,
    "C7": False,
}


class EventBuilder:
    def __init__(self, task_key: str, episode_id: str, condition: str, effect_id: str):
        self.task_key = task_key
        self.episode_id = "%s:%s:%s" % (task_key, episode_id, condition)
        self.source_episode = episode_id
        self.condition = condition
        self.effect_id = effect_id
        self.events: List[Event] = []

    def add(
        self,
        event_type: EventType,
        *,
        frame_index: Optional[int] = None,
        sensor: str = "agentview",
        digest_override: Optional[str] = None,
        evidence_id_suffix: str = "",
        parent_ids: Optional[Sequence[str]] = None,
        payload: Optional[Mapping[str, object]] = None,
        effect_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Event:
        index = len(self.events)
        active_effect = (
            None
            if event_type == EventType.IRRELEVANT
            else (self.effect_id if effect_id is None else effect_id)
        )
        event_id = "%s:e%03d:%s" % (self.episode_id, index, event_type.value)
        if parent_ids is None:
            parent_ids = (self.events[-1].event_id,) if self.events else ()
        receipt = None
        if event_type in PHYSICAL_EVENT_TYPES:
            if frame_index is None:
                raise ValueError("physical event requires source frame")
            digest = digest_override or frame_digest(
                TASKS[self.task_key], self.source_episode, frame_index, sensor
            )
            receipt = EvidenceReceipt(
                evidence_id=event_id + ":receipt" + evidence_id_suffix,
                episode_id=self.episode_id,
                event_index=index,
                timestamp=float(timestamp if timestamp is not None else index),
                sensor_identity=sensor,
                frame_digest=digest,
                effect_id=active_effect,
                evidence_type=event_type.value,
            )
        event = Event(
            event_id=event_id,
            episode_id=self.episode_id,
            event_index=index,
            timestamp=float(timestamp if timestamp is not None else index),
            event_type=event_type,
            effect_id=active_effect,
            parent_ids=tuple(parent_ids),
            receipt=receipt,
            payload=dict(payload or {}),
        )
        event.validate()
        self.events.append(event)
        return event


def _positive_sequence(builder: EventBuilder, transition: int, length: int, alias_first: bool = False) -> None:
    command = next(event for event in reversed(builder.events) if event.event_type == EventType.COMMAND)
    observe = builder.add(
        EventType.OBSERVE_POSITIVE,
        frame_index=min(transition, length - 1),
        sensor="agentview",
        parent_ids=(command.event_id,),
    )
    if alias_first:
        builder.add(
            EventType.VERIFY_POSITIVE,
            frame_index=min(transition, length - 1),
            sensor="robot0_eye_in_hand",
            digest_override=observe.receipt.frame_digest,
            evidence_id_suffix=":alias",
            parent_ids=(observe.event_id,),
            payload={"fault": "same_physical_frame_new_evidence_id"},
        )
    verify = builder.add(
        EventType.VERIFY_POSITIVE,
        frame_index=min(transition, length - 1),
        sensor="robot0_eye_in_hand",
        parent_ids=(observe.event_id,),
    )
    builder.add(
        EventType.REALIZATION_WITNESS,
        frame_index=min(transition + 1, length - 1),
        sensor="agentview",
        parent_ids=(command.event_id, verify.event_id),
    )


def build_condition_stream(labels: DemoLabels, condition: str) -> Tuple[List[Event], str]:
    task = TASKS[labels.task_key]
    target_index = deterministic_target_effect(labels.task_key, labels.episode_id, condition)
    effect = task.effects[target_index]
    transition_map = dict(labels.stable_transition_indices)
    transition_map.update(
        {key: value for key, value in labels.transition_indices.items() if key not in transition_map}
    )
    if effect not in transition_map:
        raise RuntimeError("source demo lacks transition for %s" % effect)
    transition = int(transition_map[effect])
    pre = max(0, transition - 2)
    builder = EventBuilder(labels.task_key, labels.episode_id, condition, effect)
    builder.add(EventType.REQUEST)
    if condition == "C7":
        builder.add(EventType.IMAGINE, payload={"predicted_success": True})
    else:
        builder.add(EventType.IMAGINE, payload={"predicted_success": False})
    command = builder.add(EventType.COMMAND)

    if condition == "C0":
        _positive_sequence(builder, transition, labels.length)
    elif condition == "C1":
        builder.add(EventType.TIMEOUT, timestamp=20.0, payload={"fault": "command_noop"})
    elif condition == "C2":
        builder.add(EventType.TIMEOUT, timestamp=4.0, payload={"delay_cycles": 4})
        _positive_sequence(builder, transition, labels.length)
    elif condition == "C3":
        _positive_sequence(builder, transition, labels.length)
        contradiction = builder.add(
            EventType.CONTRADICTION,
            frame_index=pre,
            sensor="agentview",
            parent_ids=(builder.events[-1].event_id,),
            payload={"fault": "post_realization_reversal"},
        )
        # Exact replay checks ledger duplicate-event idempotency.
        builder.events.append(contradiction)
        # A semantically repeated contradiction with a fresh ID must also be idempotent.
        builder.add(
            EventType.CONTRADICTION,
            frame_index=pre,
            sensor="agentview",
            parent_ids=(contradiction.event_id,),
            payload={"fault": "post_realization_reversal", "repeat": True},
        )
    elif condition == "C4":
        builder.add(
            EventType.OBSERVE_POSITIVE,
            frame_index=pre,
            sensor="agentview",
            parent_ids=(command.event_id,),
            payload={"fault": "single_camera_false_positive"},
        )
    elif condition == "C5":
        _positive_sequence(builder, transition, labels.length, alias_first=True)
    elif condition == "C6":
        _positive_sequence(builder, transition, labels.length)
        for offset in range(40):
            builder.add(
                EventType.IRRELEVANT,
                effect_id=None,
                payload={"pressure_event": offset},
            )
    elif condition == "C7":
        builder.add(
            EventType.OBSERVE_NEGATIVE,
            frame_index=pre,
            sensor="agentview",
            parent_ids=(command.event_id,),
            payload={"fault": "imagined_success_observed_failure"},
        )
        builder.add(EventType.TIMEOUT, timestamp=20.0)
    else:
        raise ValueError("unknown condition %s" % condition)
    return builder.events, effect


def _oracle_events(events: Sequence[Event], condition: str) -> List[Event]:
    transformed = []
    last_index = len(events) - 1
    for index, event in enumerate(events):
        payload = dict(event.payload)
        if event.event_type in (
            EventType.OBSERVE_POSITIVE,
            EventType.VERIFY_POSITIVE,
            EventType.REALIZATION_WITNESS,
        ):
            payload["physical_truth"] = condition != "C4"
        elif event.event_type in (EventType.OBSERVE_NEGATIVE, EventType.CONTRADICTION):
            payload["physical_truth"] = False
        if index == last_index:
            payload["oracle_decision"] = EXPECTED_DECISION[condition].value
        transformed.append(replace(event, payload=payload))
    return transformed


def _latency_summary(values_ns: Sequence[int]) -> Tuple[float, float]:
    values_us = np.asarray(values_ns, dtype=np.float64) / 1000.0
    return float(np.percentile(values_us, 50)), float(np.percentile(values_us, 95))


def run_trace_gate(labels: Iterable[DemoLabels], output_dir: Path) -> dict:
    ontology = load_ontology()
    output_dir = Path(output_dir)
    event_rows: List[dict] = []
    output_rows: List[dict] = []
    case_rows: List[dict] = []
    stream_hashes: Dict[str, Dict[str, str]] = {}
    for source in labels:
        for condition in CONDITIONS:
            events, effect = build_condition_stream(source, condition)
            stream_key = "%s:%s:%s" % (source.task_key, source.episode_id, condition)
            stream_hash = canonical_stream_sha256(events)
            stream_hashes[stream_key] = {arm: stream_hash for arm in NON_ORACLE_ARMS}
            for event in events:
                row = event.to_dict()
                row.update(
                    {
                        "task_key": source.task_key,
                        "source_episode": source.episode_id,
                        "condition": condition,
                        "non_oracle_stream_sha256": stream_hash,
                    }
                )
                event_rows.append(row)
            for arm_name in ARMS:
                arm = MemoryArm(arm_name, source.task_key, ontology)
                selected_events = _oracle_events(events, condition) if arm_name == "B7" else events
                latencies = []
                first_contradiction_fingerprint = None
                repeat_contradiction_fingerprint = None
                final_decision = Decision.SAFE_STOP
                for event in selected_events:
                    started = time.perf_counter_ns()
                    final_decision = arm.process(event, effect)
                    latencies.append(time.perf_counter_ns() - started)
                    if event.event_type == EventType.CONTRADICTION:
                        fingerprint = arm.semantic_fingerprint(effect)
                        if first_contradiction_fingerprint is None:
                            first_contradiction_fingerprint = fingerprint
                        else:
                            repeat_contradiction_fingerprint = fingerprint
                    output_rows.append(
                        {
                            "task_key": source.task_key,
                            "source_episode": source.episode_id,
                            "condition": condition,
                            "arm": arm_name,
                            "event_id": event.event_id,
                            "decision": final_decision.value,
                            "effect_state": (
                                arm.records[effect].state.value
                                if arm.records[effect].state is not None
                                else None
                            ),
                            "resident_slot_count": len(arm.ledger.resident),
                        }
                    )
                summary = arm.current_summary()
                state = arm.records[effect].state
                predicted_realized = state == EpistemicState.REALIZED
                p50, p95 = _latency_summary(latencies)
                contradiction_idempotent = (
                    first_contradiction_fingerprint == repeat_contradiction_fingerprint
                    if condition == "C3"
                    else True
                )
                case_rows.append(
                    {
                        "task_key": source.task_key,
                        "source_episode": source.episode_id,
                        "condition": condition,
                        "arm": arm_name,
                        "target_effect": effect,
                        "expected_decision": EXPECTED_DECISION[condition].value,
                        "decision": final_decision.value,
                        "decision_correct": final_decision == EXPECTED_DECISION[condition],
                        "physical_realized": FINAL_TRUTH[condition],
                        "predicted_realized": predicted_realized,
                        "false_completion": predicted_realized and not FINAL_TRUTH[condition],
                        "premature_advance": (
                            final_decision == Decision.ADVANCE_TO_NEXT_SUBTASK
                            and not FINAL_TRUTH[condition]
                        ),
                        "contradiction_detected": (
                            condition == "C3"
                            and state == EpistemicState.INVALIDATED_REALIZATION
                        ),
                        "contradiction_recovered": (
                            condition == "C3"
                            and bool(arm.records[effect].recovery_route)
                            and final_decision == Decision.ROLLBACK_OR_REPLAN
                        ),
                        "recovery_routing_correct": (
                            final_decision == EXPECTED_DECISION[condition]
                            if condition in ("C1", "C3", "C7")
                            else True
                        ),
                        "unnecessary_recovery": (
                            FINAL_TRUTH[condition]
                            and final_decision
                            in (Decision.RETRY_CURRENT_EFFECT, Decision.ROLLBACK_OR_REPLAN)
                        ),
                        "evidence_alias_acceptance": summary["alias_acceptances"],
                        "evidence_alias_rejections": summary["alias_rejections"],
                        "dangling_parent_count": summary["dangling_parent_count"],
                        "duplicate_event_count": summary["duplicate_event_count"],
                        "duplicate_contradiction_idempotent": contradiction_idempotent,
                        "resident_slot_count": summary["resident_slot_count"],
                        "resident_slot_count_max": summary["resident_slot_count_max"],
                        "latency_p50_us": p50,
                        "latency_p95_us": p95,
                        "transition_violations": summary["transition_violations"],
                    }
                )

    write_jsonl(output_dir / "trace_events.jsonl", event_rows)
    write_jsonl(output_dir / "memory_outputs.jsonl", output_rows)
    write_jsonl(output_dir / "trace_case_results.jsonl", case_rows)
    write_json(output_dir / "non_oracle_stream_hashes.json", stream_hashes)
    metrics = summarize_trace_metrics(case_rows)
    write_json(output_dir / "actor_free_metrics.json", metrics)
    return metrics


def _rate(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def summarize_trace_metrics(rows: Sequence[Mapping[str, object]]) -> dict:
    by_arm = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        true_positive = sum(row["predicted_realized"] and row["physical_realized"] for row in subset)
        predicted_positive = sum(row["predicted_realized"] for row in subset)
        actual_positive = sum(row["physical_realized"] for row in subset)
        contradiction = [row for row in subset if row["condition"] == "C3"]
        clean = [row for row in subset if row["condition"] == "C0"]
        latency50 = [float(row["latency_p50_us"]) for row in subset]
        latency95 = [float(row["latency_p95_us"]) for row in subset]
        by_arm[arm] = {
            "case_count": len(subset),
            "false_completion_rate": _rate(subset, "false_completion"),
            "premature_advance_rate": _rate(subset, "premature_advance"),
            "realized_precision": true_positive / max(predicted_positive, 1),
            "realized_recall": true_positive / max(actual_positive, 1),
            "contradiction_detection_recall": _rate(contradiction, "contradiction_detected"),
            "contradiction_recovery_recall": _rate(contradiction, "contradiction_recovered"),
            "recovery_routing_accuracy": _rate(
                [row for row in subset if row["condition"] in ("C1", "C3", "C7")],
                "recovery_routing_correct",
            ),
            "evidence_alias_acceptance": int(
                sum(int(row["evidence_alias_acceptance"]) for row in subset)
            ),
            "evidence_alias_rejections": int(
                sum(int(row["evidence_alias_rejections"]) for row in subset)
            ),
            "dangling_parent_count": int(sum(int(row["dangling_parent_count"]) for row in subset)),
            "duplicate_event_idempotency": _rate(subset, "duplicate_contradiction_idempotent"),
            "clean_decision_accuracy": _rate(clean, "decision_correct"),
            "unnecessary_recovery_rate": _rate(subset, "unnecessary_recovery"),
            "resident_slot_count_max": max(int(row["resident_slot_count_max"]) for row in subset),
            "latency_p50_us": float(np.percentile(latency50, 50)),
            "latency_p95_us": float(np.percentile(latency95, 95)),
            "decision_accuracy": _rate(subset, "decision_correct"),
            "safe_or_reobserve_rate": float(
                np.mean(
                    [row["decision"] in (Decision.SAFE_STOP.value, Decision.REOBSERVE.value) for row in subset]
                )
            ),
            "transition_violation_count": sum(len(row["transition_violations"]) for row in subset),
        }
    b6 = by_arm["B6"]
    gates = {
        "evidence_alias_acceptance_zero": b6["evidence_alias_acceptance"] == 0,
        "evidence_alias_was_actually_challenged": b6["evidence_alias_rejections"] > 0,
        "dangling_parent_count_zero": b6["dangling_parent_count"] == 0,
        "duplicate_contradictions_idempotent": b6["duplicate_event_idempotency"] == 1.0,
        "contradicted_realized_has_recovery": b6["contradiction_recovery_recall"] == 1.0,
        "resident_slots_at_most_32": all(value["resident_slot_count_max"] <= 32 for value in by_arm.values()),
        "B6_outperforms_B4_B5_contradiction_recovery": (
            b6["contradiction_recovery_recall"] > by_arm["B4"]["contradiction_recovery_recall"]
            and b6["contradiction_recovery_recall"] > by_arm["B5"]["contradiction_recovery_recall"]
        ),
        "nondegenerate_B6_decisions": b6["safe_or_reobserve_rate"] < 0.50,
        "explicit_transition_rules_hold": b6["transition_violation_count"] == 0,
    }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "arms": by_arm,
        "correctness_gates": gates,
        "actor_free_gate_pass": all(gates.values()),
    }

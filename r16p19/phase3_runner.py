"""Qualification, formal replay, and six-arm Phase-3 matrix runner."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

from .artifacts import atomic_text, write_json
from .config import TASKS
from .ontology import load_ontology
from .phase1b_evaluation import _write_video
from .phase3_baselines import ABLATION_ARMS, MAIN_ARMS, make_phase3_arm
from .phase3_event_broker import Phase3EventBroker
from .phase3_replay_backend import (
    ExecutionMode,
    FrozenEffectReplayBackend,
)
from .phase3_snapshot_bank import observation_sha256
from .types import Decision, Event


MAIN_CONDITIONS = ("C0", "C1", "C3", "C4", "C7")
DELAYED_CONDITIONS = ("D1",)
CONDITION_NAMES = {
    "C0": "CLEAN",
    "C1": "COMMAND_NOOP",
    "C3": "POST_REALIZATION_REVERSAL",
    "C4": "SINGLE_VIEW_FALSE_POSITIVE",
    "C7": "IMAGINED_SUCCESS_OBSERVED_FAILURE",
    "D1": "DELAYED_RECEIPT",
}
MAX_RETRIES = 2
MAX_REOBSERVES = 4
MAX_ROLLBACKS = 1
MAX_DECISIONS = 32


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return float(np.mean(materialized)) if materialized else None


def _unit_seed(
    task_key: str, source_episode: str, chain_ordinal: int, condition: str
) -> int:
    task_ordinal = list(TASKS).index(task_key)
    demo_index = int(source_episode.rsplit("_", 1)[1])
    condition_ordinal = list(MAIN_CONDITIONS + DELAYED_CONDITIONS).index(condition)
    return (
        1619
        + task_ordinal * 100000
        + demo_index * 1000
        + int(chain_ordinal) * 100
        + condition_ordinal
    )


def _frame_digests(observation: Mapping[str, object]) -> Dict[str, str]:
    source = observation_sha256(observation)
    return {
        sensor: hashlib.sha256((source + "|" + sensor).encode()).hexdigest()
        for sensor in (
            "agentview",
            "robot0_eye_in_hand",
            "effect_witness",
            "contradiction_sensor",
        )
    }


def _chain_map(contract: Mapping[str, object]) -> Dict[str, dict]:
    return {str(row["chain_id"]): dict(row) for row in contract["candidate_chains"]}


def _target_effect(chain: Mapping[str, object], source_episode: str, condition: str) -> str:
    effects = list(chain["effects"])
    digest = hashlib.sha256(
        (str(chain["chain_id"]) + "|" + source_episode + "|" + condition).encode()
    ).digest()
    return effects[int.from_bytes(digest[:8], "big") % len(effects)]


def paired_action_budget(
    backend: FrozenEffectReplayBackend,
    split: str,
    chain: Mapping[str, object],
    source_episode: str,
) -> int:
    """Return the frozen identical action budget for one paired chain unit."""

    lengths = [
        backend.segment(
            split,
            str(chain["task_key"]),
            source_episode,
            str(effect_id),
        ).action_count
        for effect_id in chain["effects"]
    ]
    return int(4 * sum(lengths) + 8 * MAX_REOBSERVES * len(lengths))


def _intervened_decision(
    natural: Decision,
    decision_index: int,
    forced_decisions: Mapping[int, Decision | str] | None,
) -> tuple[Decision, bool]:
    if forced_decisions is None or decision_index not in forced_decisions:
        return natural, False
    return Decision(forced_decisions[decision_index]), True


def _deliver(
    arm,
    events: Sequence[Event],
    current_effect: str,
) -> Decision:
    decision = Decision.REOBSERVE
    for event in events:
        decision = arm.process(event, current_effect)
    return decision


def _setup_attempt(
    broker: Phase3EventBroker,
    arm,
    effect_id: str,
    decision_index: int,
    *,
    imagined: bool,
) -> None:
    events = [broker.request(effect_id, decision_index)]
    if imagined:
        events.append(broker.imagine_success(effect_id, decision_index))
    events.append(broker.command(effect_id, decision_index))
    _deliver(arm, events, effect_id)


def _decision_state_record(
    backend: FrozenEffectReplayBackend,
    broker: Phase3EventBroker,
    arm,
    task_key: str,
    effect_id: str,
    decision_index: int,
    decision: Decision,
    event_start: int,
    action_trace_length: int,
    *,
    natural_decision: Decision | None = None,
    intervention: bool = False,
) -> dict:
    state = backend.current_state(task_key)
    summary = arm.current_summary()
    return {
        "decision_index": int(decision_index),
        "effect_id": effect_id,
        "decision": decision.value,
        "natural_decision": (natural_decision or decision).value,
        "decision_intervention": bool(intervention),
        "physical_truth_evaluation_only": backend.current_truth(task_key, effect_id),
        "simulator_state": state.tolist(),
        "simulator_state_sha256": hashlib.sha256(
            np.asarray(state, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "event_record_start": int(event_start),
        "event_record_stop_exclusive": len(broker.records),
        "event_prefix_sha256": broker.stream_hash(),
        "action_trace_length": int(action_trace_length),
        "memory_summary": summary,
    }


def _evidence_after_execution(
    backend: FrozenEffectReplayBackend,
    broker: Phase3EventBroker,
    arm,
    task_key: str,
    effect_id: str,
    decision_index: int,
    *,
    force_single_view_positive: bool = False,
) -> Decision:
    digests = _frame_digests(backend.current_observation(task_key))
    if force_single_view_positive:
        event = broker.single_view_false_positive(
            effect_id, decision_index, digests["agentview"]
        )
        return _deliver(arm, [event], effect_id)
    if backend.current_truth(task_key, effect_id):
        return _deliver(
            arm,
            broker.positive_receipts(effect_id, decision_index, digests),
            effect_id,
        )
    return _deliver(
        arm,
        broker.negative_receipts(effect_id, decision_index, digests),
        effect_id,
    )


def _execute_segment(
    backend: FrozenEffectReplayBackend,
    split: str,
    chain: Mapping[str, object],
    source_episode: str,
    effect_id: str,
    mode: ExecutionMode,
    seed: int,
    *,
    suppress_actions: bool,
    collect_frames: bool,
):
    segment = backend.segment(split, str(chain["task_key"]), source_episode, effect_id)
    return backend.execute_effect(
        split,
        str(chain["task_key"]),
        source_episode,
        effect_id,
        backend.snapshot_id(segment),
        mode,
        seed=seed,
        suppress_actions=suppress_actions,
        collect_frames=collect_frames,
    )


def run_chain_rollout(
    backend: FrozenEffectReplayBackend,
    split: str,
    chain: Mapping[str, object],
    chain_ordinal: int,
    source_episode: str,
    condition: str,
    arm_name: str,
    persistence_k: int,
    *,
    save_video_path: Path | None = None,
    forced_decisions: Mapping[int, Decision | str] | None = None,
) -> dict:
    if condition not in MAIN_CONDITIONS + DELAYED_CONDITIONS:
        raise ValueError("unknown condition %s" % condition)
    task_key = str(chain["task_key"])
    effects = tuple(str(value) for value in chain["effects"])
    unit_id = "%s|%s|%s" % (chain["chain_id"], source_episode, condition)
    seed = _unit_seed(task_key, source_episode, chain_ordinal, condition)
    ontology = load_ontology()
    arm = make_phase3_arm(
        arm_name, task_key, ontology, TASKS[task_key].effects, persistence_k
    )
    broker = Phase3EventBroker(unit_id, source_episode)
    target_effect = _target_effect(chain, source_episode, condition)
    collect_frames = save_video_path is not None
    frames: List[np.ndarray] = []
    action_trace: List[dict] = []
    decision_trace: List[dict] = []
    effect_success = {effect: False for effect in effects}
    action_steps = 0
    retry_count = 0
    reobserve_count = 0
    rollback_count = 0
    false_completion_count = 0
    advance_truths: List[bool] = []
    failure_type = None
    recovery_attempted = False
    recovery_success = False
    contradiction_presented = False
    contradiction_detected = False
    contradiction_recovery_decision = False
    imagined_as_realized = False
    invalidated_realization_correct = None
    fault_consumed = False
    decision_index = 0
    action_budget = paired_action_budget(backend, split, chain, source_episode)
    action_budget_exceeded = False

    def record_execution(execution) -> bool:
        nonlocal action_steps, action_budget_exceeded
        action_steps += execution.action_steps
        frames.extend(execution.frames)
        action_trace.append(execution.public_record())
        if action_steps > action_budget:
            action_budget_exceeded = True
            return False
        return True

    for effect_position, effect_id in enumerate(effects):
        effect_retry_count = 0
        effect_reobserve_count = 0
        effect_rollback_count = 0
        initial_target_fault = condition != "C0" and effect_id == target_effect
        _setup_attempt(
            broker,
            arm,
            effect_id,
            decision_index,
            imagined=bool(condition == "C7" and initial_target_fault),
        )
        suppress = bool(
            initial_target_fault and condition in ("C1", "C4", "C7")
        )
        execution = _execute_segment(
            backend,
            split,
            chain,
            source_episode,
            effect_id,
            ExecutionMode.EXECUTE,
            seed,
            suppress_actions=suppress,
            collect_frames=collect_frames,
        )
        if not record_execution(execution):
            failure_type = "TIMEOUT"
            break
        if suppress:
            fault_consumed = True

        if condition == "D1" and initial_target_fault and execution.physical_truth_after:
            advanced_during_delay = False
            for _ in range(3):
                event_start = len(broker.records)
                delayed_event = broker.delayed_tick(decision_index)
                natural_decision = _deliver(arm, [delayed_event], effect_id)
                decision, intervened = _intervened_decision(
                    natural_decision, decision_index, forced_decisions
                )
                decision_trace.append(
                    _decision_state_record(
                        backend,
                        broker,
                        arm,
                        task_key,
                        effect_id,
                        decision_index,
                        decision,
                        event_start,
                        len(action_trace),
                        natural_decision=natural_decision,
                        intervention=intervened,
                    )
                )
                decision_index += 1
                if decision == Decision.ADVANCE_TO_NEXT_SUBTASK:
                    truth = backend.current_truth(task_key, effect_id)
                    advance_truths.append(truth)
                    if not truth:
                        false_completion_count += 1
                        failure_type = "PREMATURE_ADVANCE"
                    else:
                        effect_success[effect_id] = True
                        advanced_during_delay = True
                    break
                if decision == Decision.REOBSERVE:
                    if effect_reobserve_count >= MAX_REOBSERVES:
                        failure_type = "TIMEOUT"
                        break
                    effect_reobserve_count += 1
                    reobserve_count += 1
                    delayed = _execute_segment(
                        backend,
                        split,
                        chain,
                        source_episode,
                        effect_id,
                        ExecutionMode.REOBSERVE,
                        seed,
                        suppress_actions=False,
                        collect_frames=collect_frames,
                    )
                    if not record_execution(delayed):
                        failure_type = "TIMEOUT"
                        break
                elif decision in (Decision.RETRY_CURRENT_EFFECT, Decision.ROLLBACK_OR_REPLAN):
                    recovery_attempted = True
                    mode = (
                        ExecutionMode.RETRY
                        if decision == Decision.RETRY_CURRENT_EFFECT
                        else ExecutionMode.ROLLBACK_REPLAY
                    )
                    if mode == ExecutionMode.RETRY:
                        if effect_retry_count >= MAX_RETRIES:
                            failure_type = "TIMEOUT"
                            break
                        effect_retry_count += 1
                        retry_count += 1
                    else:
                        if effect_rollback_count >= MAX_ROLLBACKS:
                            failure_type = "TIMEOUT"
                            break
                        effect_rollback_count += 1
                        rollback_count += 1
                    _setup_attempt(
                        broker, arm, effect_id, decision_index, imagined=False
                    )
                    delayed = _execute_segment(
                        backend,
                        split,
                        chain,
                        source_episode,
                        effect_id,
                        mode,
                        seed,
                        suppress_actions=False,
                        collect_frames=collect_frames,
                    )
                    if not record_execution(delayed):
                        failure_type = "TIMEOUT"
                        break
                elif decision == Decision.SAFE_STOP:
                    failure_type = "OVERCONSERVATIVE_STOP"
                    break
            if failure_type:
                break
            if advanced_during_delay:
                continue

        event_start = len(broker.records)
        natural_decision = _evidence_after_execution(
            backend,
            broker,
            arm,
            task_key,
            effect_id,
            decision_index,
            force_single_view_positive=bool(
                condition == "C4" and initial_target_fault
            ),
        )
        decision, intervened = _intervened_decision(
            natural_decision, decision_index, forced_decisions
        )

        if (
            condition == "C3"
            and initial_target_fault
            and execution.physical_truth_after
        ):
            segment = backend.segment(split, task_key, source_episode, effect_id)
            backend.reset_to_segment(segment, seed=seed)
            if backend.current_truth(task_key, effect_id):
                failure_type = "FAULT_INJECTOR_ERROR"
                break
            digests = _frame_digests(backend.current_observation(task_key))
            contradiction = broker.contradiction(
                effect_id, decision_index, digests["contradiction_sensor"]
            )
            natural_decision = _deliver(arm, [contradiction], effect_id)
            decision, intervened = _intervened_decision(
                natural_decision, decision_index, forced_decisions
            )
            contradiction_presented = True
            contradiction_recovery_decision = decision in (
                Decision.ROLLBACK_OR_REPLAN,
                Decision.RETRY_CURRENT_EFFECT,
            )
            contradiction_detected = natural_decision != Decision.ADVANCE_TO_NEXT_SUBTASK
            invalidated_realization_correct = bool(
                arm.current_summary().get("effects", {})
                .get(effect_id, {})
                .get("state")
                == "INVALIDATED_REALIZATION"
            ) if arm_name in ("B6_FULL", "B6_NO_PROVENANCE") else None
            fault_consumed = True

        while True:
            if decision_index >= MAX_DECISIONS:
                failure_type = "TIMEOUT"
                break
            decision_trace.append(
                _decision_state_record(
                    backend,
                    broker,
                    arm,
                    task_key,
                    effect_id,
                    decision_index,
                    decision,
                    event_start,
                    len(action_trace),
                    natural_decision=natural_decision,
                    intervention=intervened,
                )
            )
            decision_index += 1
            if decision == Decision.ADVANCE_TO_NEXT_SUBTASK:
                truth = backend.current_truth(task_key, effect_id)
                advance_truths.append(truth)
                if condition == "C7" and initial_target_fault and not truth:
                    imagined_as_realized = True
                if not truth:
                    false_completion_count += 1
                    failure_type = "PREMATURE_ADVANCE"
                else:
                    effect_success[effect_id] = True
                    if recovery_attempted and initial_target_fault:
                        recovery_success = True
                break
            if decision == Decision.SAFE_STOP:
                failure_type = "OVERCONSERVATIVE_STOP"
                break
            if decision == Decision.RETRY_CURRENT_EFFECT:
                if effect_retry_count >= MAX_RETRIES:
                    failure_type = "TIMEOUT"
                    break
                effect_retry_count += 1
                retry_count += 1
                recovery_attempted = recovery_attempted or initial_target_fault
                mode = ExecutionMode.RETRY
            elif decision == Decision.ROLLBACK_OR_REPLAN:
                if effect_rollback_count >= MAX_ROLLBACKS:
                    failure_type = "TIMEOUT"
                    break
                effect_rollback_count += 1
                rollback_count += 1
                recovery_attempted = recovery_attempted or initial_target_fault
                mode = ExecutionMode.ROLLBACK_REPLAY
            elif decision == Decision.REOBSERVE:
                if effect_reobserve_count >= MAX_REOBSERVES:
                    failure_type = "TIMEOUT"
                    break
                effect_reobserve_count += 1
                reobserve_count += 1
                mode = ExecutionMode.REOBSERVE
            else:
                raise AssertionError(decision)

            if mode != ExecutionMode.REOBSERVE:
                _setup_attempt(
                    broker, arm, effect_id, decision_index, imagined=False
                )
            next_execution = _execute_segment(
                backend,
                split,
                chain,
                source_episode,
                effect_id,
                mode,
                seed,
                suppress_actions=False,
                collect_frames=collect_frames,
            )
            if not record_execution(next_execution):
                failure_type = "TIMEOUT"
                break
            event_start = len(broker.records)
            natural_decision = _evidence_after_execution(
                backend,
                broker,
                arm,
                task_key,
                effect_id,
                decision_index,
                force_single_view_positive=False,
            )
            decision, intervened = _intervened_decision(
                natural_decision, decision_index, forced_decisions
            )
        if failure_type:
            break

    chain_success = bool(all(effect_success.values()) and failure_type is None)
    if save_video_path is not None and frames:
        _write_video(Path(save_video_path), frames)
    summary = arm.current_summary()
    first_effect, second_effect = effects
    return {
        "record_type": "phase3_effect_boundary_chain_rollout",
        "split": split,
        "chain_id": chain["chain_id"],
        "task_key": task_key,
        "source_episode": source_episode,
        "condition": condition,
        "condition_name": CONDITION_NAMES[condition],
        "arm": arm_name,
        "persistence_k": persistence_k if arm_name == "PERSISTENCE_RECOVERY" else None,
        "target_effect": target_effect,
        "rollout_seed": seed,
        "chain_success": chain_success,
        "current_effect_success": bool(effect_success[first_effect]),
        "next_effect_success": bool(effect_success[second_effect]),
        "effect_success": effect_success,
        "faulted_chain_success": chain_success if condition in ("C1", "C3", "C4", "C7") else None,
        "clean_chain_success": chain_success if condition == "C0" else None,
        "recovery_success": bool(recovery_success),
        "recovery_attempted": bool(recovery_attempted),
        "repeated_loop": failure_type == "TIMEOUT" and retry_count >= MAX_RETRIES,
        "grounded_advance_true_count": int(sum(advance_truths)),
        "advance_count": len(advance_truths),
        "false_completion": bool(false_completion_count),
        "false_completion_count": false_completion_count,
        "premature_advance": failure_type == "PREMATURE_ADVANCE",
        "contradiction_presented": contradiction_presented,
        "contradiction_detected": contradiction_detected,
        "contradiction_recovery_decision": contradiction_recovery_decision,
        "single_view_false_positive_advance": bool(
            condition == "C4" and failure_type == "PREMATURE_ADVANCE"
        ),
        "imagined_as_realized": imagined_as_realized,
        "invalidated_realization_accuracy": invalidated_realization_correct,
        "retry_count": retry_count,
        "reobserve_count": reobserve_count,
        "rollback_count": rollback_count,
        "unnecessary_retry": bool(condition == "C0" and retry_count),
        "unnecessary_recovery": bool(
            condition == "C0" and (retry_count or rollback_count)
        ),
        "safe_stop": failure_type == "OVERCONSERVATIVE_STOP",
        "action_steps": action_steps,
        "action_budget": action_budget,
        "action_budget_exceeded": action_budget_exceeded,
        "completion_latency": action_steps,
        "failure_type": failure_type,
        "fault_consumed": fault_consumed,
        "event_stream_sha256": broker.stream_hash(),
        "event_records": broker.event_records(),
        "decision_trace": decision_trace,
        "action_trace": action_trace,
        "final_memory_state": summary,
        "resident_slot_count_max": summary.get("resident_slot_count_max", 0),
        "dangling_parent_count": summary.get("dangling_parent_count", 0),
        "transition_violation_count": len(summary.get("transition_violations", [])),
        "fault_or_truth_leakage_count": sum(
            int(
                bool(
                    {"physical_truth", "fault_identity", "condition"}.intersection(
                        record["event"].get("payload", {})
                    )
                )
            )
            for record in broker.event_records()
        ),
        "video_path": str(save_video_path) if save_video_path is not None else None,
    }


def run_replay_qualification(
    backend: FrozenEffectReplayBackend,
    split: str,
    chains: Sequence[Mapping[str, object]],
    source_episodes: Sequence[str],
    output_path: Path,
    repetitions: int = 5,
) -> tuple[List[dict], dict]:
    output_path = Path(output_path)
    rows = _read_jsonl(output_path)
    unique = []
    for chain in chains:
        for effect_id in chain["effects"]:
            key = (chain["task_key"], effect_id)
            if key not in unique:
                unique.append(key)
    expected = {
        (task_key, episode, effect_id, repetition)
        for task_key, effect_id in unique
        for episode in source_episodes
        for repetition in range(int(repetitions))
    }
    observed = {
        (row["task_key"], row["source_episode"], row["effect_id"], row["repetition"])
        for row in rows
    }
    if len(observed) != len(rows) or not observed.issubset(expected):
        raise RuntimeError("replay qualification resume contains duplicate or unexpected rows")
    for task_key, effect_id in unique:
        for source_episode in source_episodes:
            for repetition in range(int(repetitions)):
                key = (task_key, source_episode, effect_id, repetition)
                if key in observed:
                    continue
                success = False
                error = None
                result_record = None
                try:
                    segment = backend.segment(split, task_key, source_episode, effect_id)
                    result = backend.execute_effect(
                        split,
                        task_key,
                        source_episode,
                        effect_id,
                        backend.snapshot_id(segment),
                        ExecutionMode.EXECUTE,
                        seed=1619 + repetition,
                    )
                    success = bool(
                        result.physical_truth_after
                        and result.predicate_stability_duration >= 5
                    )
                    result_record = result.public_record()
                except Exception as exc:  # invalid/missing segment is a scientific failure row
                    error = "%s:%s" % (type(exc).__name__, exc)
                row = {
                    "record_type": "phase3_replay_qualification_effect",
                    "split": split,
                    "task_key": task_key,
                    "source_episode": source_episode,
                    "effect_id": effect_id,
                    "repetition": repetition,
                    "success": success,
                    "error": error,
                    "execution": result_record,
                }
                _append_jsonl(output_path, row)
                rows.append(row)
                observed.add(key)
                print(
                    "PHASE3_REPLAY_CELL split=%s task=%s episode=%s effect=%s rep=%d success=%s"
                    % (split, task_key, source_episode, effect_id, repetition, success),
                    flush=True,
                )

    if observed != expected:
        raise RuntimeError("replay qualification did not complete exact frozen grid")

    segment_success: Dict[tuple[str, str, str], float] = {}
    for task_key, effect_id in unique:
        for source_episode in source_episodes:
            subset = [
                row
                for row in rows
                if row["task_key"] == task_key
                and row["effect_id"] == effect_id
                and row["source_episode"] == source_episode
            ]
            segment_success[(task_key, source_episode, effect_id)] = float(
                np.mean([row["success"] for row in subset])
            )
    chain_rows = []
    for chain in chains:
        values = []
        cumulative_reach = [0 for _ in chain["effects"]]
        action_steps = []
        for source_episode in source_episodes:
            for repetition in range(int(repetitions)):
                effects_ok = []
                for effect_id in chain["effects"]:
                    row = next(
                        value
                        for value in rows
                        if value["task_key"] == chain["task_key"]
                        and value["source_episode"] == source_episode
                        and value["effect_id"] == effect_id
                        and value["repetition"] == repetition
                    )
                    effects_ok.append(bool(row["success"]))
                    if row.get("execution") is not None:
                        action_steps.append(int(row["execution"]["action_steps"]))
                running = True
                for effect_index, effect_ok in enumerate(effects_ok):
                    running = bool(running and effect_ok)
                    cumulative_reach[effect_index] += int(running)
                values.append(all(effects_ok))
        segment_values = [
            segment_success[(chain["task_key"], episode, effect_id)]
            for episode in source_episodes
            for effect_id in chain["effects"]
        ]
        chain_rows.append(
            {
                "chain_id": chain["chain_id"],
                "task_key": chain["task_key"],
                "effects": list(chain["effects"]),
                "chain_success": float(np.mean(values)),
                "cumulative_chain_reach": [
                    float(value / len(values)) for value in cumulative_reach
                ],
                "minimum_source_segment_success": float(min(segment_values)),
                "mean_action_steps": (
                    float(np.mean(action_steps)) if action_steps else None
                ),
                "eligible": bool(min(segment_values) >= 0.95 and np.mean(values) >= 0.90),
            }
        )
    effect_summary = {}
    for task_key, effect_id in unique:
        subset = [
            row
            for row in rows
            if row["task_key"] == task_key and row["effect_id"] == effect_id
        ]
        effect_summary.setdefault(task_key, {})[effect_id] = float(
            np.mean([row["success"] for row in subset])
        )
    summary = {
        "schema_version": 1,
        "status": "REPLAY_QUALIFICATION_COMPLETE",
        "split": split,
        "effect_replay_count": len(rows),
        "repetitions_per_source_segment": int(repetitions),
        "conditional_effect_success_given_entry": effect_summary,
        "action_steps_per_effect": {
            task_key: {
                effect_id: _mean_or_none(
                    row["execution"]["action_steps"]
                    for row in rows
                    if row["task_key"] == task_key
                    and row["effect_id"] == effect_id
                    and row.get("execution") is not None
                )
                for candidate_task, effect_id in unique
                if candidate_task == task_key
            }
            for task_key in TASKS
        },
        "chains": chain_rows,
        "eligible_chain_count": sum(row["eligible"] for row in chain_rows),
    }
    return rows, summary


def select_chains(
    chain_contract: Mapping[str, object], qualification: Mapping[str, object]
) -> dict:
    candidate_order = {
        row["chain_id"]: index
        for index, row in enumerate(chain_contract["candidate_chains"])
    }
    rows = [dict(row) for row in qualification["chains"]]
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["chain_success"]),
            -float(row["minimum_source_segment_success"]),
            (
                float(row["mean_action_steps"])
                if row.get("mean_action_steps") is not None
                else 1.0e30
            ),
            candidate_order[row["chain_id"]],
        ),
    )
    eligible = [row for row in ordered if row["eligible"]]
    selected: List[dict] = []
    for task_key in TASKS:
        match = next((row for row in eligible if row["task_key"] == task_key), None)
        if match is not None and match not in selected:
            selected.append(match)
    for row in eligible:
        if row not in selected and len(selected) < int(chain_contract["maximum_selected_chains"]):
            selected.append(row)
    qualification_pass = bool(
        len(selected) >= int(chain_contract["minimum_eligible_chains"])
        and len({row["task_key"] for row in selected}) == len(TASKS)
    )
    analysis = list(selected)
    if not qualification_pass:
        analysis = []
        for task_key in TASKS:
            match = next(row for row in ordered if row["task_key"] == task_key)
            analysis.append(match)
        for row in ordered:
            if row not in analysis and len(analysis) < 3:
                analysis.append(row)
    return {
        "schema_version": 1,
        "freeze_status": "FROZEN_BEFORE_FORMAL_DEMO_ACCESS",
        "qualification_pass": qualification_pass,
        "selected_confirmatory_chain_ids": [row["chain_id"] for row in selected],
        "analysis_continuation_chain_ids": [row["chain_id"] for row in analysis],
        "continuation_is_diagnostic": not qualification_pass,
        "selection_rows": rows,
    }


def select_persistence_k(rows_by_k: Mapping[int, Sequence[Mapping[str, object]]]) -> dict:
    """Apply the frozen qualification-only lexicographic K selection rule."""

    candidates = []
    for value in (2, 4, 8):
        rows = list(rows_by_k.get(value, ()))
        if not rows:
            raise RuntimeError("missing persistence calibration rows for K=%d" % value)
        if any(int(row.get("persistence_k", -1)) != value for row in rows):
            raise RuntimeError("persistence calibration contains a mismatched K")
        candidates.append(
            {
                "k": value,
                "rollout_count": len(rows),
                "mean_chain_success": float(np.mean([row["chain_success"] for row in rows])),
                "false_completion_rate": float(
                    np.mean([row["false_completion"] for row in rows])
                ),
                "mean_action_steps": float(np.mean([row["action_steps"] for row in rows])),
                "backend_failure_count": sum(
                    row.get("failure_type") == "REPLAY_BACKEND_FAILURE" for row in rows
                ),
            }
        )
    ordered = sorted(
        candidates,
        key=lambda row: (
            -row["mean_chain_success"],
            row["false_completion_rate"],
            row["mean_action_steps"],
            row["k"],
        ),
    )
    return {
        "schema_version": 1,
        "status": "PERSISTENCE_K_FROZEN_BEFORE_FORMAL_DEMO_ACCESS",
        "selection_split": "qualification",
        "candidate_k": [2, 4, 8],
        "selection_order": [
            "descending_mean_chain_success",
            "ascending_false_completion_rate",
            "ascending_mean_action_steps",
            "ascending_k",
        ],
        "candidates": candidates,
        "selected_k": int(ordered[0]["k"]),
    }


def build_formal_replay_gate(
    replay_rows: Sequence[Mapping[str, object]],
    replay_summary: Mapping[str, object],
    selected_chain_ids: Sequence[str],
) -> dict:
    """Evaluate the frozen formal replay-only thresholds without reselection."""

    by_chain = {row["chain_id"]: row for row in replay_summary["chains"]}
    selected = []
    for chain_id in selected_chain_ids:
        chain = by_chain[chain_id]
        task_key = chain["task_key"]
        effects = list(chain["effects"])
        episodes = sorted(
            {
                row["source_episode"]
                for row in replay_rows
                if row["task_key"] == task_key and row["effect_id"] in effects
            },
            key=lambda value: int(str(value).rsplit("_", 1)[1]),
        )
        segment_rows = []
        valid_units = 0
        for episode in episodes:
            unit_ok = True
            for effect_id in effects:
                subset = [
                    row
                    for row in replay_rows
                    if row["task_key"] == task_key
                    and row["source_episode"] == episode
                    and row["effect_id"] == effect_id
                ]
                rate = float(np.mean([row["success"] for row in subset])) if subset else 0.0
                no_backend_error = bool(subset and all(row.get("error") is None for row in subset))
                segment_rows.append(
                    {
                        "source_episode": episode,
                        "effect_id": effect_id,
                        "conditional_success": rate,
                        "no_backend_error": no_backend_error,
                        "passes_0_90": bool(rate >= 0.90 and no_backend_error),
                    }
                )
                unit_ok = bool(unit_ok and no_backend_error)
            valid_units += int(unit_ok)
        selected.append(
            {
                "chain_id": chain_id,
                "task_key": task_key,
                "valid_formal_units": valid_units,
                "valid_formal_units_pass_8": valid_units >= 8,
                "replay_chain_success": float(chain["chain_success"]),
                "chain_pass_0_85": float(chain["chain_success"]) >= 0.85,
                "segments": segment_rows,
                "all_segments_pass_0_90": all(
                    row["passes_0_90"] for row in segment_rows
                ),
            }
        )
    passed = bool(
        selected
        and all(
            row["valid_formal_units_pass_8"]
            and row["chain_pass_0_85"]
            and row["all_segments_pass_0_90"]
            for row in selected
        )
    )
    return {
        "schema_version": 1,
        "status": "FORMAL_REPLAY_GATE_PASS" if passed else "FORMAL_REPLAY_GATE_FAIL",
        "passed": passed,
        "chain_selection_mutated_after_formal_access": False,
        "selected_chain_ids": list(selected_chain_ids),
        "effect_segment_threshold": 0.90,
        "chain_threshold": 0.85,
        "minimum_valid_formal_units_per_chain": 8,
        "chains": selected,
        "downstream_experiments_continued_under_user_override": True,
        "downstream_interpretation": "confirmatory" if passed else "diagnostic_only",
    }


def run_matrix(
    backend: FrozenEffectReplayBackend,
    split: str,
    chain_contract: Mapping[str, object],
    chain_ids: Sequence[str],
    source_episodes: Sequence[str],
    conditions: Sequence[str],
    arms: Sequence[str],
    persistence_k: int,
    output_path: Path,
    video_root: Path | None = None,
) -> List[dict]:
    chains = _chain_map(chain_contract)
    rows = _read_jsonl(output_path)
    expected_keys = {
        (chain_id, episode, condition, arm)
        for chain_id in chain_ids
        for episode in source_episodes
        for condition in conditions
        for arm in arms
    }
    observed = {
        (row["chain_id"], row["source_episode"], row["condition"], row["arm"])
        for row in rows
    }
    if len(observed) != len(rows) or not observed.issubset(expected_keys):
        raise RuntimeError("matrix resume contains duplicate or unexpected rows")
    for chain_ordinal, chain_id in enumerate(chain_ids):
        chain = chains[chain_id]
        for episode in source_episodes:
            for condition in conditions:
                for arm in arms:
                    key = (chain_id, episode, condition, arm)
                    if key in observed:
                        continue
                    video = None
                    if video_root is not None:
                        video = Path(video_root) / condition / arm / (
                            "%s_%s.mp4" % (chain_id, episode)
                        )
                    try:
                        row = run_chain_rollout(
                            backend,
                            split,
                            chain,
                            chain_ordinal,
                            episode,
                            condition,
                            arm,
                            persistence_k,
                            save_video_path=video,
                        )
                    except Exception as exc:
                        row = {
                            "record_type": "phase3_effect_boundary_chain_rollout",
                            "split": split,
                            "chain_id": chain_id,
                            "task_key": chain["task_key"],
                            "source_episode": episode,
                            "condition": condition,
                            "condition_name": CONDITION_NAMES[condition],
                            "arm": arm,
                            "persistence_k": persistence_k if arm == "PERSISTENCE_RECOVERY" else None,
                            "target_effect": _target_effect(chain, episode, condition),
                            "rollout_seed": _unit_seed(
                                str(chain["task_key"]), episode, chain_ordinal, condition
                            ),
                            "chain_success": False,
                            "current_effect_success": False,
                            "next_effect_success": False,
                            "effect_success": {effect: False for effect in chain["effects"]},
                            "faulted_chain_success": False if condition in ("C1", "C3", "C4", "C7") else None,
                            "clean_chain_success": False if condition == "C0" else None,
                            "recovery_success": False,
                            "recovery_attempted": False,
                            "repeated_loop": False,
                            "grounded_advance_true_count": 0,
                            "advance_count": 0,
                            "false_completion": False,
                            "false_completion_count": 0,
                            "premature_advance": False,
                            "contradiction_presented": False,
                            "contradiction_detected": False,
                            "contradiction_recovery_decision": False,
                            "single_view_false_positive_advance": False,
                            "imagined_as_realized": False,
                            "invalidated_realization_accuracy": None,
                            "retry_count": 0,
                            "reobserve_count": 0,
                            "rollback_count": 0,
                            "unnecessary_retry": False,
                            "unnecessary_recovery": False,
                            "safe_stop": False,
                            "action_steps": 0,
                            "action_budget": None,
                            "action_budget_exceeded": False,
                            "completion_latency": 0,
                            "failure_type": "REPLAY_BACKEND_FAILURE",
                            "fault_consumed": False,
                            "error": "%s:%s" % (type(exc).__name__, exc),
                            "event_stream_sha256": None,
                            "event_records": [],
                            "decision_trace": [],
                            "action_trace": [],
                            "final_memory_state": {},
                            "resident_slot_count_max": 0,
                            "dangling_parent_count": 0,
                            "transition_violation_count": 0,
                            "fault_or_truth_leakage_count": 0,
                            "video_path": None,
                        }
                    _append_jsonl(output_path, row)
                    rows.append(row)
                    observed.add(key)
                    print(
                        "PHASE3_MATRIX_CELL chain=%s episode=%s condition=%s arm=%s success=%s failure=%s"
                        % (
                            chain_id,
                            episode,
                            condition,
                            arm,
                            row["chain_success"],
                            row.get("failure_type"),
                        ),
                        flush=True,
                    )
    if observed != expected_keys:
        raise RuntimeError("matrix did not complete exact frozen grid")
    return rows


def paired_unit_audit(rows: Sequence[Mapping[str, object]]) -> List[dict]:
    groups: MutableMapping[tuple[str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["chain_id"], row["source_episode"], row["condition"])].append(row)
    audits = []
    compared = (
        "B3_MONOLITHIC",
        "POSTCHECK_RECOVERY",
        "PERSISTENCE_RECOVERY",
        "TYPED_MATCHED_RECOVERY",
        "B6_FULL",
    )
    for key, values in sorted(groups.items()):
        by_arm = {row["arm"]: row for row in values}
        if not set(compared).issubset(by_arm):
            continue
        traces = {
            arm: [entry["decision"] for entry in by_arm[arm]["decision_trace"]]
            for arm in compared
        }
        minimum = min(len(value) for value in traces.values())
        divergence = None
        for index in range(minimum):
            if len({traces[arm][index] for arm in compared}) > 1:
                divergence = index
                break
        if divergence is None and len({len(value) for value in traces.values()}) > 1:
            divergence = minimum
        prefix_event_hashes = {}
        prefix_action_hashes = {}
        prefix_state_hashes = {}
        event_prefix_records = []
        divergence_state = None
        divergence_effect = None
        if divergence is not None and divergence < minimum:
            for arm in compared:
                trace = by_arm[arm]["decision_trace"][divergence]
                prefix_event_hashes[arm] = trace["event_prefix_sha256"]
                prefix_state_hashes[arm] = trace["simulator_state_sha256"]
                action_count = int(trace["action_trace_length"])
                prefix_action_hashes[arm] = [
                    item["executed_action_sha256"]
                    for item in by_arm[arm]["action_trace"][:action_count]
                ]
            reference_arm = "B6_FULL"
            reference_trace = by_arm[reference_arm]["decision_trace"][divergence]
            divergence_state = reference_trace["simulator_state"]
            divergence_effect = reference_trace["effect_id"]
            event_prefix_records = by_arm[reference_arm]["event_records"][: int(
                reference_trace["event_record_stop_exclusive"]
            )]
        prefix_identical = bool(
            divergence is None
            or (
                len(set(prefix_event_hashes.values())) == 1
                and len({tuple(value) for value in prefix_action_hashes.values()}) == 1
                and len(set(prefix_state_hashes.values())) == 1
            )
        )
        audits.append(
            {
                "chain_id": key[0],
                "source_episode": key[1],
                "condition": key[2],
                "first_decision_divergence_index": divergence,
                "decision_sequences": traces,
                "prefix_event_hashes": prefix_event_hashes,
                "prefix_action_hashes": prefix_action_hashes,
                "prefix_simulator_state_hashes": prefix_state_hashes,
                "divergence_simulator_state": divergence_state,
                "divergence_effect_id": divergence_effect,
                "event_prefix_records": event_prefix_records,
                "unique_decisions": sorted(
                    {
                        value[divergence]
                        for value in traces.values()
                        if divergence is not None and divergence < len(value)
                    }
                ),
                "paired_prefix_event_and_action_bytes_identical": prefix_identical,
            }
        )
    return audits


def write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_text(Path(path), text)

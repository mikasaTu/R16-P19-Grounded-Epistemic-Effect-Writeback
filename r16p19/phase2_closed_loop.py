"""Paired, executor-decoupled LIBERO Phase-2 closed-loop evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import atomic_text, write_json, write_jsonl
from .config import ACTION_DIM, TASKS
from .memory import MemoryArm
from .ontology import load_ontology
from .phase2_executor import (
    ExecutionMode,
    RetargetedGeometricSkillExecutor,
    executor_input_hash,
    extract_geometric_snapshot,
)
from .phase2_evaluation import _video_frame, _write_video, clean_rollout_seed
from .simulator import (
    deterministic_target_effect,
    effect_truths,
    load_init_states,
    make_env,
    reset_to_state,
)
from .types import Decision, Event, EventType, EvidenceReceipt, canonical_stream_sha256


MEMORY_ARMS = ("B2", "B3", "B5", "B6")
CONDITIONS = ("C0", "C1", "C2", "C3", "C7")
FORMAL_INITS = tuple(range(20))
EXECUTED_PREFIX = 4
REOBSERVE_STEPS = 8
MAX_ACTION_STEPS = 700
MAX_CHUNKS_PER_EFFECT = 40
MAX_ATTEMPTS_PER_EFFECT = 4


def _camera_array(observation: Mapping[str, object], sensor: str) -> np.ndarray:
    candidates = {
        "agentview": ("agentview_image", "agentview_rgb"),
        "robot0_eye_in_hand": (
            "robot0_eye_in_hand_image",
            "eye_in_hand_rgb",
        ),
    }[sensor]
    for key in candidates:
        if key in observation:
            return np.asarray(observation[key])
    raise KeyError("camera %s missing" % sensor)


class PairedEventFactory:
    """Arm-independent event identity until behavior actually diverges."""

    def __init__(self, task_key: str, init_index: int, condition: str):
        self.episode_id = "%s:init_%02d:%s" % (task_key, init_index, condition)
        self.index = 0

    def make(
        self,
        event_type: EventType,
        effect: str,
        parent_ids: Sequence[str] = (),
        observation: Optional[Mapping[str, object]] = None,
        sensor: str = "agentview",
        payload: Optional[Mapping[str, object]] = None,
    ) -> Event:
        event_id = "%s:e%05d:%s" % (
            self.episode_id,
            self.index,
            event_type.value,
        )
        receipt = None
        if event_type in (
            EventType.OBSERVE_POSITIVE,
            EventType.VERIFY_POSITIVE,
            EventType.REALIZATION_WITNESS,
            EventType.OBSERVE_NEGATIVE,
            EventType.CONTRADICTION,
        ):
            if observation is None:
                raise ValueError("physical event requires an observation")
            receipt = EvidenceReceipt(
                evidence_id=event_id + ":receipt",
                episode_id=self.episode_id,
                event_index=self.index,
                timestamp=self.index / 20.0,
                sensor_identity=sensor,
                frame_digest=hashlib.sha256(
                    _camera_array(observation, sensor).tobytes()
                ).hexdigest(),
                effect_id=effect,
                evidence_type=event_type.value,
            )
        event = Event(
            event_id=event_id,
            episode_id=self.episode_id,
            event_index=self.index,
            timestamp=self.index / 20.0,
            event_type=event_type,
            effect_id=effect,
            parent_ids=tuple(parent_ids),
            receipt=receipt,
            payload=dict(payload or {}),
        )
        self.index += 1
        return event


def _process_positive(
    factory: PairedEventFactory,
    arm: MemoryArm,
    effect: str,
    command_id: str,
    observation: Mapping[str, object],
) -> Decision:
    observed = factory.make(
        EventType.OBSERVE_POSITIVE,
        effect,
        (command_id,),
        observation,
        "agentview",
    )
    arm.process(observed, effect)
    verified = factory.make(
        EventType.VERIFY_POSITIVE,
        effect,
        (observed.event_id,),
        observation,
        "robot0_eye_in_hand",
    )
    arm.process(verified, effect)
    witness = factory.make(
        EventType.REALIZATION_WITNESS,
        effect,
        (command_id, verified.event_id),
        observation,
        "agentview",
    )
    return arm.process(witness, effect)


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_rows(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(value, dtype="<f4").tobytes(order="C")
    ).hexdigest()


_EFFECT_PHYSICAL_JOINTS = {
    "STOVE_TURNED_ON": ("flat_stove_1_button",),
    "MOKA_GRASPED": ("moka_pot_1_joint0",),
    "MOKA_ON_STOVE": ("moka_pot_1_joint0",),
    "MOKA_RELEASED_ON_STOVE": ("moka_pot_1_joint0",),
    "BOWL_GRASPED": ("akita_black_bowl_1_joint0",),
    "BOWL_IN_BOTTOM_DRAWER": ("akita_black_bowl_1_joint0",),
    "BOWL_RELEASED_IN_DRAWER": ("akita_black_bowl_1_joint0",),
    "BOTTOM_DRAWER_CLOSED": ("white_cabinet_1_bottom_level",),
}


def _address_indices(address) -> np.ndarray:
    if isinstance(address, tuple):
        return np.arange(int(address[0]), int(address[1]), dtype=np.int64)
    return np.asarray((int(address),), dtype=np.int64)


def _capture_effect_physical_state(env, effect_id: str) -> dict:
    values = []
    for joint_name in _EFFECT_PHYSICAL_JOINTS[effect_id]:
        qpos_indices = _address_indices(env.sim.model.get_joint_qpos_addr(joint_name))
        qvel_indices = _address_indices(env.sim.model.get_joint_qvel_addr(joint_name))
        values.append(
            {
                "joint_name": joint_name,
                "qpos_indices": qpos_indices.tolist(),
                "qvel_indices": qvel_indices.tolist(),
                "qpos": np.asarray(env.sim.data.qpos[qpos_indices]).tolist(),
                "qvel": np.asarray(env.sim.data.qvel[qvel_indices]).tolist(),
            }
        )
    return {"effect_id": effect_id, "joints": values}


def _restore_effect_physical_state(env, snapshot: Mapping[str, object]):
    for value in snapshot["joints"]:
        qpos_indices = np.asarray(value["qpos_indices"], dtype=np.int64)
        qvel_indices = np.asarray(value["qvel_indices"], dtype=np.int64)
        env.sim.data.qpos[qpos_indices] = np.asarray(value["qpos"], dtype=np.float64)
        env.sim.data.qvel[qvel_indices] = np.asarray(value["qvel"], dtype=np.float64)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations(force_update=True)


def _decision_record(
    decision_index: int,
    effect: str,
    decision: Decision,
    physical_truth: bool,
    arm: MemoryArm,
    trigger: str,
) -> dict:
    return {
        "decision_index": decision_index,
        "effect_id": effect,
        "decision": decision.value,
        "physical_effect_truth": bool(physical_truth),
        "trigger": trigger,
        "memory_state": arm.current_summary(),
    }


def _run_one_rollout(
    env,
    executor: RetargetedGeometricSkillExecutor,
    executor_name: str,
    executor_identity: Mapping[str, object],
    task_key: str,
    init_index: int,
    condition: str,
    arm_name: str,
    initial_state: np.ndarray,
    video_path: Path,
) -> dict:
    task = TASKS[task_key]
    executor.reset_episode()
    rollout_seed = clean_rollout_seed(task_key, init_index)
    observation = reset_to_state(env, initial_state, seed=rollout_seed)
    for _ in range(5):
        observation, _, _, _ = env.step(np.zeros((ACTION_DIM,), dtype=np.float32))
    initial_state_sha256 = _array_sha256(initial_state)
    state_history = deque(maxlen=4)
    history_effect = None
    frames = [_video_frame(observation)]
    ontology = load_ontology()
    arm = MemoryArm(arm_name, task_key, ontology)
    factory = PairedEventFactory(task_key, init_index, condition)
    target_phase = deterministic_target_effect(
        task_key, "init_%02d" % init_index, condition
    )
    delay_decision_cycles = 3
    phase = 0
    action_steps = 0
    executor_call_index = 0
    retry_count = 0
    reobserve_count = 0
    premature = 0
    safe_stop = False
    repeated_loop = False
    fault_consumed = False
    fault_injector_failure = False
    effect_success = {effect: False for effect in task.effects}
    chunks_per_effect = {effect: 0 for effect in task.effects}
    attempts_per_effect = {effect: 0 for effect in task.effects}
    executor_calls: List[dict] = []
    decisions: List[dict] = []
    decision_snapshots: List[dict] = []
    failure_components: List[str] = []
    recovery_chain: List[dict] = []
    next_mode = ExecutionMode.EXECUTE
    while phase < len(task.effects) and action_steps < MAX_ACTION_STEPS and not safe_stop:
        effect = task.effects[phase]
        attempt = attempts_per_effect[effect]
        chunks_this_attempt = 0
        if history_effect != effect:
            state_history.clear()
            state_history.append(
                extract_geometric_snapshot(env, observation, task_key, effect)
            )
            history_effect = effect
        request = factory.make(EventType.REQUEST, effect)
        arm.process(request, effect)
        if condition == "C7" and phase == target_phase and not fault_consumed:
            imagined = factory.make(
                EventType.IMAGINE,
                effect,
                (request.event_id,),
                payload={"predicted_success": True},
            )
            arm.process(imagined, effect)
        command = factory.make(
            EventType.COMMAND,
            effect,
            (request.event_id,),
            payload={"attempt": attempt, "execution_mode": next_mode.value},
        )
        arm.process(command, effect)
        pre_effect_state = None
        if condition == "C3" and phase == target_phase and not fault_consumed:
            pre_effect_state = _capture_effect_physical_state(env, effect)
        reached = bool(effect_truths(env, task)[effect])
        truth_observation = observation if reached else None
        attempt_suppressed = (
            condition in ("C1", "C7")
            and phase == target_phase
            and not fault_consumed
        )
        while (
            not reached
            and action_steps < MAX_ACTION_STEPS
            and chunks_this_attempt < MAX_CHUNKS_PER_EFFECT
        ):
            input_hash = executor_input_hash(
                state_history, task_key, effect, next_mode, attempt
            )
            chunk = np.asarray(
                executor.action_chunk(
                    state_history, task_key, effect, next_mode, attempt
                ),
                dtype=np.float32,
            )
            if chunk.shape != (8, ACTION_DIM) or not np.isfinite(chunk).all():
                raise RuntimeError("executor emitted an invalid action chunk")
            suppress_chunk = attempt_suppressed
            executed_values = []
            for action in chunk[:EXECUTED_PREFIX]:
                executed = (
                    np.zeros((ACTION_DIM,), dtype=np.float32)
                    if suppress_chunk
                    else np.asarray(action, dtype=np.float32)
                )
                observation, _, _, _ = env.step(np.asarray(executed, dtype=np.float64))
                executed_values.append(executed.tolist())
                action_steps += 1
                state_history.append(
                    extract_geometric_snapshot(env, observation, task_key, effect)
                )
                if action_steps % 2 == 0:
                    frames.append(_video_frame(observation))
                reached = bool(effect_truths(env, task)[effect])
                if reached:
                    truth_observation = observation
                    break
                if action_steps >= MAX_ACTION_STEPS:
                    break
            executor_calls.append(
                {
                    "executor_call_index": executor_call_index,
                    "decision_epoch": len(decisions),
                    "effect_id": effect,
                    "execution_mode": next_mode.value,
                    "retry_index": attempt,
                    "executor_input_sha256": input_hash,
                    "action_chunk_sha256": _array_sha256(chunk),
                    "action_chunk": chunk.tolist(),
                    "executed_prefix": executed_values,
                    "suppressed_by_fault": bool(suppress_chunk),
                }
            )
            executor_call_index += 1
            chunks_this_attempt += 1
            chunks_per_effect[effect] += 1
            if suppress_chunk and reached:
                fault_injector_failure = True
            if attempt_suppressed:
                break
        if reached:
            effect_success[effect] = True
            if condition == "C2" and phase == target_phase and not fault_consumed:
                # The physical effect has already happened.  Hold the exact
                # observation that would back the positive receipts, expose
                # three arm-independent broker decision ticks, and only then
                # release receipts derived from the held observation.  No
                # action or simulator mutation is used to manufacture delay.
                for delay_cycle in range(1, delay_decision_cycles + 1):
                    withheld = factory.make(
                        EventType.IRRELEVANT,
                        effect,
                        payload={
                            "receipt_broker": "C2_WITHHELD",
                            "delay_decision_cycle": delay_cycle,
                            "physical_effect_truth": True,
                        },
                    )
                    withheld_decision = arm.process(withheld, effect)
                    record = _decision_record(
                        len(decisions),
                        effect,
                        withheld_decision,
                        True,
                        arm,
                        "delayed_receipt_cycle_%d" % delay_cycle,
                    )
                    record["execution_deferred_by_receipt_broker"] = True
                    decisions.append(record)
                    simulator_state = np.asarray(env.get_sim_state(), dtype=np.float32)
                    decision_snapshots.append(
                        {
                            "decision_index": record["decision_index"],
                            "effect_id": effect,
                            "phase": phase,
                            "action_steps": action_steps,
                            "simulator_state_sha256": _array_sha256(simulator_state),
                            "simulator_state": simulator_state.tolist(),
                        }
                    )
                fault_consumed = True
            decision = _process_positive(
                factory, arm, effect, command.event_id, truth_observation
            )
            trigger = (
                "delayed_positive_realization_witness"
                if condition == "C2" and phase == target_phase
                else "positive_realization_witness"
            )
            if condition == "C3" and phase == target_phase and not fault_consumed:
                observation = _restore_effect_physical_state(env, pre_effect_state)
                state_history.clear()
                state_history.append(
                    extract_geometric_snapshot(env, observation, task_key, effect)
                )
                frames.append(_video_frame(observation))
                physical_after_reset = bool(effect_truths(env, task)[effect])
                if physical_after_reset:
                    fault_injector_failure = True
                contradiction = factory.make(
                    EventType.CONTRADICTION,
                    effect,
                    (list(arm.ledger.events)[-1],),
                    observation,
                    "agentview",
                    {"fault": "post_realization_reversal"},
                )
                decision = arm.process(contradiction, effect)
                trigger = "post_realization_contradiction"
                fault_consumed = True
                effect_success[effect] = False
                recovery_chain.append(
                    {
                        "effect_id": effect,
                        "contradiction_event_id": contradiction.event_id,
                        "state_after_contradiction": arm.records[effect].to_dict(),
                        "recovery_decision": decision.value,
                    }
                )
        else:
            negative = factory.make(
                EventType.OBSERVE_NEGATIVE,
                effect,
                (command.event_id,),
                observation,
                "agentview",
                {
                    "fault": condition if attempt_suppressed else None,
                    "chunks_executed_this_attempt": chunks_this_attempt,
                    "chunks_executed_total": chunks_per_effect[effect],
                },
            )
            arm.process(negative, effect)
            timeout = factory.make(
                EventType.TIMEOUT,
                effect,
                (negative.event_id,),
                payload={
                    "fault": condition if attempt_suppressed else None,
                    "chunks_executed_this_attempt": chunks_this_attempt,
                    "chunks_executed_total": chunks_per_effect[effect],
                },
            )
            decision = arm.process(timeout, effect)
            trigger = "fault_timeout" if attempt_suppressed else "executor_timeout"
            if attempt_suppressed:
                fault_consumed = True
            elif (
                chunks_this_attempt >= MAX_CHUNKS_PER_EFFECT
                and attempt + 1 >= MAX_ATTEMPTS_PER_EFFECT
            ):
                repeated_loop = True
                if "timeout / repeated loop" not in failure_components:
                    failure_components.append("timeout / repeated loop")
                if "executor skill failure" not in failure_components:
                    failure_components.append("executor skill failure")
        physical_now = bool(effect_truths(env, task)[effect])
        decision_record = _decision_record(
            len(decisions), effect, decision, physical_now, arm, trigger
        )
        decisions.append(decision_record)
        simulator_state = np.asarray(env.get_sim_state(), dtype=np.float32)
        decision_snapshots.append(
            {
                "decision_index": decision_record["decision_index"],
                "effect_id": effect,
                "phase": phase,
                "action_steps": action_steps,
                "simulator_state_sha256": _array_sha256(simulator_state),
                "simulator_state": simulator_state.tolist(),
            }
        )
        if reached and condition != "C3" and decision != Decision.ADVANCE_TO_NEXT_SUBTASK:
            if "effect verifier failure" not in failure_components:
                failure_components.append("effect verifier failure")
        if decision == Decision.ADVANCE_TO_NEXT_SUBTASK:
            if not physical_now:
                premature += 1
                if "memory decision failure" not in failure_components:
                    failure_components.append("memory decision failure")
            phase += 1
            next_mode = ExecutionMode.EXECUTE
        elif decision == Decision.RETRY_CURRENT_EFFECT:
            retry_count += 1
            attempts_per_effect[effect] += 1
            next_mode = ExecutionMode.RETRY
            recovery_chain.append(
                {
                    "effect_id": effect,
                    "state_before_retry": arm.records[effect].to_dict(),
                    "recovery_decision": decision.value,
                    "executor_mode": ExecutionMode.RETRY.value,
                }
            )
        elif decision == Decision.ROLLBACK_OR_REPLAN:
            retry_count += 1
            attempts_per_effect[effect] += 1
            state_history.clear()
            state_history.append(
                extract_geometric_snapshot(env, observation, task_key, effect)
            )
            next_mode = ExecutionMode.RETRY
            recovery_chain.append(
                {
                    "effect_id": effect,
                    "state_before_retry": arm.records[effect].to_dict(),
                    "recovery_decision": decision.value,
                    "executor_history_reset": True,
                    "executor_mode": ExecutionMode.RETRY.value,
                }
            )
        elif decision == Decision.REOBSERVE:
            reobserve_count += 1
            for _ in range(REOBSERVE_STEPS):
                observation, _, _, _ = env.step(
                    np.zeros((ACTION_DIM,), dtype=np.float64)
                )
                action_steps += 1
                state_history.append(
                    extract_geometric_snapshot(env, observation, task_key, effect)
                )
                if action_steps % 2 == 0:
                    frames.append(_video_frame(observation))
                if action_steps >= MAX_ACTION_STEPS:
                    break
            attempts_per_effect[effect] += 1
            next_mode = ExecutionMode.EXECUTE
        elif decision == Decision.SAFE_STOP:
            safe_stop = True
            if not reached and "memory decision failure" not in failure_components:
                failure_components.append("memory decision failure")
        if repeated_loop and not reached:
            safe_stop = True
    full_task_success = bool(env.check_success() and phase == len(task.effects))
    if fault_injector_failure:
        failure_components.insert(0, "fault injector failure")
    if not full_task_success and not failure_components:
        failure_components.append("executor skill failure")
    failure_components = list(dict.fromkeys(failure_components))
    _write_video(video_path, frames)
    final_memory = arm.current_summary()
    recovery_success = bool(
        condition in ("C1", "C3", "C7")
        and fault_consumed
        and retry_count > 0
        and full_task_success
    )
    if recovery_success:
        recovery_chain.append(
            {
                "effect_id": task.effects[target_phase],
                "effect_re_realized": True,
                "task_completion": True,
            }
        )
    row = {
        "record_type": "closed_loop",
        "executor": executor_name,
        "executor_manifest_sha256": executor_identity["executor_manifest_sha256"],
        "executor_source_sha256": executor_identity["executor_source_sha256"],
        "executor_seed": 1619,
        "task_key": task_key,
        "init_index": int(init_index),
        "condition": condition,
        "arm": arm_name,
        "initial_state_sha256": initial_state_sha256,
        "rollout_seed": rollout_seed,
        "target_effect": task.effects[target_phase],
        "fault_schedule": {
            "target_effect_index": target_phase,
            "delayed_decision_cycles": (
                delay_decision_cycles if condition == "C2" else None
            ),
            "command_noop_chunks": 1 if condition in ("C1", "C7") else 0,
        },
        "max_action_steps": MAX_ACTION_STEPS,
        "reobserve_steps": REOBSERVE_STEPS,
        "recovery_mapping": {
            "ADVANCE_TO_NEXT_SUBTASK": "switch_effect_skill",
            "RETRY_CURRENT_EFFECT": "current_state_retry_mode",
            "REOBSERVE": "eight_step_arm_independent_noop",
            "ROLLBACK_OR_REPLAN": "reset_history_current_state_retry_mode",
            "SAFE_STOP": "terminate_episode",
        },
        "executor_calls": executor_calls,
        "decision_trace": decisions,
        "decision_snapshots": decision_snapshots,
        "event_stream_sha256": canonical_stream_sha256(list(arm.ledger.events.values())),
        "final_memory_state": final_memory,
        "task_success": full_task_success,
        "effects_reached": effect_success,
        "false_completion": premature > 0,
        "premature_subtask_transitions": premature,
        "repeated_action_loop": repeated_loop,
        "recovery_success": recovery_success,
        "recovery_chain": recovery_chain,
        "retry_count": retry_count,
        "reobserve_count": reobserve_count,
        "action_steps": action_steps,
        "safe_stop": safe_stop,
        "resident_slot_count_max": final_memory["resident_slot_count_max"],
        "dangling_parent_count": final_memory["dangling_parent_count"],
        "fault_consumed": fault_consumed,
        "failure_types": failure_components,
        "failure_type": failure_components[0] if failure_components else None,
        "video_path": str(video_path),
        "video_frame_stride": 2,
    }
    return row


def _first_sequence_divergence(sequences: Sequence[Sequence[object]]) -> Optional[int]:
    longest = max(len(value) for value in sequences)
    for index in range(longest):
        values = [value[index] if index < len(value) else None for value in sequences]
        if len(set(values)) != 1:
            return index
    return None


def audit_paired_unit(rows: Sequence[Mapping[str, object]]) -> dict:
    if {row["arm"] for row in rows} != set(MEMORY_ARMS) or len(rows) != 4:
        raise RuntimeError("paired unit must contain exactly B2/B3/B5/B6")
    ordered = [next(row for row in rows if row["arm"] == arm) for arm in MEMORY_ARMS]
    decision_sequences = [
        [entry["decision"] for entry in row["decision_trace"]] for row in ordered
    ]
    decision_divergence = _first_sequence_divergence(decision_sequences)
    action_sequences = []
    input_sequences = []
    for row in ordered:
        actions = []
        inputs = []
        for call in row["executor_calls"]:
            inputs.append(call["executor_input_sha256"])
            actions.extend(_array_sha256(np.asarray([action], dtype=np.float32)) for action in call["executed_prefix"])
        action_sequences.append(actions)
        input_sequences.append(inputs)
    action_divergence = _first_sequence_divergence(action_sequences)
    executor_call_divergence = _first_sequence_divergence(input_sequences)
    prefix_call_count = min(
        len(
            [
                call
                for call in row["executor_calls"]
                if decision_divergence is None
                or int(call["decision_epoch"]) <= decision_divergence
            ]
        )
        for row in ordered
    )
    prefix_input_sequences = [values[:prefix_call_count] for values in input_sequences]
    prefix_chunk_sequences = [
        [call["action_chunk_sha256"] for call in row["executor_calls"][:prefix_call_count]]
        for row in ordered
    ]
    prefix_identical = (
        len({tuple(value) for value in prefix_input_sequences}) == 1
        and len({tuple(value) for value in prefix_chunk_sequences}) == 1
    )
    if not prefix_identical:
        raise RuntimeError("paired executor prefix invariant violated")
    if decision_divergence is None and (
        executor_call_divergence is not None or action_divergence is not None
    ):
        raise RuntimeError("actions diverged without a memory-decision divergence")
    return {
        "task_key": ordered[0]["task_key"],
        "init_index": ordered[0]["init_index"],
        "condition": ordered[0]["condition"],
        "initial_state_sha256": ordered[0]["initial_state_sha256"],
        "executor_manifest_sha256": ordered[0]["executor_manifest_sha256"],
        "executor_source_sha256": ordered[0]["executor_source_sha256"],
        "target_effect": ordered[0]["target_effect"],
        "first_decision_divergence_step": decision_divergence,
        "first_executor_input_divergence_call": executor_call_divergence,
        "first_action_divergence_step": action_divergence,
        "paired_prefix_executor_input_and_action_bytes_identical": prefix_identical,
        "arms": {
            row["arm"]: {
                "task_success": row["task_success"],
                "failure_type": row["failure_type"],
                "retry_count": row["retry_count"],
                "action_steps": row["action_steps"],
                "video_path": row["video_path"],
            }
            for row in ordered
        },
    }


def paired_bootstrap(rows: Sequence[Mapping[str, object]], repetitions=10_000, seed=1619) -> dict:
    units = []
    for task_key in TASKS:
        for init_index in FORMAL_INITS:
            values = {}
            for arm in ("B2", "B3", "B5", "B6"):
                subset = [
                    row
                    for row in rows
                    if row["task_key"] == task_key
                    and int(row["init_index"]) == init_index
                    and row["condition"] in ("C1", "C3", "C7")
                    and row["arm"] == arm
                ]
                values[arm] = float(np.mean([row["task_success"] for row in subset]))
            units.append({"task_key": task_key, "init_index": init_index, **values})
    rng = np.random.RandomState(seed)

    def interval(values: np.ndarray) -> dict:
        samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        return {
            "estimate": float(values.mean()),
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        }

    b6_b3 = np.asarray([unit["B6"] - unit["B3"] for unit in units], dtype=np.float64)
    b6_b5 = np.asarray([unit["B6"] - unit["B5"] for unit in units], dtype=np.float64)
    b6_b2 = np.asarray([unit["B6"] - unit["B2"] for unit in units], dtype=np.float64)
    task_specific = {}
    for task_key in TASKS:
        selected = [index for index, unit in enumerate(units) if unit["task_key"] == task_key]
        task_specific[task_key] = {
            "B6_minus_B3_faulted_success": interval(b6_b3[selected]),
            "B6_minus_B5_faulted_success": interval(b6_b5[selected]),
            "B6_minus_B2_faulted_success": interval(b6_b2[selected]),
        }
    return {
        "paired_unit": "task_and_initial_state_index",
        "repetitions": repetitions,
        "seed": seed,
        "unit_count": len(units),
        "B6_minus_B3_faulted_success": interval(b6_b3),
        "B6_minus_B5_faulted_success": interval(b6_b5),
        "B6_minus_B2_faulted_success": interval(b6_b2),
        "task_specific": task_specific,
    }


def _paired_mcnemar(
    rows: Sequence[Mapping[str, object]], left: str, right: str
) -> dict:
    left_only = 0
    right_only = 0
    for task_key in TASKS:
        for init_index in FORMAL_INITS:
            for condition in CONDITIONS[1:]:
                pair = {
                    row["arm"]: bool(row["task_success"])
                    for row in rows
                    if row["task_key"] == task_key
                    and int(row["init_index"]) == init_index
                    and row["condition"] == condition
                    and row["arm"] in (left, right)
                }
                left_only += int(pair[left] and not pair[right])
                right_only += int(pair[right] and not pair[left])
    discordant = left_only + right_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2.0**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "left_arm": left,
        "right_arm": right,
        "left_success_right_failure": left_only,
        "left_failure_right_success": right_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": p_value,
    }


def summarize_behavior(
    rows: Sequence[Mapping[str, object]],
    bootstrap: Mapping[str, object],
    decision_causal_win_rate: float,
) -> dict:
    arms = {}
    for arm in MEMORY_ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        clean = [row for row in subset if row["condition"] == "C0"]
        faulted = [row for row in subset if row["condition"] != "C0"]
        target = [row for row in subset if row["condition"] in ("C1", "C3", "C7")]
        arms[arm] = {
            "rollout_count": len(subset),
            "full_task_success": float(np.mean([row["task_success"] for row in subset])),
            "clean_task_success": float(np.mean([row["task_success"] for row in clean])),
            "faulted_task_success": float(np.mean([row["task_success"] for row in faulted])),
            "target_fault_task_success": float(np.mean([row["task_success"] for row in target])),
            "false_completion_rate": float(np.mean([row["false_completion"] for row in subset])),
            "premature_transition_rate": float(np.mean([int(row["premature_subtask_transitions"] > 0) for row in subset])),
            "repeated_action_loop_rate": float(np.mean([row["repeated_action_loop"] for row in subset])),
            "recovery_success_rate": float(np.mean([row["recovery_success"] for row in target])),
            "mean_retries": float(np.mean([row["retry_count"] for row in subset])),
            "mean_action_steps": float(np.mean([row["action_steps"] for row in subset])),
            "safe_stop_rate": float(np.mean([row["safe_stop"] for row in subset])),
        }
    b3_false = arms["B3"]["false_completion_rate"]
    relative_reduction = (
        (b3_false - arms["B6"]["false_completion_rate"]) / b3_false
        if b3_false
        else 0.0
    )
    b6_c3 = [row for row in rows if row["arm"] == "B6" and row["condition"] == "C3"]
    contradiction_recovery = float(np.mean([row["recovery_success"] for row in b6_c3]))
    clean_degradation = arms["B3"]["clean_task_success"] - arms["B6"]["clean_task_success"]
    ci_b3 = bootstrap["B6_minus_B3_faulted_success"]["ci95"]
    ci_b2 = bootstrap["B6_minus_B2_faulted_success"]["ci95"]
    gates = {
        "B6_false_completion_relative_reduction_at_least_0_50": relative_reduction >= 0.50,
        "B6_contradiction_recovery_at_least_0_80": contradiction_recovery >= 0.80,
        "B6_fault_success_exceeds_B3": arms["B6"]["target_fault_task_success"] > arms["B3"]["target_fault_task_success"],
        "B6_fault_success_exceeds_B5": arms["B6"]["target_fault_task_success"] > arms["B5"]["target_fault_task_success"],
        "paired_bootstrap_B6_minus_B3_lower_gt_0": ci_b3[0] > 0.0,
        "clean_degradation_at_most_0_03": clean_degradation <= 0.03,
        "B6_statistically_distinguishable_from_B2": ci_b2[0] > 0.0 or ci_b2[1] < 0.0,
    }
    condition_effects = {}
    for condition in CONDITIONS:
        condition_effects[condition] = {}
        for comparison in ("B6_minus_B3", "B6_minus_B5", "B6_minus_B2"):
            other = comparison.rsplit("_", 1)[-1]
            b6 = [row["task_success"] for row in rows if row["condition"] == condition and row["arm"] == "B6"]
            baseline = [row["task_success"] for row in rows if row["condition"] == condition and row["arm"] == other]
            condition_effects[condition][comparison] = float(np.mean(b6) - np.mean(baseline))
    margin = arms["B6"]["target_fault_task_success"] - max(
        arms["B3"]["target_fault_task_success"], arms["B5"]["target_fault_task_success"]
    )
    b6_clean = [
        row for row in rows if row["arm"] == "B6" and row["condition"] == "C0"
    ]
    clean_unnecessary_recovery_rate = float(
        np.mean(
            [
                bool(row["retry_count"] or row["reobserve_count"])
                for row in b6_clean
            ]
        )
    )
    clean_safe_stop_rate = float(np.mean([row["safe_stop"] for row in b6_clean]))
    gates.update(
        {
            "B6_faulted_success_margin_at_least_0_10": margin >= 0.10,
            "decision_causal_win_rate_at_least_0_70": (
                decision_causal_win_rate >= 0.70
            ),
            "B6_clean_unnecessary_recovery_rate_at_most_0_10": (
                clean_unnecessary_recovery_rate <= 0.10
            ),
            "B6_clean_safe_stop_rate_zero": clean_safe_stop_rate == 0.0,
        }
    )
    return {
        "arms": arms,
        "B6_false_completion_relative_reduction_vs_B3": relative_reduction,
        "B6_contradiction_recovery_recall": contradiction_recovery,
        "clean_success_degradation_vs_B3": clean_degradation,
        "B6_absolute_target_fault_margin_vs_max_B3_B5": margin,
        "number_needed_to_recover": (1.0 / margin if margin > 0 else None),
        "decision_causal_win_rate": decision_causal_win_rate,
        "B6_clean_unnecessary_recovery_rate": clean_unnecessary_recovery_rate,
        "B6_clean_safe_stop_rate": clean_safe_stop_rate,
        "condition_effect_sizes": condition_effects,
        "task_specific_confidence_intervals": bootstrap["task_specific"],
        "paired_mcnemar": {
            "B6_vs_B3": _paired_mcnemar(rows, "B6", "B3"),
            "B6_vs_B5": _paired_mcnemar(rows, "B6", "B5"),
            "B6_vs_B2": _paired_mcnemar(rows, "B6", "B2"),
        },
        "behavior_gates": gates,
        "behavior_gate_pass": all(gates.values()),
    }


def correctness_summary(rows: Sequence[Mapping[str, object]], paired_units: Sequence[Mapping[str, object]]) -> dict:
    b6_c3 = [row for row in rows if row["arm"] == "B6" and row["condition"] == "C3"]
    b5_c3 = [row for row in rows if row["arm"] == "B5" and row["condition"] == "C3"]
    gates = {
        "evidence_alias_acceptance_zero": all(row["final_memory_state"]["alias_acceptances"] == 0 for row in rows),
        "dangling_parent_count_zero": all(row["dangling_parent_count"] == 0 for row in rows),
        "resident_slot_count_max_32": max(row["resident_slot_count_max"] for row in rows) <= 32,
        "B6_contradictions_have_recovery_route": all(any(item.get("recovery_decision") == "ROLLBACK_OR_REPLAN" for item in row["recovery_chain"]) for row in b6_c3),
        "B6_outperforms_B5_contradiction_recovery": np.mean([row["recovery_success"] for row in b6_c3]) > np.mean([row["recovery_success"] for row in b5_c3]),
        "degenerate_safe_decision_policy_forbidden": any(not row["safe_stop"] for row in rows if row["arm"] == "B6"),
        "paired_prefix_invariant": all(unit["paired_prefix_executor_input_and_action_bytes_identical"] for unit in paired_units),
    }
    return {"correctness_gates": gates, "correctness_gate_pass": all(gates.values())}


def mechanism_mediation(rows: Sequence[Mapping[str, object]]) -> dict:
    categories = (
        "memory decision failure",
        "executor skill failure",
        "effect verifier failure",
        "fault injector failure",
        "timeout / repeated loop",
    )
    counts = {
        category: int(sum(category in row["failure_types"] for row in rows))
        for category in categories
    }
    b6_recoveries = [
        {
            "task_key": row["task_key"],
            "init_index": row["init_index"],
            "condition": row["condition"],
            "target_effect": row["target_effect"],
            "recovery_chain": row["recovery_chain"],
            "video_path": row["video_path"],
        }
        for row in rows
        if row["arm"] == "B6" and row["recovery_success"]
    ]
    return {
        "failure_category_counts_nonexclusive": counts,
        "B6_successful_recovery_count": len(b6_recoveries),
        "B6_successful_recovery_chains": b6_recoveries,
    }


def _write_failure_cases(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    failures = [row for row in rows if not row["task_success"]]
    lines = [
        "# Phase-2 failure cases",
        "",
        "Failed rollouts: %d / %d." % (len(failures), len(rows)),
        "",
    ]
    for row in failures:
        lines.append(
            "- %s init %02d %s %s: `%s`; retries=%d, steps=%d, video=`%s`."
            % (
                row["task_key"],
                row["init_index"],
                row["condition"],
                row["arm"],
                ", ".join(row["failure_types"]),
                row["retry_count"],
                row["action_steps"],
                row["video_path"],
            )
        )
    atomic_text(output_dir / "failure_cases.md", "\n".join(lines) + "\n")


def _write_sha256sums_recursive(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and "wandb" not in path.parts
    )
    lines = [
        "%s  %s" % (hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(output_dir))
        for path in paths
    ]
    atomic_text(output_dir / "SHA256SUMS", "\n".join(lines) + "\n")


def _replay_decision(
    env,
    executor: RetargetedGeometricSkillExecutor,
    task_key: str,
    simulator_state: np.ndarray,
    effect_id: str,
    decision: str,
    horizon: int = 120,
) -> dict:
    task = TASKS[task_key]
    observation = reset_to_state(env, np.asarray(simulator_state, dtype=np.float64))
    executor.reset_episode()
    initial_truth = bool(effect_truths(env, task)[effect_id])
    action_steps = 0
    if decision == Decision.REOBSERVE.value:
        for _ in range(min(REOBSERVE_STEPS, horizon)):
            observation, _, _, _ = env.step(
                np.zeros((ACTION_DIM,), dtype=np.float64)
            )
            action_steps += 1
    elif decision in (
        Decision.RETRY_CURRENT_EFFECT.value,
        Decision.ROLLBACK_OR_REPLAN.value,
    ):
        history = deque(
            [extract_geometric_snapshot(env, observation, task_key, effect_id)],
            maxlen=4,
        )
        while action_steps < horizon and not effect_truths(env, task)[effect_id]:
            chunk = executor.action_chunk(
                history, task_key, effect_id, ExecutionMode.RETRY, 1
            )
            for action in chunk[:EXECUTED_PREFIX]:
                observation, _, _, _ = env.step(
                    np.asarray(action, dtype=np.float64)
                )
                action_steps += 1
                history.append(
                    extract_geometric_snapshot(env, observation, task_key, effect_id)
                )
                if effect_truths(env, task)[effect_id] or action_steps >= horizon:
                    break
    final_truth = bool(effect_truths(env, task)[effect_id])
    premature_advance = bool(
        decision == Decision.ADVANCE_TO_NEXT_SUBTASK.value and not initial_truth
    )
    safe_stopped_unrealized = bool(
        decision == Decision.SAFE_STOP.value and not initial_truth
    )
    return {
        "decision": decision,
        "initial_target_effect_truth": initial_truth,
        "immediate_target_effect_completion": final_truth,
        "recoverability_at_horizon": bool(
            final_truth
            or decision
            in (
                Decision.REOBSERVE.value,
                Decision.RETRY_CURRENT_EFFECT.value,
                Decision.ROLLBACK_OR_REPLAN.value,
            )
        ),
        "irreversible_failure": bool(premature_advance or safe_stopped_unrealized),
        "extra_action_steps": action_steps,
    }


def first_divergence_causal_replays(
    executor: RetargetedGeometricSkillExecutor,
    rows: Sequence[Mapping[str, object]],
) -> Tuple[List[dict], dict]:
    replays = []
    for task_key, task in TASKS.items():
        env = make_env(task, camera_obs=False)
        try:
            for init_index in FORMAL_INITS:
                for condition in CONDITIONS:
                    subset = {
                        row["arm"]: row
                        for row in rows
                        if row["task_key"] == task_key
                        and int(row["init_index"]) == init_index
                        and row["condition"] == condition
                        and row["arm"] in ("B3", "B5", "B6")
                    }
                    sequences = [
                        [item["decision"] for item in subset[arm]["decision_trace"]]
                        for arm in ("B3", "B5", "B6")
                    ]
                    divergence = _first_sequence_divergence(sequences)
                    if divergence is None:
                        replays.append(
                            {
                                "task_key": task_key,
                                "init_index": init_index,
                                "condition": condition,
                                "first_decision_divergence_step": None,
                                "status": "NO_DIVERGENCE",
                                "b6_causal_win": False,
                            }
                        )
                        continue
                    snapshots = {
                        arm: subset[arm]["decision_snapshots"][divergence]
                        for arm in ("B3", "B5", "B6")
                    }
                    snapshot_hashes = {
                        value["simulator_state_sha256"] for value in snapshots.values()
                    }
                    if len(snapshot_hashes) != 1:
                        raise RuntimeError(
                            "pre-decision simulator state differs across paired arms"
                        )
                    decisions = {
                        arm: sequences[index][divergence]
                        for index, arm in enumerate(("B3", "B5", "B6"))
                    }
                    reference = snapshots["B6"]
                    outcomes = {
                        decision: _replay_decision(
                            env,
                            executor,
                            task_key,
                            np.asarray(reference["simulator_state"]),
                            reference["effect_id"],
                            decision,
                        )
                        for decision in sorted(set(decisions.values()))
                    }

                    def score(outcome: Mapping[str, object]) -> tuple:
                        return (
                            int(outcome["immediate_target_effect_completion"]),
                            int(outcome["recoverability_at_horizon"]),
                            -int(outcome["irreversible_failure"]),
                            -int(outcome["extra_action_steps"]),
                        )

                    b6_score = score(outcomes[decisions["B6"]])
                    other_scores = [
                        score(outcomes[decisions[arm]]) for arm in ("B3", "B5")
                    ]
                    replays.append(
                        {
                            "task_key": task_key,
                            "init_index": init_index,
                            "condition": condition,
                            "first_decision_divergence_step": divergence,
                            "status": "REPLAY_COMPLETE",
                            "simulator_state_sha256": next(iter(snapshot_hashes)),
                            "effect_id": reference["effect_id"],
                            "arm_decisions": decisions,
                            "distinct_decision_outcomes": outcomes,
                            "b6_causal_win": bool(
                                all(b6_score > value for value in other_scores)
                            ),
                        }
                    )
        finally:
            env.close()
    completed = [row for row in replays if row["status"] == "REPLAY_COMPLETE"]
    summary = {
        "paired_unit_count": len(replays),
        "divergent_unit_count": len(completed),
        "decision_causal_win_rate": (
            float(np.mean([row["b6_causal_win"] for row in completed]))
            if completed
            else 0.0
        ),
        "fixed_horizon_action_steps": 120,
    }
    return replays, summary


def finalize_closed_loop(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    executor: RetargetedGeometricSkillExecutor,
) -> dict:
    if len(rows) != 800:
        raise RuntimeError("closed-loop factorial incomplete: %d != 800" % len(rows))
    paired_units = []
    for task_key in TASKS:
        for init_index in FORMAL_INITS:
            for condition in CONDITIONS:
                subset = [
                    row
                    for row in rows
                    if row["task_key"] == task_key
                    and int(row["init_index"]) == init_index
                    and row["condition"] == condition
                ]
                paired_units.append(audit_paired_unit(subset))
    write_jsonl(output_dir / "paired_units.jsonl", paired_units)
    bootstrap = paired_bootstrap(rows)
    divergence_replays, causal_summary = first_divergence_causal_replays(
        executor, rows
    )
    behavior = summarize_behavior(
        rows, bootstrap, causal_summary["decision_causal_win_rate"]
    )
    correctness = correctness_summary(rows, paired_units)
    mediation = mechanism_mediation(rows)
    write_json(output_dir / "paired_bootstrap.json", bootstrap)
    write_json(output_dir / "behavior_summary.json", behavior)
    write_json(output_dir / "correctness_summary.json", correctness)
    write_json(output_dir / "mechanism_mediation.json", mediation)
    write_jsonl(output_dir / "first_divergence_replays.jsonl", divergence_replays)
    write_json(output_dir / "causal_replay_summary.json", causal_summary)
    _write_failure_cases(output_dir, rows)
    final_status = (
        "PASS_PHASE2_ORACLE_BEHAVIOR"
        if behavior["behavior_gate_pass"] and correctness["correctness_gate_pass"]
        else "REJECT_CORE_MECHANISM"
    )
    final = {
        "final_status": final_status,
        "executor_formal_gate_pass": True,
        "closed_loop_rollout_count": len(rows),
        "correctness_gate_pass": correctness["correctness_gate_pass"],
        "behavior_gate_pass": behavior["behavior_gate_pass"],
    }
    write_json(output_dir / "final_status.json", final)
    lines = [
        "# Phase-2 final decision",
        "",
        "Final status: **%s**" % final_status,
        "",
        "The frozen executor formal gate passed and all 800 preregistered rollouts completed.",
        "Correctness gate: **%s**; behavior gate: **%s**."
        % (correctness["correctness_gate_pass"], behavior["behavior_gate_pass"]),
        "",
        "This result is limited to the two frozen LIBERO-10 tasks and is not a VLA-improvement claim.",
    ]
    atomic_text(output_dir / "FINAL_DECISION.md", "\n".join(lines) + "\n")
    _write_sha256sums_recursive(output_dir)
    return final


def run_closed_loop_matrix(
    executor: RetargetedGeometricSkillExecutor,
    executor_name: str,
    executor_identity: Mapping[str, object],
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formal_pass = output_dir / "formal_executor_gate_pass.json"
    if not formal_pass.is_file():
        raise RuntimeError("800-run matrix forbidden before formal executor gate pass")
    gate = json.loads(formal_pass.read_text(encoding="utf-8"))
    if gate["executor_manifest"] != executor_identity["runtime_manifest"]:
        raise RuntimeError("formal executor gate differs from matrix executor")
    marker_path = output_dir / "CLOSED_LOOP_800_STARTED.json"
    marker = {
        "executor": executor_name,
        "executor_manifest_sha256": executor_identity["executor_manifest_sha256"],
        "executor_source_sha256": executor_identity["executor_source_sha256"],
        "tasks": list(TASKS),
        "init_indices": list(FORMAL_INITS),
        "conditions": list(CONDITIONS),
        "arms": list(MEMORY_ARMS),
        "expected_rollouts": 800,
    }
    if marker_path.is_file():
        if json.loads(marker_path.read_text(encoding="utf-8")) != marker:
            raise RuntimeError("closed-loop resume identity drift")
    else:
        atomic_text(marker_path, json.dumps(marker, indent=2, sort_keys=True) + "\n")
    results_path = output_dir / "closed_loop_results.jsonl"
    rows = _read_rows(results_path)
    observed = {
        (row["task_key"], int(row["init_index"]), row["condition"], row["arm"])
        for row in rows
    }
    if len(observed) != len(rows):
        raise RuntimeError("closed-loop resume has duplicate cells")
    video_dir = output_dir / "closed_loop_videos"
    for task_key, task in TASKS.items():
        env = make_env(task, camera_obs=True)
        initial_states = load_init_states(task)
        try:
            for init_index in FORMAL_INITS:
                for condition in CONDITIONS:
                    for arm_name in MEMORY_ARMS:
                        key = (task_key, init_index, condition, arm_name)
                        if key in observed:
                            continue
                        video_path = video_dir / (
                            "%s_init_%02d_%s_%s.mp4"
                            % (task_key, init_index, condition, arm_name)
                        )
                        row = _run_one_rollout(
                            env,
                            executor,
                            executor_name,
                            executor_identity,
                            task_key,
                            init_index,
                            condition,
                            arm_name,
                            initial_states[init_index],
                            video_path,
                        )
                        _append_row(results_path, row)
                        rows.append(row)
                        observed.add(key)
                        print(
                            "PHASE2_CLOSED_LOOP_PERSISTED count=%d task=%s init=%d condition=%s arm=%s success=%s"
                            % (
                                len(rows),
                                task_key,
                                init_index,
                                condition,
                                arm_name,
                                row["task_success"],
                            ),
                            flush=True,
                        )
        finally:
            env.close()
    return finalize_closed_loop(output_dir, rows, executor)

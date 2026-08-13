"""Clean executor gates and development ablations for LIBERO Phase-2."""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import numpy as np

from .artifacts import atomic_text, write_json
from .config import ACTION_DIM, TASKS
from .phase1b_evaluation import _video_frame, _write_video
from .phase2_executor import (
    ExecutionMode,
    RetargetedGeometricSkillExecutor,
    executor_input_hash,
    extract_geometric_snapshot,
)
from .simulator import effect_truths, load_init_states, make_env, reset_to_state


QUALIFICATION_THRESHOLD = 0.90
FULL_TASK_THRESHOLD = 0.80
FORMAL_THRESHOLD = 0.80
MAX_ACTION_STEPS = 700
MAX_CHUNKS_PER_EFFECT = 40
EXECUTED_PREFIX = 4


def clean_rollout_seed(task_key: str, init_index: int) -> int:
    return 1619 + 1000 * list(TASKS).index(task_key) + int(init_index)


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
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def summarize_clean_gate(
    rows: Sequence[Mapping[str, object]],
    init_indices: Sequence[int],
    gate_name: str,
    executor_manifest: Mapping[str, object],
) -> dict:
    expected = len(TASKS) * len(init_indices)
    if len(rows) != expected:
        raise RuntimeError("clean gate incomplete: %d != %d" % (len(rows), expected))
    per_effect = {}
    per_task_full = {}
    for task_key, task in TASKS.items():
        subset = [row for row in rows if row["task_key"] == task_key]
        if len(subset) != len(init_indices):
            raise RuntimeError("clean gate task cell incomplete")
        per_effect[task_key] = {
            effect_id: float(np.mean([row["effect_success"][effect_id] for row in subset]))
            for effect_id in task.effects
        }
        per_task_full[task_key] = float(
            np.mean([row["full_task_success"] for row in subset])
        )
    minimum = min(value for task in per_effect.values() for value in task.values())
    loop_rate = float(np.mean([row["repeated_action_loop"] for row in rows]))
    summary = {
        "status": "CLEAN_GATE_COMPLETE",
        "gate_name": gate_name,
        "executor_manifest": dict(executor_manifest),
        "init_indices": [int(value) for value in init_indices],
        "rollout_count": len(rows),
        "per_effect_success": per_effect,
        "min_per_effect_success": float(minimum),
        "per_task_full_success": per_task_full,
        "full_task_success_rate": float(np.mean([row["full_task_success"] for row in rows])),
        "repeated_action_loop_rate": loop_rate,
        "mean_action_steps": float(np.mean([row["action_steps"] for row in rows])),
        "failure_video_count": int(sum(bool(row.get("failure_video")) for row in rows)),
        "selection_used_gate_results": False,
    }
    if gate_name == "qualification":
        summary.update(
            {
                "minimum_per_effect_threshold": QUALIFICATION_THRESHOLD,
                "minimum_full_task_success_each_task_threshold": FULL_TASK_THRESHOLD,
                "maximum_repeated_action_loop_rate": 0.10,
                "pass": bool(
                    minimum >= QUALIFICATION_THRESHOLD
                    and min(per_task_full.values()) >= FULL_TASK_THRESHOLD
                    and loop_rate <= 0.10
                ),
            }
        )
    elif gate_name == "formal_competence":
        summary.update(
            {
                "minimum_per_effect_threshold": FORMAL_THRESHOLD,
                "pass": bool(minimum >= FORMAL_THRESHOLD),
            }
        )
    return summary


def run_clean_gate(
    executor: RetargetedGeometricSkillExecutor,
    init_indices: Sequence[int],
    output_dir: Path,
    gate_name: str,
    save_failure_videos: bool = False,
    executed_prefix: int = EXECUTED_PREFIX,
    max_action_steps: int = MAX_ACTION_STEPS,
    max_attempts_per_effect: int = 1,
) -> Tuple[List[dict], dict]:
    """Run a resumable clean gate or development ablation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / (gate_name + "_rollouts.jsonl")
    summary_path = output_dir / (gate_name + "_summary.json")
    marker_path = output_dir / (gate_name.upper() + "_STARTED.json")
    executor_manifest = executor.frozen_manifest()
    marker = {
        "gate_name": gate_name,
        "executor_manifest": executor_manifest,
        "init_indices": [int(value) for value in init_indices],
        "max_action_steps": int(max_action_steps),
        "max_chunks_per_effect": MAX_CHUNKS_PER_EFFECT,
        "max_attempts_per_effect": int(max_attempts_per_effect),
        "executed_prefix": int(executed_prefix),
        "save_failure_videos": bool(save_failure_videos),
    }
    if marker_path.is_file():
        observed = json.loads(marker_path.read_text(encoding="utf-8"))
        if observed != marker:
            raise RuntimeError("clean gate marker identity drift")
    else:
        atomic_text(marker_path, json.dumps(marker, indent=2, sort_keys=True) + "\n")

    rows = _read_rows(results_path)
    expected_keys = {
        (task_key, int(init_index))
        for task_key in TASKS
        for init_index in init_indices
    }
    observed_keys = {(row["task_key"], int(row["init_index"])) for row in rows}
    if len(observed_keys) != len(rows) or not observed_keys.issubset(expected_keys):
        raise RuntimeError("clean gate resume rows contain duplicates or unexpected cells")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(rows) != len(expected_keys):
            raise RuntimeError("clean gate summary exists with incomplete rows")
        return rows, summary

    videos_dir = output_dir / (gate_name + "_failure_videos")
    for task_key, task in TASKS.items():
        pending = [
            int(init_index)
            for init_index in init_indices
            if (task_key, int(init_index)) not in observed_keys
        ]
        if not pending:
            continue
        env = make_env(task, camera_obs=save_failure_videos)
        initial_states = load_init_states(task)
        try:
            for init_index in pending:
                executor.reset_episode()
                initial_state = np.asarray(initial_states[init_index]).copy()
                rollout_seed = clean_rollout_seed(task_key, init_index)
                observation = reset_to_state(env, initial_state, seed=rollout_seed)
                for _ in range(5):
                    observation, _, _, _ = env.step(
                        np.zeros((ACTION_DIM,), dtype=np.float32)
                    )
                frames = [_video_frame(observation)] if save_failure_videos else []
                effect_success = {effect_id: False for effect_id in task.effects}
                action_steps = 0
                repeated_loop = False
                failure_type = None
                chunk_trace = []
                retry_count = 0
                for effect_id in task.effects:
                    if effect_truths(env, task)[effect_id]:
                        effect_success[effect_id] = True
                        continue
                    state_history = deque(maxlen=4)
                    state_history.append(
                        extract_geometric_snapshot(env, observation, task_key, effect_id)
                    )
                    reached = False
                    effect_exhausted_attempt = False
                    for retry_index in range(int(max_attempts_per_effect)):
                        mode = (
                            ExecutionMode.EXECUTE
                            if retry_index == 0
                            else ExecutionMode.RETRY
                        )
                        if retry_index:
                            retry_count += 1
                        for chunk_index in range(MAX_CHUNKS_PER_EFFECT):
                            input_hash = executor_input_hash(
                                state_history,
                                task_key,
                                effect_id,
                                mode,
                                retry_index,
                            )
                            chunk = np.asarray(
                                executor.action_chunk(
                                    state_history,
                                    task_key,
                                    effect_id,
                                    mode,
                                    retry_index,
                                ),
                                dtype=np.float32,
                            )
                            action_hash = _array_sha256(chunk)
                            trace = dict(executor.last_trace)
                            trace.update(
                                {
                                    "effect_id": effect_id,
                                    "attempt_index": retry_index,
                                    "chunk_index": chunk_index,
                                    "executor_input_sha256": input_hash,
                                    "action_chunk_sha256": action_hash,
                                }
                            )
                            chunk_trace.append(trace)
                            for action in chunk[: int(executed_prefix)]:
                                observation, _, _, _ = env.step(
                                    np.asarray(action, dtype=np.float64)
                                )
                                action_steps += 1
                                state_history.append(
                                    extract_geometric_snapshot(
                                        env, observation, task_key, effect_id
                                    )
                                )
                                if save_failure_videos and action_steps % 2 == 0:
                                    frames.append(_video_frame(observation))
                                if effect_truths(env, task)[effect_id]:
                                    effect_success[effect_id] = True
                                    reached = True
                                    break
                                if action_steps >= int(max_action_steps):
                                    failure_type = "MAX_ACTION_STEPS"
                                    break
                            if reached or failure_type is not None:
                                break
                        if not reached and failure_type is None:
                            effect_exhausted_attempt = True
                        if reached or failure_type is not None:
                            break
                    if not reached:
                        repeated_loop = bool(
                            effect_exhausted_attempt
                            or failure_type == "MAX_ACTION_STEPS"
                        )
                        if failure_type is None:
                            failure_type = "EFFECT_CHUNK_LIMIT:%s" % effect_id
                        break
                full_task_success = bool(
                    all(effect_success.values()) and env.check_success()
                )
                if not full_task_success and failure_type is None:
                    failure_type = "FINAL_TASK_PREDICATE_FALSE"
                video_path = None
                if save_failure_videos and not full_task_success:
                    video_path = videos_dir / (
                        "%s_%s_init_%02d.mp4" % (gate_name, task_key, init_index)
                    )
                    _write_video(video_path, frames)
                row = {
                    "record_type": "phase2_clean_executor_gate",
                    "gate_name": gate_name,
                    "executor_variant": executor.variant.value,
                    "executor_manifest_sha256": executor.manifest_sha256,
                    "task_key": task_key,
                    "init_index": init_index,
                    "initial_state_sha256": _array_sha256(initial_state),
                    "rollout_seed": rollout_seed,
                    "effect_success": effect_success,
                    "full_task_success": full_task_success,
                    "action_steps": action_steps,
                    "repeated_action_loop": repeated_loop,
                    "retry_count": retry_count,
                    "failure_type": failure_type,
                    "failure_video": (
                        str(video_path.relative_to(output_dir)) if video_path else None
                    ),
                    "chunk_trace": chunk_trace,
                }
                _append_row(results_path, row)
                rows.append(row)
                observed_keys.add((task_key, init_index))
                print(
                    "PHASE2_ROLLOUT_PERSISTED gate=%s task=%s init=%d full=%s steps=%d failure=%s"
                    % (
                        gate_name,
                        task_key,
                        init_index,
                        full_task_success,
                        action_steps,
                        failure_type,
                    ),
                    flush=True,
                )
        finally:
            env.close()
    rows = sorted(rows, key=lambda row: (row["task_key"], int(row["init_index"])))
    summary = summarize_clean_gate(rows, init_indices, gate_name, executor_manifest)
    write_json(summary_path, summary)
    print(
        "PHASE2_CLEAN_GATE gate=%s min_per_effect=%.3f full=%.3f loop=%.3f pass=%s"
        % (
            gate_name,
            summary["min_per_effect_success"],
            summary["full_task_success_rate"],
            summary["repeated_action_loop_rate"],
            summary.get("pass", "n/a"),
        ),
        flush=True,
    )
    return rows, summary

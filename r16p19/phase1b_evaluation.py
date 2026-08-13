"""Frozen clean actor gates for LIBERO Phase-1B."""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np

from .artifacts import atomic_text, write_json
from .config import ACTION_DIM, TASKS
from .phase1b_actor import ExecutionMode, SkillActor, actor_input_hash
from .simulator import effect_truths, load_init_states, make_env, reset_to_state


ACTOR_GATE_THRESHOLD = 0.80
MAX_ACTION_STEPS = 700
MAX_CHUNKS_PER_EFFECT = 30
EXECUTED_PREFIX = 4


def _camera(observation: Mapping[str, object], key_candidates: Sequence[str]) -> np.ndarray:
    for key in key_candidates:
        if key in observation:
            value = np.asarray(observation[key])
            if value.dtype != np.uint8:
                value = np.clip(value, 0, 255).astype(np.uint8)
            return value
    raise KeyError("camera observation missing from %r" % sorted(observation))


def _video_frame(observation: Mapping[str, object]) -> np.ndarray:
    agent = _camera(observation, ("agentview_image", "agentview_rgb"))
    wrist = _camera(
        observation, ("robot0_eye_in_hand_image", "eye_in_hand_rgb")
    )
    if agent.shape[:2] != wrist.shape[:2]:
        raise ValueError("qualification camera dimensions differ")
    return np.concatenate((agent, wrist), axis=1)


def _write_video(path: Path, frames: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.mp4" % path.stem)
    imageio.mimsave(str(temporary), list(frames), fps=20, macro_block_size=1)
    os.replace(str(temporary), str(path))


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


def summarize_actor_gate(
    rows: Sequence[Mapping[str, object]],
    actor_name: str,
    checkpoint_identity: Mapping[str, object],
    init_indices: Sequence[int],
) -> dict:
    expected = len(TASKS) * len(init_indices)
    if len(rows) != expected:
        raise RuntimeError("actor gate incomplete: %d != %d" % (len(rows), expected))
    per_effect = {}
    for task_key, task in TASKS.items():
        subset = [row for row in rows if row["task_key"] == task_key]
        if len(subset) != len(init_indices):
            raise RuntimeError("actor gate task cell incomplete")
        per_effect[task_key] = {
            effect: float(np.mean([row["effect_success"][effect] for row in subset]))
            for effect in task.effects
        }
    minimum = min(value for task in per_effect.values() for value in task.values())
    return {
        "status": "ACTOR_GATE_COMPLETE",
        "actor": actor_name,
        "checkpoint_identity": dict(checkpoint_identity),
        "init_indices": [int(value) for value in init_indices],
        "rollout_count": len(rows),
        "per_effect_success": per_effect,
        "min_per_effect_success": float(minimum),
        "full_task_success_rate": float(
            np.mean([row["full_task_success"] for row in rows])
        ),
        "threshold": ACTOR_GATE_THRESHOLD,
        "pass": bool(minimum >= ACTOR_GATE_THRESHOLD),
        "failure_video_count": int(sum(bool(row.get("failure_video")) for row in rows)),
        "selection_used_gate_results": False,
    }


def run_actor_gate(
    actor: SkillActor,
    actor_name: str,
    checkpoint_identity: Mapping[str, object],
    init_indices: Sequence[int],
    output_dir: Path,
    gate_name: str,
    max_action_steps: int = MAX_ACTION_STEPS,
) -> Tuple[List[dict], dict]:
    """Run a resumable qualification or one-shot formal clean gate."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / (gate_name + "_rollouts.jsonl")
    summary_path = output_dir / (gate_name + "_summary.json")
    marker_path = output_dir / (gate_name.upper() + "_STARTED.json")
    marker = {
        "gate_name": gate_name,
        "actor": actor_name,
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "init_indices": [int(value) for value in init_indices],
        "max_action_steps": int(max_action_steps),
        "max_chunks_per_effect": MAX_CHUNKS_PER_EFFECT,
        "executed_prefix": EXECUTED_PREFIX,
    }
    if marker_path.is_file():
        observed = json.loads(marker_path.read_text(encoding="utf-8"))
        if observed != marker:
            raise RuntimeError("actor gate marker identity drift")
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
        raise RuntimeError("actor gate resume rows have duplicate or unexpected cells")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(rows) != len(expected_keys):
            raise RuntimeError("complete actor-gate summary has incomplete rows")
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
        env = make_env(task, camera_obs=True)
        initial_states = load_init_states(task)
        try:
            for init_index in pending:
                observation = reset_to_state(env, initial_states[init_index])
                for _ in range(5):
                    observation, _, _, _ = env.step(
                        np.zeros((ACTION_DIM,), dtype=np.float32)
                    )
                state_history = deque(
                    [np.asarray(env.get_sim_state()).copy()], maxlen=4
                )
                frames = [_video_frame(observation)]
                effect_success = {effect: False for effect in task.effects}
                chunk_trace = []
                action_steps = 0
                failure_type = None
                for effect_id in task.effects:
                    if effect_truths(env, task)[effect_id]:
                        effect_success[effect_id] = True
                        continue
                    reached = False
                    for chunk_index in range(MAX_CHUNKS_PER_EFFECT):
                        mode = (
                            ExecutionMode.EXECUTE
                            if chunk_index == 0
                            else ExecutionMode.RETRY
                        )
                        input_hash = actor_input_hash(
                            state_history, task_key, effect_id, mode
                        )
                        chunk = np.asarray(
                            actor.action_chunk(
                                state_history, task_key, effect_id, mode
                            ),
                            dtype=np.float32,
                        )
                        if chunk.shape != (8, ACTION_DIM) or not np.isfinite(chunk).all():
                            raise RuntimeError("actor emitted an invalid action chunk")
                        chunk_trace.append(
                            {
                                "effect_id": effect_id,
                                "chunk_index": chunk_index,
                                "execution_mode": mode.value,
                                "actor_input_sha256": input_hash,
                                "action_chunk_sha256": hashlib.sha256(
                                    np.asarray(chunk, dtype="<f4").tobytes(order="C")
                                ).hexdigest(),
                            }
                        )
                        for action in chunk[:EXECUTED_PREFIX]:
                            observation, _, _, _ = env.step(
                                np.asarray(action, dtype=np.float64)
                            )
                            action_steps += 1
                            state_history.append(
                                np.asarray(env.get_sim_state()).copy()
                            )
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
                    if not reached:
                        if failure_type is None:
                            failure_type = "EFFECT_CHUNK_LIMIT:%s" % effect_id
                        break
                full_task_success = bool(
                    all(effect_success.values()) and env.check_success()
                )
                if not full_task_success and failure_type is None:
                    failure_type = "FINAL_TASK_PREDICATE_FALSE"
                video_path = None
                if not full_task_success:
                    video_path = videos_dir / (
                        "%s_%s_init_%02d.mp4" % (gate_name, task_key, init_index)
                    )
                    _write_video(video_path, frames)
                row = {
                    "record_type": "actor_gate",
                    "gate_name": gate_name,
                    "actor": actor_name,
                    "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
                    "normalization_sha256": checkpoint_identity[
                        "normalization_sha256"
                    ],
                    "task_key": task_key,
                    "init_index": init_index,
                    "effect_success": effect_success,
                    "full_task_success": full_task_success,
                    "action_steps": action_steps,
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
                    "PHASE1B_ROLLOUT_PERSISTED gate=%s task=%s init=%d full=%s steps=%d failure=%s"
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
    summary = summarize_actor_gate(
        rows, actor_name, checkpoint_identity, init_indices
    )
    write_json(summary_path, summary)
    print(
        "PHASE1B_ACTOR_GATE gate=%s min_per_effect=%.3f pass=%s"
        % (gate_name, summary["min_per_effect_success"], summary["pass"]),
        flush=True,
    )
    return rows, summary

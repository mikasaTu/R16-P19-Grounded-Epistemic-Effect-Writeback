"""Exact LIBERO demonstration effect-boundary snapshot extraction for Phase-3.

The extractor resets once to the official demonstration's initial simulator
state and then uses only normal ``env.step`` transitions. Existing Phase-1
labels are deliberately not loaded: simulator predicates are independently
recomputed for every replayed action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import h5py
import numpy as np

from .artifacts import atomic_text, write_json
from .config import TASKS
from .simulator import effect_truths, make_env, reset_to_state


STABLE_TRUTH_STEPS = 5
SEGMENT_PRE_ROLL_ACTIONS = 4
OFFICIAL_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SPLITS: Mapping[str, tuple[int, ...]] = {
    "development": tuple(range(0, 20)),
    "calibration": tuple(range(20, 30)),
    "qualification": tuple(range(30, 40)),
    "formal": tuple(range(40, 50)),
}
DATASET_SHA256 = {
    "stove_moka": "6b30906a52a5741e98ef447d27e7066d6c0be4a5f7acd7ecaf1cb7468aca4aa9",
    "bowl_drawer": "703950f48a3c49dfde61be489ade91527f16e1449b4f29a85f2e51153cef3638",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def observation_sha256(observation: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        value = np.ascontiguousarray(np.asarray(observation[key]))
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def first_true_index(values: Sequence[bool]) -> int | None:
    for index, value in enumerate(values):
        if value:
            return index
    return None


def first_stable_true_index(
    values: Sequence[bool], stable_steps: int = STABLE_TRUTH_STEPS
) -> int | None:
    if stable_steps <= 0:
        raise ValueError("stable_steps must be positive")
    streak = 0
    for index, value in enumerate(values):
        streak = streak + 1 if value else 0
        if streak >= stable_steps:
            return index - stable_steps + 1
    return None


def _episode_index(episode_id: str) -> int:
    match = re.fullmatch(r"demo_(\d+)", episode_id)
    if match is None:
        raise ValueError("invalid episode id %r" % episode_id)
    return int(match.group(1))


def _capture_state(env) -> np.ndarray:
    return np.asarray(env.sim.get_state().flatten(), dtype="<f8").copy()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_append(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class ReplayedDemo:
    task_key: str
    episode_id: str
    actions: np.ndarray
    pre_states: tuple[np.ndarray, ...]
    post_states: tuple[np.ndarray, ...]
    pre_observation_sha256: tuple[str, ...]
    post_observation_sha256: tuple[str, ...]
    pre_truths: tuple[Mapping[str, bool], ...]
    post_truths: tuple[Mapping[str, bool], ...]
    final_task_success: bool


def replay_official_demo(env, task_key: str, episode_id: str) -> ReplayedDemo:
    """Replay one explicit demo group without touching any other HDF5 group."""

    task = TASKS[task_key]
    demo_index = _episode_index(episode_id)
    with h5py.File(str(task.dataset_path), "r") as handle:
        demo = handle["data"][episode_id]
        initial_state = np.asarray(demo["states"][0], dtype="<f8")
        actions = np.asarray(demo["actions"], dtype="<f4")

    observation = reset_to_state(env, initial_state, seed=1619 + demo_index)
    pre_states: List[np.ndarray] = []
    post_states: List[np.ndarray] = []
    pre_hashes: List[str] = []
    post_hashes: List[str] = []
    pre_truths: List[Mapping[str, bool]] = []
    post_truths: List[Mapping[str, bool]] = []
    for action in actions:
        pre_states.append(_capture_state(env))
        pre_hashes.append(observation_sha256(observation))
        pre_truths.append(dict(effect_truths(env, task)))
        observation, _, _, _ = env.step(np.asarray(action, dtype=np.float64))
        post_states.append(_capture_state(env))
        post_hashes.append(observation_sha256(observation))
        post_truths.append(dict(effect_truths(env, task)))
    return ReplayedDemo(
        task_key=task_key,
        episode_id=episode_id,
        actions=actions,
        pre_states=tuple(pre_states),
        post_states=tuple(post_states),
        pre_observation_sha256=tuple(pre_hashes),
        post_observation_sha256=tuple(post_hashes),
        pre_truths=tuple(pre_truths),
        post_truths=tuple(post_truths),
        final_task_success=bool(env.check_success()),
    )


def segment_boundaries(replay: ReplayedDemo) -> Dict[str, dict]:
    task = TASKS[replay.task_key]
    stable_indices: Dict[str, int | None] = {}
    first_indices: Dict[str, int | None] = {}
    for effect_id in task.effects:
        timeline = [bool(value[effect_id]) for value in replay.post_truths]
        first_indices[effect_id] = first_true_index(timeline)
        stable_indices[effect_id] = first_stable_true_index(timeline)

    result: Dict[str, dict] = {}
    for effect_index, effect_id in enumerate(task.effects):
        stable_start = stable_indices[effect_id]
        if effect_index == 0:
            first_action = 0
            start = 0
        else:
            previous = stable_indices[task.effects[effect_index - 1]]
            # ``previous`` is the first action in the five-step stable suffix.
            # The next effect begins only after that suffix has completed.  The
            # stored entry includes exactly four preceding actions so replay
            # reconstructs the same transition context without pretending that
            # those pre-roll actions belong to the next effect.
            first_action = (
                0 if previous is None else previous + STABLE_TRUTH_STEPS
            )
            start = max(0, first_action - SEGMENT_PRE_ROLL_ACTIONS)
        stop = None if stable_start is None else min(
            len(replay.actions), stable_start + STABLE_TRUTH_STEPS
        )
        precondition_effects = task.effects[:effect_index]
        precondition_index = min(first_action, max(0, len(replay.pre_truths) - 1))
        entry_precondition_truth = (
            {
                value: bool(replay.pre_truths[start][value])
                for value in precondition_effects
            }
            if replay.pre_truths
            else {}
        )
        precondition_truth = (
            {
                value: bool(replay.pre_truths[precondition_index][value])
                for value in precondition_effects
            }
            if replay.pre_truths
            else {}
        )
        target_truth_at_entry = bool(
            replay.pre_truths and replay.pre_truths[start][effect_id]
        )
        valid = bool(
            stop is not None
            and stop - start >= 2
            and first_action < int(stop)
            and all(precondition_truth.values())
            and not target_truth_at_entry
        )
        result[effect_id] = {
            "effect_index": effect_index,
            "entry_action_index": start,
            "first_action_index_belonging_to_effect": first_action,
            "first_true_action_index": first_indices[effect_id],
            "stable_true_start_action_index": stable_start,
            "stop_action_index_exclusive": stop,
            "entry_precondition_truth": entry_precondition_truth,
            "precondition_truth": precondition_truth,
            "precondition_evaluation_action_index": precondition_index,
            "target_truth_at_entry": target_truth_at_entry,
            "valid": valid,
            "invalid_reason": (
                None
                if valid
                else (
                    "missing_stable_truth_or_precondition_or_minimum_two_actions_"
                    "or_target_already_true"
                )
            ),
        }
    return result


def persist_demo_segments(
    replay: ReplayedDemo,
    split: str,
    output_root: Path,
) -> dict:
    output_root = Path(output_root)
    task = TASKS[replay.task_key]
    boundaries = segment_boundaries(replay)
    rows = []
    for effect_index, effect_id in enumerate(task.effects):
        boundary = boundaries[effect_id]
        start = int(boundary["entry_action_index"])
        stop_value = boundary["stop_action_index_exclusive"]
        snapshot_rel = Path("snapshots") / split / replay.task_key / replay.episode_id / (
            effect_id + ".npz"
        )
        snapshot_path = output_root / snapshot_rel
        if stop_value is None:
            actions = np.empty((0, 7), dtype="<f4")
            stable_post_state = replay.post_states[-1]
            timeline = np.asarray([], dtype=np.bool_)
        else:
            stop = int(stop_value)
            actions = np.asarray(replay.actions[start:stop], dtype="<f4")
            stable_post_state = replay.post_states[stop - 1]
            timeline = np.asarray(
                [value[effect_id] for value in replay.post_truths[start:stop]],
                dtype=np.bool_,
            )
        next_entry_state = (
            replay.pre_states[boundaries[task.effects[effect_index + 1]]["entry_action_index"]]
            if effect_index + 1 < len(task.effects)
            else stable_post_state
        )
        _atomic_npz(
            snapshot_path,
            entry_state=np.asarray(replay.pre_states[start], dtype="<f8"),
            actions=actions,
            stable_post_state=np.asarray(stable_post_state, dtype="<f8"),
            next_effect_entry_state=np.asarray(next_entry_state, dtype="<f8"),
            effect_truth_timeline=timeline,
        )
        row: MutableMapping[str, object] = {
            "schema_version": 1,
            "split": split,
            "task_key": replay.task_key,
            "source_episode": replay.episode_id,
            "source_demo_index": _episode_index(replay.episode_id),
            "effect_id": effect_id,
            **boundary,
            "snapshot_path": str(snapshot_rel),
            "snapshot_sha256": sha256_file(snapshot_path),
            "entry_state_sha256": array_sha256(replay.pre_states[start], "<f8"),
            "action_segment_sha256": array_sha256(actions, "<f4"),
            "stable_post_state_sha256": array_sha256(stable_post_state, "<f8"),
            "next_effect_entry_state_sha256": array_sha256(next_entry_state, "<f8"),
            "entry_observation_sha256": replay.pre_observation_sha256[start],
            "stable_post_observation_sha256": (
                replay.post_observation_sha256[int(stop_value) - 1]
                if stop_value is not None
                else replay.post_observation_sha256[-1]
            ),
            "effect_truth_timeline": timeline.astype(int).tolist(),
            "stable_completion_truth": bool(
                len(timeline) >= STABLE_TRUTH_STEPS
                and bool(np.all(timeline[-STABLE_TRUTH_STEPS:]))
            ),
            "action_count": int(len(actions)),
            "action_dtype": "<f4",
            "simulator_state_dtype": "<f8",
            "source_dataset_path": str(task.dataset_path),
            "source_dataset_sha256": DATASET_SHA256[replay.task_key],
            "official_libero_commit": OFFICIAL_LIBERO_COMMIT,
            "candidate_label_used_as_truth": False,
            "boundary_method": "single_reset_then_normal_env_step_independent_predicates",
            "full_demo_replay_task_success": replay.final_task_success,
        }
        rows.append(dict(row))
    cell = {
        "schema_version": 1,
        "split": split,
        "task_key": replay.task_key,
        "source_episode": replay.episode_id,
        "segment_count": len(rows),
        "valid_segment_count": sum(bool(row["valid"]) for row in rows),
        "full_demo_replay_task_success": replay.final_task_success,
        "segments": rows,
    }
    cell_path = output_root / "cells" / split / replay.task_key / (
        replay.episode_id + ".json"
    )
    write_json(cell_path, cell)
    return cell


def _formal_access_paths(output_root: Path, task_key: str, episode_id: str):
    root = Path(output_root) / "formal_access_ledger" / task_key
    return root / (episode_id + ".started.json"), root / (episode_id + ".complete.json")


def build_snapshot_split(
    split: str,
    output_root: Path,
    task_keys: Iterable[str] = tuple(TASKS),
) -> dict:
    if split not in SPLITS:
        raise ValueError("unknown split %s" % split)
    output_root = Path(output_root)
    task_keys = tuple(task_keys)
    output_root.mkdir(parents=True, exist_ok=True)
    cells: List[dict] = []
    for task_key in task_keys:
        task = TASKS[task_key]
        env = make_env(task, camera_obs=False)
        try:
            for demo_index in SPLITS[split]:
                episode_id = "demo_%d" % demo_index
                cell_path = output_root / "cells" / split / task_key / (
                    episode_id + ".json"
                )
                started = complete = None
                if split == "formal":
                    started, complete = _formal_access_paths(
                        output_root, task_key, episode_id
                    )
                if cell_path.is_file():
                    if split == "formal":
                        if not complete.is_file():
                            raise RuntimeError(
                                "formal cell exists without a complete access ledger: %s:%s"
                                % (task_key, episode_id)
                            )
                        completion = json.loads(complete.read_text(encoding="utf-8"))
                        if sha256_file(cell_path) != completion.get("cell_sha256"):
                            raise RuntimeError("formal cell hash differs from access ledger")
                    cells.append(json.loads(cell_path.read_text(encoding="utf-8")))
                    continue
                if split == "formal":
                    if started.is_file() and not complete.is_file():
                        raise RuntimeError(
                            "formal demo access was interrupted and may not be repeated: %s:%s"
                            % (task_key, episode_id)
                        )
                    write_json(
                        started,
                        {
                            "task_key": task_key,
                            "source_episode": episode_id,
                            "status": "FORMAL_ACCESS_STARTED",
                            "repeat_access_forbidden": True,
                        },
                    )
                replay = replay_official_demo(env, task_key, episode_id)
                cell = persist_demo_segments(replay, split, output_root)
                cells.append(cell)
                if complete is not None:
                    write_json(
                        complete,
                        {
                            "task_key": task_key,
                            "source_episode": episode_id,
                            "status": "FORMAL_ACCESS_COMPLETE",
                            "cell_path": str(cell_path.relative_to(output_root)),
                            "cell_sha256": sha256_file(cell_path),
                        },
                    )
                print(
                    "PHASE3_SNAPSHOT_CELL split=%s task=%s episode=%s valid=%d/%d"
                    % (
                        split,
                        task_key,
                        episode_id,
                        cell["valid_segment_count"],
                        cell["segment_count"],
                    ),
                    flush=True,
                )
        finally:
            env.close()
    cells.sort(key=lambda value: (value["task_key"], value["source_episode"]))
    rows = [row for cell in cells for row in cell["segments"]]
    manifest = {
        "schema_version": 1,
        "status": "SNAPSHOT_SPLIT_COMPLETE",
        "split": split,
        "demo_indices": list(SPLITS[split]),
        "task_keys": sorted(task_keys),
        "episode_count": len(cells),
        "segment_count": len(rows),
        "valid_segment_count": sum(bool(row["valid"]) for row in rows),
        "stable_truth_steps": STABLE_TRUTH_STEPS,
        "segment_pre_roll_actions": SEGMENT_PRE_ROLL_ACTIONS,
        "formal_demo_accessed": split == "formal",
        "segments": rows,
    }
    manifest_path = output_root / ("snapshot_bank_%s_manifest.json" % split)
    write_json(manifest_path, manifest)
    return manifest


def combine_snapshot_manifests(output_root: Path, splits: Sequence[str]) -> dict:
    output_root = Path(output_root)
    manifests = []
    for split in splits:
        path = output_root / ("snapshot_bank_%s_manifest.json" % split)
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    rows = [row for manifest in manifests for row in manifest["segments"]]
    result = {
        "schema_version": 1,
        "status": "SNAPSHOT_BANK_COMPLETE",
        "splits": list(splits),
        "segment_count": len(rows),
        "valid_segment_count": sum(bool(row["valid"]) for row in rows),
        "formal_demo_accessed": "formal" in splits,
        "source_manifests": [
            {
                "split": split,
                "path": "snapshot_bank_%s_manifest.json" % split,
                "sha256": sha256_file(
                    output_root / ("snapshot_bank_%s_manifest.json" % split)
                ),
            }
            for split in splits
        ],
        "segments": sorted(
            rows,
            key=lambda row: (
                row["split"],
                row["task_key"],
                row["source_demo_index"],
                row["effect_index"],
            ),
        ),
    }
    write_json(output_root / "snapshot_bank_manifest.json", result)
    return result


def append_access_audit(path: Path, value: Mapping[str, object]) -> None:
    """Public fsynced JSONL helper used by the stage launcher."""

    _fsync_append(path, value)

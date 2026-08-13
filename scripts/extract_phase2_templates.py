#!/usr/bin/env python3
"""Extract the preregistered Phase-2 geometric skill templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "glx")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r16p19.config import TASKS  # noqa: E402
from r16p19.phase2_executor import (  # noqa: E402
    extract_geometric_snapshot,
    pose_to_local,
)
from r16p19.simulator import make_env  # noqa: E402


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "r16p19_libero_phase2"
LABELS_PATH = (
    PROJECT_ROOT
    / "artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment/demo_effect_labels.jsonl"
)
TRAIN_EPISODES = tuple("demo_%d" % index for index in range(30))
MAX_TEMPLATES = 3
MAX_WAYPOINTS = 48
ARC_STEP_M = 0.015


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_labels(path: Path) -> Dict[tuple, dict]:
    values = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            values[(row["task_key"], row["episode_id"])] = row
    return values


def _transition(label: Mapping[str, object], effect_id: str):
    stable = label.get("stable_transition_indices", {})
    transitions = label.get("transition_indices", {})
    if effect_id in stable:
        return int(stable[effect_id])
    if effect_id in transitions:
        return int(transitions[effect_id])
    return None


def _segment(label: Mapping[str, object], effects: Sequence[str], effect_index: int, length: int):
    current = _transition(label, effects[effect_index])
    if current is None:
        return None
    if effect_index == 0:
        start = 0
    else:
        previous = _transition(label, effects[effect_index - 1])
        if previous is None:
            return None
        start = max(0, previous - 4)
    stop = min(int(length), current + 5)
    return (start, stop) if stop - start >= 2 else None


def _sample_indices(positions: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    count = len(positions)
    required = {0, count - 1}
    changed = np.flatnonzero(gripper[1:] != gripper[:-1]) + 1
    for index in changed:
        required.add(max(0, int(index) - 1))
        required.add(int(index))
        required.add(min(count - 1, int(index) + 1))
    last = 0
    distance = 0.0
    for index in range(1, count):
        distance += float(np.linalg.norm(positions[index] - positions[index - 1]))
        if distance >= ARC_STEP_M:
            required.add(index)
            last = index
            distance = 0.0
    required.add(last)
    selected = sorted(required)
    if len(selected) > MAX_WAYPOINTS:
        protected = sorted({0, count - 1}.union(int(value) for value in changed))
        remaining = [value for value in selected if value not in protected]
        slots = MAX_WAYPOINTS - len(protected)
        if slots < 0:
            raise RuntimeError("gripper transitions exceed waypoint budget")
        if slots:
            keep = np.linspace(0, len(remaining) - 1, slots, dtype=np.int64)
            selected = sorted(protected + [remaining[int(index)] for index in keep])
        else:
            selected = protected
    return np.asarray(selected, dtype=np.int64)


def _descriptor(candidate: Mapping[str, object]) -> np.ndarray:
    local_positions = np.asarray(candidate["local_positions"], dtype=np.float64)
    local_quaternions = np.asarray(candidate["local_quaternions_xyzw"], dtype=np.float64)
    return np.concatenate(
        (local_positions[0], local_quaternions[0], local_positions[-1], local_quaternions[-1])
    )


def _select_representatives(candidates: Sequence[dict], count: int = MAX_TEMPLATES) -> List[dict]:
    ordered = sorted(candidates, key=lambda value: value["source_episode"])
    descriptors = np.stack([_descriptor(value) for value in ordered])
    scale = np.maximum(descriptors.std(axis=0), 1e-6)
    normalized = (descriptors - descriptors.mean(axis=0)) / scale
    distances = np.linalg.norm(normalized[:, None] - normalized[None], axis=-1)
    first = int(np.argmin(distances.sum(axis=1)))
    selected = [first]
    while len(selected) < min(count, len(ordered)):
        remaining = [index for index in range(len(ordered)) if index not in selected]
        nearest = [float(np.min(distances[index, selected])) for index in remaining]
        maximum = max(nearest)
        tied = [index for index, value in zip(remaining, nearest) if abs(value - maximum) < 1e-12]
        selected.append(min(tied, key=lambda index: ordered[index]["source_episode"]))
    return [ordered[index] for index in selected]


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _extract_candidates(task_key: str, labels: Mapping[tuple, dict]) -> Dict[str, List[dict]]:
    task = TASKS[task_key]
    result = {effect_id: [] for effect_id in task.effects}
    env = make_env(task, camera_obs=False)
    try:
        with h5py.File(str(task.dataset_path), "r") as handle:
            for episode_id in TRAIN_EPISODES:
                label = labels[(task_key, episode_id)]
                if not bool(label["final_success"]):
                    continue
                demo = handle["data"][episode_id]
                states = np.asarray(demo["states"], dtype=np.float64)
                actions = np.asarray(demo["actions"], dtype=np.float32)
                length = min(len(states), len(actions))
                for effect_index, effect_id in enumerate(task.effects):
                    if effect_id in label.get("inferred_transition_methods", {}):
                        continue
                    segment = _segment(label, task.effects, effect_index, length)
                    if segment is None:
                        continue
                    start, stop = segment
                    observation = env.set_init_state(states[start])
                    source = extract_geometric_snapshot(env, observation, task_key, effect_id)
                    world_positions = []
                    world_quaternions = []
                    for index in range(start, stop):
                        observation = env.set_init_state(states[index])
                        snapshot = extract_geometric_snapshot(env, observation, task_key, effect_id)
                        world_positions.append(snapshot.eef_position)
                        world_quaternions.append(snapshot.eef_quaternion_xyzw)
                    world_positions_array = np.asarray(world_positions, dtype=np.float32)
                    world_quaternions_array = np.asarray(world_quaternions, dtype=np.float32)
                    gripper = np.where(actions[start:stop, 6] > 0.0, 1.0, -1.0).astype(np.float32)
                    indices = _sample_indices(world_positions_array, gripper)
                    local_positions = []
                    local_quaternions = []
                    for position, quaternion in zip(
                        world_positions_array[indices], world_quaternions_array[indices]
                    ):
                        local_position, local_quaternion = pose_to_local(
                            position,
                            quaternion,
                            source.effect_fixture_position,
                            source.effect_fixture_quaternion_xyzw,
                        )
                        local_positions.append(local_position)
                        local_quaternions.append(local_quaternion)
                    candidate = {
                        "schema_version": 1,
                        "template_id": "%s__%s__%s" % (task_key, effect_id, episode_id),
                        "task_key": task_key,
                        "effect_id": effect_id,
                        "source_episode": episode_id,
                        "source_segment": {"start": start, "stop_exclusive": stop},
                        "source_frame_position": source.effect_fixture_position.tolist(),
                        "source_frame_quaternion_xyzw": source.effect_fixture_quaternion_xyzw.tolist(),
                        "local_positions": np.asarray(local_positions).tolist(),
                        "local_quaternions_xyzw": np.asarray(local_quaternions).tolist(),
                        "world_positions": world_positions_array[indices].tolist(),
                        "world_quaternions_xyzw": world_quaternions_array[indices].tolist(),
                        "actions": actions[start:stop][indices].tolist(),
                        "gripper": gripper[indices].tolist(),
                        "descriptor": [],
                        "waypoint_source_indices": (indices + start).tolist(),
                    }
                    candidate["descriptor"] = _descriptor(candidate).tolist()
                    result[effect_id].append(candidate)
    finally:
        env.close()
    return result


def extract(output_root: Path) -> dict:
    output_root = Path(output_root)
    template_root = output_root / "skill_templates"
    template_root.mkdir(parents=True, exist_ok=True)
    labels = _read_labels(LABELS_PATH)
    manifest_rows = []
    candidate_summary = {}
    for task_key in TASKS:
        candidates = _extract_candidates(task_key, labels)
        candidate_summary[task_key] = {}
        for effect_id, values in candidates.items():
            selected = _select_representatives(values)
            candidate_summary[task_key][effect_id] = {
                "candidate_count": len(values),
                "selected_source_episodes": [value["source_episode"] for value in selected],
            }
            for value in selected:
                filename = value["template_id"] + ".json"
                path = template_root / filename
                raw = _canonical_json(value)
                path.write_bytes(raw)
                manifest_rows.append(
                    {
                        "task_key": task_key,
                        "effect_id": effect_id,
                        "template_id": value["template_id"],
                        "source_episode": value["source_episode"],
                        "waypoint_count": len(value["local_positions"]),
                        "path": "skill_templates/" + filename,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    manifest = {
        "schema_version": 1,
        "executor_family": "RetargetedGeometricSkillExecutor",
        "official_libero_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "source_demo_indices": list(range(30)),
        "calibration_demo_indices_used_for_extraction": [],
        "diagnostic_demo_indices_used": [],
        "source_labels_path": str(LABELS_PATH.relative_to(PROJECT_ROOT)),
        "source_labels_sha256": _sha256(LABELS_PATH),
        "maximum_templates_per_effect": MAX_TEMPLATES,
        "templates": sorted(
            manifest_rows,
            key=lambda value: (value["task_key"], value["effect_id"], value["sha256"]),
        ),
    }
    (output_root / "skill_template_extraction_manifest.json").write_bytes(
        _canonical_json(manifest)
    )
    # Before calibration the complete extraction manifest is also executable.
    # The calibration script later replaces this alias with its selected
    # subset while retaining the complete extraction manifest above.
    (output_root / "skill_template_manifest.json").write_bytes(
        _canonical_json(manifest)
    )
    (output_root / "skill_template_candidates.json").write_bytes(
        _canonical_json(
            {
                "schema_version": 1,
                "selection_algorithm": "deterministic_farthest_first_medoid",
                "selection_seed": 1619,
                "summary": candidate_summary,
            }
        )
    )
    return manifest


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args(argv)
    manifest = extract(args.output_root)
    print(
        "PHASE2_TEMPLATES_EXTRACTED count=%d effects=%d"
        % (len(manifest["templates"]), len({row["effect_id"] for row in manifest["templates"]}))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Deterministic Phase-1B actor-data and official-init audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r16p19.config import CALIBRATION_EPISODES, TASKS, TRACE_TEST_EPISODES, TRAIN_EPISODES


DEFAULT_LABELS = (
    ROOT
    / "artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment"
    / "demo_effect_labels.jsonl"
)
DEFAULT_BASELINE_RESULTS = (
    ROOT
    / "artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment"
    / "state_bc_results.jsonl"
)
DEFAULT_OUTPUT = ROOT / "experiments/r16p19_libero_phase1b/actor_data_audit.json"
HISTORY_LENGTH = 4
ACTION_HORIZON = 8
EXECUTED_PREFIX = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(values: Sequence[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    return {
        "count": int(len(array)),
        "min": int(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def transition_index(label: Mapping[str, object], effect: str):
    stable = label.get("stable_transition_indices", {})
    transitions = label.get("transition_indices", {})
    if effect in stable:
        return int(stable[effect])
    if effect in transitions:
        return int(transitions[effect])
    return None


def effect_segment(label: Mapping[str, object], effects: Sequence[str], effect_index: int, length: int):
    """Return the causal action segment for one effect.

    The segment ends at the registered physical transition and overlaps the
    preceding boundary by the fixed executed-prefix length. This gives a
    realizable release segment when placement and release share a transition,
    without copying an episode into another source split.
    """

    current = transition_index(label, effects[effect_index])
    if current is None:
        return None
    if effect_index == 0:
        start = 0
    else:
        previous = transition_index(label, effects[effect_index - 1])
        start = max(0, int(previous or 0) - EXECUTED_PREFIX)
    stop = min(int(length), int(current) + 1)
    if stop <= start:
        return None
    return start, stop


def split_mapping() -> Dict[str, Sequence[str]]:
    return {
        "train": TRAIN_EPISODES,
        "calibration": CALIBRATION_EPISODES,
        "trace_test": TRACE_TEST_EPISODES,
    }


def alignment_and_effect_stats(task_key: str, labels: Mapping[str, dict]) -> dict:
    task = TASKS[task_key]
    result = {"splits": {}}
    with h5py.File(str(task.dataset_path), "r") as handle:
        source_episodes = set(handle["data"].keys())
        for split_name, episodes in split_mapping().items():
            mismatches = []
            transition_summaries = {}
            effect_lengths: Dict[str, List[int]] = {effect: [] for effect in task.effects}
            padded_chunks = Counter()
            full_horizon_chunks = Counter()
            missing_effect_episodes: Dict[str, List[str]] = {effect: [] for effect in task.effects}
            for episode in episodes:
                if episode not in source_episodes:
                    mismatches.append({"episode": episode, "error": "missing_source_episode"})
                    continue
                demo = handle["data"][episode]
                state_count = int(len(demo["states"]))
                action_count = int(len(demo["actions"]))
                if state_count != action_count:
                    mismatches.append(
                        {"episode": episode, "state_count": state_count, "action_count": action_count}
                    )
                label = labels[episode]
                for effect_index, effect in enumerate(task.effects):
                    segment = effect_segment(label, task.effects, effect_index, min(state_count, action_count))
                    if segment is None:
                        missing_effect_episodes[effect].append(episode)
                        continue
                    segment_length = int(segment[1] - segment[0])
                    effect_lengths[effect].append(segment_length)
                    padded_chunks[effect] += segment_length
                    full_horizon_chunks[effect] += max(0, segment_length - ACTION_HORIZON + 1)
            for effect in task.effects:
                indices = [
                    transition_index(labels[episode], effect)
                    for episode in episodes
                    if transition_index(labels[episode], effect) is not None
                ]
                transition_summaries[effect] = {
                    **summarize(indices),
                    "missing_episodes": missing_effect_episodes[effect],
                }
            result["splits"][split_name] = {
                "episode_count": len(episodes),
                "state_action_alignment": {
                    "aligned_episode_count": len(episodes) - len(mismatches),
                    "mismatch_count": len(mismatches),
                    "mismatches": mismatches,
                },
                "transition_boundaries": transition_summaries,
                "effect_segment_length_distribution": {
                    effect: summarize(effect_lengths[effect]) for effect in task.effects
                },
                "training_frames_per_effect": {
                    effect: int(sum(effect_lengths[effect])) for effect in task.effects
                },
                "padded_action_chunks_per_effect": {
                    effect: int(padded_chunks[effect]) for effect in task.effects
                },
                "full_horizon_action_chunks_per_effect": {
                    effect: int(full_horizon_chunks[effect]) for effect in task.effects
                },
            }
    return result


def action_statistics() -> dict:
    per_task = {}
    combined = []
    for task_key, task in TASKS.items():
        values = []
        with h5py.File(str(task.dataset_path), "r") as handle:
            for episode in TRAIN_EPISODES:
                values.append(np.asarray(handle["data"][episode]["actions"], dtype=np.float64))
        actions = np.concatenate(values, axis=0)
        combined.append(actions)
        per_task[task_key] = action_summary(actions)
    return {"per_task": per_task, "combined_train": action_summary(np.concatenate(combined, axis=0))}


def action_summary(actions: np.ndarray) -> dict:
    gripper = actions[:, 6]
    positive = int(np.sum(gripper > 0.5))
    negative = int(np.sum(gripper < -0.5))
    neutral = int(len(gripper) - positive - negative)
    binary_total = positive + negative
    weights = {
        "positive": float(binary_total / (2 * positive)) if positive else None,
        "negative": float(binary_total / (2 * negative)) if negative else None,
    }
    return {
        "action_count": int(len(actions)),
        "dimension_labels": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "gripper_class_balance": {
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "positive_fraction": float(positive / len(gripper)),
            "negative_fraction": float(negative / len(gripper)),
            "neutral_fraction": float(neutral / len(gripper)),
            "inverse_frequency_binary_loss_weights": weights,
        },
    }


def baseline_failure_audit(rows: Iterable[Mapping[str, object]]) -> dict:
    selected = [row for row in rows if row.get("actor") == "retrieval_augmented_tiny_mlp"]
    failed = [row for row in selected if not row.get("full_task_success")]
    per_effect = Counter()
    first_failed = Counter()
    for row in failed:
        task = TASKS[str(row["task_key"])]
        failures = [effect for effect in task.effects if not row["effect_success"][effect]]
        per_effect.update(failures)
        if failures:
            first_failed[failures[0]] += 1
    video_extensions = {".mp4", ".avi", ".mov", ".webm"}
    formal_root = DEFAULT_BASELINE_RESULTS.parents[1]
    videos = sorted(
        str(path.relative_to(ROOT))
        for path in formal_root.rglob("*")
        if path.is_file() and path.suffix.lower() in video_extensions
    )
    return {
        "actor": "retrieval_augmented_tiny_mlp",
        "rollout_count": len(selected),
        "failed_full_task_rollouts": len(failed),
        "effect_failure_count": dict(sorted(per_effect.items())),
        "first_failed_effect_count": dict(sorted(first_failed.items())),
        "failure_video_inventory": {
            "count": len(videos),
            "paths": videos,
            "interpretation": "Phase-1 did not persist competence videos" if not videos else "available",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    label_rows = read_jsonl(args.labels)
    if len(label_rows) != 100:
        raise RuntimeError("expected 100 frozen demo labels, got %d" % len(label_rows))
    by_task: Dict[str, Dict[str, dict]] = {task_key: {} for task_key in TASKS}
    for row in label_rows:
        by_task[row["task_key"]][row["episode_id"]] = row

    splits = {name: set(values) for name, values in split_mapping().items()}
    overlap = {
        "train_calibration": sorted(splits["train"] & splits["calibration"]),
        "train_trace_test": sorted(splits["train"] & splits["trace_test"]),
        "calibration_trace_test": sorted(splits["calibration"] & splits["trace_test"]),
    }
    if any(overlap.values()):
        raise RuntimeError("source episode split overlap: %r" % overlap)

    tasks = {}
    for task_key, task in TASKS.items():
        initial_states = np.asarray(torch.load(str(task.init_path), map_location="cpu"))
        tasks[task_key] = {
            "dataset_path": str(task.dataset_path),
            "dataset_sha256": sha256_file(task.dataset_path),
            "init_path": str(task.init_path),
            "init_sha256": sha256_file(task.init_path),
            "available_initial_states": int(initial_states.shape[0]),
            "initial_state_shape": list(initial_states.shape),
            **alignment_and_effect_stats(task_key, by_task[task_key]),
        }

    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "audit_source": {
            "audit_script": str(Path(__file__).resolve().relative_to(ROOT)),
            "audit_script_sha256": sha256_file(Path(__file__).resolve()),
            "frozen_demo_labels": str(args.labels.relative_to(ROOT)),
            "frozen_demo_labels_sha256": sha256_file(args.labels),
            "baseline_actor_results": str(args.baseline_results.relative_to(ROOT)),
            "baseline_actor_results_sha256": sha256_file(args.baseline_results),
        },
        "actor_training_sample_policy": {
            "history_length": HISTORY_LENGTH,
            "action_horizon": ACTION_HORIZON,
            "executed_prefix": EXECUTED_PREFIX,
            "effect_segment_start": "previous_effect_transition_minus_executed_prefix_or_episode_start",
            "effect_segment_stop": "current_effect_transition_plus_one_exclusive",
            "missing_effect_policy": "exclude_missing_effect_segment_without_fabrication",
            "short_segment_policy": "pad_action_chunk_with_last_action_inside_same_effect_segment",
            "cross_split_episode_copying": False,
        },
        "episode_split_disjointness": {
            "overlaps": overlap,
            "all_pairwise_disjoint": not any(overlap.values()),
            "train_count": len(TRAIN_EPISODES),
            "calibration_count": len(CALIBRATION_EPISODES),
            "trace_test_count": len(TRACE_TEST_EPISODES),
        },
        "tasks": tasks,
        "train_action_statistics": action_statistics(),
        "baseline_actor_failures": baseline_failure_audit(read_jsonl(args.baseline_results)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print("PHASE1B_DATA_AUDIT_COMPLETE output=%s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

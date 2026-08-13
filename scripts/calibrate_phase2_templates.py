#!/usr/bin/env python3
"""Calibrate the extracted Phase-2 templates only on demo 30--39."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r16p19.artifacts import atomic_text, write_json  # noqa: E402
from r16p19.config import ACTION_DIM, TASKS  # noqa: E402
from r16p19.phase2_executor import (  # noqa: E402
    ExecutionMode,
    ExecutorVariant,
    RetargetedGeometricSkillExecutor,
    extract_geometric_snapshot,
)
from r16p19.simulator import effect_truths, make_env, reset_to_state  # noqa: E402


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/r16p19_libero_phase2"
LABELS_PATH = (
    PROJECT_ROOT
    / "artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment/"
    "demo_effect_labels.jsonl"
)
CALIBRATION_EPISODES = tuple("demo_%d" % index for index in range(30, 40))
MAX_CHUNKS = 40
EXECUTED_PREFIX = 4


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
            if line.strip():
                row = json.loads(line)
                values[(row["task_key"], row["episode_id"])] = row
    return values


def _append_row(path: Path, row: Mapping[str, object]) -> None:
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


def _transition(label: Mapping[str, object], effect_id: str) -> Optional[int]:
    stable = label.get("stable_transition_indices", {})
    transitions = label.get("transition_indices", {})
    if effect_id in stable:
        return int(stable[effect_id])
    if effect_id in transitions:
        return int(transitions[effect_id])
    return None


def _segment_start(
    label: Mapping[str, object], effects: Sequence[str], effect_index: int
) -> Optional[int]:
    if _transition(label, effects[effect_index]) is None:
        return None
    if effect_index == 0:
        return 0
    previous = _transition(label, effects[effect_index - 1])
    return None if previous is None else max(0, previous - 4)


def _run_template_unit(
    env,
    executor: RetargetedGeometricSkillExecutor,
    task_key: str,
    effect_id: str,
    template_sha256: str,
    state: np.ndarray,
    seed: int,
) -> dict:
    task = TASKS[task_key]
    executor.reset_episode()
    executor._template_order_sha256[(task_key, effect_id)] = (template_sha256,)
    observation = reset_to_state(env, state, seed=seed)
    for _ in range(5):
        observation, _, _, _ = env.step(
            np.zeros((ACTION_DIM,), dtype=np.float32)
        )
    history = deque(
        [extract_geometric_snapshot(env, observation, task_key, effect_id)],
        maxlen=4,
    )
    reached = bool(effect_truths(env, task)[effect_id])
    action_steps = 0
    chunks = 0
    while not reached and chunks < MAX_CHUNKS:
        chunk = executor.action_chunk(
            history, task_key, effect_id, ExecutionMode.EXECUTE, 0
        )
        for action in chunk[:EXECUTED_PREFIX]:
            observation, _, _, _ = env.step(np.asarray(action, dtype=np.float64))
            action_steps += 1
            history.append(
                extract_geometric_snapshot(env, observation, task_key, effect_id)
            )
            reached = bool(effect_truths(env, task)[effect_id])
            if reached:
                break
        chunks += 1
    return {
        "success": bool(reached),
        "action_steps": action_steps,
        "chunks": chunks,
    }


def calibrate(
    extraction_manifest_path: Path,
    output_manifest_path: Path,
    output_dir: Path,
) -> dict:
    extraction_manifest_path = Path(extraction_manifest_path)
    output_manifest_path = Path(output_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    labels = _read_labels(LABELS_PATH)
    rule_path = EXPERIMENT_ROOT / "template_calibration_rule.json"
    marker = {
        "schema_version": 1,
        "extraction_manifest_sha256": _sha256(extraction_manifest_path),
        "calibration_rule_sha256": _sha256(rule_path),
        "calibration_episodes": list(CALIBRATION_EPISODES),
        "maximum_chunks": MAX_CHUNKS,
        "executed_prefix": EXECUTED_PREFIX,
        "controller": {
            "position_gain": 8.0,
            "orientation_gain": 2.0,
            "position_tolerance_m": 0.010,
        },
    }
    marker_path = output_dir / "TEMPLATE_CALIBRATION_STARTED.json"
    if marker_path.is_file():
        if json.loads(marker_path.read_text(encoding="utf-8")) != marker:
            raise RuntimeError("template calibration resume identity drift")
    else:
        atomic_text(marker_path, json.dumps(marker, indent=2, sort_keys=True) + "\n")
    results_path = output_dir / "template_calibration_results.jsonl"
    rows: List[dict] = _read_rows(results_path)
    observed = {
        (
            row["task_key"],
            row["effect_id"],
            row["template_sha256"],
            row["calibration_episode"],
        )
        for row in rows
    }
    if len(observed) != len(rows):
        raise RuntimeError("duplicate template calibration resume cells")
    executor = RetargetedGeometricSkillExecutor(
        extraction_manifest_path,
        variant=ExecutorVariant.FROZEN_FULL,
        position_gain=8.0,
        orientation_gain=2.0,
        position_tolerance_m=0.010,
        retry_reapproach=False,
    )
    for task_ordinal, (task_key, task) in enumerate(TASKS.items()):
        env = make_env(task, camera_obs=False)
        try:
            with h5py.File(str(task.dataset_path), "r") as handle:
                for effect_index, effect_id in enumerate(task.effects):
                    template_rows = [
                        row
                        for row in extraction["templates"]
                        if row["task_key"] == task_key
                        and row["effect_id"] == effect_id
                    ]
                    for template_row in template_rows:
                        for episode_id in CALIBRATION_EPISODES:
                            cell_key = (
                                task_key,
                                effect_id,
                                template_row["sha256"],
                                episode_id,
                            )
                            if cell_key in observed:
                                continue
                            label = labels[(task_key, episode_id)]
                            exclusion = label.get("inferred_transition_methods", {})
                            if effect_id in exclusion:
                                continue
                            start = _segment_start(label, task.effects, effect_index)
                            if start is None:
                                continue
                            states = handle["data"][episode_id]["states"]
                            seed = (
                                2619
                                + 1000 * task_ordinal
                                + int(episode_id.split("_")[1])
                            )
                            result = _run_template_unit(
                                env,
                                executor,
                                task_key,
                                effect_id,
                                template_row["sha256"],
                                np.asarray(states[start], dtype=np.float64),
                                seed,
                            )
                            row = {
                                "record_type": "phase2_template_calibration",
                                "task_key": task_key,
                                "effect_id": effect_id,
                                "template_id": template_row["template_id"],
                                "template_sha256": template_row["sha256"],
                                "calibration_episode": episode_id,
                                "segment_start": int(start),
                                "rollout_seed": seed,
                                **result,
                            }
                            _append_row(results_path, row)
                            rows.append(row)
                            observed.add(cell_key)
                            print(
                                "PHASE2_TEMPLATE_CALIBRATION task=%s effect=%s "
                                "template=%s episode=%s success=%s steps=%d"
                                % (
                                    task_key,
                                    effect_id,
                                    template_row["template_id"],
                                    episode_id,
                                    result["success"],
                                    result["action_steps"],
                                ),
                                flush=True,
                            )
        finally:
            env.close()

    selected_hashes = set()
    summary: Dict[str, Dict[str, object]] = {}
    for task_key, task in TASKS.items():
        summary[task_key] = {}
        for effect_id in task.effects:
            candidates = []
            for template_row in extraction["templates"]:
                if (
                    template_row["task_key"] != task_key
                    or template_row["effect_id"] != effect_id
                ):
                    continue
                subset = [
                    row
                    for row in rows
                    if row["template_sha256"] == template_row["sha256"]
                ]
                success_rate = float(np.mean([row["success"] for row in subset]))
                success_steps = [
                    row["action_steps"] for row in subset if row["success"]
                ]
                candidates.append(
                    {
                        "template_id": template_row["template_id"],
                        "template_sha256": template_row["sha256"],
                        "calibration_count": len(subset),
                        "success_rate": success_rate,
                        "mean_success_action_steps": (
                            float(np.mean(success_steps)) if success_steps else None
                        ),
                    }
                )
            if not candidates:
                raise RuntimeError("no calibration candidates for %s/%s" % (task_key, effect_id))
            best = max(value["success_rate"] for value in candidates)
            threshold = max(best - 0.10, 0.90 if best >= 0.90 else 0.0)
            retained = [
                value
                for value in candidates
                if value["success_rate"] + 1e-12 >= threshold
            ]
            retained.sort(
                key=lambda value: (
                    -value["success_rate"],
                    value["mean_success_action_steps"]
                    if value["mean_success_action_steps"] is not None
                    else float("inf"),
                    value["template_sha256"],
                )
            )
            selected_hashes.update(value["template_sha256"] for value in retained)
            summary[task_key][effect_id] = {
                "best_success_rate": best,
                "retention_threshold": threshold,
                "candidates": candidates,
                "retained_template_sha256": [
                    value["template_sha256"] for value in retained
                ],
            }
    selected_rows = [
        row for row in extraction["templates"] if row["sha256"] in selected_hashes
    ]
    selected_manifest = dict(extraction)
    selected_manifest["template_manifest_role"] = "demo_30_to_39_calibrated_subset"
    selected_manifest["extraction_manifest_path"] = str(
        extraction_manifest_path.relative_to(PROJECT_ROOT)
    )
    selected_manifest["extraction_manifest_sha256"] = _sha256(
        extraction_manifest_path
    )
    selected_manifest["calibration_rule_path"] = (
        "experiments/r16p19_libero_phase2/template_calibration_rule.json"
    )
    selected_manifest["calibration_demo_indices_used"] = list(range(30, 40))
    selected_manifest["templates"] = sorted(
        selected_rows,
        key=lambda value: (
            value["task_key"],
            value["effect_id"],
            value["sha256"],
        ),
    )
    write_json(output_dir / "template_calibration_summary.json", summary)
    write_json(output_manifest_path, selected_manifest)
    result = {
        "calibration_rollout_count": len(rows),
        "extracted_template_count": len(extraction["templates"]),
        "selected_template_count": len(selected_rows),
        "selected_manifest_path": str(output_manifest_path),
        "selected_manifest_sha256": _sha256(output_manifest_path),
        "summary": summary,
    }
    write_json(output_dir / "template_calibration_selection.json", result)
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extraction-manifest",
        type=Path,
        default=EXPERIMENT_ROOT / "skill_template_extraction_manifest.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=EXPERIMENT_ROOT / "skill_template_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/phase2_seeded/template_calibration",
    )
    args = parser.parse_args(argv)
    result = calibrate(
        args.extraction_manifest, args.output_manifest, args.output_dir
    )
    print(
        "PHASE2_TEMPLATE_CALIBRATION_COMPLETE rollouts=%d selected=%d manifest=%s"
        % (
            result["calibration_rollout_count"],
            result["selected_template_count"],
            result["selected_manifest_sha256"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

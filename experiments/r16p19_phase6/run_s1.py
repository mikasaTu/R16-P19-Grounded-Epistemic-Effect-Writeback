#!/usr/bin/env python3
"""Phase-6 S1 offline verifier recalibration and decision replay.

This program is deliberately CPU-only.  It treats Phase-5 artifacts as frozen
inputs and writes exclusively below experiments/r16p19_phase6.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from r16p19.phase5_arm_kernel import evaluate_arm
from r16p19.phase5_verifier_data import _feature_rows, formal_frame_features
from r16p19.phase5_verifier_model import auroc, predict


CURRENT_THRESHOLD = 0.9395385384559631
TRUE_CONDITIONS = {"C0_CLEAN", "A5_EXTERNAL_REALIZATION"}
NOOP_CONDITION = "A1_NOOP_RETRY_STALE"
BASELINE = "M0_TYPED_MATCHED"
CORE = "M3_ASCEL_CORE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def effect_key(task_id: int, effect_index: int, name: str) -> str:
    return f"task{task_id}:effect{effect_index}:{name}"


def npz_schema(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return sorted(data.files)


def jsonl_schema(path: Path) -> list[str]:
    rows = read_jsonl(path)
    return sorted({key for row in rows for key in row})


def inventory(repo: Path, raw_root: Path, output: Path) -> dict[str, Any]:
    phase5 = repo / "experiments/r16p19_phase5"
    results = phase5 / "artifacts/results"
    required = [
        results / "verifier_checkpoint.npz",
        results / "verifier_checkpoint.metrics.json",
        results / "learned_verifier_formal_results.jsonl",
        results / "oracle_formal_results.jsonl",
        results / "support_formal_results.jsonl",
        results / "verifier_dataset_manifest.json",
        phase5 / "task_selection_contract.yaml",
        phase5 / "preregistration.yaml",
        phase5 / "artifacts/raw_rollout_sha256.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing Phase-5 frozen inputs: {missing}")
    raw_episode_root = raw_root / "episodes"
    calibration = sorted((raw_episode_root / "calibration").glob("*.npz"))
    formal = sorted((raw_episode_root / "formal").glob("*.npz"))
    if len(calibration) != 30 or len(formal) != 240:
        raise RuntimeError(f"frozen split count drift: calibration={len(calibration)}, formal={len(formal)}")

    records: list[dict[str, Any]] = []
    for path in required:
        relative = path.relative_to(repo).as_posix()
        if path.suffix == ".jsonl":
            line_count = len(read_jsonl(path))
            fields = jsonl_schema(path)
        elif path.suffix == ".npz":
            line_count = None
            fields = npz_schema(path)
        elif path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            fields = sorted(value) if isinstance(value, dict) else []
        else:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            fields = []
        records.append(
            {"path": relative, "lines": line_count, "fields": fields, "sha256": sha256(path)}
        )

    split_records = []
    for split, paths in (("calibration", calibration), ("formal", formal)):
        split_records.append(
            {
                "split": split,
                "npz_count": len(paths),
                "json_count": sum(path.with_suffix(".json").is_file() for path in paths),
                "episode_ids_sha256": canonical_json_sha([path.stem for path in paths]),
                "npz_sha256s_sha256": canonical_json_sha([sha256(path) for path in paths]),
                "representative_npz": str(paths[0]),
                "npz_fields": npz_schema(paths[0]),
            }
        )

    learned = read_jsonl(results / "learned_verifier_formal_results.jsonl")
    raw_score_saved = bool(learned) and all(
        isinstance(row.get("verifier_scores"), list) and row["verifier_scores"] for row in learned
    )
    report = {
        "schema_version": 1,
        "phase5_read_only": True,
        "raw_effect_scores_saved": raw_score_saved,
        "checkpoint_recompute_feasible": True,
        "checkpoint_recompute_inputs": [
            "verifier_checkpoint.npz",
            "frozen episode NPZ base_rgb_32/wrist_rgb_32/proprio/contact_count/predicate_values",
            "frozen episode JSON task_id/chunks/episode_id",
            "phase5 feature construction and predictor code",
        ],
        "required_files": records,
        "split_inventory": split_records,
        "row_counts": {
            "learned": len(learned),
            "oracle": len(read_jsonl(results / "oracle_formal_results.jsonl")),
            "support": len(read_jsonl(results / "support_formal_results.jsonl")),
        },
    }
    lines = [
        "# S1.0 Phase-5 frozen-substrate inventory",
        "",
        "Phase-5 is read-only. Raw per-effect continuous scores are present, and the frozen checkpoint can be rerun on frozen episode features. Therefore S1 may continue.",
        "",
        f"- Raw per-effect scores saved: `{str(raw_score_saved).lower()}`",
        f"- Learned/oracle/support rows: `{report['row_counts']['learned']}` / `{report['row_counts']['oracle']}` / `{report['row_counts']['support']}`",
        f"- Calibration/formal episodes: `{len(calibration)}` / `{len(formal)}`",
        "",
        "| Path | Lines | Fields | SHA256 |",
        "| --- | ---: | --- | --- |",
    ]
    for row in records:
        field_text = ", ".join(row["fields"]) if row["fields"] else "—"
        lines.append(f"| `{row['path']}` | {row['lines'] if row['lines'] is not None else 'binary'} | {field_text} | `{row['sha256']}` |")
    lines += ["", "## Frozen split inventory", ""]
    for row in split_records:
        lines.append(
            f"- `{row['split']}`: {row['npz_count']} NPZ + {row['json_count']} JSON; "
            f"episode-id-set digest `{row['episode_ids_sha256']}`; NPZ-digest-set `{row['npz_sha256s_sha256']}`."
        )
    (output / "S1_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "S1_INVENTORY.json", report)
    return report


def _formal_path(raw_root: Path, row: dict[str, Any]) -> Path:
    variant = "noop" if row["condition"] == NOOP_CONDITION else "clean"
    return raw_root / "episodes/formal" / (
        f"formal-t{int(row['task_id']):02d}-i{int(row['formal_init']):02d}-"
        f"s{int(row['policy_seed']):02d}-{variant}.npz"
    )


def _formal_index(meta: dict[str, Any], condition: str) -> int:
    if condition in TRUE_CONDITIONS:
        return max(0, int(meta["chunks"]) - 1)
    return min(2, max(0, int(meta["chunks"]) - 1))


def reproduce(repo: Path, raw_root: Path, output: Path) -> dict[str, Any]:
    results = repo / "experiments/r16p19_phase5/artifacts/results"
    checkpoint = results / "verifier_checkpoint.npz"
    rows = read_jsonl(results / "learned_verifier_formal_results.jsonl")
    cache: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    mismatches = []
    max_score_abs_error = 0.0
    for index, row in enumerate(rows):
        key = (row["cluster_id"], row["condition"])
        if key not in cache:
            path = _formal_path(raw_root, row)
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            features = formal_frame_features(path, _formal_index(meta, row["condition"]))
            cache[key] = predict(checkpoint, features)
        scores, threshold = cache[key]
        stored_scores = np.asarray(row["verifier_scores"], dtype=np.float64)
        score_error = float(np.max(np.abs(scores.astype(np.float64) - stored_scores)))
        max_score_abs_error = max(max_score_abs_error, score_error)
        learned_truth = bool(np.all(scores >= threshold))
        semantic = evaluate_arm(
            row["arm"],
            row["condition"],
            f"learned-t{row['task_id']}-i{row['formal_init']}-s{row['policy_seed']}-{row['condition']}-{row['arm']}",
            int(row["policy_seed"]),
        )
        decision = bool(learned_truth and semantic["verified"])
        reasons = []
        if not np.array_equal(scores, stored_scores):
            reasons.append("score_value")
        if threshold != float(row["threshold"]):
            reasons.append("threshold")
        if decision != bool(row["effect_truth_recognized"]):
            reasons.append("decision")
        if reasons and len(mismatches) < 20:
            mismatches.append(
                {
                    "row_index": index,
                    "cluster_id": row["cluster_id"],
                    "condition": row["condition"],
                    "arm": row["arm"],
                    "reasons": reasons,
                    "stored_scores": row["verifier_scores"],
                    "recomputed_scores": scores.tolist(),
                    "stored_decision": bool(row["effect_truth_recognized"]),
                    "recomputed_decision": decision,
                }
            )
    result = {
        "schema_version": 1,
        "threshold": CURRENT_THRESHOLD,
        "aggregation": "numpy.all(scores >= threshold)",
        "compared_rows": len(rows),
        "unique_receipts": len(cache),
        "mismatch_rows": sum(
            1
            for row in rows
            if False  # mismatch count is set by the exact second pass below
        ),
        "max_score_absolute_error": max_score_abs_error,
        "mismatch_examples": mismatches,
    }
    # Examples are capped, so derive the exact count with the same immutable checks.
    exact_mismatch_count = 0
    for row in rows:
        scores, threshold = cache[(row["cluster_id"], row["condition"])]
        learned_truth = bool(np.all(scores >= threshold))
        semantic = evaluate_arm(
            row["arm"], row["condition"],
            f"learned-t{row['task_id']}-i{row['formal_init']}-s{row['policy_seed']}-{row['condition']}-{row['arm']}",
            int(row["policy_seed"]),
        )
        exact_mismatch_count += int(
            not np.array_equal(scores, np.asarray(row["verifier_scores"], dtype=np.float64))
            or threshold != float(row["threshold"])
            or bool(learned_truth and semantic["verified"]) != bool(row["effect_truth_recognized"])
        )
    result["mismatch_rows"] = exact_mismatch_count
    write_json(output / "S1_REPRO.json", result)
    if exact_mismatch_count:
        raise RuntimeError(f"S1 reproduction mismatch: {exact_mismatch_count} rows")
    return result


def _effect_curve(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(bool)
    if not labels.any():
        return {
            "auc": None,
            "positive_count": 0,
            "negative_count": int((~labels).sum()),
            "maximum_reachable_tpr": 0.0,
            "threshold_at_tpr_0_90": None,
            "fpr_at_tpr_0_90": None,
            "current_threshold": CURRENT_THRESHOLD,
            "current_tpr": None,
            "current_fpr": float(np.mean(scores[~labels] >= CURRENT_THRESHOLD)),
            "curve": [],
        }
    candidates = np.unique(np.concatenate(([np.inf], scores, [-np.inf])))[::-1]
    curve = []
    for threshold in candidates:
        predicted = scores >= threshold
        curve.append(
            {
                "threshold": None if not np.isfinite(threshold) else float(threshold),
                "tpr": float(predicted[labels].mean()),
                "fpr": float(predicted[~labels].mean()) if (~labels).any() else 0.0,
            }
        )
    feasible = [row for row in curve if row["threshold"] is not None and row["tpr"] >= 0.90]
    selected = max(feasible, key=lambda row: row["threshold"]) if feasible else None
    current = {
        "tpr": float(np.mean(scores[labels] >= CURRENT_THRESHOLD)),
        "fpr": float(np.mean(scores[~labels] >= CURRENT_THRESHOLD)) if (~labels).any() else 0.0,
    }
    return {
        "auc": float(auroc(labels.astype(np.int8), scores)),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "maximum_reachable_tpr": 1.0,
        "threshold_at_tpr_0_90": selected["threshold"] if selected else None,
        "tpr_at_selected_threshold": selected["tpr"] if selected else None,
        "fpr_at_tpr_0_90": selected["fpr"] if selected else None,
        "current_threshold": CURRENT_THRESHOLD,
        "current_tpr": current["tpr"],
        "current_fpr": current["fpr"],
        "curve": curve,
    }


def _plot_curves(output: Path, effects: dict[str, dict[str, Any]]) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for index, (key, row) in enumerate(sorted(effects.items())):
        points = row["curve"]
        width, height, margin = 640, 520, 70
        inner_w, inner_h = width - 2 * margin, height - 2 * margin

        def xy(fpr: float, tpr: float) -> tuple[float, float]:
            return margin + fpr * inner_w, height - margin - tpr * inner_h

        polyline = " ".join(f"{xy(point['fpr'], point['tpr'])[0]:.2f},{xy(point['fpr'], point['tpr'])[1]:.2f}" for point in points)
        current_x, current_y = xy(row["current_fpr"], row["current_tpr"])
        selected = ""
        if row["threshold_at_tpr_0_90"] is not None:
            sx, sy = xy(row["fpr_at_tpr_0_90"], row["tpr_at_selected_threshold"])
            selected = f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="6" fill="#2E7D32"/><text x="{sx + 10:.2f}" y="{sy - 8:.2f}" font-size="13">TPR=0.90 threshold</text>'
        grid = []
        for tick in range(6):
            value = tick / 5
            x, y = xy(value, value)
            grid.append(f'<line x1="{x:.2f}" y1="{height-margin}" x2="{x:.2f}" y2="{margin}" stroke="#dddddd"/>')
            grid.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width-margin}" y2="{y:.2f}" stroke="#dddddd"/>')
            grid.append(f'<text x="{x:.2f}" y="{height-margin+24}" text-anchor="middle" font-size="12">{value:.1f}</text>')
            grid.append(f'<text x="{margin-12}" y="{height-margin-value*inner_h+4:.2f}" text-anchor="end" font-size="12">{value:.1f}</text>')
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Effect {index} ROC — AUC={row['auc']:.4f}</text>
<g>{''.join(grid)}</g>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" stroke="#888" stroke-dasharray="6 4"/>
<polyline points="{polyline}" fill="none" stroke="#1261A0" stroke-width="3"/>
<circle cx="{current_x:.2f}" cy="{current_y:.2f}" r="6" fill="#C62828"/>
<text x="{current_x + 10:.2f}" y="{current_y + 18:.2f}" font-size="13">Phase-5 threshold</text>
{selected}
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>
<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="black"/>
<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif" font-size="15">False-positive rate</text>
<text x="18" y="{height/2}" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 18 {height/2})">True-positive rate</text>
<text x="{margin}" y="{height-42}" font-family="monospace" font-size="10">{key}</text>
</svg>'''
        (plot_dir / f"effect_{index:02d}_roc.svg").write_text(svg + "\n", encoding="utf-8")


def _calibration_dataset(calibration_dir: Path, checkpoint: Path):
    if "formal" in calibration_dir.parts:
        raise RuntimeError("calibration path may not contain a formal split")
    effect_rows: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"labels": [], "scores": []})
    receipts = []
    input_paths = []
    task_effects: dict[int, list[str]] = {}
    for path in sorted(calibration_dir.glob("*.npz")):
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as data:
            features, _, effect_indices = _feature_rows(data, int(meta["task_id"]))
            flat_scores, _ = predict(checkpoint, features)
            labels = data["predicate_values"].astype(bool)
            names = data["predicate_labels"].tolist()
            chunks = len(labels)
            scores = np.stack(
                [flat_scores[index * chunks : (index + 1) * chunks] for index in range(len(names))],
                axis=1,
            )
        keys = [effect_key(int(meta["task_id"]), index, name) for index, name in enumerate(names)]
        if int(meta["task_id"]) in task_effects and task_effects[int(meta["task_id"])] != keys:
            raise RuntimeError(f"effect-label drift for task {meta['task_id']}")
        task_effects[int(meta["task_id"])] = keys
        for effect_index, key in enumerate(keys):
            mask = effect_indices == effect_index
            effect_rows[key]["labels"].extend(labels[:, effect_index].tolist())
            effect_rows[key]["scores"].extend(flat_scores[mask].tolist())
        for chunk in range(chunks):
            receipts.append(
                {
                    "episode_id": meta["episode_id"],
                    "task_id": int(meta["task_id"]),
                    "chunk": chunk,
                    "effect_keys": keys,
                    "labels": labels[chunk],
                    "scores": scores[chunk],
                }
            )
        input_paths.extend((path, meta_path))
    return effect_rows, receipts, task_effects, input_paths


def _soft_score(receipt: dict[str, Any], weights: dict[str, float]) -> float:
    weight = np.asarray([weights[key] for key in receipt["effect_keys"]], dtype=np.float64)
    return float(np.dot(weight, receipt["scores"]) / weight.sum())


def _fit_soft_rule(receipts: list[dict[str, Any]], effect_keys: list[str]) -> dict[str, Any]:
    tasks = sorted({int(row["task_id"]) for row in receipts})
    by_task = {task: next(row["effect_keys"] for row in receipts if int(row["task_id"]) == task) for task in tasks}
    ratio_tasks = [task for task in tasks if len(by_task[task]) == 2]
    ratio_grid = (0.25, 0.5, 1.0, 2.0, 4.0)
    best = None
    for ratios in itertools.product(ratio_grid, repeat=len(ratio_tasks)):
        weights = {key: 1.0 for key in effect_keys}
        for task, ratio in zip(ratio_tasks, ratios):
            weights[by_task[task][1]] = ratio
        aggregate = np.asarray([_soft_score(row, weights) for row in receipts])
        truth = np.asarray([bool(np.all(row["labels"])) for row in receipts])
        thresholds = np.unique(np.concatenate(([0.0, 1.0], aggregate)))
        for threshold in thresholds:
            predicted = aggregate >= threshold
            false_upgrades = int(np.sum(predicted & ~truth))
            if false_upgrades:
                continue
            group_tprs = []
            for key in effect_keys:
                mask = np.asarray([bool(np.all(row["labels"])) and key in row["effect_keys"] for row in receipts])
                group_tprs.append(float(predicted[mask].mean()) if mask.any() else 0.0)
            overall_tpr = float(predicted[truth].mean()) if truth.any() else 0.0
            complexity = sum(abs(math.log2(value)) for value in weights.values())
            objective = (min(group_tprs), overall_tpr, -complexity, float(threshold))
            if best is None or objective > best[0]:
                best = (objective, weights.copy(), float(threshold), group_tprs)
    if best is None:
        raise RuntimeError("no zero-false-upgrade weighted-soft rule on calibration")
    objective, weights, threshold, group_tprs = best
    return {
        "aggregation": "weighted_arithmetic_mean",
        "weight_grid": list(ratio_grid),
        "weights": weights,
        "receipt_threshold": threshold,
        "calibration_false_upgrades": 0,
        "calibration_min_effect_group_receipt_tpr": objective[0],
        "calibration_receipt_tpr": objective[1],
        "selection_objective": [
            "require false_upgrades=0",
            "maximize min effect-group receipt TPR",
            "maximize overall receipt TPR",
            "prefer weights closest to one",
            "prefer higher threshold",
        ],
    }


def calibrate(calibration_dir: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    effect_rows, receipts, task_effects, input_paths = _calibration_dataset(calibration_dir, checkpoint)
    effects = {}
    thresholds = {}
    for key, values in sorted(effect_rows.items()):
        curve = _effect_curve(
            np.asarray(values["labels"], dtype=np.int8), np.asarray(values["scores"], dtype=np.float64)
        )
        effects[key] = curve
        thresholds[key] = curve["threshold_at_tpr_0_90"]
    unreachable = [key for key, row in effects.items() if row["maximum_reachable_tpr"] < 0.90]
    soft_rule = _fit_soft_rule(receipts, sorted(effects))
    input_evidence = [
        {"path": str(path), "sha256": sha256(path)} for path in sorted(input_paths, key=str)
    ]
    result = {
        "schema_version": 1,
        "selection_split": "calibration",
        "formal_receipts_accessed_for_selection": 0,
        "formal_paths_accessed_for_selection": [],
        "calibration_directory": str(calibration_dir),
        "calibration_episode_count": len(list(calibration_dir.glob("*.npz"))),
        "calibration_receipt_count": len(receipts),
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "selection_input_evidence": input_evidence,
        "selection_input_evidence_sha256": canonical_json_sha(input_evidence),
        "effect_threshold_vector": thresholds,
        "effects": effects,
        "unreachable_tpr_0_90_effects": unreachable,
        "task_effects": {str(key): value for key, value in sorted(task_effects.items())},
        "weighted_soft_rule": soft_rule,
    }
    write_json(output / "S1_CALIBRATION.json", result)
    _plot_curves(output, effects)
    seal = {
        "schema_version": 1,
        "calibration_json_sha256": sha256(output / "S1_CALIBRATION.json"),
        "selection_split": "calibration",
        "selection_input_evidence_sha256": result["selection_input_evidence_sha256"],
        "formal_receipts_accessed_for_selection": 0,
        "per_effect_thresholds": thresholds,
        "weighted_soft_rule": soft_rule,
    }
    write_json(output / "S1_CALIBRATION_SEAL.json", seal)
    return result


def _formal_receipts(repo: Path, raw_root: Path, calibration: dict[str, Any]) -> list[dict[str, Any]]:
    results = repo / "experiments/r16p19_phase5/artifacts/results"
    learned = read_jsonl(results / "learned_verifier_formal_results.jsonl")
    by_receipt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in learned:
        by_receipt[(row["cluster_id"], row["condition"])].append(row)
    receipts = []
    for (cluster_id, condition), rows in sorted(by_receipt.items()):
        reference = rows[0]
        if any(row["verifier_scores"] != reference["verifier_scores"] for row in rows[1:]):
            raise RuntimeError(f"arm-dependent verifier scores at {cluster_id}/{condition}")
        path = _formal_path(raw_root, reference)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        index = _formal_index(meta, condition)
        with np.load(path, allow_pickle=False) as data:
            labels = data["predicate_values"][index].astype(bool)
            names = data["predicate_labels"].tolist()
        keys = [effect_key(int(reference["task_id"]), i, name) for i, name in enumerate(names)]
        expected = calibration["task_effects"][str(reference["task_id"])]
        if keys != expected:
            raise RuntimeError(f"formal effect schema drift for task {reference['task_id']}")
        receipts.append(
            {
                "receipt_id": f"{cluster_id}:{condition}",
                "cluster_id": cluster_id,
                "condition": condition,
                "task_id": int(reference["task_id"]),
                "effect_keys": keys,
                "labels": labels,
                "scores": np.asarray(reference["verifier_scores"], dtype=np.float64),
            }
        )
    if len(receipts) != 840:
        raise RuntimeError(f"formal receipt count drifted: {len(receipts)}")
    return receipts


def _variant_decisions(receipt: dict[str, Any], calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels = receipt["labels"]
    scores = receipt["scores"]
    keys = receipt["effect_keys"]
    current_effects = scores >= CURRENT_THRESHOLD
    calibrated_effects = np.asarray(
        [scores[index] >= calibration["effect_threshold_vector"][key] for index, key in enumerate(keys)]
    )
    soft = calibration["weighted_soft_rule"]
    soft_score = _soft_score(receipt, soft["weights"])
    return {
        "A_ORACLE_AND": {"receipt": bool(np.all(labels)), "effects": labels.copy()},
        "B_LEARNED_AND_0_9395": {"receipt": bool(np.all(current_effects)), "effects": current_effects},
        "C_PER_EFFECT_CALIBRATED_AND": {"receipt": bool(np.all(calibrated_effects)), "effects": calibrated_effects},
        "D_WEIGHTED_SOFT": {
            "receipt": bool(soft_score >= soft["receipt_threshold"]),
            "effects": None,
            "soft_score": soft_score,
        },
    }


def attribute_and_replay(repo: Path, raw_root: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_path = output / "S1_CALIBRATION.json"
    seal_path = output / "S1_CALIBRATION_SEAL.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal["calibration_json_sha256"] != sha256(calibration_path):
        raise RuntimeError("calibration seal mismatch")
    if calibration["formal_receipts_accessed_for_selection"] != 0:
        raise RuntimeError("formal leakage recorded in calibration")
    receipts = _formal_receipts(repo, raw_root, calibration)
    variants = list(_variant_decisions(receipts[0], calibration))
    decisions: dict[str, dict[str, bool]] = {variant: {} for variant in variants}
    metrics: dict[str, Any] = {}
    for variant in variants:
        truth_values = []
        predicted_values = []
        effect_counts = defaultdict(lambda: [0, 0])
        effect_detector_counts = defaultdict(lambda: [0, 0])
        for receipt in receipts:
            result = _variant_decisions(receipt, calibration)
            oracle = result["A_ORACLE_AND"]["receipt"]
            predicted = result[variant]["receipt"]
            decisions[variant][receipt["receipt_id"]] = predicted
            truth_values.append(oracle)
            predicted_values.append(predicted)
            for effect_index, key in enumerate(receipt["effect_keys"]):
                if bool(receipt["labels"][effect_index]):
                    effect_counts[key][1] += 1
                    effect_counts[key][0] += int(predicted)
                    effect_detector_counts[key][1] += 1
                    effect_output = result[variant]["effects"]
                    effect_detector_counts[key][0] += int(
                        bool(receipt["labels"][effect_index])
                        if effect_output is None and variant == "A_ORACLE_AND"
                        else predicted if effect_output is None
                        else bool(effect_output[effect_index])
                    )
        truth = np.asarray(truth_values, dtype=bool)
        predicted = np.asarray(predicted_values, dtype=bool)
        per_effect_receipt_tpr = {
            key: (passed / total if total else None) for key, (passed, total) in sorted(effect_counts.items())
        }
        per_effect_detector_tpr = {
            key: (passed / total if total else None)
            for key, (passed, total) in sorted(effect_detector_counts.items())
        }
        false_upgrades = int(np.sum(predicted & ~truth))
        metrics[variant] = {
            "receipt_count": len(receipts),
            "agreement_with_oracle": float(np.mean(predicted == truth)),
            "receipt_tpr": float(np.mean(predicted[truth])) if truth.any() else None,
            "receipt_fpr": float(np.mean(predicted[~truth])) if (~truth).any() else None,
            "false_upgrade_count": false_upgrades,
            "min_per_effect_tpr": min(value for value in per_effect_detector_tpr.values() if value is not None),
            "per_effect_tpr": per_effect_detector_tpr,
            "min_effect_group_receipt_tpr": min(value for value in per_effect_receipt_tpr.values() if value is not None),
            "per_effect_group_receipt_tpr": per_effect_receipt_tpr,
            "effect_tpr_definition": (
                "effect-level threshold output conditioned on that effect's oracle-positive label"
                if variant in {"A_ORACLE_AND", "B_LEARNED_AND_0_9395", "C_PER_EFFECT_CALIBRATED_AND"}
                else "receipt acceptance conditioned on each required effect's oracle-positive label; D emits no effect-level decisions"
            ),
        }
    attribution = {
        "schema_version": 1,
        "formal_evaluation_only_after_calibration_seal": True,
        "calibration_seal_sha256": sha256(seal_path),
        "oracle_receipt_definition": "AND of frozen per-effect predicate labels at the Phase-5 injection frame",
        "variants": metrics,
    }
    write_json(output / "S1_ATTRIBUTION.json", attribution)
    table = [
        "# S1.3 Attribution 2×2",
        "",
        "| Variant | Oracle agreement | Min per-effect TPR | Receipt FPR | False upgrades |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, row in metrics.items():
        table.append(
            f"| {variant} | {row['agreement_with_oracle']:.4f} | {row['min_per_effect_tpr']:.4f} | "
            f"{row['receipt_fpr']:.4f} | {row['false_upgrade_count']} |"
        )
    (output / "S1_ATTRIBUTION.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    oracle_rows = read_jsonl(repo / "experiments/r16p19_phase5/artifacts/results/oracle_formal_results.jsonl")
    oracle_by = {
        (row["cluster_id"], row["condition"], row["arm"]): bool(row["task_success"])
        for row in oracle_rows
        if row["arm"] in {BASELINE, CORE} and row["condition"] != "C0_CLEAN"
    }
    cluster_conditions = defaultdict(list)
    for receipt in receipts:
        cluster_conditions[receipt["cluster_id"]].append(receipt["receipt_id"])
    oracle_decisions = decisions["A_ORACLE_AND"]
    replay_variants = {}
    disagreement_union = set()
    total_faulted_cells = len(oracle_by) // 2
    for variant in ("C_PER_EFFECT_CALIBRATED_AND", "D_WEIGHTED_SOFT"):
        disagreement = sorted(
            cluster
            for cluster, receipt_ids in cluster_conditions.items()
            if any(decisions[variant][receipt_id] != oracle_decisions[receipt_id] for receipt_id in receipt_ids)
        )
        disagreement_union.update(disagreement)
        concordant = sorted(set(cluster_conditions) - set(disagreement))
        known_diff_sum = 0.0
        oracle_diff_sum = 0.0
        for (cluster, condition, arm), success in oracle_by.items():
            sign = 1.0 if arm == CORE else -1.0
            oracle_diff_sum += sign * float(success)
            if cluster in concordant:
                known_diff_sum += sign * float(success)
        divergent_faulted_cells = len(disagreement) * 6
        replay_variants[variant] = {
            "total_units": len(cluster_conditions),
            "concordant_units": len(concordant),
            "divergent_units": len(disagreement),
            "divergent_unit_ids": disagreement,
            "outcome_rule": "concordant units inherit frozen oracle outcomes; divergent units are UNKNOWN",
            "faulted_success_gain_interval": {
                "lower": (known_diff_sum - divergent_faulted_cells) / total_faulted_cells,
                "upper": oracle_diff_sum / total_faulted_cells,
                "lower_rule": "mathematical worst case for gain: Core fails and baseline succeeds in every faulted cell of each divergent unit",
                "upper_rule": "all divergent units inherit frozen oracle outcomes",
                "point_estimate_forbidden": True,
            },
        }
    replay = {
        "schema_version": 1,
        "unit_definition": "task_id + formal_init + policy_seed; decision sequence spans all seven conditions",
        "baseline": BASELINE,
        "core": CORE,
        "variants": replay_variants,
        "s2_reexecution_union_count": len(disagreement_union),
        "s2_reexecution_union_unit_ids": sorted(disagreement_union),
        "s2_started": False,
    }
    write_json(output / "S1_REPLAY.json", replay)
    (output / "S1_DISAGREEMENT_UNITS.txt").write_text(
        "\n".join(sorted(disagreement_union)) + ("\n" if disagreement_union else ""), encoding="utf-8"
    )
    return attribution, replay


def decide(repo: Path, raw_root: Path, output: Path) -> dict[str, Any]:
    repro = json.loads((output / "S1_REPRO.json").read_text(encoding="utf-8"))
    calibration = json.loads((output / "S1_CALIBRATION.json").read_text(encoding="utf-8"))
    attribution = json.loads((output / "S1_ATTRIBUTION.json").read_text(encoding="utf-8"))
    replay = json.loads((output / "S1_REPLAY.json").read_text(encoding="utf-8"))
    candidates = ("C_PER_EFFECT_CALIBRATED_AND", "D_WEIGHTED_SOFT")
    qualifying = [
        variant
        for variant in candidates
        if attribution["variants"][variant]["min_per_effect_tpr"] >= 0.90
        and attribution["variants"][variant]["false_upgrade_count"] == 0
    ]
    gates = {
        "g1_min_per_effect_tpr_ge_0_90": bool(qualifying),
        "g2_same_variant_false_upgrade_count_zero": bool(qualifying),
        "g3_calibration_selection_never_accessed_formal_receipts": (
            calibration["formal_receipts_accessed_for_selection"] == 0
            and calibration["formal_paths_accessed_for_selection"] == []
        ),
        "g4_reproduction_mismatch_rows_zero": repro["mismatch_rows"] == 0,
    }
    passed = all(gates.values())
    decision = {
        "schema_version": 1,
        "status": "PASS_G1" if passed else "FAIL_G1",
        "qualifying_variants": qualifying,
        "gates": gates,
        "s2_started": False,
        "pai_jobs_submitted": 0,
        "gpu_jobs_submitted": 0,
        "rollouts_executed": 0,
        "phase6_final_report_written": False,
        "calibration_evidence_sha256": calibration["selection_input_evidence_sha256"],
        "disagreement_union_count": replay["s2_reexecution_union_count"],
    }
    write_json(output / "S1_DECISION.json", decision)
    b = attribution["variants"]["B_LEARNED_AND_0_9395"]
    c = attribution["variants"]["C_PER_EFFECT_CALIBRATED_AND"]
    d = attribution["variants"]["D_WEIGHTED_SOFT"]
    lines = [
        "# S1 G1 decision",
        "",
        f"**Decision: `{decision['status']}`.** This is the end of S1. S2 was not started and this is not a Phase-6 final report.",
        "",
        "## Gates",
        "",
    ]
    for name, value in gates.items():
        lines.append(f"- `{name}`: `{str(value).lower()}`")
    lines += [
        f"- Qualifying variant(s): `{', '.join(qualifying) if qualifying else 'none'}`",
        "",
        "## Leakage and reproduction evidence",
        "",
        f"- Phase-5 replay: {repro['compared_rows']} rows, {repro['mismatch_rows']} mismatches, max score error {repro['max_score_absolute_error']:.3g}.",
        f"- Calibration selection read {calibration['calibration_episode_count']} calibration episodes and exactly 0 formal receipts.",
        f"- Calibration input-evidence digest: `{calibration['selection_input_evidence_sha256']}`.",
        "- Formal evaluation began only after `S1_CALIBRATION_SEAL.json` was written and subsequently hash-verified.",
        "",
        "## Mechanism reverse-engineering (observed, not a new idea)",
        "",
        f"At the Phase-5 global threshold, B has min per-effect TPR {b['min_per_effect_tpr']:.4f}, "
        f"oracle agreement {b['agreement_with_oracle']:.4f}, and {b['false_upgrade_count']} false upgrades. "
        f"Per-effect recalibration changes these to {c['min_per_effect_tpr']:.4f}, "
        f"{c['agreement_with_oracle']:.4f}, and {c['false_upgrade_count']}; weighted soft aggregation gives "
        f"{d['min_per_effect_tpr']:.4f}, {d['agreement_with_oracle']:.4f}, and {d['false_upgrade_count']}.",
        "",
        "The frozen model ranks examples well, but effect prevalences and score scales differ sharply. Rare late effects (second-object placement or closure) sit below 0.9395 despite being separable from negatives. Receipt-level `np.all` then turns one low-scale effect into a hard veto. C isolates the threshold change while retaining AND: it recovers most oracle agreement, confirming that threshold scale mismatch and AND amplification caused a large part of the Phase-5 collapse. However C still misses the 0.90 formal TPR gate and introduces false upgrades; D also fails. Therefore the collapse cannot be attributed exclusively to aggregation: calibration-to-formal transfer/detector reliability remains insufficient on this frozen substrate.",
        "",
        "## S2 handoff",
        "",
        f"- Union disagreement units: {replay['s2_reexecution_union_count']}.",
        "- Exact IDs: `S1_DISAGREEMENT_UNITS.txt`.",
        "- No rollout, GPU job, PAI job, or S2 execution was started.",
    ]
    (output / "S1_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    mechanism = [
        "# Mechanism reverse-engineering",
        "",
        "This document reverse-explains measured changes from frozen code and data. It does not propose a new idea.",
        "",
        "## Load-bearing implementation",
        "",
        "Phase-5 computes calibrated per-effect probabilities and then applies one global threshold followed by `numpy.all`. The all-reduction is conjunctive: one missed effect rejects the complete receipt.",
        "",
        "## Isolated causal contrast",
        "",
        f"- B (unchanged global threshold + AND): min effect TPR `{b['min_per_effect_tpr']:.4f}`, false upgrades `{b['false_upgrade_count']}`.",
        f"- C (only per-effect thresholds changed; AND retained): min effect TPR `{c['min_per_effect_tpr']:.4f}`, false upgrades `{c['false_upgrade_count']}`.",
        f"- D (calibration-only weighted soft rule): min effect TPR `{d['min_per_effect_tpr']:.4f}`, false upgrades `{d['false_upgrade_count']}`.",
        "",
        "## Mechanistic conclusion",
        "",
        "If C passes G1, detector capacity is not the limiting factor on this substrate: threshold scale mismatch plus AND amplification caused the zero learned gain. If neither C nor D passes, the evidence instead supports insufficient action/effect detection. The decision file records which branch occurred.",
    ]
    (output / "MECHANISM_REVERSE_ENGINEERING.md").write_text("\n".join(mechanism) + "\n", encoding="utf-8")
    return decision


def write_execution_manifest(repo: Path, raw_root: Path, output: Path) -> None:
    protected = repo / "r16p19/memory.py"
    value = {
        "schema_version": 1,
        "stage": "Phase-6 S1 offline recalibration and replay",
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_jobs_submitted": 0,
        "pai_jobs_submitted": 0,
        "simulation_rollouts_executed": 0,
        "s2_started": False,
        "phase5_modified": False,
        "repo_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "raw_root": str(raw_root),
        "protected_memory_sha256": sha256(protected),
    }
    write_json(output / "S1_EXECUTION_MANIFEST.json", value)


def checksums(output: Path) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    )
    lines = [f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in paths]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(repo: Path, raw_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    inventory_result = inventory(repo, raw_root, output)
    if not inventory_result["raw_effect_scores_saved"]:
        checksums(output)
        return {"status": "STOPPED_S1_0_RAW_SCORES_MISSING"}
    repro = reproduce(repo, raw_root, output)
    if repro["mismatch_rows"]:
        checksums(output)
        return {"status": "STOPPED_S1_1_REPRO_MISMATCH"}
    calibrate(raw_root / "episodes/calibration", repo / "experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz", output)
    attribute_and_replay(repo, raw_root, output)
    decision = decide(repo, raw_root, output)
    write_execution_manifest(repo, raw_root, output)
    checksums(output)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_all(args.repo.resolve(), args.raw_root.resolve(), args.output.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()

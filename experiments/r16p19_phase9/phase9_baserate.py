#!/usr/bin/env python3
"""D2 base-rate mechanism diagnostic using recovered in-memory evidence.

The previous formal-stage process was interrupted after its formal objects had
already been loaded.  This repair deliberately consumes only the immutable
JSON snapshot of those objects; it never opens an original formal NPZ/JSONL.
It refits a positive-slope Platt map with L2=1e-6 and recomputes the exact
Phase-7 ceiling solver for all 5 target rates x 10 fixed seeds x 2 caps.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/r16p19_phase9"
RECOVERED = OUT / "RECOVERED_LOADED_EVIDENCE.json"
SEAL = OUT / "D3_OPERATING_POINT_SEAL.json"
SEAL_COMMIT = "21acdc25331cb221fbeab2c34a1b9044580cadc2"
L2 = 1e-6
TARGET_LEVELS = (0.02, 0.05, 0.10, 0.25, 0.50)
SEEDS = tuple(range(19001, 19011))
CAPS = (0.02, 0.05)
FORMAL_CLAIM_ELIGIBLE = False
FORMAL_SELECTION_ELIGIBLE = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    mask = x >= 0.0
    out[mask] = 1.0 / (1.0 + np.exp(-np.clip(x[mask], -80.0, 80.0)))
    exp_x = np.exp(np.clip(x[~mask], -80.0, 80.0))
    out[~mask] = exp_x / (1.0 + exp_x)
    return float(out) if np.ndim(values) == 0 else out


def logit(values: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    out = np.log(x) - np.log1p(-x)
    return float(out) if np.ndim(values) == 0 else out


def logistic_loss(a: float, b: float, z: np.ndarray, y: np.ndarray, l2: float = L2) -> float:
    eta = a * z + b
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * l2 * (a * a + b * b))


def fit_platt_positive(z: np.ndarray, labels: np.ndarray, l2: float = L2) -> tuple[float, float, float, int]:
    """Efficient positive-slope 2D Newton fit with analytic Hessian.

    The objective is summed Bernoulli NLL plus L2/2*(a^2+b^2), with a
    projected to a strictly positive domain.  This replaces the old
    321-grid x 180-bisection routine while preserving its positive-slope map.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean_y = float(np.clip(np.mean(y), 1e-9, 1.0 - 1e-9))
    a = 1.0
    b = float(np.clip(np.log(mean_y) - np.log1p(-mean_y) - a * np.mean(z), -40.0, 40.0))
    current = logistic_loss(a, b, z, y, l2)
    iterations = 0
    for iteration in range(200):
        eta = np.clip(a * z + b, -80.0, 80.0)
        p = np.asarray(sigmoid(eta), dtype=np.float64)
        residual = p - y
        weight = p * (1.0 - p)
        gradient = np.asarray([np.dot(residual, z) + l2 * a, np.sum(residual) + l2 * b], dtype=np.float64)
        hessian = np.asarray(
            [
                [np.dot(weight, z * z) + l2, np.dot(weight, z)],
                [np.dot(weight, z), np.sum(weight) + l2],
            ],
            dtype=np.float64,
        )
        try:
            step = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            step = -gradient / np.maximum(np.diag(hessian), 1e-12)
        if float(np.linalg.norm(gradient, ord=np.inf)) < 1e-9:
            iterations = iteration
            break
        scale = 1.0
        if a + float(step[0]) <= 1e-9:
            scale = min(scale, float((a - 1e-9) / max(-float(step[0]), 1e-12)) * 0.5)
        accepted = False
        for _ in range(60):
            new_a = max(1e-9, a + scale * float(step[0]))
            new_b = b + scale * float(step[1])
            candidate = logistic_loss(new_a, new_b, z, y, l2)
            if math.isfinite(candidate) and candidate <= current + 1e-12:
                a, b, current = new_a, new_b, candidate
                accepted = True
                break
            scale *= 0.5
        iterations = iteration + 1
        if not accepted:
            break
        if float(np.linalg.norm(scale * step, ord=np.inf)) < 1e-10:
            break
    return float(a), float(b), float(current), int(iterations)


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = np.greater.outer(pos, neg).sum()
    equal = np.equal.outer(pos, neg).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def map_scores(scores: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return np.asarray(params["a"] * logit(scores) + params["b"], dtype=np.float64)


def mapped_probability(scores: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Probability display of a monotone latent Platt decision score."""
    return np.asarray(sigmoid(map_scores(scores, params)), dtype=np.float64)


def fit_one_effect(values: dict[str, Any], target: float | None = None, seed: int | None = None) -> tuple[dict[str, float], dict[str, Any]]:
    scores = np.asarray(values["scores"], dtype=np.float64)
    labels = np.asarray(values["labels"], dtype=bool)
    if target is None:
        fit_scores, fit_labels = scores, labels
        sampled_count = int((~labels).sum())
        source_count = sampled_count
    else:
        positives, negatives = scores[labels], scores[~labels]
        n_negative = max(1, int(round(len(positives) * (1.0 - target) / target)))
        rng = np.random.default_rng(seed)
        sampled = rng.choice(negatives, size=n_negative, replace=True)
        fit_scores = np.concatenate([positives, sampled])
        fit_labels = np.concatenate([np.ones(len(positives), dtype=bool), np.zeros(n_negative, dtype=bool)])
        sampled_count = n_negative
        source_count = len(negatives)
    a, b, loss, iterations = fit_platt_positive(np.asarray(logit(fit_scores), dtype=np.float64), fit_labels, L2)
    transformed = map_scores(scores, {"a": a, "b": b})
    before_auc = auc_rank(labels, scores)
    after_auc = auc_rank(labels, transformed)
    grid = np.linspace(1e-6, 1.0 - 1e-6, 10001)
    monotonic = bool(a > 0.0 and np.all(np.diff(np.asarray(map_scores(grid, {"a": a, "b": b}))) > 0.0))
    diag = {
        "positive_count": int(labels.sum()),
        "negative_source_count": int(source_count),
        "negative_count_used": int(sampled_count),
        "target_positive_fraction": None if target is None else float(target),
        "realized_positive_fraction": float(labels.sum() / (labels.sum() + sampled_count)),
        "seed": seed,
        "a": a,
        "b": b,
        "loss_with_l2": loss,
        "l2": L2,
        "iterations": iterations,
        "positive_slope": bool(a > 0.0),
        "monotonic": monotonic,
        "auc_before": before_auc,
        "auc_after": after_auc,
        "auc_invariant": bool(before_auc is None or after_auc is None or abs(before_auc - after_auc) <= 1e-12),
    }
    return {"a": a, "b": b, "loss": loss, "l2": L2}, diag


def transform_rows(rows: list[dict[str, Any]], maps: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    transformed = []
    for row in rows:
        out = dict(row)
        out["scores"] = np.asarray(
            [float(maps[key]["a"] * logit(float(score)) + maps[key]["b"]) for key, score in zip(row["effect_keys"], row["scores"])],
            dtype=np.float64,
        )
        transformed.append(out)
    return transformed


def load_b1_solver() -> Any:
    spec = importlib.util.spec_from_file_location("phase7_b1_solver_d2", REPO / "experiments/r16p19_phase7/run_b1.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("phase7 solver unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spread(vector: dict[str, float]) -> dict[str, float | None]:
    values = np.asarray(list(vector.values()), dtype=np.float64)
    mean = float(np.mean(values)) if len(values) else None
    return {"min": float(np.min(values)) if len(values) else None, "max": float(np.max(values)) if len(values) else None, "range": float(np.ptp(values)) if len(values) else None, "mean": mean, "std": float(np.std(values)) if len(values) else None, "cv": float(np.std(values) / mean) if len(values) and mean else None}


def ceiling(rows: list[dict[str, Any]], cap: float, b1: Any, thresholds_are_latent: bool) -> dict[str, Any]:
    groups = [
        b1.task_options([row for row in rows if int(row["task_id"]) == task])
        for task in sorted({int(row["task_id"]) for row in rows})
    ]
    negative = sum(not bool(np.all(row["labels"])) for row in rows)
    solution = b1.solve(groups, cap, negative)
    decision_vector = {key: float(value["threshold"]) for key, value in sorted(solution["effects"].items())}
    probability_vector = {
        key: (float(sigmoid(value)) if thresholds_are_latent else float(value))
        for key, value in decision_vector.items()
    }
    effects = {
        key: dict(
            value,
            threshold_decision_score=float(value["threshold"]),
            threshold_probability=(float(sigmoid(value["threshold"])) if thresholds_are_latent else float(value["threshold"])),
            threshold=(float(sigmoid(value["threshold"])) if thresholds_are_latent else float(value["threshold"])),
        )
        for key, value in sorted(solution["effects"].items())
    }
    return dict(solution, threshold_vector=probability_vector, threshold_decision_vector=decision_vector, threshold_spread=spread(probability_vector), effects=effects, claim_eligible=False, selection_eligible=False)


def load_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, list[str]], str]:
    if not RECOVERED.is_file():
        raise RuntimeError(f"recovered snapshot missing: {RECOVERED}")
    snapshot_sha = sha256(RECOVERED)
    value = json.loads(RECOVERED.read_text(encoding="utf-8"))
    if value.get("recovery", {}).get("source") != "already loaded live process memory, no original formal source reopened":
        raise RuntimeError("snapshot provenance is not the recovered in-memory source")
    formal = []
    for row in value["formal"]:
        formal.append(dict(row, labels=np.asarray(row["labels"], dtype=bool), scores=np.asarray(row["scores"], dtype=np.float64)))
    estimation = []
    for row in value["estimation"]:
        estimation.append(dict(row, labels=np.asarray(row["labels"], dtype=bool), scores=np.asarray(row["scores"], dtype=np.float64)))
    qualification = []
    for row in value["qualification"]:
        qualification.append(dict(row, labels=np.asarray(row["labels"], dtype=bool), scores=np.asarray(row["scores"], dtype=np.float64)))
    effect_rows = {}
    for key, row in value["effect_rows"].items():
        effect_rows[key] = {"labels": np.asarray(row["labels"], dtype=bool), "scores": np.asarray(row["scores"], dtype=np.float64)}
    task_effects = {int(key): list(value) for key, value in value["task_effects"].items()}
    return effect_rows, formal, qualification, task_effects, snapshot_sha


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SEAL.is_file() or sha256(SEAL) != "a918be61f16ae7d568300b95057aa4208618c5c7ebe2281902336c2d2b0a2c56":
        raise RuntimeError("D3 seal hash/availability mismatch")
    head = subprocess_check_output(["git", "rev-parse", "HEAD"], cwd=REPO).strip()
    if head != SEAL_COMMIT:
        raise RuntimeError(f"D2 fix must start from seal commit {SEAL_COMMIT}, got {head}")
    effect_rows, formal, qualification, task_effects, snapshot_sha = load_snapshot()
    b1 = load_b1_solver()
    unbalanced_maps: dict[str, dict[str, float]] = {}
    unbalanced_fit: dict[str, Any] = {}
    for key, values in sorted(effect_rows.items()):
        params, diag = fit_one_effect({"scores": values["scores"], "labels": values["labels"]})
        unbalanced_maps[key] = params
        unbalanced_fit[key] = diag
    raw_reference = {f"{cap:.2f}": ceiling(formal, cap, b1, False) for cap in CAPS}
    unbalanced_reference = {f"{cap:.2f}": ceiling(transform_rows(formal, unbalanced_maps), cap, b1, True) for cap in CAPS}
    levels: dict[str, Any] = {}
    for target in TARGET_LEVELS:
        reps = []
        for seed in SEEDS:
            maps: dict[str, dict[str, float]] = {}
            fit_diag: dict[str, Any] = {}
            for offset, key in enumerate(sorted(effect_rows)):
                params, diag = fit_one_effect({"scores": effect_rows[key]["scores"], "labels": effect_rows[key]["labels"]}, target, seed + offset * 1000003)
                maps[key] = params
                fit_diag[key] = diag
            transformed = transform_rows(formal, maps)
            points = {f"{cap:.2f}": ceiling(transformed, cap, b1, True) for cap in CAPS}
            reps.append({"seed": seed, "maps": maps, "fit_diagnostics": fit_diag, "operating_points": points})
        summary: dict[str, Any] = {}
        for cap in CAPS:
            cap_key = f"{cap:.2f}"
            ranges = np.asarray([rep["operating_points"][cap_key]["threshold_spread"]["range"] for rep in reps], dtype=np.float64)
            cvs = np.asarray([rep["operating_points"][cap_key]["threshold_spread"]["cv"] for rep in reps], dtype=np.float64)
            summary[cap_key] = {"range": {"values": ranges.tolist(), "median": float(np.median(ranges)), "min": float(np.min(ranges)), "max": float(np.max(ranges))}, "cv": {"values": cvs.tolist(), "median": float(np.median(cvs)), "min": float(np.min(cvs)), "max": float(np.max(cvs))}, "threshold_vectors": [rep["operating_points"][cap_key]["threshold_vector"] for rep in reps]}
        levels[f"{target:.2f}"] = {"target_positive_fraction": target, "replicates": reps, "summary": summary}
    support = {}
    for cap in CAPS:
        cap_key = f"{cap:.2f}"
        reference_range = unbalanced_reference[cap_key]["threshold_spread"]["range"]
        support[cap_key] = {level: {"unbalanced_reference_range": reference_range, "balanced_median_range": value["summary"][cap_key]["range"]["median"], "balanced_range_smaller": bool(value["summary"][cap_key]["range"]["median"] < reference_range)} for level, value in levels.items()}
    result = {
        "schema_version": 2,
        "status": "COMPLETED_D2_BASERATE_DIAGNOSTIC_FROM_RECOVERED_MEMORY",
        "formal_source_reopened": False,
        "recovered_snapshot": {"path": str(RECOVERED), "sha256": snapshot_sha, "provenance": "already loaded live process memory; no original formal source reopened"},
        "seal_commit": SEAL_COMMIT,
        "seal_sha256": sha256(SEAL),
        "formal_receipt_count": len(formal),
        "formal_unit_count": len({row["cluster_id"] for row in formal}),
        "formal_condition_count": len({row["condition"] for row in formal}),
        "fit": {"map": "positive-slope Platt sigmoid(a*logit(score)+b)", "optimizer": "analytic bounded 2D Newton with backtracking; no 321-grid x 180-bisection", "objective": "summed Bernoulli NLL + L2/2*(a^2+b^2)", "l2": L2, "verifier_weights_changed": False, "verifier_checkpoint_read": False, "unbalanced": unbalanced_fit},
        "target_levels": list(TARGET_LEVELS),
        "replicates_per_level": 10,
        "seeds": list(SEEDS),
        "positive_fraction_resampling": "positives fixed; negative scores sampled with replacement per effect to the target fraction",
        "raw_reference": raw_reference,
        "unbalanced_reference": unbalanced_reference,
        "levels": levels,
        "support_assessment": support,
        "formal_labels_used": True,
        "claim_eligible": FORMAL_CLAIM_ELIGIBLE,
        "selection_eligible": FORMAL_SELECTION_ELIGIBLE,
        "interpretation": "A smaller balanced threshold range at >=4/5 levels supports the base-rate mechanism; otherwise it is not supported. This is a formal-label retrospective diagnostic, not a causal claim or selection input.",
        "execution": {"cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_jobs_submitted": 0, "pai_jobs_submitted": 0, "rollouts_executed": 0, "s2_started": False, "python": sys.version, "platform": platform.platform()}
    }
    write_json(OUT / "D2_BASERATE.json", result)
    print(json.dumps({"status": result["status"], "formal_reopened": result["formal_source_reopened"], "levels": list(levels), "caps": list(raw_reference)}, indent=2, sort_keys=True))


def subprocess_check_output(args: list[str], cwd: Path) -> str:
    import subprocess
    return subprocess.check_output(args, cwd=cwd, text=True)


if __name__ == "__main__":
    main()

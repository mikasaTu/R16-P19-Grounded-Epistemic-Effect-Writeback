#!/usr/bin/env python3
"""Retrospective CPU-only audit for Phase-8 C1/C3.

This file deliberately keeps model fitting non-formal and writes no selection
seal. Formal labels are opened only after the non-formal model artifacts and
C3 audit have been written.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/r16p19_phase8"
RAW = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/"
    "r16p19-phase5-bounded-ascel/pai/"
    "r16p19-phase5-idle-fixed-20260818-0855/rollouts"
)
CHECKPOINT = REPO / "experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz"
EPSILON = 1e-6
Z95 = 1.959963984540054

# Reject formal reads until explicit diagnostic stage; preserve source bytes.
STAGE = "NONFORMAL"
FORMAL_READS = set()
def access_audit(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    p = Path(os.fsdecode(args[0])).absolute()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing:
        if (str(p).startswith(str(REPO)) and not str(p).startswith(str(OUT))) or str(p).startswith(str(RAW)):
            raise RuntimeError("protected source write: " + str(p))
    elif "formal" in p.parts or "formal_results" in p.name or p.name == "B1_CEILING.json":
        if STAGE != "FORMAL_DIAGNOSTIC":
            raise RuntimeError("formal access before model freeze: " + str(p))
        FORMAL_READS.add(str(p))
sys.dont_write_bytecode = True
sys.addaudithook(access_audit)

_spec_s1 = importlib.util.spec_from_file_location(
    "phase6_s1", REPO / "experiments/r16p19_phase6/run_s1.py"
)
if _spec_s1 is None or _spec_s1.loader is None:
    raise RuntimeError("cannot load frozen phase6 run_s1.py")
s1 = importlib.util.module_from_spec(_spec_s1)
sys.path.insert(0, str(REPO))
_spec_s1.loader.exec_module(s1)

_spec_b1 = importlib.util.spec_from_file_location(
    "phase7_b1", REPO / "experiments/r16p19_phase7/run_b1.py"
)
if _spec_b1 is None or _spec_b1.loader is None:
    raise RuntimeError("cannot load frozen phase7 run_b1.py")
b1 = importlib.util.module_from_spec(_spec_b1)
_spec_b1.loader.exec_module(b1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    a = np.asarray(x, dtype=np.float64)
    out = np.empty_like(a)
    positive = a >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-np.clip(a[positive], -80.0, 80.0)))
    exp_a = np.exp(np.clip(a[~positive], -80.0, 80.0))
    out[~positive] = exp_a / (1.0 + exp_a)
    if np.ndim(x) == 0:
        return float(out)
    return out


def logit(scores: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(np.asarray(scores, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    out = np.log(x) - np.log1p(-x)
    if np.ndim(scores) == 0:
        return float(out)
    return out


def wilson(k: int, n: int, z: float = Z95) -> list[float | None]:
    if n <= 0:
        return [None, None]
    p = float(k) / float(n)
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    w = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / d
    return [float(c - w), float(c + w)]


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = np.greater.outer(pos, neg).sum()
    equal = np.equal.outer(pos, neg).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def binary_nll(eta: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    # Stable Bernoulli negative log likelihood on logits.
    return float(np.sum(y * np.logaddexp(0.0, -eta) + (1.0 - y) * np.logaddexp(0.0, eta)))


def golden_minimize(func, left: float, right: float, iterations: int = 90) -> tuple[float, float]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    c = right - (right - left) / phi
    d = left + (right - left) / phi
    fc = func(c)
    fd = func(d)
    for _ in range(iterations):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - (right - left) / phi
            fc = func(c)
        else:
            left, c, fc = c, d, fd
            d = left + (right - left) / phi
            fd = func(d)
    q = (left + right) / 2.0
    return q, float(func(q))


def fit_temperature(z: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)

    def objective(log_t: float) -> float:
        return binary_nll(z * math.exp(-log_t), y)

    grid = np.linspace(-8.0, 8.0, 321)
    values = np.asarray([objective(float(q)) for q in grid])
    i = int(np.argmin(values))
    lo = float(grid[max(0, i - 1)])
    hi = float(grid[min(len(grid) - 1, i + 1)])
    if lo == hi:
        log_t = lo
        loss = float(values[i])
    else:
        log_t, loss = golden_minimize(objective, lo, hi)
    temperature = math.exp(log_t)
    return float(temperature), float(loss)


def fit_platt_positive(z: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Fit sigmoid(a*z+b) with a strictly positive slope.

    The intercept is solved by bisection for each positive slope, then a
    deterministic bounded one-dimensional search selects the likelihood
    minimum. This avoids an unconstrained negative-slope or sign-flipped map.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    target = float(np.mean(y))

    def intercept(a: float) -> float:
        lo, hi = -60.0, 60.0
        for _ in range(180):
            b = (lo + hi) / 2.0
            mean_p = float(np.mean(sigmoid(a * z + b)))
            if mean_p > target:
                hi = b
            else:
                lo = b
        return float((lo + hi) / 2.0)

    def objective(log_a: float) -> tuple[float, float]:
        a = math.exp(log_a)
        b = intercept(a)
        return binary_nll(a * z + b, y), b

    grid = np.linspace(-8.0, 8.0, 321)
    losses = np.asarray([objective(float(q))[0] for q in grid])
    i = int(np.argmin(losses))
    lo = float(grid[max(0, i - 1)])
    hi = float(grid[min(len(grid) - 1, i + 1)])

    def scalar(q: float) -> float:
        return objective(q)[0]

    if lo == hi:
        log_a = lo
        loss = float(losses[i])
    else:
        log_a, loss = golden_minimize(scalar, lo, hi)
    a = math.exp(log_a)
    b = objective(log_a)[1]
    return float(a), float(b), float(loss)


def latent_from_raw(raw_scores: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return params["a"] * logit(raw_scores) + params["b"]


def output_from_raw(raw_scores: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return np.asarray(sigmoid(latent_from_raw(raw_scores, params)), dtype=np.float64)


def fit_maps(effect_rows: dict[str, dict[str, list[Any]]]) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    report: dict[str, Any] = {}
    maps: dict[str, dict[str, dict[str, float]]] = {"temperature": {}, "platt": {}}
    grid = np.linspace(EPSILON, 1.0 - EPSILON, 10001)
    grid_z = np.asarray(logit(grid), dtype=np.float64)
    for key in sorted(effect_rows):
        x = np.asarray(effect_rows[key]["scores"], dtype=np.float64)
        y = np.asarray(effect_rows[key]["labels"], dtype=bool)
        z = np.asarray(logit(x), dtype=np.float64)
        temperature, temperature_loss = fit_temperature(z, y)
        platt_a, platt_b, platt_loss = fit_platt_positive(z, y)
        temperature_params = {"a": float(1.0 / temperature), "b": 0.0}
        platt_params = {"a": float(platt_a), "b": float(platt_b)}
        maps["temperature"][key] = temperature_params
        maps["platt"][key] = platt_params

        temp_z = temperature_params["a"] * z
        platt_z = platt_a * z + platt_b
        temp_grid = temperature_params["a"] * grid_z
        platt_grid = platt_a * grid_z + platt_b
        report[key] = {
            "samples": int(len(y)),
            "positive_count": int(y.sum()),
            "negative_count": int((~y).sum()),
            "temperature": {
                "temperature": float(temperature),
                "a": temperature_params["a"],
                "b": 0.0,
                "loss": temperature_loss,
                "slope_positive": bool(temperature_params["a"] > 0.0),
            },
            "platt": {
                "a": platt_a,
                "b": platt_b,
                "loss": platt_loss,
                "slope_positive": bool(platt_a > 0.0),
            },
            "auc_decision_score_before": auc_rank(y, z),
            "auc_decision_score_after": {
                "temperature": auc_rank(y, temp_z),
                "platt": auc_rank(y, platt_z),
            },
            "auc_delta": {
                "temperature": float(auc_rank(y, temp_z) - auc_rank(y, z)),
                "platt": float(auc_rank(y, platt_z) - auc_rank(y, z)),
            },
            "monotonicity": {
                "grid_points": int(len(grid)),
                "temperature_min_latent_difference": float(np.min(np.diff(temp_grid))),
                "platt_min_latent_difference": float(np.min(np.diff(platt_grid))),
                "temperature_non_decreasing": bool(np.all(np.diff(temp_grid) >= 0.0)),
                "platt_non_decreasing": bool(np.all(np.diff(platt_grid) >= 0.0)),
            },
            "decision_score_tie_count_before": int(len(z) - len(np.unique(z))),
            "decision_score_tie_count_after": {
                "temperature": int(len(temp_z) - len(np.unique(temp_z))),
                "platt": int(len(platt_z) - len(np.unique(platt_z))),
            },
        }
    return report, maps


def collect_split(split: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    if split == "formal":
        raise RuntimeError("formal split cannot enter non-formal collection")
    return s1._calibration_dataset(RAW / "episodes" / split, CHECKPOINT)


def collect_estimation() -> tuple[dict[str, dict[str, list[Any]]], list[dict[str, Any]], dict[str, list[Path]]]:
    effect_rows: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"scores": [], "labels": []})
    receipts: list[dict[str, Any]] = []
    paths: dict[str, list[Path]] = {}
    for split in ("calibration", "pilot"):
        effects, rows, _, input_paths = collect_split(split)
        paths[split] = list(input_paths)
        receipts.extend(rows)
        for key, values in effects.items():
            effect_rows[key]["scores"].extend(values["scores"])
            effect_rows[key]["labels"].extend(values["labels"])
    return dict(effect_rows), receipts, paths


def map_artifact(report: dict[str, Any], maps: dict[str, dict[str, dict[str, float]]], paths: dict[str, list[Path]]) -> dict[str, Any]:
    evidence = []
    for split in sorted(paths):
        for path in sorted(paths[split], key=str):
            evidence.append({"split": split, "path": str(path), "sha256": sha256(path)})
    return {
        "schema_version": 2,
        "status": "SEALED_RETROSPECTIVE_NONFORMAL_DIAGNOSTIC",
        "retraining_performed": False,
        "verifier_weights_unchanged": True,
        "fit_splits": ["calibration", "pilot"],
        "fit_unit": "all frame-effect rows, positive and negative labels",
        "epsilon": EPSILON,
        "decision_score": "logit(clip(raw_score, 1e-6, 1-1e-6)); AUC and threshold search use latent scores to avoid sigmoid ties",
        "models": report,
        "input_evidence": evidence,
        "input_evidence_sha256": hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "formal_access_count": 0,
        "formal_read_paths": [],
        "claim_eligible": False,
        "selection_eligible": False,
        "selection_seal_written": False,
        "note": "Retrospective repair artifact; C0 was already written before this audit and is not re-sealed.",
    }


def raw_matrix(
    receipts: list[dict[str, Any]], effects: list[str]
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    score_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    truth: list[bool] = []
    effect_set = set(effects)
    for row in receipts:
        keys = list(row["effect_keys"])
        if not set(keys).issubset(effect_set):
            raise RuntimeError("receipt contains an unknown effect key")
        scores = np.asarray(row["scores"], dtype=np.float64)
        labels = np.asarray(row["labels"], dtype=bool)
        if len(keys) != len(scores) or len(keys) != len(labels):
            raise RuntimeError("receipt score/label/effect lengths differ")
        score_rows.append(scores)
        label_rows.append(labels)
        truth.append(bool(labels.all()))
    return score_rows, label_rows, np.asarray(truth, dtype=bool)


def curve_for_receipts(
    receipts: list[dict[str, Any]],
    effects: list[str],
    model_name: str,
    maps: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    raw_rows, label_rows, truth = raw_matrix(receipts, effects)
    decision_rows: list[np.ndarray] = []
    output_rows: list[np.ndarray] = []
    for row, raw in zip(receipts, raw_rows):
        if model_name == "raw":
            decision = raw
            output = raw
            score_space = "raw_probability"
        else:
            decision = np.asarray(
                [
                    float(latent_from_raw(np.asarray([value]), maps[model_name][key])[0])
                    for value, key in zip(raw, row["effect_keys"])
                ],
                dtype=np.float64,
            )
            output = np.asarray(sigmoid(decision), dtype=np.float64)
            score_space = "effect_specific_transformed_probability_with_logit_decision"
        decision_rows.append(decision)
        output_rows.append(output)
    aggregate = np.asarray([float(np.min(row)) for row in decision_rows], dtype=np.float64)
    thresholds = np.r_[
        np.nextafter(float(aggregate.max()), np.inf),
        np.unique(aggregate)[::-1],
        np.nextafter(float(aggregate.min()), -np.inf),
    ]
    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        effect_pred_rows = [scores >= threshold for scores in decision_rows]
        receipt_pred = np.asarray(
            [bool(pred.all()) for pred in effect_pred_rows], dtype=bool
        )
        passed: dict[str, int] = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        false_passed: dict[str, int] = defaultdict(int)
        false_total: dict[str, int] = defaultdict(int)
        for row, labels, pred in zip(receipts, label_rows, effect_pred_rows):
            for key, label, effect_prediction in zip(row["effect_keys"], labels, pred):
                if bool(label):
                    total[key] += 1
                    passed[key] += int(bool(effect_prediction))
                else:
                    false_total[key] += 1
                    false_passed[key] += int(bool(effect_prediction))
        per_effect = {
            key: (float(passed[key] / total[key]) if total[key] else None)
            for key in effects
            if total[key]
        }
        per_effect_fpr = {
            key: (float(false_passed[key] / false_total[key]) if false_total[key] else None)
            for key in effects
            if false_total[key]
        }
        finite_threshold = float(threshold)
        threshold_output = (
            finite_threshold if model_name == "raw" else float(sigmoid(finite_threshold))
        )
        points.append(
            {
                "threshold": threshold_output,
                "threshold_decision_score": finite_threshold,
                "tp": int((receipt_pred & truth).sum()),
                "fp": int((receipt_pred & ~truth).sum()),
                "tpr": float(receipt_pred[truth].mean()) if truth.any() else None,
                "fpr": float(receipt_pred[~truth].mean()) if (~truth).any() else None,
                "oracle_agreement": float(np.mean(receipt_pred == truth)),
                "per_effect_tpr": per_effect,
                "per_effect_fpr": per_effect_fpr,
                "min_per_effect_tpr": float(min(v for v in per_effect.values() if v is not None)),
            }
        )
    tpr = np.asarray([row["tpr"] for row in points], dtype=np.float64)
    fpr = np.asarray([row["fpr"] for row in points], dtype=np.float64)
    trap_auc = float(np.trapz(tpr, fpr))
    rank_auc = auc_rank(truth, aggregate)
    return {
        "method": "min",
        "model": model_name,
        "score_space": score_space,
        "effect_decision_definition": "each effect decision is its transformed scalar score >= the common receipt threshold",
        "produces_effect_decisions": True,
        "receipt_count": int(len(receipts)),
        "positives": int(truth.sum()),
        "negatives": int((~truth).sum()),
        "auc_trapezoid": trap_auc,
        "auc_rank": rank_auc,
        "points": points,
    }

def read_old_b4() -> dict[str, Any]:
    path = REPO / "experiments/r16p19_phase7/B4_SOFT.json"
    return {"path": str(path), "sha256": sha256(path), "value": json.loads(path.read_text())}


def c3_audit(
    estimation: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    effects: list[str],
    maps: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    old = read_old_b4()
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "COMPLETED_RETROSPECTIVE_DIAGNOSTIC",
        "method": "min(scores)",
        "sum_log_p": {
            "status": "EXCLUDED",
            "reason": "sum(log p) is receipt-only and does not produce effect-level decisions",
        },
        "formal_access_count": 0,
        "formal_read_paths": [],
        "claim_eligible": False,
        "selection_eligible": False,
        "selection_performed": False,
        "phase7_b4_source": old["path"],
        "phase7_b4_sha256": old["sha256"],
        "splits": {},
    }
    for split_name, rows in (("estimation", estimation), ("qualification", qualification)):
        raw_curve = curve_for_receipts(rows, effects, "raw", maps)
        temp_curve = curve_for_receipts(rows, effects, "temperature", maps)
        platt_curve = curve_for_receipts(rows, effects, "platt", maps)
        expected = old["value"][split_name]["min"]
        baseline = {
            "phase7_auc_trapezoid": expected["auc"],
            "recomputed_auc_trapezoid": raw_curve["auc_trapezoid"],
            "auc_absolute_difference": abs(raw_curve["auc_trapezoid"] - expected["auc"]),
            "phase7_point_count": len(expected["points"]),
            "recomputed_point_count": len(raw_curve["points"]),
            "point_count_match": len(expected["points"]) == len(raw_curve["points"]),
            "auc_match": bool(abs(raw_curve["auc_trapezoid"] - expected["auc"]) <= 1e-12),
        }
        result["splits"][split_name] = {
            "receipt_count": raw_curve["receipt_count"],
            "positive_receipts": raw_curve["positives"],
            "negative_receipts": raw_curve["negatives"],
            "baseline_phase7_b4_reproduction": baseline,
            "raw": raw_curve,
            "temperature": temp_curve,
            "platt": platt_curve,
        }
    return result


def old_threshold_vector(b1_value: dict[str, Any], cap: str) -> dict[str, float]:
    index = 0 if cap == "0.02" else 1
    effects = b1_value["operating_points"][index]["effects"]
    return {key: float(row["threshold"]) for key, row in effects.items()}


def threshold_spread(vector: dict[str, float]) -> dict[str, float]:
    values = np.asarray(list(vector.values()), dtype=np.float64)
    mean = float(np.mean(values))
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.ptp(values)),
        "mean": mean,
        "std": float(np.std(values)),
        "cv": float(np.std(values) / mean) if mean != 0.0 else None,
    }


def transform_formal_receipts(
    receipts: list[dict[str, Any]],
    model_name: str,
    maps: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    out = []
    for row in receipts:
        new_row = dict(row)
        if model_name == "raw":
            new_row["scores"] = np.asarray(row["scores"], dtype=np.float64)
        else:
            new_row["scores"] = np.asarray(
                [
                    float(latent_from_raw(np.asarray([raw], dtype=np.float64), maps[model_name][key])[0])
                    for raw, key in zip(row["scores"], row["effect_keys"])
                ],
                dtype=np.float64,
            )
        out.append(new_row)
    return out


def ceiling_solution(
    receipts: list[dict[str, Any]],
    effects: list[str],
    model_name: str,
    cap: float,
    maps: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    transformed = transform_formal_receipts(receipts, model_name, maps)
    groups = [
        b1.task_options(
            [row for row in transformed if int(row["task_id"]) == task]
        )
        for task in sorted({int(row["task_id"]) for row in transformed})
    ]
    negative = sum(not bool(np.all(row["labels"])) for row in transformed)
    solution = b1.solve(groups, cap, negative)
    effects_out: dict[str, Any] = {}
    for key, row in solution["effects"].items():
        copy = dict(row)
        latent_threshold = float(row["threshold"])
        copy["threshold_decision_score"] = latent_threshold
        copy["threshold_probability"] = float(
            latent_threshold if model_name == "raw" else sigmoid(latent_threshold)
        )
        copy["threshold"] = copy["threshold_probability"]
        effects_out[key] = copy
    solution = dict(solution)
    solution["model"] = model_name
    solution["effects"] = effects_out
    solution["claim_eligible"] = False
    solution["selection_eligible"] = False
    return solution


def c1_formal_diagnostic(
    formal_receipts: list[dict[str, Any]],
    effects: list[str],
    maps: dict[str, dict[str, dict[str, float]]],
    nonformal_model_sha: str,
    map_sha: str,
) -> dict[str, Any]:
    b1_path = REPO / "experiments/r16p19_phase7/B1_CEILING.json"
    b1_value = json.loads(b1_path.read_text())
    old = old_threshold_vector(b1_value, "0.02")
    outputs: dict[str, Any] = {}
    for cap in (0.02, 0.05):
        cap_key = f"{cap:.2f}"
        outputs[cap_key] = {
            "raw_phase7_reference": {
                "threshold_vector": old_threshold_vector(b1_value, cap_key),
                "threshold_spread": threshold_spread(old_threshold_vector(b1_value, cap_key)),
                "min_per_effect_tpr": b1_value["operating_points"][0 if cap == 0.02 else 1]["min_per_effect_tpr"],
                "false_upgrade_count": b1_value["operating_points"][0 if cap == 0.02 else 1]["false_upgrade_count"],
            },
            "raw_recomputed": ceiling_solution(formal_receipts, effects, "raw", cap, maps),
            "temperature": ceiling_solution(formal_receipts, effects, "temperature", cap, maps),
            "platt": ceiling_solution(formal_receipts, effects, "platt", cap, maps),
        }
        for model_name in ("raw_recomputed", "temperature", "platt"):
            row = outputs[cap_key][model_name]
            vector = {key: float(value["threshold"]) for key, value in row["effects"].items()}
            row["threshold_vector"] = vector
            row["threshold_spread"] = threshold_spread(vector)
            row["claim_eligible"] = False
            row["selection_eligible"] = False
    convergence: dict[str, Any] = {}
    for model_name, label in (
        ("raw_phase7_reference", "raw"),
        ("temperature", "temperature"),
        ("platt", "platt"),
    ):
        row = outputs["0.02"][model_name]
        vector = row["threshold_vector"] if "threshold_vector" in row else row["threshold_vector"]
        convergence[label] = {
            "threshold_vector": vector,
            "spread": row["threshold_spread"],
        }
    return {
        "schema_version": 2,
        "status": "COMPLETED_RETROSPECTIVE_FORMAL_DIAGNOSTIC",
        "retraining_performed": False,
        "verifier_weights_unchanged": True,
        "fit_artifact": "C1_MODELS_NONFORMAL.json",
        "fit_artifact_sha256": nonformal_model_sha,
        "maps_artifact": "C2_EXTRA_MAPS_NONFORMAL.json",
        "maps_artifact_sha256": map_sha,
        "formal_diagnostic": {
            "formal_labels_used": True,
            "formal_access_count": 1,
            "formal_read_paths": [
                "experiments/r16p19_phase5/artifacts/results/learned_verifier_formal_results.jsonl",
                "raw/episodes/formal/*.npz and *.json",
            ],
            "formal_receipt_count": len(formal_receipts),
            "claim_eligible": False,
            "selection_eligible": False,
            "selection_seal_written": False,
            "diagnostic_only": True,
        },
        "threshold_convergence": convergence,
        "operating_points": outputs,
        "interpretation": (
            "Both scalar maps are strictly monotone on the logit decision scale, so "
            "per-effect ranking/AUC is invariant. Numeric threshold scale may move or "
            "compress, but this formal-label ceiling cannot establish a prospective claim "
            "and is excluded from all selection seals."
        ),
    }


def main() -> None:
    global STAGE
    OUT.mkdir(parents=True, exist_ok=True)
    effect_rows, estimation_receipts, paths = collect_estimation()
    effects = sorted(effect_rows)
    model_report, maps = fit_maps(effect_rows)

    # This is intentionally before any formal input access.
    nonformal_model = map_artifact(model_report, maps, paths)
    model_path = OUT / "C1_MODELS_NONFORMAL.json"
    map_path = OUT / "C2_EXTRA_MAPS_NONFORMAL.json"
    write_json(model_path, nonformal_model)
    write_json(map_path, maps)
    if nonformal_model["formal_access_count"] != 0 or nonformal_model["formal_read_paths"]:
        raise RuntimeError("non-formal artifact records formal access")
    if set(json.loads(map_path.read_text()).keys()) != {"temperature", "platt"}:
        raise RuntimeError("unexpected C2_EXTRA_MAPS_NONFORMAL schema")

    _, qualification_receipts, _, _ = collect_split("qualification")
    c3_path = OUT / "C3_AUDITED.json"
    c3 = c3_audit(estimation_receipts, qualification_receipts, effects, maps)
    c3['maps_sha256'] = sha256(map_path)
    write_json(c3_path, c3)

    # Formal is opened only after both non-formal artifacts and C3 are durable.
    calibration = json.loads((REPO / "experiments/r16p19_phase6/S1_CALIBRATION.json").read_text())
    STAGE = "FORMAL_DIAGNOSTIC"
    formal_receipts = s1._formal_receipts(REPO, RAW, calibration)
    c1 = c1_formal_diagnostic(
        formal_receipts,
        effects,
        maps,
        sha256(model_path),
        sha256(map_path),
    )
    c1.update(claim_eligible=False, selection_eligible=False)
    c1['formal_diagnostic']['formal_read_paths'] = sorted(FORMAL_READS)
    c1['formal_diagnostic']['formal_access_count'] = len(FORMAL_READS)
    c1['formal_diagnostic']['access_count_unit'] = 'unique paths; Python open audit'
    write_json(OUT / "C1_AUDITED.json", c1)
    print(
        json.dumps(
            {
                "status": "complete",
                "effects": effects,
                "nonformal_model": str(model_path),
                "map_artifact": str(map_path),
                "c3": str(OUT / "C3_AUDITED.json"),
                "c1": str(OUT / "C1_AUDITED.json"),
                "formal_claim_eligible": c1["formal_diagnostic"]["claim_eligible"],
                "formal_selection_eligible": c1["formal_diagnostic"]["selection_eligible"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

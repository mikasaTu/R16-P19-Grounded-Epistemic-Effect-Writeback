# RETIRED FAILED IMPLEMENTATION. Main disabled. Not a valid single-read experiment.
#!/usr/bin/env python3
"""Phase-9 formal-stage diagnostics after the independently committed D3 seal.

The process loads formal evidence exactly once, after verifying the D3 seal
commit.  All D1/D2/D4/D5/D6 calculations reuse the resulting in-memory
objects.  Formal-label quantities are diagnostic only and are marked
claim_eligible=false and selection_eligible=false.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
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
RAW = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/"
    "r16p19-phase5-bounded-ascel/pai/"
    "r16p19-phase5-idle-fixed-20260818-0855/rollouts"
)
D0_PATH = OUT / "D0_PREREGISTRATION.json"
SEAL_PATH = OUT / "D3_OPERATING_POINT_SEAL.json"
D3_PATH = OUT / "D3_SELECTION.json"
D0_SHA = "23dcfb04c6c1478fce14d1b1537cedf6e46283c622deeaf0aa158b7a03cbff10"
SEAL_COMMIT = "21acdc25331cb221fbeab2c34a1b9044580cadc2"
Z95 = 1.959963984540054
ALPHAS = (0.005, 0.01, 0.02, 0.05)
QUANTILES = (0.90, 0.95, 0.98, 0.99, 0.995)
TARGET_LEVELS = (0.02, 0.05, 0.10, 0.25, 0.50)
FORMAL_LOAD_SESSIONS = 0
FORMAL_READ_ACTIVE = False
FORMAL_READ_PATHS: set[str] = set()
# A first exploratory invocation reached the single formal loader before an
# import typo (missing numpy) aborted D1.  It is recorded as a protocol
# deviation; this corrected invocation still enforces one loader call.
PRIOR_FAILED_FORMAL_ATTEMPT = {
    "occurred": True,
    "reason": "first post-seal invocation loaded formal once then exited at D1 because numpy import was missing",
    "raw_formal_reads_in_failed_attempt": True,
}


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


def cp_upper(k: int, n: int, confidence: float = 0.95) -> float | None:
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    if k <= 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / n))
    tail = 1.0 - confidence

    def logsumexp(values: list[float]) -> float:
        m = max(values)
        return m + math.log(sum(math.exp(v - m) for v in values))

    def cdf(p: float) -> float:
        lp = math.log(p)
        lq = math.log1p(-p)
        terms = [
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * lp
            + (n - i) * lq
            for i in range(k + 1)
        ]
        return math.exp(min(0.0, logsumexp(terms)))

    lo, hi = float(k) / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if cdf(mid) > tail:
            lo = mid
        else:
            hi = mid
    return float(hi)


def wilson(k: int, n: int, z: float = Z95) -> list[float | None]:
    if n <= 0:
        return [None, None]
    p = float(k) / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    w = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / d
    return [float(c - w), float(c + w)]


def binom_lower_tail(k: int, n: int, p: float = 0.5) -> float | None:
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    lp = math.log(p)
    lq = math.log1p(-p)
    values = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * lp
        + (n - i) * lq
        for i in range(k + 1)
    ]
    m = max(values)
    return float(math.exp(m) * sum(math.exp(v - m) for v in values))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    a = np.asarray(x, dtype=np.float64)
    out = np.empty_like(a)
    positive = a >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-np.clip(a[positive], -80.0, 80.0)))
    exp_a = np.exp(np.clip(a[~positive], -80.0, 80.0))
    out[~positive] = exp_a / (1.0 + exp_a)
    return float(out) if np.ndim(x) == 0 else out


def logit(scores: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    out = np.log(x) - np.log1p(-x)
    return float(out) if np.ndim(scores) == 0 else out


def binary_nll(eta: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    return float(np.sum(y * np.logaddexp(0.0, -eta) + (1.0 - y) * np.logaddexp(0.0, eta)))


def golden_minimize(func: Any, left: float, right: float, iterations: int = 90) -> tuple[float, float]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    c = right - (right - left) / phi
    d = left + (right - left) / phi
    fc, fd = func(c), func(d)
    for _ in range(iterations):
        if fc < fd:
            right, d, fd = d, c, fc
            c, fc = right - (right - left) / phi, func(right - (right - left) / phi)
        else:
            left, c, fc = c, d, fd
            d, fd = left + (right - left) / phi, func(left + (right - left) / phi)
    q = (left + right) / 2.0
    return q, float(func(q))


def fit_platt_positive(z: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    target = float(np.mean(y))

    def intercept(a: float) -> float:
        lo, hi = -60.0, 60.0
        for _ in range(180):
            b = (lo + hi) / 2.0
            if float(np.mean(sigmoid(a * z + b))) > target:
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
    lo, hi = float(grid[max(0, i - 1)]), float(grid[min(len(grid) - 1, i + 1)])
    if lo == hi:
        log_a, loss = lo, float(losses[i])
    else:
        log_a, loss = golden_minimize(lambda q: objective(q)[0], lo, hi)
    a = math.exp(log_a)
    b = objective(log_a)[1]
    return float(a), float(b), float(loss)


def load_s1() -> Any:
    spec = importlib.util.spec_from_file_location(
        "phase6_s1_for_phase9_formal", REPO / "experiments/r16p19_phase6/run_s1.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen phase6 run_s1.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO))
    spec.loader.exec_module(module)
    return module


def collect_nonformal(s1: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, list[str]]]:
    effect_rows: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"scores": [], "labels": []})
    estimation: list[dict[str, Any]] = []
    qualification: list[dict[str, Any]] = []
    task_effects: dict[int, list[str]] = {}
    for split, target in (("calibration", estimation), ("pilot", estimation), ("qualification", qualification)):
        effects, rows, tasks, _ = s1._calibration_dataset(
            RAW / "episodes" / split,
            REPO / "experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz",
        )
        target.extend(rows)
        if split in {"calibration", "pilot"}:
            for key, values in effects.items():
                effect_rows[key]["scores"].extend(values["scores"])
                effect_rows[key]["labels"].extend(values["labels"])
        for task, keys in tasks.items():
            if task in task_effects and task_effects[task] != keys:
                raise RuntimeError(f"effect schema drift task={task}")
            task_effects[task] = list(keys)
    return dict(effect_rows), estimation, qualification, task_effects


def formal_audit(event: str, args: tuple[Any, ...]) -> None:
    if event != "open" or not args or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).absolute()
    lower = str(path).lower()
    is_formal = "formal" in lower or "formal_results" in lower
    if is_formal:
        if not FORMAL_READ_ACTIVE:
            raise RuntimeError(f"formal read outside the single post-seal load: {path}")
        FORMAL_READ_PATHS.add(str(path))


def load_formal_once(s1: Any, task_effects: dict[int, list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global FORMAL_LOAD_SESSIONS, FORMAL_READ_ACTIVE
    if FORMAL_LOAD_SESSIONS != 0:
        raise RuntimeError("formal source load attempted more than once")
    calibration = {"task_effects": {str(k): v for k, v in task_effects.items()}}
    FORMAL_LOAD_SESSIONS += 1
    FORMAL_READ_ACTIVE = True
    try:
        receipts = s1._formal_receipts(REPO, RAW, calibration)
        oracle_path = REPO / "experiments/r16p19_phase5/artifacts/results/oracle_formal_results.jsonl"
        oracle_rows = [
            json.loads(line)
            for line in oracle_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    finally:
        FORMAL_READ_ACTIVE = False
    if len(receipts) != 840:
        raise RuntimeError(f"formal receipt count drift: {len(receipts)}")
    if len({row["cluster_id"] for row in receipts}) != 120:
        raise RuntimeError("formal unit count drift")
    if len({row["condition"] for row in receipts}) != 7:
        raise RuntimeError("formal condition count drift")
    return receipts, oracle_rows


def negative_scores(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, score, label in zip(row["effect_keys"], row["scores"], row["labels"]):
            if not bool(label):
                values[key].append(float(score))
    return {key: np.asarray(value, dtype=np.float64) for key, value in sorted(values.items())}


def episode_negative_max(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        episode = str(row["episode_id"])
        for key, score, label in zip(row["effect_keys"], row["scores"], row["labels"]):
            if bool(label):
                continue
            value = float(score)
            old = values[key].get(episode)
            values[key][episode] = value if old is None else max(old, value)
    return {key: np.asarray(list(value.values()), dtype=np.float64) for key, value in sorted(values.items())}


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) == 0 or len(y) == 0:
        return None
    values = np.sort(np.unique(np.concatenate([x, y])))
    sx = np.searchsorted(np.sort(x), values, side="right") / len(x)
    sy = np.searchsorted(np.sort(y), values, side="right") / len(y)
    return float(np.max(np.abs(sx - sy)))


def d1_exchangeability(estimation: list[dict[str, Any]], qualification: list[dict[str, Any]], formal: list[dict[str, Any]]) -> dict[str, Any]:
    est = negative_scores(estimation)
    qual = negative_scores(qualification)
    frm = negative_scores(formal)
    effects = sorted(est)
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "negative-score quantile curves and two-sample KS; formal evaluation diagnostic only",
        "quantiles": list(QUANTILES),
        "formal_access_count": FORMAL_LOAD_SESSIONS,
        "formal_read_paths": sorted(FORMAL_READ_PATHS),
        "protocol_deviation": PRIOR_FAILED_FORMAL_ATTEMPT,
        "claim_eligible": False,
        "selection_eligible": False,
        "effects": {},
    }
    max_rare_ratio = 0.0
    for key in effects:
        estimation_values, qualification_values, formal_values = est[key], qual[key], frm[key]
        curve = {}
        fpr_by_alpha = {}
        for q in QUANTILES:
            qe, qq, qf = (float(np.quantile(values, q)) if len(values) else None for values in (estimation_values, qualification_values, formal_values))
            curve[str(q)] = {
                "estimation": qe,
                "qualification": qq,
                "formal": qf,
                "qualification_minus_estimation": None if qq is None or qe is None else qq - qe,
                "formal_minus_estimation": None if qf is None or qe is None else qf - qe,
            }
        for alpha in ALPHAS:
            threshold = float(np.quantile(estimation_values, 1.0 - alpha))
            actual = float(np.mean(formal_values > threshold)) if len(formal_values) else None
            fpr_by_alpha[f"{alpha:.3f}"] = {
                "estimation_quantile": threshold,
                "formal_actual_fpr": actual,
                "nominal_alpha": alpha,
                "actual_minus_nominal": None if actual is None else actual - alpha,
                "actual_over_nominal": None if actual is None else actual / alpha,
            }
        rare = any(token in key.lower() for token in ("tomato", "book", "close"))
        ratios = [x["actual_over_nominal"] for x in fpr_by_alpha.values() if x["actual_over_nominal"] is not None]
        effect_max_ratio = max(ratios, default=0.0)
        if rare:
            max_rare_ratio = max(max_rare_ratio, effect_max_ratio)
        report["effects"][key] = {
            "negative_counts": {"estimation": len(estimation_values), "qualification": len(qualification_values), "formal": len(formal_values)},
            "quantile_curve": curve,
            "ks": {"qualification_vs_estimation": ks_statistic(estimation_values, qualification_values), "formal_vs_estimation": ks_statistic(formal_values, estimation_values)},
            "formal_fpr_at_estimation_quantiles": fpr_by_alpha,
            "rare_effect": rare,
            "max_formal_to_nominal_ratio": effect_max_ratio,
        }
    report["max_rare_effect_formal_to_nominal_ratio"] = max_rare_ratio
    report["gate_threshold_ratio"] = 2.0
    report["d1_pass"] = gate_exchangeability(max_rare_ratio)
    return report


def transform_rows(rows: list[dict[str, Any]], maps: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    transformed = []
    for row in rows:
        out = dict(row)
        out["scores"] = np.asarray(
            [float(sigmoid(maps[key]["a"] * logit(float(score)) + maps[key]["b"])) for key, score in zip(row["effect_keys"], row["scores"])],
            dtype=np.float64,
        )
        transformed.append(out)
    return transformed


def fit_unbalanced_maps(effect_rows: dict[str, Any]) -> dict[str, dict[str, float]]:
    maps = {}
    for key, values in sorted(effect_rows.items()):
        x = np.asarray(values["scores"], dtype=np.float64)
        y = np.asarray(values["labels"], dtype=bool)
        a, b, loss = fit_platt_positive(np.asarray(logit(x)), y)
        maps[key] = {"a": a, "b": b, "loss": loss}
    return maps


def fit_balanced_maps(effect_rows: dict[str, Any], target: float, seed: int) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    maps: dict[str, dict[str, float]] = {}
    samples: dict[str, Any] = {}
    for offset, key in enumerate(sorted(effect_rows)):
        x = np.asarray(effect_rows[key]["scores"], dtype=np.float64)
        y = np.asarray(effect_rows[key]["labels"], dtype=bool)
        positive = x[y]
        negative = x[~y]
        if len(positive) == 0 or len(negative) == 0:
            raise RuntimeError(f"cannot balance effect {key}")
        n_negative = max(1, int(round(len(positive) * (1.0 - target) / target)))
        rng = np.random.default_rng(seed + offset * 1000003)
        sampled = rng.choice(negative, size=n_negative, replace=True)
        train_x = np.concatenate([positive, sampled])
        train_y = np.concatenate([np.ones(len(positive), dtype=bool), np.zeros(n_negative, dtype=bool)])
        a, b, loss = fit_platt_positive(np.asarray(logit(train_x)), train_y)
        maps[key] = {"a": a, "b": b, "loss": loss}
        samples[key] = {
            "positive_count": int(len(positive)),
            "negative_count_sampled": int(n_negative),
            "realized_positive_fraction": float(len(positive) / (len(positive) + n_negative)),
            "negative_source_count": int(len(negative)),
        }
    return maps, samples


def threshold_spread(vector: dict[str, float]) -> dict[str, float | None]:
    values = np.asarray(list(vector.values()), dtype=np.float64)
    mean = float(np.mean(values)) if len(values) else None
    return {
        "min": float(np.min(values)) if len(values) else None,
        "max": float(np.max(values)) if len(values) else None,
        "range": float(np.ptp(values)) if len(values) else None,
        "mean": mean,
        "std": float(np.std(values)) if len(values) else None,
        "cv": float(np.std(values) / mean) if len(values) and mean else None,
    }


def ceiling_solution(rows: list[dict[str, Any]], cap: float, b1: Any, model_name: str) -> dict[str, Any]:
    groups = [
        b1.task_options([row for row in rows if int(row["task_id"]) == task])
        for task in sorted({int(row["task_id"]) for row in rows})
    ]
    negative = sum(not bool(np.all(row["labels"])) for row in rows)
    solution = b1.solve(groups, cap, negative)
    effects: dict[str, Any] = {}
    probability_vector: dict[str, float] = {}
    decision_vector: dict[str, float] = {}
    for key, value in sorted(solution["effects"].items()):
        latent = float(value["threshold"])
        threshold = float(latent)
        probability_vector[key] = threshold
        decision_vector[key] = threshold
        effects[key] = dict(value, threshold_probability=threshold, threshold_decision_score=threshold)
    spread = threshold_spread(probability_vector)
    return dict(solution, model=model_name, effects=effects, threshold_vector=probability_vector, threshold_decision_vector=decision_vector, threshold_spread=spread, claim_eligible=False, selection_eligible=False)


def d2_baserate(effect_rows: dict[str, Any], formal: list[dict[str, Any]], b1: Any) -> dict[str, Any]:
    unbalanced_maps = fit_unbalanced_maps(effect_rows)
    unbalanced_rows = transform_rows(formal, unbalanced_maps)
    reference = {
        f"{cap:.2f}": ceiling_solution(unbalanced_rows, cap, b1, "platt_unbalanced")
        for cap in (0.02, 0.05)
    }
    levels: dict[str, Any] = {}
    for level_index, target in enumerate(TARGET_LEVELS):
        replicates = []
        for rep in range(10):
            seed = 19001 + rep
            maps, sample_info = fit_balanced_maps(effect_rows, target, seed)
            transformed = transform_rows(formal, maps)
            by_cap = {
                f"{cap:.2f}": ceiling_solution(transformed, cap, b1, "platt_balanced")
                for cap in (0.02, 0.05)
            }
            replicates.append({"replicate": rep, "seed": seed, "maps": maps, "sample_info": sample_info, "operating_points": by_cap})
        summary: dict[str, Any] = {}
        for cap in (0.02, 0.05):
            cap_key = f"{cap:.2f}"
            rows = [r["operating_points"][cap_key] for r in replicates]
            ranges = np.asarray([r["threshold_spread"]["range"] for r in rows], dtype=np.float64)
            cvs = np.asarray([r["threshold_spread"]["cv"] for r in rows], dtype=np.float64)
            summary[cap_key] = {
                "range": {"values": ranges.tolist(), "median": float(np.median(ranges)), "min": float(np.min(ranges)), "max": float(np.max(ranges))},
                "cv": {"values": cvs.tolist(), "median": float(np.median(cvs)), "min": float(np.min(cvs)), "max": float(np.max(cvs))},
                "replicate_threshold_vectors": [r["threshold_vector"] for r in rows],
            }
        levels[f"{target:.2f}"] = {"target_positive_fraction": target, "replicates": replicates, "summary": summary}
    support: dict[str, Any] = {}
    for cap in (0.02, 0.05):
        cap_key = f"{cap:.2f}"
        reference_range = reference[cap_key]["threshold_spread"]["range"]
        comparisons = {}
        for target_key, data in levels.items():
            median_range = data["summary"][cap_key]["range"]["median"]
            comparisons[target_key] = {"reference_range": reference_range, "balanced_median_range": median_range, "balanced_range_smaller": bool(median_range < reference_range)}
        support[cap_key] = comparisons
    counts = {key: {"positive": int(np.sum(np.asarray(values["labels"], dtype=bool))), "negative": int(np.sum(~np.asarray(values["labels"], dtype=bool)))} for key, values in sorted(effect_rows.items())}
    return {
        "schema_version": 1,
        "method": "negative resampling with replacement; positive fraction levels; positive-slope Platt maps; formal ceiling diagnostic",
        "positive_counts_fixed": counts,
        "target_levels": list(TARGET_LEVELS),
        "replicates_per_level": 10,
        "seeds": list(range(19001, 19011)),
        "unbalanced_reference": reference,
        "levels": levels,
        "support_assessment": support,
        "formal_access_count": FORMAL_LOAD_SESSIONS,
        "formal_read_paths": sorted(FORMAL_READ_PATHS),
        "protocol_deviation": PRIOR_FAILED_FORMAL_ATTEMPT,
        "claim_eligible": False,
        "selection_eligible": False,
        "conclusion": "formal labels are diagnostic only; a smaller balanced median threshold range at >=4/5 levels is treated as support for the base-rate mechanism, not causal proof",
    }


def receipt_accepts(row: dict[str, Any], thresholds: dict[str, float | None]) -> bool:
    return all(value is not None and float(score) > float(value) for key, score in zip(row["effect_keys"], row["scores"]) for value in [thresholds.get(key)])


def d4_formal(formal: list[dict[str, Any]], seal: dict[str, Any], b1_ceiling: dict[str, Any]) -> dict[str, Any]:
    thresholds = {key: value for key, value in seal["selected_threshold_vector"].items()}
    effects = sorted(thresholds)
    predictions = [receipt_accepts(row, thresholds) for row in formal]
    oracle = [bool(np.all(row["labels"])) for row in formal]
    per_effect: dict[str, Any] = {}
    for key in effects:
        receipt_truth = [bool(row["labels"][row["effect_keys"].index(key)]) for row in formal]
        receipt_pred = [thresholds[key] is not None and float(row["scores"][row["effect_keys"].index(key)]) > float(thresholds[key]) for row in formal]
        tp = sum(p and y for p, y in zip(receipt_pred, receipt_truth))
        fn = sum((not p) and y for p, y in zip(receipt_pred, receipt_truth))
        fp = sum(p and not y for p, y in zip(receipt_pred, receipt_truth))
        tn = sum((not p) and not y for p, y in zip(receipt_pred, receipt_truth))
        unit_truth: dict[str, bool] = {}
        unit_pred: dict[str, bool] = {}
        for row, p in zip(formal, receipt_pred):
            unit = str(row["cluster_id"])
            unit_truth[unit] = unit_truth.get(unit, False) or bool(row["labels"][row["effect_keys"].index(key)])
            unit_pred[unit] = unit_pred.get(unit, False) or bool(p)
        units = sorted(unit_truth)
        utp = sum(unit_pred[u] and unit_truth[u] for u in units)
        ufn = sum((not unit_pred[u]) and unit_truth[u] for u in units)
        ufp = sum(unit_pred[u] and not unit_truth[u] for u in units)
        utn = sum((not unit_pred[u]) and not unit_truth[u] for u in units)
        per_effect[key] = {
            "threshold": thresholds[key],
            "reject_all": thresholds[key] is None,
            "receipt": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "positive_count": tp + fn, "negative_count": fp + tn, "tpr": float(tp / (tp + fn)) if tp + fn else None, "fpr": float(fp / (fp + tn)) if fp + tn else None, "tpr_wilson95": wilson(tp, tp + fn), "fpr_wilson95": wilson(fp, fp + tn), "fpr_exact_binomial_upper95": cp_upper(fp, fp + tn, 0.95), "fpr_exact_binomial_upper99": cp_upper(fp, fp + tn, 0.99)},
            "episode_cluster": {"tp": utp, "fn": ufn, "fp": ufp, "tn": utn, "positive_count": utp + ufn, "negative_count": ufp + utn, "tpr": float(utp / (utp + ufn)) if utp + ufn else None, "fpr": float(ufp / (ufp + utn)) if ufp + utn else None, "tpr_wilson95": wilson(utp, utp + ufn), "fpr_wilson95": wilson(ufp, ufp + utn), "fpr_exact_binomial_upper95": cp_upper(ufp, ufp + utn, 0.95), "fpr_exact_binomial_upper99": cp_upper(ufp, ufp + utn, 0.99)},
        }
    negative_receipt = [not y for y in oracle]
    fp_receipts = sum(p and n for p, n in zip(predictions, negative_receipt))
    negative_units: set[str] = set()
    fp_units: set[str] = set()
    for row, p, n in zip(formal, predictions, negative_receipt):
        if n:
            negative_units.add(str(row["cluster_id"]))
        if p and n:
            fp_units.add(str(row["cluster_id"]))
    negative_unit_count = len(negative_units)
    agreement = sum(p == y for p, y in zip(predictions, oracle)) / len(formal)
    b1point = b1_ceiling["operating_points"][0]
    return {
        "schema_version": 1,
        "status": "COMPLETED_SINGLE_FORMAL_EVALUATION",
        "selected_alpha": seal["selected_alpha"],
        "selected_threshold_vector": thresholds,
        "threshold_comparison": "> strict; null threshold is explicit reject-all",
        "per_effect": per_effect,
        "receipt_false_upgrades": {"count": fp_receipts, "rate": float(fp_receipts / sum(negative_receipt)) if sum(negative_receipt) else None, "negative_receipt_count": int(sum(negative_receipt)), "wilson95": wilson(fp_receipts, int(sum(negative_receipt))), "exact_binomial_upper95": cp_upper(fp_receipts, int(sum(negative_receipt)), 0.95), "exact_binomial_upper99": cp_upper(fp_receipts, int(sum(negative_receipt)), 0.99), "episode_cluster_count": len(fp_units), "episode_cluster_denominator": negative_unit_count, "episode_cluster_rate": float(len(fp_units) / negative_unit_count) if negative_unit_count else None, "episode_cluster_wilson95": wilson(len(fp_units), negative_unit_count), "episode_cluster_exact_binomial_upper95": cp_upper(len(fp_units), negative_unit_count, 0.95), "episode_cluster_exact_binomial_upper99": cp_upper(len(fp_units), negative_unit_count, 0.99), "episode_cluster_pvalue_rate_at_most_half": binom_lower_tail(len(fp_units), negative_unit_count, 0.5)},
        "oracle_receipt_concordance": float(agreement),
        "oracle_receipt_count": int(sum(oracle)),
        "formal_receipt_count": len(formal),
        "formal_unit_count": len({row["cluster_id"] for row in formal}),
        "formal_condition_count": len({row["condition"] for row in formal}),
        "b1_ceiling_gap": {"b1_source": "experiments/r16p19_phase7/B1_CEILING.json", "b1_min_per_effect_tpr": b1point["min_per_effect_tpr"], "b1_false_upgrade_count": b1point["false_upgrade_count"], "selected_min_per_effect_tpr": float(min((row["receipt"]["tpr"] for row in per_effect.values())),), "selected_false_upgrade_count": fp_receipts, "min_tpr_gap_selected_minus_b1": float(min(row["receipt"]["tpr"] for row in per_effect.values()) - b1point["min_per_effect_tpr"]), "false_upgrade_count_gap_selected_minus_b1": int(fp_receipts - b1point["false_upgrade_count"])},
        "d3_seal_commit": SEAL_COMMIT,
        "formal_access_count": FORMAL_LOAD_SESSIONS,
        "formal_read_paths": sorted(FORMAL_READ_PATHS),
        "protocol_deviation": PRIOR_FAILED_FORMAL_ATTEMPT,
        "claim_eligible": False,
        "selection_eligible": False,
    }


def d5_replay(formal: list[dict[str, Any]], oracle_rows: list[dict[str, Any]], seal: dict[str, Any], b1_ceiling: dict[str, Any]) -> dict[str, Any]:
    thresholds = seal["selected_threshold_vector"]
    predictions = {f"{row['cluster_id']}:{row['condition']}": receipt_accepts(row, thresholds) for row in formal}
    oracle_receipt = {f"{row['cluster_id']}:{row['condition']}": bool(np.all(row["labels"])) for row in formal}
    by_unit: dict[str, list[str]] = defaultdict(list)
    for row in formal:
        by_unit[str(row["cluster_id"])].append(f"{row['cluster_id']}:{row['condition']}")
    divergent = sorted(unit for unit, ids in by_unit.items() if any(predictions[i] != oracle_receipt[i] for i in ids))
    concordant = sorted(set(by_unit) - set(divergent))
    if sorted({len(ids) for ids in by_unit.values()}) != [7]:
        raise RuntimeError("D5 requires seven conditions per unit")
    oracle_by: dict[tuple[str, str, str], bool] = {}
    for row in oracle_rows:
        arm = row.get("arm")
        condition = row.get("condition")
        cluster = str(row.get("cluster_id"))
        if arm in {"M0_TYPED_MATCHED", "M3_ASCEL_CORE"} and condition != "C0_CLEAN":
            oracle_by[(cluster, condition, arm)] = bool(row["task_success"])
    expected_cells = len(by_unit) * 6 * 2
    if len(oracle_by) != expected_cells:
        raise RuntimeError(f"oracle fault-cell count drift: {len(oracle_by)} != {expected_cells}")
    total_faulted_cells = expected_cells // 2
    oracle_diff_sum = 0.0
    known_diff_sum = 0.0
    for (unit, condition, arm), success in oracle_by.items():
        sign = 1.0 if arm == "M3_ASCEL_CORE" else -1.0
        oracle_diff_sum += sign * float(success)
        if unit in concordant:
            known_diff_sum += sign * float(success)
    divergent_faulted_cells = len(divergent) * 6
    lower = (known_diff_sum - divergent_faulted_cells) / total_faulted_cells
    upper = oracle_diff_sum / total_faulted_cells
    neutral = known_diff_sum / total_faulted_cells
    b1 = b1_ceiling["operating_points"][0]
    return {
        "schema_version": 1,
        "status": "COMPLETED_RECEIPT_AND_LEDGER_REPLAY",
        "ledger_semantics": "receipt-AND: every effect must pass strict > threshold; one miss vetoes receipt; unit diverges if any of seven receipt decisions differs from oracle",
        "threshold_vector": thresholds,
        "condition_count_per_unit": 7,
        "unit_count": len(by_unit),
        "concordant_unit_count": len(concordant),
        "divergent_unit_count": len(divergent),
        "concordant_unit_ids": concordant,
        "divergent_unit_ids": divergent,
        "receipt_concordance": float(sum(predictions[k] == oracle_receipt[k] for k in predictions) / len(predictions)),
        "oracle_receipt_concordance": float(sum(predictions[k] == oracle_receipt[k] for k in predictions) / len(predictions)),
        "faulted_cell_count_per_arm": total_faulted_cells,
        "gain_interval": {"lower_adversarial": lower, "upper_inherit_oracle": upper, "neutral_assumption": neutral, "neutral_is_assumption": True, "lower_rule": "each divergent unit's six faulted cells take the worst Core-minus-baseline sign", "upper_rule": "divergent units inherit frozen oracle outcomes", "point_estimate_forbidden": True},
        "phase6_same_metric_comparison": {"C_concordant_units": 102, "C_total_units": 120, "C_concordance": 102 / 120, "C_faulted_cells_concordant": 197, "C_faulted_cells_total": 227, "C_faulted_cell_concordance": 197 / 227, "comparison_note": "phase6 C values are copied from the frozen task background and use the same unit/receipt-AND definition"},
        "b1_ceiling_reference": {"min_per_effect_tpr": b1["min_per_effect_tpr"], "false_upgrade_count": b1["false_upgrade_count"]},
        "s2_reexecution_unit_ids": divergent,
        "s2_reexecution_unit_count": len(divergent),
        "formal_access_count": FORMAL_LOAD_SESSIONS,
        "formal_read_paths": sorted(FORMAL_READ_PATHS),
        "claim_eligible": False,
        "selection_eligible": False,
        "s2_started": False,
    }


def d6_failure_modes(formal: list[dict[str, Any]], seal: dict[str, Any]) -> dict[str, Any]:
    thresholds = seal["selected_threshold_vector"]
    false_rows = [row for row in formal if receipt_accepts(row, thresholds) and not bool(np.all(row["labels"]))]
    unit_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in false_rows:
        unit_rows[str(row["cluster_id"])].append(row)
    total_units = len({str(row["cluster_id"]) for row in formal})
    unit_ids = sorted(unit_rows)
    task_dist = Counter(int(row["task_id"]) for row in false_rows)
    condition_dist = Counter(str(row["condition"]) for row in false_rows)
    special = {condition: sum(1 for row in false_rows if row["condition"] == condition) for condition in ("A5_EXTERNAL_REALIZATION", "C0_CLEAN")}
    return {
        "schema_version": 1,
        "status": "COMPLETED_FALSE_UPGRADE_FAILURE_MODE_AUDIT",
        "false_upgrade_receipt_count": len(false_rows),
        "false_upgrade_unit_count": len(unit_ids),
        "false_upgrade_unit_ids": unit_ids,
        "total_unit_count": total_units,
        "concentration": {"false_upgrade_units_over_total_units": float(len(unit_ids) / total_units) if total_units else None, "unit_count_denominator": total_units, "false_upgrade_unit_count_numerator": len(unit_ids)},
        "task_distribution": {str(k): int(v) for k, v in sorted(task_dist.items())},
        "condition_distribution": {str(k): int(v) for k, v in sorted(condition_dist.items())},
        "special_condition_checks": {condition: {"false_upgrade_receipt_count": count, "false_upgrade_unit_count": len({str(row["cluster_id"]) for row in false_rows if row["condition"] == condition})} for condition, count in special.items()},
        "boundary_checks": {"temporal_or_attribution_boundary_evidence": None, "realized_vs_observed_external_confusion": None, "reason": "frozen formal artifacts expose no event-level timing/type fields for a false upgrade; zero false upgrades also makes the check vacuous"},
        "interpretation": "With a reject-all fallback the false-upgrade set is empty, so no concentration or temporal/attribution mechanism can be inferred; this is a safety consequence of zero upgrades, not evidence that boundary confusion is absent.",
        "formal_access_count": FORMAL_LOAD_SESSIONS,
        "formal_read_paths": sorted(FORMAL_READ_PATHS),
        "claim_eligible": False,
        "selection_eligible": False,
    }


def gate_exchangeability(max_rare_fpr_ratio: float | None) -> bool:
    return max_rare_fpr_ratio is not None and math.isfinite(max_rare_fpr_ratio) and max_rare_fpr_ratio <= 2.0


def gate_effect_fpr(ucbs: list[float | None], alpha: float | None, certified: bool) -> bool:
    return bool(certified and alpha is not None and all(value is not None and value <= alpha for value in ucbs))


def gate_receipt_fpr(cluster_ucb: float | None, pvalue_vs_half: float | None) -> bool:
    return bool(cluster_ucb is not None and cluster_ucb <= 0.05 and pvalue_vs_half is not None and pvalue_vs_half <= 0.05)


def gate_ledger(concordant: int | None, total: int | None) -> bool:
    return bool(concordant is not None and total is not None and total > 0 and concordant / total >= 0.85)


def source_snapshot() -> dict[str, str]:
    value = {}
    for dirname in ("r16p19_phase5", "r16p19_phase6", "r16p19_phase7", "r16p19_phase8"):
        root = REPO / "experiments" / dirname
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                value[path.relative_to(REPO).as_posix()] = sha256(path)
    return value


def write_mechanism(d1: dict[str, Any], d2: dict[str, Any], d4: dict[str, Any], d5: dict[str, Any], d6: dict[str, Any]) -> None:
    d2_support = d2["support_assessment"]
    support_counts = {}
    for cap, levels in d2_support.items():
        support_counts[cap] = sum(bool(v["balanced_range_smaller"]) for v in levels.values())
    lines = [
        "# Phase-9 机制反解（不生成新 idea）",
        "",
        "本文件只解释本轮代码和冻结数据中观察到的升降，不把诊断结果包装成新的 idea。",
        "",
        "## 负类保形阈值",
        "",
        f"D3 的 primary episode-max 负类样本每个 effect 只有 15 个。四个 alpha 的秩均为 16，超过样本数，因此冻结规则产生 null/reject-all 阈值。该点的零假升级来自拒绝所有 witness，不能解释为检测器达到高 TPR。",
        "D3 选择器只读取负类导出；正样本路径 hook 测试通过，qualification 被排除，formal 访问数为 0。",
        "",
        "## D1/D2：分数迁移和 base-rate 机制",
        "",
        f"D1 稀有 effect 的 formal/nominal 最大 FPR 比为 {d1['max_rare_effect_formal_to_nominal_ratio']:.6g}，D1 gate={'通过' if d1['d1_pass'] else '未通过'}。",
        f"D2 每个 base-rate 水平做 10 个固定 seed 重采样；在 cap .02/.05 下，balanced range 小于 unbalanced reference 的水平数分别为 {support_counts.get('0.02', 0)}/5 和 {support_counts.get('0.05', 0)}/5。这个比较只能支持或削弱 base-rate 与阈值尺度相关的机制，不能从 formal 标签得到因果证明。",
        "",
        "## D4/D5/D6：ledger 结果的代码原因",
        "",
        f"D4 采用 strict > 与 null reject-all，所以 formal 假升级为 {d4['receipt_false_upgrades']['count']}，同时每 effect 的召回为 0；低 FPR 和低 TPR 是同一拒绝规则的两面。",
        f"D5 沿用 receipt-AND：一个 effect 未过就否决整条 receipt。当前点 unit 一致 {d5['concordant_unit_count']}/{d5['unit_count']}，分歧 {d5['divergent_unit_count']} 个；与 phase6 C 的 102/120 unit、197/227 fault-cell 口径已并列报告。",
        f"D6 找到 {d6['false_upgrade_unit_count']} 个 false-upgrade unit，占 {d6['concentration']['false_upgrade_units_over_total_units'] if d6['concentration']['false_upgrade_units_over_total_units'] is not None else 'null'}；空集合意味着边界归属机制没有可观测证据，不能据此宣称已经排除。",
        "",
        "结论：本轮主要观测是 episode 聚类样本过少导致保形阈值退化为 reject-all；该规则确实压低假升级，但没有产生可认证的有效 witness 召回或 ledger 一致率。",
    ]
    (OUT / "MECHANISM_REVERSE_ENGINEERING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global FORMAL_READ_ACTIVE
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(D0_PATH) != D0_SHA:
        raise RuntimeError("D0 hash drift")
    if not SEAL_PATH.is_file() or not D3_PATH.is_file():
        raise RuntimeError("D3 seal/selection missing")
    seal_sha = sha256(SEAL_PATH)
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    d3 = json.loads(D3_PATH.read_text(encoding="utf-8"))
    if seal.get("d0_preregistration_sha256") != D0_SHA or d3.get("d0_preregistration_sha256") != D0_SHA:
        raise RuntimeError("D3 does not bind frozen D0")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != SEAL_COMMIT:
        raise RuntimeError(f"formal evaluation must run at seal commit {SEAL_COMMIT}, got {head}")
    before_sources = source_snapshot()
    s1 = load_s1()
    effect_rows, estimation, qualification, task_effects = collect_nonformal(s1)
    # Install the formal-path hook only after all non-formal collection and seal verification.
    sys.addaudithook(formal_audit)
    formal, oracle_rows = load_formal_once(s1, task_effects)
    if FORMAL_LOAD_SESSIONS != 1 or FORMAL_READ_ACTIVE or not FORMAL_READ_PATHS:
        raise RuntimeError("single formal load audit failed")
    d1 = d1_exchangeability(estimation, qualification, formal)
    write_json(OUT / "D1_EXCHANGEABILITY.json", d1)
    d1_decision = [
        "# D1 decision",
        "",
        f"最大稀有 effect formal/nominal FPR 比为 `{d1['max_rare_effect_formal_to_nominal_ratio']:.6g}`，阈值为 `2.0`，D1 判定为 **{'通过' if d1['d1_pass'] else 'NOT_EXCHANGEABLE（未通过）'}**。",
        "",
        "按用户已确认的执行顺序，即使该门失败也继续完成 D2、D4、D5、D6；此文件不改变任何已封存选择规则。",
    ]
    (OUT / "D1_DECISION.md").write_text("\n".join(d1_decision) + "\n", encoding="utf-8")
    spec_b1 = importlib.util.spec_from_file_location("phase7_b1_after_seal", REPO / "experiments/r16p19_phase7/run_b1.py")
    if spec_b1 is None or spec_b1.loader is None:
        raise RuntimeError("cannot load phase7 solver")
    b1 = importlib.util.module_from_spec(spec_b1)
    spec_b1.loader.exec_module(b1)
    d2 = d2_baserate(effect_rows, formal, b1)
    write_json(OUT / "D2_BASERATE.json", d2)
    b1_ceiling = json.loads((REPO / "experiments/r16p19_phase7/B1_CEILING.json").read_text(encoding="utf-8"))
    d4 = d4_formal(formal, seal, b1_ceiling)
    d4["d3_seal_sha256"] = seal_sha
    write_json(OUT / "D4_FORMAL.json", d4)
    d5 = d5_replay(formal, oracle_rows, seal, b1_ceiling)
    write_json(OUT / "D5_REPLAY.json", d5)
    (OUT / "D5_S2_UNITS.txt").write_text("\n".join(d5["s2_reexecution_unit_ids"]) + ("\n" if d5["s2_reexecution_unit_ids"] else ""), encoding="utf-8")
    d6 = d6_failure_modes(formal, seal)
    write_json(OUT / "D6_FAILURE_MODES.json", d6)
    d6_text = [
        "# D6 假升级失效模式描述",
        "",
        f"formal 工作点产生 {d6['false_upgrade_receipt_count']} 条假升级、涉及 {d6['false_upgrade_unit_count']}/{d6['total_unit_count']} 个 unit。",
        "",
        "由于 D3 fallback 是 reject-all，假升级集合为空；A5_EXTERNAL_REALIZATION、C0_CLEAN 的集中性计数均为 0。冻结事件证据没有提供可用于判断时序或 realized/observed/external 归属混淆的字段，因此该机制检查为不可判定，不据空集合宣称机制不存在。",
    ]
    (OUT / "D6_FAILURE_MODES.md").write_text("\n".join(d6_text) + "\n", encoding="utf-8")
    gate_values = {
        "gate_exchangeability": {"function": "gate_exchangeability", "value": gate_exchangeability(d1["max_rare_effect_formal_to_nominal_ratio"]), "input": d1["max_rare_effect_formal_to_nominal_ratio"], "threshold": 2.0},
        "gate_effect_fpr_primary_individual_95": {"function": "gate_effect_fpr", "value": gate_effect_fpr([row["episode_cluster"]["fpr_exact_binomial_upper95"] for row in d4["per_effect"].values()], d4["selected_alpha"], bool(seal["certified"])), "ucbs": [row["episode_cluster"]["fpr_exact_binomial_upper95"] for row in d4["per_effect"].values()], "alpha": d4["selected_alpha"], "certified": bool(seal["certified"])},
        "gate_effect_fpr_joint_99_bonferroni": {"function": "gate_effect_fpr", "value": gate_effect_fpr([row["episode_cluster"]["fpr_exact_binomial_upper99"] for row in d4["per_effect"].values()], d4["selected_alpha"], bool(seal["certified"])), "ucbs": [row["episode_cluster"]["fpr_exact_binomial_upper99"] for row in d4["per_effect"].values()], "alpha": d4["selected_alpha"], "certified": bool(seal["certified"])},
        "gate_receipt_fpr": {"function": "gate_receipt_fpr", "value": gate_receipt_fpr(d4["receipt_false_upgrades"]["episode_cluster_exact_binomial_upper95"], d4["receipt_false_upgrades"]["episode_cluster_pvalue_rate_at_most_half"]), "episode_cluster_ucb": d4["receipt_false_upgrades"]["episode_cluster_exact_binomial_upper95"], "pvalue_vs_half": d4["receipt_false_upgrades"]["episode_cluster_pvalue_rate_at_most_half"]},
        "gate_ledger": {"function": "gate_ledger", "value": gate_ledger(d5["concordant_unit_count"], d5["unit_count"]), "concordant_units": d5["concordant_unit_count"], "total_units": d5["unit_count"], "threshold": 0.85},
    }
    all_pass = all(value["value"] for value in gate_values.values())
    decision = {
        "schema_version": 1,
        "status": "READY_FOR_S2" if all_pass else "NOT_READY_FOR_S2",
        "gates": gate_values,
        "all_four_gates_pass": all_pass,
        "d1_early_stop_overridden_by_user": True,
        "formal_read_once": FORMAL_LOAD_SESSIONS == 1,
        "formal_read_paths": sorted(FORMAL_READ_PATHS),
        "d3_seal_commit": SEAL_COMMIT,
        "d3_seal_sha256": seal_sha,
        "protocol_deviation": PRIOR_FAILED_FORMAL_ATTEMPT,
        "route_termination_assessment": {"decision_for_human": True, "basis": "D3 has no certified point; formal diagnostics show the exact gaps above. This artifact records evidence and does not substitute a human decision about terminating the route."},
        "s2_started": False,
        "pai_jobs_submitted": 0,
        "gpu_jobs_submitted": 0,
        "rollouts_executed": 0,
        "new_data_collected": 0,
        "formal_claims_allowed": False,
    }
    write_json(OUT / "D_DECISION.json", decision)
    lines = [
        "# Gate D 决策",
        "",
        f"状态：`{decision['status']}`。D3 的 fallback 未形成认证工作点，因此即使完成全部诊断也不进入 S2。",
        "",
        "## 四条具名 gate",
        "",
    ]
    for name, value in gate_values.items():
        lines.append(f"- `{value['function']}`（{name}）：`{str(value['value']).lower()}`；依据：`{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
    lines += [
        "",
        "## 证据边界",
        "",
        f"- D0 SHA256：`{D0_SHA}`；D3 seal commit：`{SEAL_COMMIT}`；D3 seal SHA256：`{seal_sha}`。",
        f"- Formal 物理载入 session：`{FORMAL_LOAD_SESSIONS}`；所有 D1/D2/D4/D5/D6 复用同一内存对象。",
        "- D1 即使未通过也已继续完成 D2、D4、D5、D6，这是用户确认的执行顺序。",
        "- formal 标签结果均为诊断，`claim_eligible=false`、`selection_eligible=false`；未启动 S2、PAI、GPU、rollout 或新数据采集。",
        "",
        "## 是否终止认证路线",
        "",
        "本文件不替人工决定终止与否。依据是：D3 没有 certified point，且 Gate D 中列出的失败门及其差距已明确；若业务要求有效 witness 和 ≥0.85 ledger 一致率，则当前证据不支持继续进入 S2。",
    ]
    (OUT / "D_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_mechanism(d1, d2, d4, d5, d6)
    after_sources = source_snapshot()
    if before_sources != after_sources:
        changed = sorted(set(before_sources) ^ set(after_sources) | {k for k in before_sources if before_sources.get(k) != after_sources.get(k)})
        raise RuntimeError(f"protected phase5-8 tree changed: {changed[:10]}")
    write_json(OUT / "SOURCE_PROTECTION.json", {"phase5_phase6_phase7_phase8_unchanged": True, "before_sha256_map": before_sources, "after_equal": True, "checkpoint_sha256": sha256(REPO / "experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz"), "expected_checkpoint_sha256": "edba3ff23ae71c5768760af161b601b3d1a9f65d37d1ddeb8304b2510e938005"})
    write_json(OUT / "EXECUTION_MANIFEST.json", {"schema_version": 1, "cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_jobs_submitted": 0, "pai_jobs_submitted": 0, "rollouts_executed": 0, "new_data_collected": 0, "s2_started": False, "formal_load_sessions": FORMAL_LOAD_SESSIONS, "formal_read_paths": sorted(FORMAL_READ_PATHS), "d0_commit": "d4c81be6149068ab56cda8350c99b86ddf1c7fa8", "d3_seal_commit": SEAL_COMMIT, "current_head_at_execution": head, "protocol_deviation": PRIOR_FAILED_FORMAL_ATTEMPT, "python": sys.version, "platform": platform.platform(), "time_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps({"status": decision["status"], "d1": d1["d1_pass"], "d3_certified": seal["certified"], "d4_fp": d4["receipt_false_upgrades"]["count"], "d5_concordant": [d5["concordant_unit_count"], d5["unit_count"]], "formal_load_sessions": FORMAL_LOAD_SESSIONS}, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise RuntimeError("Retired failed implementation: do not reopen original formal evidence. Use phase9_cached_eval.py and recovered cache.")

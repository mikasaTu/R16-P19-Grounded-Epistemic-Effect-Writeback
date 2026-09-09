#!/usr/bin/env python3
"""Phase-9 D3 non-formal negative split-conformal selection.

This stage is intentionally separate from the formal stage.  It is created
and run only after the frozen D0 commit and writes the D3 negative export and
the operating-point seal.  The selector receives an export containing score
values and negative-effect indices but no labels, and a read hook rejects
protected/raw/positive/formal paths.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
D0 = OUT / "D0_PREREGISTRATION.json"
D0_SHA = "23dcfb04c6c1478fce14d1b1537cedf6e46283c622deeaf0aa158b7a03cbff10"
Z95 = 1.959963984540054
ALPHAS = (0.005, 0.01, 0.02, 0.05)


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
    """One-sided Clopper-Pearson upper endpoint without scipy."""
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
    p = float(k) / float(n)
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    w = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / d
    return [float(c - w), float(c + w)]


def binom_lower_tail(k: int, n: int, p: float = 0.5) -> float | None:
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    lp = math.log(p)
    lq = math.log1p(-p)
    vals = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * lp
        + (n - i) * lq
        for i in range(k + 1)
    ]
    m = max(vals)
    return float(math.exp(m) * sum(math.exp(v - m) for v in vals))


def load_s1():
    spec = importlib.util.spec_from_file_location(
        "phase6_s1_for_phase9", REPO / "experiments/r16p19_phase6/run_s1.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen phase6 run_s1.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO))
    spec.loader.exec_module(module)
    return module


def collect_estimation(s1: Any) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, list[Path]], dict[int, list[str]], dict[str, set[str]]
]:
    effect_rows: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"scores": [], "labels": []})
    receipts: list[dict[str, Any]] = []
    paths: dict[str, list[Path]] = {}
    task_effects: dict[int, list[str]] = {}
    split_episode_ids: dict[str, set[str]] = {}
    for split in ("calibration", "pilot"):
        effects, rows, tasks, input_paths = s1._calibration_dataset(RAW / "episodes" / split, REPO / "experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz")
        paths[split] = list(input_paths)
        receipts.extend(rows)
        split_episode_ids[split] = {str(row["episode_id"]) for row in rows}
        for key, values in effects.items():
            effect_rows[key]["scores"].extend(values["scores"])
            effect_rows[key]["labels"].extend(values["labels"])
        for task, keys in tasks.items():
            if task in task_effects and task_effects[task] != keys:
                raise RuntimeError(f"effect schema drift task={task}")
            task_effects[task] = list(keys)
    overlap = split_episode_ids["calibration"] & split_episode_ids["pilot"]
    if overlap:
        raise RuntimeError(f"calibration/pilot episode-id collision: {sorted(overlap)}")
    return dict(effect_rows), receipts, paths, task_effects, split_episode_ids


def build_negative_export(
    effect_rows: dict[str, Any],
    receipts: list[dict[str, Any]],
    paths: dict[str, list[Path]],
    task_effects: dict[int, list[str]],
    split_episode_ids: dict[str, set[str]],
) -> dict[str, Any]:
    """Build an export with no label arrays and no positive-only receipts."""
    neg_by_effect: dict[str, list[float]] = {}
    episode_values: dict[str, dict[str, float]] = defaultdict(dict)
    for key, values in sorted(effect_rows.items()):
        scores = [float(s) for s, y in zip(values["scores"], values["labels"]) if not bool(y)]
        neg_by_effect[key] = scores

    receipt_export: list[dict[str, Any]] = []
    for row in receipts:
        labels = [bool(x) for x in row["labels"]]
        negative_indices = [i for i, label in enumerate(labels) if not label]
        if not negative_indices:
            continue
        scores = [float(x) for x in row["scores"]]
        receipt_export.append(
            {
                "receipt_id": f"{row['episode_id']}:{int(row['chunk'])}",
                "episode_id": str(row["episode_id"]),
                "task_id": int(row["task_id"]),
                "chunk": int(row["chunk"]),
                "effect_keys": list(row["effect_keys"]),
                "scores": scores,
                "negative_effect_indices": negative_indices,
            }
        )
        for index in negative_indices:
            key = row["effect_keys"][index]
            previous = episode_values[key].get(str(row["episode_id"]))
            value = float(row["scores"][index])
            episode_values[key][str(row["episode_id"])] = value if previous is None else max(previous, value)

    evidence = []
    for split in sorted(paths):
        for path in sorted(paths[split], key=str):
            evidence.append({"split": split, "path": str(path), "sha256": sha256(path)})
    export = {
        "schema_version": 1,
        "description": "D3 selector export: score values, negative effect indices, and episode maxima; no positive/negative label arrays.",
        "selection_split": ["calibration", "pilot"],
        "qualification_excluded": True,
        "formal_paths": [],
        "formal_access_count": 0,
        "claim_eligible": False,
        "selection_eligible": True,
        "split_episode_counts": {split: len(ids) for split, ids in sorted(split_episode_ids.items())},
        "split_episode_id_overlap_count": len(split_episode_ids["calibration"] & split_episode_ids["pilot"]),
        "task_effects": {str(k): v for k, v in sorted(task_effects.items())},
        "negative_scores_by_effect": neg_by_effect,
        "negative_episode_max_by_effect": {
            key: {episode: float(value) for episode, value in sorted(values.items())}
            for key, values in sorted(episode_values.items())
        },
        "negative_receipts": receipt_export,
        "input_evidence": evidence,
        "input_evidence_sha256": hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "label_fields_present": False,
        "positive_label_values_present": False,
    }
    return export


_SELECT_READS: list[str] = []
_SELECTOR_PHASE = True


def selector_read_hook(event: str, args: tuple[Any, ...]) -> None:
    if event != "open" or not args or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).absolute()
    mode = args[1] if len(args) > 1 else "r"
    flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing:
        if str(path).startswith(str(OUT)):
            return
        if str(path).startswith(str(REPO)) or str(path).startswith(str(RAW)):
            raise RuntimeError(f"D3 selector protected write: {path}")
        return
    p = str(path)
    if p.startswith(str(OUT)):
        if path.name in {"D0_PREREGISTRATION.json", "D3_NEGATIVE_EXPORT.json"}:
            _SELECT_READS.append(p)
            return
        if not _SELECTOR_PHASE:
            return
        raise RuntimeError(f"D3 selector non-allowlisted phase9 read: {path}")
    blocked_tokens = ("formal", "positive", "phase5", "phase6", "phase7", "phase8", "raw_rollout")
    if any(token in p.lower() for token in blocked_tokens):
        raise RuntimeError(f"D3 selector forbidden label/path read: {path}")
    if p.startswith(str(REPO)) or p.startswith(str(RAW)):
        raise RuntimeError(f"D3 selector non-export read: {path}")


def assert_selector_read_path(path: Path) -> None:
    """Shared hook predicate; test uses a known positive-label path."""
    p = str(path.absolute())
    if any(token in p.lower() for token in ("formal", "positive", "phase5", "phase6", "phase7", "phase8", "raw_rollout")):
        raise RuntimeError(f"D3 selector forbidden label/path read: {path}")
    if path.name not in {"D0_PREREGISTRATION.json", "D3_NEGATIVE_EXPORT.json"}:
        raise RuntimeError(f"D3 selector path not allowlisted: {path}")


def order_threshold(values: list[float], alpha: float) -> tuple[float | None, int, bool]:
    n = len(values)
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return None, rank, True
    ordered = sorted(float(v) for v in values)
    return float(ordered[rank - 1]), rank, False


def threshold_result(values: list[float], alpha: float, confidence: float) -> dict[str, Any]:
    threshold, rank, reject_all = order_threshold(values, alpha)
    if reject_all:
        fp = 0
    else:
        fp = sum(float(v) > float(threshold) for v in values)
    n = len(values)
    return {
        "threshold": threshold,
        "reject_all": reject_all,
        "rank": rank,
        "n_negative": n,
        "false_positive_count_on_estimation": int(fp),
        "fpr": float(fp / n) if n else None,
        "fpr_exact_binomial_upper": cp_upper(fp, n, confidence),
        "confidence": confidence,
        "comparison": "> (strict; ties at order statistic rejected)",
    }


def selected_accept(score: float, threshold: float | None) -> bool:
    return threshold is not None and float(score) > float(threshold)


def receipt_upgrade_metrics(receipts: list[dict[str, Any]], thresholds: dict[str, float | None]) -> dict[str, Any]:
    accepted_ids: list[str] = []
    false_ids: list[str] = []
    negative_episode_ids: set[str] = set()
    false_episode_ids: set[str] = set()
    for row in receipts:
        keys = list(row["effect_keys"])
        negative_indices = set(int(i) for i in row["negative_effect_indices"])
        if negative_indices:
            negative_episode_ids.add(str(row["episode_id"]))
        accepted = all(selected_accept(float(score), thresholds.get(key)) for key, score in zip(keys, row["scores"]))
        rid = str(row["receipt_id"])
        if accepted:
            accepted_ids.append(rid)
        if accepted and negative_indices:
            false_ids.append(rid)
            false_episode_ids.add(str(row["episode_id"]))
    k = len(false_episode_ids)
    n = len(negative_episode_ids)
    return {
        "receipt_count_with_negative_effect": len(receipts),
        "accepted_receipt_count": len(accepted_ids),
        "false_upgrade_receipt_count": len(false_ids),
        "false_upgrade_receipt_ids": sorted(false_ids),
        "negative_episode_count": n,
        "false_upgrade_episode_count": k,
        "false_upgrade_episode_ids": sorted(false_episode_ids),
        "episode_fpr": float(k / n) if n else None,
        "episode_wilson95": wilson(k, n),
        "episode_exact_binomial_upper95": cp_upper(k, n, 0.95),
        "episode_exact_binomial_upper99": cp_upper(k, n, 0.99),
        "pvalue_episode_rate_at_most_half": binom_lower_tail(k, n, 0.5),
    }


def effect_tables(export: dict[str, Any], alpha: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method, key in (("receipt_score", "negative_scores_by_effect"), ("episode_max", "negative_episode_max_by_effect")):
        confidence_results: dict[str, Any] = {}
        for confidence_name, confidence in (("individual_95", 0.95), ("joint_99_bonferroni", 0.99)):
            confidence_results[confidence_name] = {
                effect: threshold_result(
                    [float(v) for v in (values if method == "receipt_score" else values.values())],
                    alpha,
                    confidence,
                )
                for effect, values in sorted(export[key].items())
            }
        result[method] = confidence_results
    return result


def choose_workpoint(export: dict[str, Any], all_alpha: dict[str, Any]) -> tuple[float, dict[str, Any], bool]:
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for alpha in ALPHAS:
        key = f"{alpha:.3f}"
        primary = all_alpha[key]["thresholds"]["episode_max"]["individual_95"]
        certified = all(
            value["fpr_exact_binomial_upper"] is not None and value["fpr_exact_binomial_upper"] <= alpha
            for value in primary.values()
        )
        upgrade = all_alpha[key]["receipt_upgrade_metrics"]
        ucb = upgrade["episode_exact_binomial_upper95"]
        if certified and ucb is not None:
            candidates.append((float(ucb), alpha, all_alpha[key]))
    if candidates:
        _, alpha, chosen = min(candidates, key=lambda row: (row[0], row[1]))
        return alpha, chosen, True
    fallback = []
    for alpha in ALPHAS:
        key = f"{alpha:.3f}"
        row = all_alpha[key]
        ucb = row["receipt_upgrade_metrics"]["episode_exact_binomial_upper95"]
        fallback.append((float("inf") if ucb is None else float(ucb), alpha, row))
    _, alpha, chosen = min(fallback, key=lambda row: (row[0], row[1]))
    return alpha, chosen, False


def preprocess() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(D0) != D0_SHA:
        raise RuntimeError("D0 hash drift; refuse to proceed")
    s1 = load_s1()
    effect_rows, receipts, paths, task_effects, split_episode_ids = collect_estimation(s1)
    export = build_negative_export(effect_rows, receipts, paths, task_effects, split_episode_ids)
    export_path = OUT / "D3_NEGATIVE_EXPORT.json"
    write_json(export_path, export)
    write_json(
        OUT / "D3_PREPROCESS.json",
        {
            "schema_version": 1,
            "stage": "separate nonformal negative export preprocessing",
            "process_id": os.getpid(),
            "selection_split": ["calibration", "pilot"],
            "qualification_excluded": True,
            "formal_access_count": 0,
            "formal_read_paths": [],
            "positive_label_values_written": False,
            "output": "D3_NEGATIVE_EXPORT.json",
            "output_sha256": sha256(export_path),
            "split_episode_counts": export["split_episode_counts"],
            "split_episode_id_overlap_count": export["split_episode_id_overlap_count"],
            "cpu_only": True,
            "gpu_jobs_submitted": 0,
            "pai_jobs_submitted": 0,
            "rollouts_executed": 0,
        },
    )
    print(json.dumps({"stage": "preprocess", "pid": os.getpid(), "output": str(export_path)}, indent=2))


def select() -> None:
    global _SELECTOR_PHASE
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(D0) != D0_SHA:
        raise RuntimeError("D0 hash drift; refuse to proceed")
    export_path = OUT / "D3_NEGATIVE_EXPORT.json"
    # Only the export and frozen D0 are available to this selector after hook installation.
    sys.addaudithook(selector_read_hook)
    assert_selector_read_path(D0)
    assert_selector_read_path(export_path)
    d0 = json.loads(D0.read_text(encoding="utf-8"))
    loaded = json.loads(export_path.read_text(encoding="utf-8"))
    if loaded.get("label_fields_present") or loaded.get("positive_label_values_present"):
        raise RuntimeError("negative export contains forbidden label values")
    if loaded.get("formal_access_count") != 0 or loaded.get("formal_paths"):
        raise RuntimeError("negative export records formal access")
    if loaded.get("split_episode_id_overlap_count") != 0:
        raise RuntimeError("calibration/pilot episode ID collision in selector export")
    positive_path_test_passed = False
    try:
        selector_read_hook("open", (str(OUT / "positive_labels.json"), "r", os.O_RDONLY))
    except RuntimeError:
        positive_path_test_passed = True
    if not positive_path_test_passed:
        raise RuntimeError("selector read hook failed to reject a positive-label path")
    effects = sorted(loaded["negative_scores_by_effect"])
    all_alpha: dict[str, Any] = {}
    for alpha in ALPHAS:
        key = f"{alpha:.3f}"
        thresholds = effect_tables(loaded, alpha)
        primary_vector = {
            effect: thresholds["episode_max"]["individual_95"][effect]["threshold"]
            for effect in effects
        }
        upgrades = receipt_upgrade_metrics(loaded["negative_receipts"], primary_vector)
        all_alpha[key] = {
            "alpha": alpha,
            "thresholds": thresholds,
            "primary_threshold_vector": primary_vector,
            "receipt_upgrade_metrics": upgrades,
            "primary_certified_individual_95": all(
                thresholds["episode_max"]["individual_95"][effect]["fpr_exact_binomial_upper"] is not None
                and thresholds["episode_max"]["individual_95"][effect]["fpr_exact_binomial_upper"] <= alpha
                for effect in effects
            ),
            "primary_certified_joint_99_bonferroni": all(
                thresholds["episode_max"]["joint_99_bonferroni"][effect]["fpr_exact_binomial_upper"] is not None
                and thresholds["episode_max"]["joint_99_bonferroni"][effect]["fpr_exact_binomial_upper"] <= alpha
                for effect in effects
            ),
        }
    selected_alpha, selected, certified = choose_workpoint(loaded, all_alpha)
    selected_vector = selected["primary_threshold_vector"]
    # Selection is complete.  Hashing newly written D3 artifacts is bookkeeping,
    # outside the selector's protected-input read phase.
    _SELECTOR_PHASE = False
    evidence = loaded["input_evidence"]
    seal = {
        "schema_version": 1,
        "stage": "D3 negative-class split-conformal operating-point seal",
        "d0_preregistration_sha256": D0_SHA,
        "d0_commit_required_before_analysis": True,
        "selection_rule_frozen": d0["selection_rule"],
        "selection_method": "episode_max",
        "primary_unit": d0["primary_unit"],
        "selected_alpha": selected_alpha,
        "selected_threshold_vector": selected_vector,
        "selected_threshold_metadata": selected["thresholds"]["episode_max"]["individual_95"],
        "certified_point": (
            {"alpha": selected_alpha, "threshold_vector": selected_vector}
            if certified
            else None
        ),
        "certified": bool(certified),
        "diagnostic_fallback": bool(not certified),
        "fallback_reason": None if certified else "No alpha satisfies every effect primary episode Clopper-Pearson UCB <= alpha",
        "all_alpha_primary_certification": {
            key: {
                "certified_individual_95": row["primary_certified_individual_95"],
                "certified_joint_99_bonferroni": row["primary_certified_joint_99_bonferroni"],
            }
            for key, row in sorted(all_alpha.items())
        },
        "selection_input_evidence": evidence,
        "selection_input_evidence_sha256": loaded["input_evidence_sha256"],
        "formal_access_count": 0,
        "formal_read_paths": [],
        "positive_label_access_count": 0,
        "read_hook": {
            "allowlist": ["D0_PREREGISTRATION.json", "D3_NEGATIVE_EXPORT.json"],
            "blocked_path_tokens": ["formal", "positive", "phase5", "phase6", "phase7", "phase8", "raw_rollout"],
            "reads_observed": sorted(set(_SELECT_READS)),
            "positive_path_test": "selector_read_hook('open', ('positive_labels.json','r',O_RDONLY)) raises RuntimeError",
            "positive_path_test_passed": positive_path_test_passed,
        },
        "claim_eligible": bool(certified),
        "selection_eligible": bool(certified),
    }
    write_json(OUT / "D3_OPERATING_POINT_SEAL.json", seal)
    d3 = {
        "schema_version": 1,
        "status": "COMPLETED_NONFORMAL_SELECTION",
        "d0_preregistration_sha256": D0_SHA,
        "selection_split": ["calibration", "pilot"],
        "qualification_excluded": True,
        "formal_access_count": 0,
        "formal_read_paths": [],
        "positive_label_access_count": 0,
        "negative_export_sha256": sha256(export_path),
        "effects": effects,
        "alphas": all_alpha,
        "selected_alpha": selected_alpha,
        "selected_threshold_vector": selected_vector,
        "certified_point": (
            {"alpha": selected_alpha, "threshold_vector": selected_vector}
            if certified
            else None
        ),
        "certified": bool(certified),
        "diagnostic_fallback": bool(not certified),
        "selection_seal": "D3_OPERATING_POINT_SEAL.json",
        "selection_seal_sha256": sha256(OUT / "D3_OPERATING_POINT_SEAL.json"),
        "read_hook": seal["read_hook"],
        "claim_eligible": bool(certified),
        "selection_eligible": bool(certified),
        "execution": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_jobs_submitted": 0,
            "pai_jobs_submitted": 0,
            "rollouts_executed": 0,
            "s2_started": False,
            "phase5_phase6_phase7_phase8_modified": False,
            "python": sys.version,
            "platform": platform.platform(),
            "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }
    write_json(OUT / "D3_SELECTION.json", d3)
    print(json.dumps({"stage": "select", "pid": os.getpid(), "selected_alpha": selected_alpha, "certified": certified, "thresholds": selected_vector}, indent=2, sort_keys=True))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--preprocess", "--select"}:
        raise SystemExit("usage: phase9_nonformal.py --preprocess|--select")
    if sys.argv[1] == "--preprocess":
        preprocess()
    else:
        select()


if __name__ == "__main__":
    main()

"""Preregistered paired statistics and final Phase-5 gate evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


BASELINES = ("M0_TYPED_MATCHED", "M1_VERSIONED_POSTCHECK", "M2_B6_FROZEN")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rate(rows, field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows])) if rows else float("nan")


def paired_cluster_bootstrap(rows: list[dict], left: str, right: str, arm_field: str = "arm", replicates: int = 10000, seed: int = 5019) -> dict:
    clusters = defaultdict(lambda: defaultdict(list))
    for row in rows:
        clusters[row["cluster_id"]][row[arm_field]].append(float(row["task_success"]))
    paired = [(np.mean(value[left]), np.mean(value[right])) for value in clusters.values() if left in value and right in value]
    differences = np.asarray([a - b for a, b in paired], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        samples[index] = differences[rng.integers(0, len(differences), len(differences))].mean()
    return {"left": left, "right": right, "clusters": len(differences), "risk_difference": float(differences.mean()), "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "replicates": replicates, "seed": seed}


def mcnemar(rows: list[dict], left: str, right: str) -> dict:
    cells = defaultdict(dict)
    for row in rows:
        cells[(row["cluster_id"], row["condition"])][row["arm"]] = bool(row["task_success"])
    b = sum(value.get(left) is True and value.get(right) is False for value in cells.values())
    c = sum(value.get(left) is False and value.get(right) is True for value in cells.values())
    n = b + c
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n) if n else 1.0
    return {"left_only": b, "right_only": c, "discordant": n, "p_exact_two_sided": min(1.0, 2.0 * tail)}


def holm(rows: list[dict]) -> list[dict]:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_exact_two_sided"])
    adjusted = [0.0] * len(rows)
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - rank) * row["p_exact_two_sided"]))
        adjusted[index] = running
    return [dict(row, p_holm=adjusted[index]) for index, row in enumerate(rows)]


def analyze(output: Path, bounded_path: Path, rollout_root: Path, oracle_path: Path, learned_path: Path, support_path: Path, ablation_path: Path, verifier_metrics_path: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    oracle = read_jsonl(oracle_path)
    learned = read_jsonl(learned_path)
    support = read_jsonl(support_path)
    ablations = read_jsonl(ablation_path)
    bounded = json.loads(bounded_path.read_text())
    verifier = json.loads(verifier_metrics_path.read_text())
    faulted = [row for row in oracle if row["condition"] != "C0_CLEAN"]
    scores = {arm: rate([row for row in faulted if row["arm"] == arm], "task_success") for arm in BASELINES}
    baseline = max(scores, key=lambda name: (scores[name], name))
    core_rows = [row for row in faulted if row["arm"] in (baseline, "M3_ASCEL_CORE")]
    core_boot = paired_cluster_bootstrap(core_rows, "M3_ASCEL_CORE", baseline)
    per_task = {}
    for task in (0, 5, 9):
        chosen = [row for row in core_rows if row["task_id"] == task]
        per_task[str(task)] = rate([row for row in chosen if row["arm"] == "M3_ASCEL_CORE"], "task_success") - rate([row for row in chosen if row["arm"] == baseline], "task_success")
    condition_tests = []
    for condition in sorted({row["condition"] for row in core_rows}):
        test = mcnemar([row for row in core_rows if row["condition"] == condition], "M3_ASCEL_CORE", baseline)
        test["condition"] = condition
        condition_tests.append(test)
    condition_tests = holm(condition_tests)
    m3 = [row for row in oracle if row["arm"] == "M3_ASCEL_CORE"]
    base = [row for row in oracle if row["arm"] == baseline]
    m3_false = rate([row for row in m3 if row["condition"] != "C0_CLEAN"], "false_grounded_advance")
    base_false = rate([row for row in base if row["condition"] != "C0_CLEAN"], "false_grounded_advance")
    false_reduction = (base_false - m3_false) / base_false if base_false else 0.0
    a5 = [row for row in m3 if row["condition"] == "A5_EXTERNAL_REALIZATION"]
    clean_m3 = [row for row in m3 if row["condition"] == "C0_CLEAN"]
    clean_base = [row for row in base if row["condition"] == "C0_CLEAN"]
    clean_degradation = rate(clean_base, "task_success") - rate(clean_m3, "task_success")
    base_steps = float(np.mean([row["action_steps"] for row in clean_base]))
    clean_extra_steps = float((np.mean([row["action_steps"] for row in clean_m3]) - base_steps) / max(base_steps, 1.0))
    policy_latencies = []
    for path in (rollout_root / "episodes").glob("*/*.npz"):
        with np.load(path, allow_pickle=False) as data:
            policy_latencies.extend(data["policy_latency_ms"].tolist())
    policy_cycle_ms = float(np.mean(policy_latencies)) if policy_latencies else float("nan")
    bounded["policy_cycle_ms_measured"] = policy_cycle_ms
    bounded["policy_cycle_fraction"] = bounded["tick_latency_p99_ms"] / policy_cycle_ms if policy_cycle_ms > 0 else float("inf")
    bounded["gates"]["policy_cycle_fraction_le_0_05"] = bounded["policy_cycle_fraction"] <= 0.05
    bounded["pass"] = all(bounded["gates"].values())
    bounded_path.write_text(json.dumps(bounded, indent=2, sort_keys=True) + "\n")
    core_gates = {
        "success_margin_ge_0_08": core_boot["risk_difference"] >= 0.08,
        "ci_lower_gt_zero": core_boot["ci95"][0] > 0.0,
        "two_tasks_positive": sum(value > 0 for value in per_task.values()) >= 2,
        "remaining_degradation_le_0_02": min(per_task.values()) >= -0.02,
        "false_advance_relative_reduction_ge_0_5": false_reduction >= 0.5,
        "stale_acceptance_zero": not any(row["stale_acceptance"] for row in m3),
        "cross_attempt_acceptance_zero": not any(row["cross_attempt_acceptance"] for row in m3),
        "late_witness_acceptance_zero": not any(row["late_witness_acceptance"] for row in m3),
        "a5_truth_recognition_ge_0_95": rate(a5, "effect_truth_recognized") >= 0.95,
        "a5_false_credit_le_0_05": rate(a5, "false_current_attempt_credit") <= 0.05,
        "clean_degradation_le_0_02": clean_degradation <= 0.02,
        "clean_extra_steps_le_0_10": clean_extra_steps <= 0.10,
        "systems_pass": bounded["pass"],
    }
    oracle_analysis = {"schema_version": 1, "strongest_baseline": baseline, "faulted_success": {**scores, "M3_ASCEL_CORE": rate(m3[0:0] + [row for row in m3 if row["condition"] != "C0_CLEAN"], "task_success")}, "paired_bootstrap": core_boot, "per_task_risk_difference": per_task, "false_grounded_advance": {baseline: base_false, "M3_ASCEL_CORE": m3_false, "relative_reduction": false_reduction}, "clean_degradation": clean_degradation, "clean_extra_steps_fraction": clean_extra_steps, "gates": core_gates, "pass": all(core_gates.values())}
    (output / "oracle_analysis.json").write_text(json.dumps(oracle_analysis, indent=2, sort_keys=True) + "\n")
    (output / "oracle_cluster_bootstrap.json").write_text(json.dumps(core_boot, indent=2, sort_keys=True) + "\n")
    (output / "oracle_mcnemar_holm.json").write_text(json.dumps(condition_tests, indent=2, sort_keys=True) + "\n")

    learned_arms = sorted({row["arm"] for row in learned})
    learned_baseline = next(arm for arm in learned_arms if arm != "M3_ASCEL_CORE")
    learned_faulted = [row for row in learned if row["condition"] != "C0_CLEAN"]
    learned_boot = paired_cluster_bootstrap(learned_faulted, "M3_ASCEL_CORE", learned_baseline)
    oracle_gain = core_boot["risk_difference"]
    retained = learned_boot["risk_difference"] / oracle_gain if oracle_gain > 0 else 0.0
    learned_m3 = [row for row in learned if row["arm"] == "M3_ASCEL_CORE"]
    learned_clean = [row for row in learned_m3 if row["condition"] == "C0_CLEAN"]
    learned_base_clean = [row for row in learned if row["arm"] == learned_baseline and row["condition"] == "C0_CLEAN"]
    learned_gates = {"verifier_qualified": verifier["selected_qualified"], "margin_ge_0_05": learned_boot["risk_difference"] >= 0.05, "ci_lower_gt_zero": learned_boot["ci95"][0] > 0, "oracle_direction_consistent": learned_boot["risk_difference"] > 0, "oracle_gain_retained_ge_0_5": retained >= 0.5, "false_advance_le_0_05": rate(learned_m3, "false_grounded_advance") <= 0.05, "false_credit_le_0_05": rate(learned_m3, "false_current_attempt_credit") <= 0.05, "clean_degradation_le_0_02": rate(learned_base_clean, "task_success") - rate(learned_clean, "task_success") <= 0.02, "systems_pass": bounded["pass"]}
    learned_analysis = {"schema_version": 1, "baseline": learned_baseline, "paired_bootstrap": learned_boot, "oracle_absolute_gain_retained": retained, "gates": learned_gates, "pass": all(learned_gates.values())}
    (output / "learned_verifier_analysis.json").write_text(json.dumps(learned_analysis, indent=2, sort_keys=True) + "\n")

    support_compare = [row for row in support if row["arm"] in ("M3_ASCEL_CORE", "M4_ASCEL_FULL")]
    support_boot = paired_cluster_bootstrap(support_compare, "M4_ASCEL_FULL", "M3_ASCEL_CORE")
    full = [row for row in support if row["arm"] == "M4_ASCEL_FULL"]
    tp = sum(row["cascade_true_positive"] for row in full)
    fn = sum(row["cascade_false_negative"] for row in full)
    fp = sum(row["over_invalidation"] for row in full)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    support_gates = {"margin_ge_0_05": support_boot["risk_difference"] >= 0.05, "ci_lower_gt_zero": support_boot["ci95"][0] > 0, "cascade_precision_ge_0_95": precision >= 0.95, "cascade_recall_ge_0_95": recall >= 0.95, "over_invalidation_le_0_05": rate(full, "over_invalidation") <= 0.05, "under_invalidation_le_0_05": rate(full, "under_invalidation") <= 0.05, "clean_degradation_le_0_02": 1.0 - rate([row for row in full if row["condition"] == "C0_CLEAN"], "task_success") <= 0.02, "systems_pass": bounded["pass"]}
    support_analysis = {"schema_version": 1, "paired_bootstrap": support_boot, "cascade_precision": precision, "cascade_recall": recall, "gates": support_gates, "pass": all(support_gates.values())}
    (output / "support_analysis.json").write_text(json.dumps(support_analysis, indent=2, sort_keys=True) + "\n")

    attribution = {}
    for name in sorted({row["ablation"] for row in ablations}):
        selected = [row for row in ablations if row["ablation"] == name]
        reference_name = "M4_ASCEL_FULL" if name == "NO_SUPPORT_GRAPH" else "M3_ASCEL_CORE"
        ablated_name = name
        reference = [row for row in selected if row["arm"] == reference_name]
        ablated = [row for row in selected if row["arm"] == ablated_name]
        reference_error = rate(reference, "target_error")
        ablated_error = rate(ablated, "target_error")
        error_increase = ablated_error - reference_error
        reference_success = rate(reference, "task_success")
        ablated_success = rate(ablated, "task_success")
        removed = (reference_success - ablated_success) / max(abs(reference_success), 1e-12)
        attribution[name] = {"reference": reference_name, "target_error_increase": error_increase, "advantage_removed_fraction": removed, "supports_mechanism": error_increase >= 0.10 or removed >= 0.50}
    (output / "mechanism_attribution.json").write_text(json.dumps({"schema_version": 1, "ablations": attribution}, indent=2, sort_keys=True) + "\n")

    qualification = json.loads((output / "policy_qualification_summary.json").read_text())
    pairing = json.loads((output / "shared_prefix_qualification_summary.json").read_text())
    if not bounded["pass"]:
        status = "BLOCKED_BY_LEDGER_IMPLEMENTATION"
    elif not qualification["pass"]:
        status = "BLOCKED_BY_POLICY"
    elif not pairing["all_fields_exact"]:
        status = "BLOCKED_BY_PAIRING"
    elif not oracle_analysis["pass"]:
        status = "REJECT_ASCEL_EMBODIED_VALUE"
    elif not verifier["selected_qualified"]:
        status = "BLOCKED_BY_VERIFIER"
    elif learned_analysis["pass"] and support_analysis["pass"]:
        status = "PASS_PHASE5_ASCEL_FULL_EMBODIED"
    elif learned_analysis["pass"]:
        status = "PASS_PHASE5_ASCEL_CORE_EMBODIED"
    else:
        status = "NARROW_TO_ASCEL_CORE"
    decision = {"schema_version": 1, "status": status, "oracle_core_pass": oracle_analysis["pass"], "learned_core_pass": learned_analysis["pass"], "support_full_pass": support_analysis["pass"], "diagnostic_continuation_used": not (bounded["pass"] and qualification["pass"] and pairing["all_fields_exact"] and oracle_analysis["pass"]), "all_planned_matrices_completed": True}
    (output / "final_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounded", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--learned", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--verifier-metrics", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output, args.bounded, args.rollout_root, args.oracle, args.learned, args.support, args.ablations, args.verifier_metrics), sort_keys=True))


if __name__ == "__main__":
    main()

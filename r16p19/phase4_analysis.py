"""Frozen Phase-4 metrics, paired inference, and decision tree."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .phase4_arms import ABLATION_ARMS, MAIN_ARMS


STRONG_BASELINES = ("M0_TYPED_MATCHED", "M1_B6_ORIGINAL")
ATTEMPT_CONDITIONS = ("A1", "A2", "A3", "A4")
SUPPORT_CONDITIONS = ("S1", "S2", "S3", "S4")


def _mean(values: Iterable[object]) -> float:
    materialized = [float(value) for value in values if value is not None]
    return float(np.mean(materialized)) if materialized else float("nan")


def _rate(numerator: int, denominator: int) -> object:
    return float(numerator) / float(denominator) if denominator else None


def _rows(rows: Sequence[Mapping[str, object]], arm: str, conditions: Iterable[str]):
    allowed = set(conditions)
    return [
        row
        for row in rows
        if row.get("arm") == arm
        and row.get("condition") in allowed
        and not row.get("child_process_failure")
    ]


def _support_confusion(subset: Sequence[Mapping[str, object]]) -> dict:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for row in subset:
        expected = set(row.get("support_expected_invalidated", []))
        actual = set(row.get("support_actual_invalidated", []))
        true_positive += len(expected & actual)
        false_positive += len(actual - expected)
        false_negative += len(expected - actual)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": _rate(true_positive, true_positive + false_positive),
        "recall": _rate(true_positive, true_positive + false_negative),
        "over_invalidation_rate": _rate(
            false_positive, true_positive + false_positive
        ),
        "under_invalidation_rate": _rate(
            false_negative, true_positive + false_negative
        ),
    }


def summarize_arm(rows: Sequence[Mapping[str, object]], arm: str) -> dict:
    subset = _rows(rows, arm, ("C0", "A1", "A2", "A3", "A4", "A5", "S1", "S2", "S3", "S4"))
    clean = [row for row in subset if row["condition"] == "C0"]
    attempt = [row for row in subset if row["condition"] in ATTEMPT_CONDITIONS]
    support = [row for row in subset if row["condition"] in SUPPORT_CONDITIONS]
    incidental = [row for row in subset if row["condition"] == "A5"]
    faulted = [row for row in subset if row["condition"] != "C0"]
    attempt_population = clean + attempt + incidental
    current_attributed = [
        row
        for row in attempt_population
        if row.get("attempt_attributed_success") is True
    ]
    final_attributed = [
        row
        for row in attempt_population
        if row.get("final_attempt_attributed_success") is True
    ]
    final_attributable_truth = [
        row
        for row in attempt_population
        if row.get("final_target_truth") is True and row["condition"] != "A5"
    ]
    support_confusion = _support_confusion(support)
    return {
        "arm": arm,
        "rollout_count": len(subset),
        "clean_chain_success": _mean(row["chain_success"] for row in clean),
        "faulted_chain_success": _mean(row["chain_success"] for row in faulted),
        "attempt_family_chain_success": _mean(
            row["chain_success"] for row in attempt
        ),
        "support_family_chain_success": _mean(
            row["chain_success"] for row in support
        ),
        "stale_witness_acceptance_rate": _mean(
            row["stale_witness_accepted"]
            for row in attempt
            if row["condition"] == "A1"
        ),
        "cross_attempt_verification_rate": _mean(
            row["cross_attempt_verification"]
            for row in attempt
            if row["condition"] == "A2"
        ),
        "superseded_command_realization_rate": _mean(
            row["superseded_command_realization"]
            for row in attempt
            if row["condition"] == "A3"
        ),
        "late_witness_after_revocation_rate": _mean(
            row["late_witness_after_revocation"]
            for row in attempt
            if row["condition"] == "A4"
        ),
        "current_attempt_realization_precision": _rate(
            sum(
                row.get("target_truth_at_decision") is True
                for row in current_attributed
            ),
            len(current_attributed),
        ),
        "attempt_attribution_precision": _rate(
            sum(
                row.get("final_target_truth") is True
                and row["condition"] != "A5"
                for row in final_attributed
            ),
            len(final_attributed),
        ),
        "attempt_attribution_recall": _rate(
            sum(
                row.get("final_attempt_attributed_success") is True
                for row in final_attributable_truth
            ),
            len(final_attributable_truth),
        ),
        "effect_truth_recognition": _mean(
            row["effect_truth_recognition"] for row in incidental
        ),
        "task_advance_correctness": _mean(
            row["task_advance_correctness"] for row in incidental
        ),
        "false_skill_credit_rate": _mean(
            row["false_skill_credit"] for row in incidental
        ),
        "missed_incidental_success_rate": _mean(
            row["missed_incidental_success"] for row in incidental
        ),
        "cascade_invalidation_precision": support_confusion["precision"],
        "cascade_invalidation_recall": support_confusion["recall"],
        "over_invalidation_rate": support_confusion["over_invalidation_rate"],
        "under_invalidation_rate": support_confusion["under_invalidation_rate"],
        "alternative_support_survival_rate": _mean(
            row["alternative_support_survived"]
            for row in support
            if row["condition"] == "S3"
        ),
        "discharged_support_false_invalidation_rate": _mean(
            row["discharged_support_false_invalidation"]
            for row in support
            if row["condition"] == "S2"
        ),
        "branch_locality_accuracy": _mean(
            row["branch_locality_correct"]
            for row in support
            if row["condition"] == "S4"
        ),
        "grounded_advance_precision": _mean(
            row["task_advance_correctness"]
            for row in subset
            if row.get("task_advance_correctness") is not None
        ),
        "recovery_success": _mean(
            row["recovery_success"]
            for row in subset
            if int(row.get("retry_count", 0)) > 0
        ),
        "premature_advance_rate": _mean(row["premature_advance"] for row in subset),
        "timeout_rate": 0.0,
        "action_steps": _mean(row["action_steps"] for row in subset),
        "clean_action_steps": _mean(row["action_steps"] for row in clean),
        "retry_count": _mean(row["retry_count"] for row in subset),
        "reobserve_count": _mean(row["reobserve_count"] for row in subset),
        "rollback_count": _mean(row["rollback_count"] for row in subset),
        "decision_latency_ns": _mean(row["decision_latency_ns"] for row in subset),
        "clean_decision_latency_ns": _mean(
            row["decision_latency_ns"] for row in clean
        ),
        "event_processing_time_ns": _mean(
            row["event_processing_time_ns"] for row in subset
        ),
        "clean_event_processing_time_ns": _mean(
            row["event_processing_time_ns"] for row in clean
        ),
        "ledger_size": _mean(row["ledger_size"] for row in subset),
        "proof_graph_size": _mean(row["proof_graph_size"] for row in subset),
        "child_process_failures": sum(
            row.get("child_process_failure", False)
            for row in rows
            if row.get("arm") == arm
        ),
        "per_condition_chain_success": {
            condition: _mean(
                row["chain_success"]
                for row in subset
                if row["condition"] == condition
            )
            for condition in ("C0", "A1", "A2", "A3", "A4", "A5", "S1", "S2", "S3", "S4")
        },
    }


def _paired_rows(
    rows: Sequence[Mapping[str, object]],
    arm_a: str,
    arm_b: str,
    conditions: Iterable[str],
) -> List[Tuple[Tuple[str, int], str, int, int]]:
    allowed = set(conditions)
    by_unit: MutableMapping[str, Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if (
            row.get("condition") in allowed
            and row.get("arm") in (arm_a, arm_b)
            and not row.get("child_process_failure")
        ):
            by_unit[str(row["unit_id"])][str(row["arm"])] = row
    result = []
    for unit_id, group in sorted(by_unit.items()):
        if arm_a not in group or arm_b not in group:
            continue
        source = group[arm_a]
        cluster = (str(source["task_id"]), int(source["seed"]))
        result.append(
            (
                cluster,
                unit_id,
                int(bool(group[arm_a]["chain_success"])),
                int(bool(group[arm_b]["chain_success"])),
            )
        )
    return result


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    arm_a: str,
    arm_b: str,
    conditions: Iterable[str],
    repetitions: int = 10000,
    seed: int = 1619,
) -> dict:
    pairs = _paired_rows(rows, arm_a, arm_b, conditions)
    clusters = sorted({row[0] for row in pairs})
    by_cluster = {
        cluster: [row for row in pairs if row[0] == cluster] for cluster in clusters
    }
    if not clusters:
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "paired_unit_count": 0,
            "cluster_count": 0,
            "observed_difference": None,
            "ci_95": [None, None],
        }
    observed = _mean(a - b for _, _, a, b in pairs)
    generator = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = generator.choice(len(clusters), size=len(clusters), replace=True)
        differences = []
        for cluster_index in selected:
            for _, _, a, b in by_cluster[clusters[int(cluster_index)]]:
                differences.append(a - b)
        draws[index] = np.mean(differences)
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "method": "paired_cluster_bootstrap_task_plus_seed",
        "repetitions": repetitions,
        "seed": seed,
        "paired_unit_count": len(pairs),
        "cluster_count": len(clusters),
        "observed_difference": observed,
        "ci_95": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "bootstrap_mean": float(np.mean(draws)),
        "draw_sha256": hashlib.sha256(draws.astype("<f8").tobytes()).hexdigest(),
    }


def exact_mcnemar(
    rows: Sequence[Mapping[str, object]],
    arm_a: str,
    arm_b: str,
    conditions: Iterable[str],
) -> dict:
    pairs = _paired_rows(rows, arm_a, arm_b, conditions)
    a_only = sum(a == 1 and b == 0 for _, _, a, b in pairs)
    b_only = sum(a == 0 and b == 1 for _, _, a, b in pairs)
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(a_only, b_only)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / float(2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "comparison": "%s_vs_%s" % (arm_a, arm_b),
        "paired_unit_count": len(pairs),
        "a_success_b_failure": a_only,
        "a_failure_b_success": b_only,
        "discordant_count": discordant,
        "absolute_success_difference": _mean(a - b for _, _, a, b in pairs),
        "exact_two_sided_p_value": p_value,
    }


def holm_adjust(tests: Sequence[Mapping[str, object]]) -> List[dict]:
    # The same arm comparison is intentionally reported once per endpoint
    # family.  Index by input row, not by the non-unique comparison label.
    ordered = sorted(
        enumerate(tests), key=lambda item: float(item[1]["exact_two_sided_p_value"])
    )
    adjusted: Dict[int, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (original_index, row) in enumerate(ordered):
        value = min(1.0, float(row["exact_two_sided_p_value"]) * (count - rank))
        running = max(running, value)
        adjusted[original_index] = running
    result = []
    for original_index, row in enumerate(tests):
        value = dict(row)
        value["holm_adjusted_p_value"] = adjusted[original_index]
        value["holm_reject_0_05"] = value["holm_adjusted_p_value"] <= 0.05
        result.append(value)
    return result


def _best_baseline(summaries: Mapping[str, Mapping[str, object]], endpoint: str) -> str:
    values = {
        arm: float(summaries[arm][endpoint]) for arm in STRONG_BASELINES
    }
    if values[STRONG_BASELINES[0]] >= values[STRONG_BASELINES[1]]:
        return STRONG_BASELINES[0]
    return STRONG_BASELINES[1]


def analyze_formal(rows: Sequence[Mapping[str, object]]) -> dict:
    summaries = {arm: summarize_arm(rows, arm) for arm in MAIN_ARMS}
    attempt_best = _best_baseline(summaries, "attempt_family_chain_success")
    support_best = _best_baseline(summaries, "support_family_chain_success")
    bootstrap = {
        "attempt": {
            baseline: paired_cluster_bootstrap(
                rows,
                "M4_ASCEL_FULL",
                baseline,
                ATTEMPT_CONDITIONS,
                seed=1619 + index,
            )
            for index, baseline in enumerate(STRONG_BASELINES)
        },
        "support": {
            baseline: paired_cluster_bootstrap(
                rows,
                "M4_ASCEL_FULL",
                baseline,
                SUPPORT_CONDITIONS,
                seed=1719 + index,
            )
            for index, baseline in enumerate(STRONG_BASELINES)
        },
    }
    tests = []
    for family, conditions in (
        ("attempt", ATTEMPT_CONDITIONS),
        ("support", SUPPORT_CONDITIONS),
    ):
        for baseline in STRONG_BASELINES:
            value = exact_mcnemar(rows, "M4_ASCEL_FULL", baseline, conditions)
            value["family"] = family
            tests.append(value)
    tests = holm_adjust(tests)
    m4 = summaries["M4_ASCEL_FULL"]
    attempt_margin = float(m4["attempt_family_chain_success"]) - float(
        summaries[attempt_best]["attempt_family_chain_success"]
    )
    support_margin = float(m4["support_family_chain_success"]) - float(
        summaries[support_best]["support_family_chain_success"]
    )
    clean_best = _best_baseline(summaries, "clean_chain_success")
    clean_degradation = float(summaries[clean_best]["clean_chain_success"]) - float(
        m4["clean_chain_success"]
    )
    baseline_actions = float(summaries[clean_best]["clean_action_steps"])
    action_overhead = (
        float(m4["clean_action_steps"]) - baseline_actions
    ) / baseline_actions
    baseline_latency = float(
        summaries[clean_best]["clean_event_processing_time_ns"]
    )
    latency_overhead = (
        float(m4["clean_event_processing_time_ns"]) - baseline_latency
    ) / baseline_latency
    attempt_gates = {
        "success_margin_ge_0_15": attempt_margin >= 0.15,
        "ci_95_lower_gt_0": bootstrap["attempt"][attempt_best]["ci_95"][0] > 0.0,
        "stale_witness_acceptance_zero": m4["stale_witness_acceptance_rate"] == 0.0,
        "cross_attempt_verification_zero": m4["cross_attempt_verification_rate"] == 0.0,
        "superseded_witness_realization_zero": m4["superseded_command_realization_rate"] == 0.0,
        "late_witness_after_revocation_zero": m4["late_witness_after_revocation_rate"] == 0.0,
    }
    support_gates = {
        "cascade_precision_ge_0_95": float(m4["cascade_invalidation_precision"]) >= 0.95,
        "cascade_recall_ge_0_95": float(m4["cascade_invalidation_recall"]) >= 0.95,
        "over_invalidation_le_0_05": float(m4["over_invalidation_rate"]) <= 0.05,
        "under_invalidation_le_0_05": float(m4["under_invalidation_rate"]) <= 0.05,
        "success_margin_ge_0_15": support_margin >= 0.15,
        "ci_95_lower_gt_0": bootstrap["support"][support_best]["ci_95"][0] > 0.0,
    }
    truth_gates = {
        "effect_truth_recognition_ge_0_95": m4["effect_truth_recognition"] >= 0.95,
        "task_advance_correctness_ge_0_95": m4["task_advance_correctness"] >= 0.95,
        "false_current_skill_credit_le_0_05": m4["false_skill_credit_rate"] <= 0.05,
    }
    clean_gates = {
        "clean_success_degradation_le_0_02": clean_degradation <= 0.02,
        "action_step_overhead_le_0_15": action_overhead <= 0.15,
        "event_processing_latency_overhead_le_0_10": latency_overhead <= 0.10,
    }
    return {
        "schema_version": 1,
        "arms": summaries,
        "best_baselines": {
            "attempt_family": attempt_best,
            "support_family": support_best,
            "clean": clean_best,
        },
        "effect_sizes": {
            "attempt_family_success_margin": attempt_margin,
            "support_family_success_margin": support_margin,
            "clean_success_degradation": clean_degradation,
            "action_step_overhead_fraction": action_overhead,
            "event_processing_latency_overhead_fraction": latency_overhead,
        },
        "cluster_bootstrap": bootstrap,
        "paired_tests": tests,
        "component_gates": {
            "attempt_scope": attempt_gates,
            "support_proof": support_gates,
            "truth_attribution": truth_gates,
            "clean": clean_gates,
        },
        "component_status": {
            "attempt_scope": "ATTEMPT_SCOPE_PASS" if all(attempt_gates.values()) else "ATTEMPT_SCOPE_FAIL",
            "support_proof": "SUPPORT_PROOF_PASS" if all(support_gates.values()) else "SUPPORT_PROOF_FAIL",
            "truth_attribution": "TRUTH_ATTRIBUTION_PASS" if all(truth_gates.values()) else "TRUTH_ATTRIBUTION_FAIL",
            "clean": "CLEAN_PASS" if all(clean_gates.values()) else "CLEAN_FAIL",
        },
    }


def analyze_ablations(
    formal_rows: Sequence[Mapping[str, object]],
    ablation_rows: Sequence[Mapping[str, object]],
    formal_analysis: Mapping[str, object],
) -> dict:
    ablations = {arm: summarize_arm(ablation_rows, arm) for arm in ABLATION_ARMS}
    formal_arms = formal_analysis["arms"]
    m4 = formal_arms["M4_ASCEL_FULL"]
    attempt_best = formal_analysis["best_baselines"]["attempt_family"]
    support_best = formal_analysis["best_baselines"]["support_family"]
    m4_attempt_advantage = float(m4["attempt_family_chain_success"]) - float(
        formal_arms[attempt_best]["attempt_family_chain_success"]
    )
    no_attempt_advantage = float(
        ablations["NO_ATTEMPT_SCOPE"]["attempt_family_chain_success"]
    ) - float(formal_arms[attempt_best]["attempt_family_chain_success"])
    attempt_removed = (
        1.0 - no_attempt_advantage / m4_attempt_advantage
        if m4_attempt_advantage > 0
        else None
    )
    m4_support_advantage = float(m4["support_family_chain_success"]) - float(
        formal_arms[support_best]["support_family_chain_success"]
    )
    no_support_advantage = float(
        ablations["NO_SUPPORT_VALIDITY"]["support_family_chain_success"]
    ) - float(formal_arms[support_best]["support_family_chain_success"])
    support_removed = (
        1.0 - no_support_advantage / m4_support_advantage
        if m4_support_advantage > 0
        else None
    )
    attribution_increase = float(
        ablations["NO_ATTRIBUTION_SPLIT"]["false_skill_credit_rate"]
    ) - float(m4["false_skill_credit_rate"])
    revocation_increase = float(
        ablations["NO_PRE_REALIZATION_REVOCATION"][
            "late_witness_after_revocation_rate"
        ]
    ) - float(m4["late_witness_after_revocation_rate"])
    return {
        "schema_version": 1,
        "arms": ablations,
        "isolated_effects": {
            "NO_ATTEMPT_SCOPE": {
                "attempt_advantage_removed_fraction": attempt_removed,
                "criterion_ge_0_50": attempt_removed is not None and attempt_removed >= 0.50,
            },
            "NO_SUPPORT_VALIDITY": {
                "support_advantage_removed_fraction": support_removed,
                "over_invalidation_rate": ablations["NO_SUPPORT_VALIDITY"]["over_invalidation_rate"],
                "criterion": (
                    support_removed is not None and support_removed >= 0.50
                ) or float(ablations["NO_SUPPORT_VALIDITY"]["over_invalidation_rate"]) >= 0.10,
            },
            "NO_ATTRIBUTION_SPLIT": {
                "false_skill_credit_absolute_increase": attribution_increase,
                "criterion_ge_0_10": attribution_increase >= 0.10,
            },
            "NO_PRE_REALIZATION_REVOCATION": {
                "A4_false_realization_absolute_increase": revocation_increase,
                "criterion_ge_0_10": revocation_increase >= 0.10,
            },
        },
    }


def final_decision(
    formal_analysis: Mapping[str, object],
    trace_pass: bool,
    executor_pass: bool,
    shared_prefix_pass: bool,
) -> dict:
    statuses = formal_analysis["component_status"]
    attempt = statuses["attempt_scope"] == "ATTEMPT_SCOPE_PASS"
    support = statuses["support_proof"] == "SUPPORT_PROOF_PASS"
    truth = statuses["truth_attribution"] == "TRUTH_ATTRIBUTION_PASS"
    clean = statuses["clean"] == "CLEAN_PASS"
    if not trace_pass:
        overall = "BLOCKED_BY_MECHANISM_IMPLEMENTATION"
    elif not executor_pass:
        overall = "BLOCKED_BY_MICROENV"
    elif not shared_prefix_pass:
        overall = "BLOCKED_BY_SHARED_PREFIX"
    elif attempt and support and truth and clean:
        overall = "PASS_PHASE4_ASC_EFFECT_LEDGER"
    elif attempt and not support:
        overall = "NARROW_TO_ATTEMPT_SCOPED_LEDGER"
    elif support and not attempt:
        overall = "NARROW_TO_SUPPORT_PROOF_LEDGER"
    elif truth and not attempt and not support:
        overall = "NARROW_TO_EFFECT_ATTRIBUTION_LEDGER"
    elif not attempt and not support and not truth:
        overall = "REJECT_R16P19_V2"
    else:
        overall = "BLOCKED_BY_IMPLEMENTATION"
    return {
        "schema_version": 1,
        "platform_gates": {
            "trace_gate": bool(trace_pass),
            "executor_gate": bool(executor_pass),
            "shared_prefix_gate": bool(shared_prefix_pass),
        },
        "component_status": dict(statuses),
        "overall_status": overall,
        "diagnostic_continuation_after_failed_gate": True,
        "failed_gate_can_be_overridden_by_downstream_results": False,
        "claim_boundary": "CPU_microbenchmark_only_not_VLA_or_external_benchmark_evidence",
    }


def failure_decomposition(rows: Sequence[Mapping[str, object]]) -> List[dict]:
    result = []
    arms = sorted({str(row["arm"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows})
    for arm in arms:
        for condition in conditions:
            subset = [
                row
                for row in rows
                if row["arm"] == arm and row["condition"] == condition
            ]
            result.append(
                {
                    "arm": arm,
                    "condition": condition,
                    "rollout_count": len(subset),
                    "chain_success": _mean(row["chain_success"] for row in subset),
                    "premature_advance_rate": _mean(
                        row.get("premature_advance", False) for row in subset
                    ),
                    "over_invalidation_count": sum(
                        int(row.get("over_invalidation_count", 0)) for row in subset
                    ),
                    "under_invalidation_count": sum(
                        int(row.get("under_invalidation_count", 0)) for row in subset
                    ),
                    "false_skill_credit_rate": _mean(
                        row.get("false_skill_credit", False) for row in subset
                    ),
                    "child_process_failure_count": sum(
                        row.get("child_process_failure", False) for row in subset
                    ),
                }
            )
    return result

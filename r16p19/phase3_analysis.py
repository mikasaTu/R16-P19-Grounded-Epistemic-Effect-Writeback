"""Frozen Phase-3 metrics, paired inference, and causal intervention analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

from .artifacts import sha256_file
if TYPE_CHECKING:
    from .phase3_replay_backend import FrozenEffectReplayBackend


FAULT_CONDITIONS = ("C1", "C3", "C4", "C7")
MAIN_CONDITIONS = ("C0", "C1", "C3", "C4", "C7")
STRONG_BASELINES = (
    "POSTCHECK_RECOVERY",
    "PERSISTENCE_RECOVERY",
    "TYPED_MATCHED_RECOVERY",
)
BEST_BASELINE_TIE_ORDER = (
    "TYPED_MATCHED_RECOVERY",
    "PERSISTENCE_RECOVERY",
    "POSTCHECK_RECOVERY",
)
PROTECTED_B6_SHA256 = "4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5"


def _chain_map(contract: Mapping[str, object]) -> Dict[str, dict]:
    return {str(row["chain_id"]): dict(row) for row in contract["candidate_chains"]}


def _mean(values: Iterable[object]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    return float(np.mean(materialized)) if materialized else None


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _read_jsonl(path: Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _unit_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row["chain_id"]),
        str(row["source_episode"]),
        str(row["condition"]),
    )


def _valid_primary_rows(rows: Sequence[Mapping[str, object]], arm: str) -> List[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row["arm"] == arm
        and row["condition"] in FAULT_CONDITIONS
        and row.get("failure_type")
        not in ("REPLAY_BACKEND_FAILURE", "RECEIPT_BROKER_ERROR", "FAULT_INJECTOR_ERROR")
    ]


def summarize_arm(rows: Sequence[Mapping[str, object]], arm: str) -> dict:
    subset = [row for row in rows if row["arm"] == arm]
    faulted = [row for row in subset if row["condition"] in FAULT_CONDITIONS]
    clean = [row for row in subset if row["condition"] == "C0"]
    c3 = [row for row in subset if row["condition"] == "C3"]
    c4 = [row for row in subset if row["condition"] == "C4"]
    c7 = [row for row in subset if row["condition"] == "C7"]
    recovery_attempted = [row for row in faulted if row.get("recovery_attempted")]
    total_advances = sum(int(row.get("advance_count", 0)) for row in subset)
    grounded_advances = sum(int(row.get("grounded_advance_true_count", 0)) for row in subset)
    expected_effect_completions = sum(len(row.get("effect_success", {})) for row in subset)
    effect_completions = sum(
        sum(bool(value) for value in row.get("effect_success", {}).values()) for row in subset
    )
    c3_presented = [row for row in c3 if row.get("contradiction_presented")]
    invalidation_rows = [
        row for row in c3_presented if row.get("invalidated_realization_accuracy") is not None
    ]
    return {
        "arm": arm,
        "rollout_count": len(subset),
        "chain_success": _mean(row["chain_success"] for row in subset),
        "current_effect_success": _mean(row["current_effect_success"] for row in subset),
        "next_effect_success": _mean(row["next_effect_success"] for row in subset),
        "faulted_chain_success": _mean(row["chain_success"] for row in faulted),
        "clean_chain_success": _mean(row["chain_success"] for row in clean),
        "recovery_success": _mean(row["recovery_success"] for row in recovery_attempted),
        "recovery_attempt_count": len(recovery_attempted),
        "repeated_loop_rate": _mean(row["repeated_loop"] for row in subset),
        "grounded_advance_precision": _rate(grounded_advances, total_advances),
        "grounded_advance_recall": _rate(effect_completions, expected_effect_completions),
        "false_completion_rate": _mean(row["false_completion"] for row in subset),
        "premature_advance_rate": _mean(row["premature_advance"] for row in subset),
        "contradiction_detection_recall": _mean(
            row.get("contradiction_detected", False) for row in c3_presented
        ),
        "contradiction_recovery_recall": _mean(
            row.get("contradiction_recovery_decision", False) for row in c3_presented
        ),
        "single_view_false_positive_advance_rate": _mean(
            row["single_view_false_positive_advance"] for row in c4
        ),
        "imagined_as_realized_rate": _mean(row["imagined_as_realized"] for row in c7),
        "invalidated_realization_accuracy": _mean(
            row["invalidated_realization_accuracy"] for row in invalidation_rows
        ),
        "retry_count": _mean(row["retry_count"] for row in subset),
        "reobserve_count": _mean(row["reobserve_count"] for row in subset),
        "rollback_count": _mean(row["rollback_count"] for row in subset),
        "unnecessary_retry_rate": _mean(row["unnecessary_retry"] for row in clean),
        "unnecessary_recovery_rate": _mean(row["unnecessary_recovery"] for row in clean),
        "safe_stop_rate": _mean(row["safe_stop"] for row in subset),
        "action_steps": _mean(row["action_steps"] for row in subset),
        "clean_action_steps": _mean(row["action_steps"] for row in clean),
        "completion_latency": _mean(row["completion_latency"] for row in subset),
        "backend_failure_count": sum(
            row.get("failure_type") == "REPLAY_BACKEND_FAILURE" for row in subset
        ),
        "fault_injector_error_count": sum(
            row.get("failure_type") == "FAULT_INJECTOR_ERROR" for row in subset
        ),
        "action_budget_exceeded_count": sum(
            bool(row.get("action_budget_exceeded")) for row in subset
        ),
        "resident_slot_count_max": max(
            [int(row.get("resident_slot_count_max", 0)) for row in subset] or [0]
        ),
        "dangling_parent_count": sum(int(row.get("dangling_parent_count", 0)) for row in subset),
        "transition_violation_count": sum(
            int(row.get("transition_violation_count", 0)) for row in subset
        ),
        "fault_or_truth_leakage_count": sum(
            int(row.get("fault_or_truth_leakage_count", 0)) for row in subset
        ),
        "per_condition_chain_success": {
            condition: _mean(
                row["chain_success"] for row in subset if row["condition"] == condition
            )
            for condition in MAIN_CONDITIONS
        },
        "per_chain_chain_success": {
            chain_id: _mean(
                row["chain_success"] for row in subset if row["chain_id"] == chain_id
            )
            for chain_id in sorted({str(row["chain_id"]) for row in subset})
        },
    }


def _paired_values(
    rows: Sequence[Mapping[str, object]], arm_a: str, arm_b: str
) -> List[tuple[str, tuple[str, str, str], int, int]]:
    by_key: MutableMapping[tuple[str, str, str], Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if row["condition"] in FAULT_CONDITIONS and row["arm"] in (arm_a, arm_b):
            by_key[_unit_key(row)][str(row["arm"])] = row
    result = []
    for key, group in sorted(by_key.items()):
        if arm_a not in group or arm_b not in group:
            continue
        if any(
            value.get("failure_type")
            in ("REPLAY_BACKEND_FAILURE", "RECEIPT_BROKER_ERROR", "FAULT_INJECTOR_ERROR")
            for value in (group[arm_a], group[arm_b])
        ):
            continue
        result.append(
            (
                key[1],
                key,
                int(bool(group[arm_a]["chain_success"])),
                int(bool(group[arm_b]["chain_success"])),
            )
        )
    return result


def cluster_bootstrap(
    rows: Sequence[Mapping[str, object]], baseline: str, repetitions: int = 10000, seed: int = 1619
) -> dict:
    pairs = _paired_values(rows, "B6_FULL", baseline)
    clusters = sorted({cluster for cluster, _, _, _ in pairs})
    by_cluster = {cluster: [row for row in pairs if row[0] == cluster] for cluster in clusters}
    if not clusters:
        return {
            "baseline": baseline,
            "repetitions": repetitions,
            "paired_unit_count": 0,
            "cluster_count": 0,
            "observed_difference": None,
            "ci_95": [None, None],
        }
    observed = float(np.mean([primary - control for _, _, primary, control in pairs]))
    generator = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = generator.choice(clusters, size=len(clusters), replace=True)
        differences = [
            primary - control
            for cluster in sampled
            for _, _, primary, control in by_cluster[str(cluster)]
        ]
        draws[repetition] = np.mean(differences)
    return {
        "baseline": baseline,
        "method": "paired_cluster_bootstrap_with_source_episode_resampling",
        "seed": seed,
        "repetitions": repetitions,
        "paired_unit_count": len(pairs),
        "cluster_count": len(clusters),
        "clusters": clusters,
        "observed_difference": observed,
        "ci_95": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "bootstrap_mean": float(np.mean(draws)),
        "draw_sha256": hashlib.sha256(draws.astype("<f8").tobytes()).hexdigest(),
    }


def exact_mcnemar(rows: Sequence[Mapping[str, object]], baseline: str) -> dict:
    pairs = _paired_values(rows, "B6_FULL", baseline)
    primary_only = sum(primary == 1 and control == 0 for _, _, primary, control in pairs)
    control_only = sum(primary == 0 and control == 1 for _, _, primary, control in pairs)
    discordant = primary_only + control_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(0, min(primary_only, control_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "comparison": "B6_FULL_vs_%s" % baseline,
        "paired_unit_count": len(pairs),
        "b6_success_baseline_failure": primary_only,
        "b6_failure_baseline_success": control_only,
        "discordant_count": discordant,
        "absolute_success_difference": _mean(
            primary - control for _, _, primary, control in pairs
        ),
        "exact_two_sided_p_value": float(p_value),
    }


def holm_adjust(tests: Sequence[Mapping[str, object]]) -> List[dict]:
    ordered = sorted(tests, key=lambda row: float(row["exact_two_sided_p_value"]))
    adjusted_by_name: Dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (count - index) * float(row["exact_two_sided_p_value"]))
        running = max(running, adjusted)
        adjusted_by_name[str(row["comparison"])] = running
    return [
        {
            **dict(row),
            "holm_adjusted_p_value": adjusted_by_name[str(row["comparison"])],
            "holm_reject_0_05": adjusted_by_name[str(row["comparison"])] <= 0.05,
        }
        for row in tests
    ]


def effect_sizes(rows: Sequence[Mapping[str, object]], baseline: str) -> dict:
    def difference(predicate) -> float | None:
        primary = [
            row["chain_success"]
            for row in rows
            if row["arm"] == "B6_FULL" and predicate(row)
        ]
        control = [
            row["chain_success"] for row in rows if row["arm"] == baseline and predicate(row)
        ]
        if not primary or not control:
            return None
        return float(np.mean(primary) - np.mean(control))

    return {
        "baseline": baseline,
        "per_condition": {
            condition: difference(lambda row, value=condition: row["condition"] == value)
            for condition in MAIN_CONDITIONS
        },
        "per_chain": {
            chain_id: difference(lambda row, value=chain_id: row["chain_id"] == value)
            for chain_id in sorted({str(row["chain_id"]) for row in rows})
        },
    }


def choose_best_strong(rows: Sequence[Mapping[str, object]]) -> tuple[str, Dict[str, float]]:
    means = {
        arm: float(np.mean([row["chain_success"] for row in _valid_primary_rows(rows, arm)]))
        if _valid_primary_rows(rows, arm)
        else float("-inf")
        for arm in STRONG_BASELINES
    }
    tie_rank = {arm: index for index, arm in enumerate(BEST_BASELINE_TIE_ORDER)}
    best = sorted(means, key=lambda arm: (-means[arm], tie_rank[arm]))[0]
    return best, means


def summarize_behavior(
    rows: Sequence[Mapping[str, object]],
    causal_rows: Sequence[Mapping[str, object]],
) -> dict:
    arms = sorted({str(row["arm"]) for row in rows})
    summaries = {arm: summarize_arm(rows, arm) for arm in arms}
    best, primary_means = choose_best_strong(rows)
    b6_mean = summaries["B6_FULL"]["faulted_chain_success"]
    best_mean = summaries[best]["faulted_chain_success"]
    causal_units = {}
    for row in causal_rows:
        causal_units.setdefault(row["paired_unit_id"], bool(row["b6_natural_decision_is_causal_winner"]))
    decision_causal_win_rate = _mean(causal_units.values())
    return {
        "schema_version": 1,
        "status": "BEHAVIOR_SUMMARY_COMPLETE",
        "rollout_count": len(rows),
        "arms": summaries,
        "best_observed_strong_baseline": best,
        "primary_faulted_chain_success": {
            "B6_FULL": b6_mean,
            **primary_means,
        },
        "B6_minus_best_strong": (
            float(b6_mean - best_mean)
            if b6_mean is not None and best_mean is not None
            else None
        ),
        "decision_causal_win_rate": decision_causal_win_rate,
        "causal_paired_unit_count": len(causal_units),
    }


def run_first_divergence_replays(
    backend: "FrozenEffectReplayBackend",
    chain_contract: Mapping[str, object],
    chain_ids: Sequence[str],
    audits: Sequence[Mapping[str, object]],
    formal_rows: Sequence[Mapping[str, object]],
    persistence_k: int,
    output_path: Path,
) -> List[dict]:
    """Rerun B6 from the exact deterministic prefix under each unique decision."""

    from .phase3_runner import run_chain_rollout, write_rows

    output_path = Path(output_path)
    existing = _read_jsonl(output_path) if output_path.is_file() else []
    observed = {
        (row["paired_unit_id"], row["forced_decision"]) for row in existing
    }
    chains = _chain_map(chain_contract)
    original = {
        _unit_key(row): row for row in formal_rows if row["arm"] == "B6_FULL"
    }
    for audit in audits:
        divergence = audit.get("first_decision_divergence_index")
        if divergence is None or not audit.get("paired_prefix_event_and_action_bytes_identical"):
            continue
        key = (
            str(audit["chain_id"]),
            str(audit["source_episode"]),
            str(audit["condition"]),
        )
        if key not in original:
            continue
        base = original[key]
        if int(divergence) >= len(base.get("decision_trace", [])):
            continue
        paired_unit_id = "|".join(key)
        natural_decision = base["decision_trace"][int(divergence)]["decision"]
        generated = []
        for forced_decision in audit.get("unique_decisions", []):
            record_key = (paired_unit_id, forced_decision)
            if record_key in observed:
                continue
            rollout = run_chain_rollout(
                backend,
                "formal",
                chains[key[0]],
                list(chain_ids).index(key[0]),
                key[1],
                key[2],
                "B6_FULL",
                persistence_k,
                forced_decisions={int(divergence): str(forced_decision)},
            )
            trace = rollout["decision_trace"]
            intervention_trace = trace[int(divergence)] if int(divergence) < len(trace) else None
            prefix_verified = bool(
                intervention_trace
                and intervention_trace["event_prefix_sha256"]
                == base["decision_trace"][int(divergence)]["event_prefix_sha256"]
                and intervention_trace["simulator_state_sha256"]
                == base["decision_trace"][int(divergence)]["simulator_state_sha256"]
            )
            after = (
                trace[int(divergence) + 1]
                if int(divergence) + 1 < len(trace)
                else intervention_trace
            )
            row = {
                "record_type": "phase3_first_divergence_decision_intervention",
                "paired_unit_id": paired_unit_id,
                "chain_id": key[0],
                "source_episode": key[1],
                "condition": key[2],
                "divergence_index": int(divergence),
                "divergence_effect_id": audit.get("divergence_effect_id"),
                "saved_simulator_state_sha256": base["decision_trace"][int(divergence)][
                    "simulator_state_sha256"
                ],
                "saved_event_prefix_sha256": base["decision_trace"][int(divergence)][
                    "event_prefix_sha256"
                ],
                "prefix_replay_verified": prefix_verified,
                "natural_b6_decision": natural_decision,
                "forced_decision": forced_decision,
                "immediate_effect_completion": bool(
                    after and after.get("physical_truth_evaluation_only")
                ),
                "eventual_chain_completion": bool(rollout["chain_success"]),
                "irreversible_failure": rollout.get("failure_type")
                in ("PREMATURE_ADVANCE", "OVERCONSERVATIVE_STOP"),
                "extra_steps_vs_natural_b6": int(rollout["action_steps"])
                - int(base["action_steps"]),
                "recovery_cost": {
                    "retry_count": rollout["retry_count"],
                    "reobserve_count": rollout["reobserve_count"],
                    "rollback_count": rollout["rollback_count"],
                    "action_steps": rollout["action_steps"],
                },
                "failure_type": rollout.get("failure_type"),
                "b6_natural_decision_is_causal_winner": False,
            }
            generated.append(row)
        if generated:
            candidates = [
                row for row in existing if row["paired_unit_id"] == paired_unit_id
            ] + generated
            def score(row: Mapping[str, object]) -> tuple[int, int, int, int]:
                return (
                    int(bool(row["eventual_chain_completion"])),
                    int(bool(row["immediate_effect_completion"])),
                    -int(bool(row["irreversible_failure"])),
                    -int(row["recovery_cost"]["action_steps"]),
                )
            best_score = max(score(row) for row in candidates)
            natural_wins = any(
                row["forced_decision"] == natural_decision and score(row) == best_score
                for row in candidates
            )
            for row in candidates:
                row["b6_natural_decision_is_causal_winner"] = natural_wins
            existing = [row for row in existing if row["paired_unit_id"] != paired_unit_id]
            existing.extend(candidates)
            write_rows(output_path, existing)
            observed.update((row["paired_unit_id"], row["forced_decision"]) for row in candidates)
    return sorted(existing, key=lambda row: (row["paired_unit_id"], row["forced_decision"]))


def paired_inference(rows: Sequence[Mapping[str, object]]) -> tuple[dict, dict]:
    best, primary_means = choose_best_strong(rows)
    bootstraps = {
        baseline: cluster_bootstrap(rows, baseline) for baseline in STRONG_BASELINES
    }
    tests = holm_adjust([exact_mcnemar(rows, baseline) for baseline in STRONG_BASELINES])
    cluster = {
        "schema_version": 1,
        "status": "CLUSTER_BOOTSTRAP_COMPLETE",
        "cluster_unit": "source_demonstration_episode_index",
        "primary_endpoint": "mean_chain_success_over_C1_C3_C4_C7",
        "best_observed_strong_baseline": best,
        "primary_means": {"B6_FULL": _mean(row["chain_success"] for row in _valid_primary_rows(rows, "B6_FULL")), **primary_means},
        "comparisons": bootstraps,
        "primary_comparison": bootstraps[best],
    }
    paired = {
        "schema_version": 1,
        "status": "PAIRED_TESTS_COMPLETE",
        "method": "exact_two_sided_McNemar_with_Holm_step_down",
        "tests": tests,
        "effect_sizes": [effect_sizes(rows, baseline) for baseline in STRONG_BASELINES],
    }
    return cluster, paired


def failure_decomposition(rows: Sequence[Mapping[str, object]]) -> List[dict]:
    failure_types = (
        "MEMORY_DECISION_ERROR",
        "REPLAY_BACKEND_FAILURE",
        "RECEIPT_BROKER_ERROR",
        "FAULT_INJECTOR_ERROR",
        "TIMEOUT",
        "PREMATURE_ADVANCE",
        "OVERCONSERVATIVE_STOP",
    )
    result = []
    for arm in sorted({str(row["arm"]) for row in rows}):
        subset = [row for row in rows if row["arm"] == arm]
        counts = Counter(row.get("failure_type") for row in subset if row.get("failure_type"))
        result.append(
            {
                "arm": arm,
                "rollout_count": len(subset),
                "success_count": sum(bool(row["chain_success"]) for row in subset),
                "failure_counts": {value: int(counts.get(value, 0)) for value in failure_types},
                "unclassified_failure_count": sum(
                    not row["chain_success"] and row.get("failure_type") not in failure_types
                    for row in subset
                ),
            }
        )
    return result


def evaluate_final_decision(
    rows: Sequence[Mapping[str, object]],
    behavior: Mapping[str, object],
    cluster: Mapping[str, object],
    qualification_pass: bool,
    formal_replay_pass: bool,
    paired_audits: Sequence[Mapping[str, object]],
    protected_b6_path: Path,
    expected_rollout_count: int | None = None,
) -> dict:
    arms = behavior["arms"]
    b6 = arms["B6_FULL"]
    best_name = behavior["best_observed_strong_baseline"]
    best = arms[best_name]
    margin = behavior["B6_minus_best_strong"]
    ci_lower = cluster["primary_comparison"]["ci_95"][0]
    clean_degradation = float(best["clean_chain_success"] - b6["clean_chain_success"])
    clean_overhead = (
        float(b6["clean_action_steps"] / best["clean_action_steps"] - 1.0)
        if best["clean_action_steps"]
        else float("inf")
    )
    protected_hash_match = sha256_file(Path(protected_b6_path)) == PROTECTED_B6_SHA256
    paired_prefix_ok = bool(
        paired_audits
        and all(row["paired_prefix_event_and_action_bytes_identical"] for row in paired_audits)
    )
    correctness = {
        "dangling_parent_count_zero": b6["dangling_parent_count"] == 0,
        "resident_slot_count_max_le_32": b6["resident_slot_count_max"] <= 32,
        "transition_violation_count_zero": b6["transition_violation_count"] == 0,
        "fault_or_truth_leakage_count_zero": b6["fault_or_truth_leakage_count"] == 0,
        "paired_prefix_event_action_state_identity": paired_prefix_ok,
        "protected_b6_source_hash_match": protected_hash_match,
    }
    gates = {
        "faulted_chain_success_margin_ge_0_10": margin is not None and margin >= 0.10,
        "cluster_bootstrap_95ci_lower_gt_0": ci_lower is not None and ci_lower > 0.0,
        "grounded_advance_precision_ge_0_95": b6["grounded_advance_precision"] is not None and b6["grounded_advance_precision"] >= 0.95,
        "C3_contradiction_recovery_recall_ge_0_80": b6["contradiction_recovery_recall"] is not None and b6["contradiction_recovery_recall"] >= 0.80,
        "C4_false_positive_advance_rate_le_0_05": b6["single_view_false_positive_advance_rate"] is not None and b6["single_view_false_positive_advance_rate"] <= 0.05,
        "clean_success_degradation_le_0_02": clean_degradation <= 0.02,
        "clean_action_step_overhead_le_0_15": clean_overhead <= 0.15,
        "decision_causal_win_rate_ge_0_70": behavior["decision_causal_win_rate"] is not None and behavior["decision_causal_win_rate"] >= 0.70,
        "all_provenance_and_resident_memory_correctness_gates": all(correctness.values()),
    }
    implementation_errors = sum(
        row.get("failure_type")
        in ("REPLAY_BACKEND_FAILURE", "RECEIPT_BROKER_ERROR", "FAULT_INJECTOR_ERROR")
        for row in rows
    )
    unique_cells = {(_unit_key(row), row["arm"]) for row in rows}
    complete_grid = bool(
        len(rows) == len(unique_cells)
        and (expected_rollout_count is None or len(rows) == expected_rollout_count)
    )
    b6_beats_weak = all(
        b6["faulted_chain_success"] > arms[arm]["faulted_chain_success"]
        for arm in ("B2_COMMAND_PROGRESS", "B3_MONOLITHIC")
    )
    if implementation_errors or not complete_grid:
        status = "BLOCKED_BY_IMPLEMENTATION"
    elif not qualification_pass or not formal_replay_pass:
        status = "BLOCKED_BY_REPLAY_BACKEND"
    elif all(gates.values()):
        status = "PASS_PHASE3_STRONG_CONTROLLED"
    elif b6_beats_weak:
        status = "WEAK_BASELINE_ONLY"
    else:
        status = "REJECT_PHASE3_INCREMENTAL_VALUE"
    return {
        "schema_version": 1,
        "final_status": status,
        "terminal_status_precedence_applied": True,
        "qualification_replay_gate_pass": bool(qualification_pass),
        "formal_replay_gate_pass": bool(formal_replay_pass),
        "formal_matrix_complete": complete_grid,
        "expected_formal_rollout_count": expected_rollout_count,
        "observed_formal_rollout_count": len(rows),
        "implementation_error_count": implementation_errors,
        "best_observed_strong_baseline": best_name,
        "observed_values": {
            "faulted_chain_success_margin": margin,
            "bootstrap_ci_95_lower": ci_lower,
            "clean_success_degradation": clean_degradation,
            "clean_action_step_overhead_fraction": clean_overhead,
            "decision_causal_win_rate": behavior["decision_causal_win_rate"],
        },
        "primary_gates": gates,
        "correctness_gates": correctness,
        "user_override_downstream_completion_honored": True,
        "claim_boundary": (
            "controlled_confirmation"
            if status == "PASS_PHASE3_STRONG_CONTROLLED"
            else "no_strong_controlled_advantage_claim"
        ),
    }


def mechanism_ablation_summary(
    main_rows: Sequence[Mapping[str, object]],
    ablation_rows: Sequence[Mapping[str, object]],
    final_status: str,
) -> dict:
    best_baseline, _ = choose_best_strong(main_rows)
    comparisons = []
    specifications = (
        ("B6_NO_PROVENANCE", ("C4", "C7")),
        ("B6_NO_INVALIDATION", ("C3",)),
    )
    for ablation, conditions in specifications:
        full = [
            row for row in main_rows if row["arm"] == "B6_FULL" and row["condition"] in conditions
        ]
        altered = [row for row in ablation_rows if row["arm"] == ablation]
        baseline = [
            row
            for row in main_rows
            if row["arm"] == best_baseline and row["condition"] in conditions
        ]
        full_rate = _mean(row["chain_success"] for row in full)
        altered_rate = _mean(row["chain_success"] for row in altered)
        baseline_rate = _mean(row["chain_success"] for row in baseline)
        full_advantage = (
            float(full_rate - baseline_rate)
            if full_rate is not None and baseline_rate is not None
            else None
        )
        ablated_advantage = (
            float(altered_rate - baseline_rate)
            if altered_rate is not None and baseline_rate is not None
            else None
        )
        absolute_drop = (
            float(full_advantage - ablated_advantage)
            if full_advantage is not None and ablated_advantage is not None
            else None
        )
        relative_drop = (
            float(absolute_drop / abs(full_advantage))
            if absolute_drop is not None and full_advantage
            else None
        )
        comparisons.append(
            {
                "ablation": ablation,
                "conditions": list(conditions),
                "B6_FULL_chain_success": full_rate,
                "ablated_chain_success": altered_rate,
                "reference_strong_baseline": best_baseline,
                "reference_strong_baseline_chain_success": baseline_rate,
                "B6_FULL_advantage": full_advantage,
                "ablated_advantage": ablated_advantage,
                "absolute_advantage_drop": absolute_drop,
                "relative_advantage_drop": relative_drop,
                "mechanism_drop_ge_50_percent": relative_drop is not None and relative_drop >= 0.50,
            }
        )
    paper_claim_allowed = bool(
        final_status == "PASS_PHASE3_STRONG_CONTROLLED"
        and any(row["mechanism_drop_ge_50_percent"] for row in comparisons)
    )
    return {
        "schema_version": 1,
        "status": "MECHANISM_ABLATIONS_COMPLETE",
        "run_policy": "unconditional_user_override",
        "interpretation": (
            "paper_level_mechanism_claim_supported"
            if paper_claim_allowed
            else "diagnostic_only_no_paper_level_mechanism_claim"
        ),
        "comparisons": comparisons,
        "paper_level_mechanism_claim_allowed": paper_claim_allowed,
    }

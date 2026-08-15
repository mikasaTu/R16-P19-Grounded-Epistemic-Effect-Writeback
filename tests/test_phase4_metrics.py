from r16p19.phase4_analysis import (
    exact_mcnemar,
    holm_adjust,
    paired_cluster_bootstrap,
    summarize_arm,
)


def _paired_rows():
    rows = []
    for seed in range(10):
        for condition in ("A1", "A2", "A3", "A4"):
            unit = "T1|%s|%d" % (condition, seed)
            for arm, success in (("M4_ASCEL_FULL", True), ("M0_TYPED_MATCHED", False)):
                rows.append(
                    {
                        "unit_id": unit,
                        "task_id": "T1",
                        "condition": condition,
                        "seed": seed,
                        "arm": arm,
                        "chain_success": success,
                        "child_process_failure": False,
                    }
                )
    return rows


def test_paired_cluster_bootstrap_is_reproducible_and_clustered():
    rows = _paired_rows()
    first = paired_cluster_bootstrap(
        rows, "M4_ASCEL_FULL", "M0_TYPED_MATCHED", ("A1", "A2", "A3", "A4"), 500
    )
    second = paired_cluster_bootstrap(
        rows, "M4_ASCEL_FULL", "M0_TYPED_MATCHED", ("A1", "A2", "A3", "A4"), 500
    )
    assert first["cluster_count"] == 10
    assert first["paired_unit_count"] == 40
    assert first["observed_difference"] == 1.0
    assert first["ci_95"] == [1.0, 1.0]
    assert first["draw_sha256"] == second["draw_sha256"]


def test_exact_mcnemar_and_holm_preserve_pairing_and_bounds():
    test = exact_mcnemar(
        _paired_rows(), "M4_ASCEL_FULL", "M0_TYPED_MATCHED", ("A1", "A2", "A3", "A4")
    )
    assert test["a_success_b_failure"] == 40
    assert test["a_failure_b_success"] == 0
    assert 0.0 <= test["exact_two_sided_p_value"] <= 1.0
    adjusted = holm_adjust(
        [
            {"comparison": "a", "exact_two_sided_p_value": 0.01},
            {"comparison": "b", "exact_two_sided_p_value": 0.04},
        ]
    )
    assert adjusted[0]["holm_adjusted_p_value"] == 0.02
    assert adjusted[1]["holm_adjusted_p_value"] == 0.04


def test_holm_keeps_same_comparison_label_separate_across_families():
    adjusted = holm_adjust(
        [
            {
                "comparison": "M4_vs_M0",
                "family": "attempt",
                "exact_two_sided_p_value": 0.001,
            },
            {
                "comparison": "M4_vs_M0",
                "family": "support",
                "exact_two_sided_p_value": 0.04,
            },
        ]
    )
    assert adjusted[0]["holm_adjusted_p_value"] == 0.002
    assert adjusted[1]["holm_adjusted_p_value"] == 0.04


def test_support_confusion_is_computed_from_exact_invalidated_sets():
    row = {
        "unit_id": "T3|S3|0",
        "task_id": "T3",
        "condition": "S3",
        "seed": 0,
        "arm": "M4_ASCEL_FULL",
        "chain_success": True,
        "child_process_failure": False,
        "support_expected_invalidated": ["LEFT"],
        "support_actual_invalidated": ["LEFT"],
        "target_truth_at_decision": True,
        "attempt_attributed_success": False,
        "alternative_support_survived": True,
        "discharged_support_false_invalidation": None,
        "branch_locality_correct": None,
        "task_advance_correctness": True,
        "effect_truth_recognition": True,
        "false_skill_credit": False,
        "missed_incidental_success": False,
        "premature_advance": False,
        "retry_count": 0,
        "reobserve_count": 0,
        "rollback_count": 0,
        "recovery_success": False,
        "action_steps": 1,
        "decision_latency_ns": 1,
        "event_processing_time_ns": 1,
        "ledger_size": 1,
        "proof_graph_size": 1,
    }
    summary = summarize_arm([row], "M4_ASCEL_FULL")
    assert summary["cascade_invalidation_precision"] == 1.0
    assert summary["cascade_invalidation_recall"] == 1.0
    assert summary["over_invalidation_rate"] == 0.0
    assert summary["under_invalidation_rate"] == 0.0

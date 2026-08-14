from r16p19.phase3_analysis import cluster_bootstrap, exact_mcnemar, holm_adjust


def _rows():
    values = {
        "demo_40": [(1, 0), (1, 1)],
        "demo_41": [(0, 0), (1, 0)],
        "demo_42": [(1, 1), (0, 1)],
    }
    rows = []
    for episode, pairs in values.items():
        for index, (primary, control) in enumerate(pairs):
            for arm, success in (
                ("B6_FULL", primary),
                ("POSTCHECK_RECOVERY", control),
            ):
                rows.append(
                    {
                        "chain_id": "chain-%d" % index,
                        "source_episode": episode,
                        "condition": "C1",
                        "arm": arm,
                        "chain_success": bool(success),
                        "failure_type": None,
                    }
                )
    return rows


def test_cluster_bootstrap_is_episode_clustered_and_reproducible():
    first = cluster_bootstrap(_rows(), "POSTCHECK_RECOVERY", repetitions=500, seed=1619)
    second = cluster_bootstrap(_rows(), "POSTCHECK_RECOVERY", repetitions=500, seed=1619)
    assert first["cluster_count"] == 3
    assert first["paired_unit_count"] == 6
    assert first["draw_sha256"] == second["draw_sha256"]
    assert first["observed_difference"] == 1 / 6


def test_exact_mcnemar_and_holm_are_bounded_and_monotone():
    test = exact_mcnemar(_rows(), "POSTCHECK_RECOVERY")
    assert 0.0 <= test["exact_two_sided_p_value"] <= 1.0
    adjusted = holm_adjust(
        [
            {"comparison": "a", "exact_two_sided_p_value": 0.01},
            {"comparison": "b", "exact_two_sided_p_value": 0.04},
            {"comparison": "c", "exact_two_sided_p_value": 0.20},
        ]
    )
    by_name = {row["comparison"]: row for row in adjusted}
    assert by_name["a"]["holm_adjusted_p_value"] == 0.03
    assert by_name["b"]["holm_adjusted_p_value"] == 0.08
    assert by_name["c"]["holm_adjusted_p_value"] == 0.20

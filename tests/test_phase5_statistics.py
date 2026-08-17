from r16p19.phase5_analysis import paired_cluster_bootstrap


def test_paired_bootstrap_is_clustered_and_directional():
    rows = []
    for cluster in range(6):
        rows += [
            {"cluster_id": str(cluster), "arm": "left", "task_success": True},
            {"cluster_id": str(cluster), "arm": "right", "task_success": cluster == 0},
        ]
    result = paired_cluster_bootstrap(rows, "left", "right", replicates=1000)
    assert result["clusters"] == 6
    assert result["risk_difference"] > 0
    assert result["ci95"][0] >= 0

from r16p19.phase5_bounded_benchmark import run


def test_latency_benchmark_reports_scaling_gate():
    result = run(events=1200, attempts=120)
    assert result["exact_reference_mismatches"] == 0
    assert "p99_scaling_ratio_le_1_2" in result["gates"]

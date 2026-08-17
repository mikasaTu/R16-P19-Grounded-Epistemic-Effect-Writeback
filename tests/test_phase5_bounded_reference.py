from r16p19.phase5_bounded_benchmark import run


def test_bounded_matches_reference():
    result = run(events=5000, attempts=500)
    assert result["exact_reference_mismatches"] == 0
    assert result["audit_chain_breaks"] == 0
    assert result["hot_memory_mb"] < 10.0

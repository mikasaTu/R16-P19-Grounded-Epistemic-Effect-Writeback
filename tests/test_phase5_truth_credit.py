from r16p19.phase5_arm_kernel import evaluate_arm


def test_external_truth_does_not_credit_active_ascel_attempt():
    row = evaluate_arm("M3_ASCEL_CORE", "A5_EXTERNAL_REALIZATION", "truth-credit", 1)
    assert row["verified"]
    assert not row["credited"]

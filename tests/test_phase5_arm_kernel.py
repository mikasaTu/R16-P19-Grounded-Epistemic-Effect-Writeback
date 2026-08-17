from r16p19.phase5_arm_kernel import ARMS, CONDITIONS, evaluate_arm


def test_arm_kernel_runs_full_contract():
    rows = [evaluate_arm(arm, condition, "u", 1) for arm in ARMS for condition in CONDITIONS]
    assert len(rows) == 35
    assert all(row["event_count"] >= 3 for row in rows)


def test_core_splits_external_truth_from_attempt_credit():
    row = evaluate_arm("M3_ASCEL_CORE", "A5_EXTERNAL_REALIZATION", "external", 2)
    assert row["verified"]
    assert not row["credited"]

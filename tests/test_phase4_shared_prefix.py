import pytest


pytest.importorskip("mujoco")

from r16p19.phase4_arms import MAIN_ARMS, PROTECTED_B6_SHA256, protected_b6_sha256
from r16p19.phase4_fork_runner import run_paired_unit


def test_all_arms_inherit_exact_prefix_and_forced_terminal_state():
    rows, audit = run_paired_unit(
        "T1_CARRY_RELEASE", "C0", 400000, MAIN_ARMS, forced_identical=True
    )
    assert len(rows) == len(MAIN_ARMS)
    assert audit["pass"] is True
    assert audit["forced_identical_terminal_state_identity"] is True
    assert all(row["pre_decision_identity_pass"] for row in rows)


def test_original_b6_bytes_remain_protected():
    assert protected_b6_sha256() == PROTECTED_B6_SHA256

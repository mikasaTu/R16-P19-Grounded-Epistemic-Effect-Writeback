from r16p19.phase5_pair_runner import audit_paired, run_paired
from r16p19.phase5_snapshot import SharedPrefixSnapshot


def test_all_arms_restore_same_prefix():
    snapshot = SharedPrefixSnapshot.capture(physics_state={"q": [1]}, controller_state={"i": 2}, policy_cache={}, numpy_rng=(1, 2), python_rng=(3, 4), observation=b"rgb", event_prefix=["e"], action_prefix=[1.0], terminal_state=False)
    rows = run_paired(snapshot, ["M0", "M1", "M2", "M3", "M4"], lambda arm, state: {"arm": arm, "q": state["physics_state"]["q"]})
    assert audit_paired(rows)["exact"]

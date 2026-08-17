from r16p19.phase5_support_runner import _contract


def test_alternative_support_is_disjunctive_and_persistent():
    _, _, support, _, child, _ = _contract("T2_ALTERNATIVE_PHYSICAL_SUPPORT")
    clauses = support[child]
    assert len(clauses) == 2
    assert all(clause[0]["type"] == "PERSISTENT" for clause in clauses)

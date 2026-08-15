from r16p19.phase4_support_graph import SupportProofGraph
from r16p19.phase4_types import SupportReference, SupportValidityType


def _ref(proof_id, kind, until=None):
    return SupportReference(
        parent_proof_id=proof_id,
        validity_type=kind,
        until_effect_id=until,
    )


def test_alternative_support_preserves_dependent_until_all_clauses_fail():
    graph = SupportProofGraph()
    graph.add_root_proof("LEFT", "left-proof")
    graph.add_root_proof("RIGHT", "right-proof")
    graph.add_supported_proof(
        "ELEVATED",
        "elevated-proof",
        [
            [_ref("left-proof", SupportValidityType.PERSISTENT)],
            [_ref("right-proof", SupportValidityType.PERSISTENT)],
        ],
    )

    assert graph.invalidate_proof("left-proof", "left-false") == ["left-proof"]
    assert graph.effect_valid("ELEVATED")
    invalidated = graph.invalidate_proof("right-proof", "right-false")
    assert set(invalidated) == {"right-proof", "elevated-proof"}
    assert not graph.effect_valid("ELEVATED")


def test_until_effect_support_is_discharged_at_contract_endpoint():
    graph = SupportProofGraph()
    graph.add_root_proof("GRASPED", "grasp-proof")
    graph.add_supported_proof(
        "LIFTED",
        "lift-proof",
        [[
            _ref(
                "grasp-proof",
                SupportValidityType.UNTIL_EFFECT_REALIZED,
                "RELEASED",
            )
        ]],
    )
    graph.realize_effect("RELEASED", "release-event")
    graph.invalidate_proof("grasp-proof", "grasp-false")
    assert graph.effect_valid("LIFTED")
    snapshot = graph.snapshot()
    assert snapshot["discharge_events"][0]["effect_id"] == "RELEASED"
    assert snapshot["clauses"][0]["references"][0]["discharged"] is True

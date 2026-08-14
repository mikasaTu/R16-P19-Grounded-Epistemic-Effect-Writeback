from pathlib import Path

from r16p19.ontology import load_ontology
from r16p19.phase3_baselines import make_phase3_arm
from r16p19.phase3_event_broker import Phase3EventBroker
from r16p19.types import Decision


EFFECTS = (
    "STOVE_TURNED_ON",
    "MOKA_GRASPED",
    "MOKA_ON_STOVE",
    "MOKA_RELEASED_ON_STOVE",
)


def _digests(value):
    return {name: (value + str(index)).ljust(64, "0")[:64] for index, name in enumerate(("agentview", "robot0_eye_in_hand", "effect_witness", "contradiction_sensor"))}


def test_persistence_counts_positive_decision_ticks_not_sensor_events():
    arm = make_phase3_arm("PERSISTENCE_RECOVERY", "stove_moka", load_ontology(), EFFECTS, 2)
    broker = Phase3EventBroker("unit", "demo_30")
    effect = EFFECTS[0]
    arm.process(broker.request(effect, 0), effect)
    arm.process(broker.command(effect, 0), effect)
    decisions = [arm.process(event, effect) for event in broker.positive_receipts(effect, 0, _digests("a"))]
    assert decisions[-1] == Decision.REOBSERVE
    decisions = [arm.process(event, effect) for event in broker.positive_receipts(effect, 1, _digests("b"))]
    assert decisions[-1] == Decision.ADVANCE_TO_NEXT_SUBTASK


def test_typed_matched_needs_two_sources_but_not_command_parent_on_witness():
    arm = make_phase3_arm("TYPED_MATCHED_RECOVERY", "stove_moka", load_ontology(), EFFECTS, 2)
    broker = Phase3EventBroker("typed", "demo_30")
    effect = EFFECTS[0]
    arm.process(broker.request(effect, 0), effect)
    arm.process(broker.command(effect, 0), effect)
    events = broker.positive_receipts(effect, 1, _digests("c"), witness_linked=False)
    decision = Decision.REOBSERVE
    for event in events:
        decision = arm.process(event, effect)
    assert decision == Decision.ADVANCE_TO_NEXT_SUBTASK


def test_b6_source_is_unmodified_and_ablation_is_a_wrapper():
    import hashlib

    source = Path(__file__).resolve().parents[1] / "r16p19" / "memory.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5"
    arm = make_phase3_arm("B6_NO_INVALIDATION", "stove_moka", load_ontology(), EFFECTS, 2)
    assert arm.public_name == "B6_NO_INVALIDATION"
    assert arm.memory.arm == "B6"

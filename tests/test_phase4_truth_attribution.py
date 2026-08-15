from r16p19.phase4_arms import make_phase4_arm
from r16p19.phase4_event_broker import Phase4EventBroker
from r16p19.types import Decision


def _arm(name):
    return make_phase4_arm(name, "UNIT", ("E",), {"E": ()}, {"E": []}, "episode")


def test_external_truth_advances_without_current_skill_credit():
    arm = _arm("M4_ASCEL_FULL")
    broker = Phase4EventBroker("external", 101)
    request, command = broker.prefix("E")
    arm.process(request)
    arm.process(command)
    decision = arm.process(broker.external_realization("E"))

    assert decision == Decision.ADVANCE_TO_NEXT_SUBTASK
    assert arm.effect_fact_verified("E")
    assert not arm.attempt_attributed_success("E")


def test_merging_truth_and_attribution_creates_false_skill_credit():
    arm = _arm("NO_ATTRIBUTION_SPLIT")
    broker = Phase4EventBroker("external-ablation", 101)
    request, command = broker.prefix("E")
    arm.process(request)
    arm.process(command)
    arm.process(broker.external_realization("E"))
    assert arm.effect_fact_verified("E")
    assert arm.attempt_attributed_success("E")

from r16p19.phase4_arms import make_phase4_arm
from r16p19.phase4_event_broker import Phase4EventBroker, to_frozen_event
from r16p19.phase3_baselines import StrongRecoveryBaseline


EFFECTS = ("EFFECT",)


def _arm(name="M4_ASCEL_FULL"):
    return make_phase4_arm(name, "UNIT", EFFECTS, {"EFFECT": ()}, {"EFFECT": []}, "episode")


def test_new_request_supersedes_attempt_and_isolates_evidence_scope():
    arm = _arm()
    broker = Phase4EventBroker("scope", 17)

    first = broker.request("EFFECT")
    arm.process(first)
    first_command = broker.command("EFFECT", first.attempt_id)
    arm.process(first_command)
    first_positive = broker.positive(
        "EFFECT", "sensor_a", first.attempt_id, first_command.event_id, True
    )
    arm.process(first_positive)

    second = broker.request("EFFECT")
    arm.process(second)
    second_command = broker.command("EFFECT", second.attempt_id)
    arm.process(second_command)
    arm.process(
        broker.positive(
            "EFFECT", "sensor_b", second.attempt_id, second_command.event_id, True
        )
    )
    arm.process(
        broker.witness(
            "EFFECT", second.attempt_id, second_command.event_id, True
        )
    )

    summary = arm.summary()["ledger"]
    assert summary["attempts"][0]["status"] == "SUPERSEDED"
    assert first_positive.evidence_id in summary["attempts"][0]["evidence_ids"]
    assert summary["facts"]["EFFECT"]["fact_state"] == "OBSERVED"
    assert not arm.effect_fact_verified("EFFECT")

    arm.process(
        broker.positive(
            "EFFECT", "sensor_a", second.attempt_id, second_command.event_id, True
        )
    )
    arm.process(
        broker.witness(
            "EFFECT", second.attempt_id, second_command.event_id, True
        )
    )
    assert arm.effect_fact_verified("EFFECT")
    assert arm.attempt_attributed_success("EFFECT")


def test_stale_attempt_receipt_is_retained_but_rejected():
    arm = _arm()
    broker = Phase4EventBroker("stale", 29)
    first, first_command = broker.prefix("EFFECT")
    arm.process(first)
    arm.process(first_command)
    second, second_command = broker.prefix("EFFECT")
    arm.process(second)
    arm.process(second_command)

    stale = broker.positive(
        "EFFECT", "sensor_a", first.attempt_id, first_command.event_id, True
    )
    arm.process(stale)
    audit = arm.summary()["ledger"]["audit_rows"][-1]
    assert audit["retained_for_audit"] is True
    assert audit["accepted"] is False
    assert not arm.effect_fact_verified("EFFECT")


def test_m0_reproduces_phase3_typed_matched_event_semantics():
    m0 = _arm("M0_TYPED_MATCHED")
    phase3 = StrongRecoveryBaseline(
        "TYPED_MATCHED_RECOVERY", "phase4_equivalence", EFFECTS, 2
    )
    broker = Phase4EventBroker("typed-equivalence", 71)

    first, first_command = broker.prefix("EFFECT")
    sensor_a = broker.positive(
        "EFFECT", "sensor_a", first.attempt_id, first_command.event_id, True
    )
    second, second_command = broker.prefix("EFFECT")
    sensor_b = broker.positive(
        "EFFECT", "sensor_b", second.attempt_id, second_command.event_id, False
    )
    witness = broker.witness(
        "EFFECT", second.attempt_id, second_command.event_id, False
    )
    contradiction = broker.contradiction("EFFECT", False)
    events = (
        first,
        first_command,
        sensor_a,
        second,
        second_command,
        sensor_b,
        witness,
        contradiction,
    )

    m0_decisions = [m0.process(event) for event in events]
    phase3_decisions = [
        phase3.process(to_frozen_event(event, "typed-equivalence"), "EFFECT")
        for event in events
    ]
    assert m0_decisions == phase3_decisions
    assert phase3.state["EFFECT"] == "FALSE"
    assert not m0.effect_fact_verified("EFFECT")

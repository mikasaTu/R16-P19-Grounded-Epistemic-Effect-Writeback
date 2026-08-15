from r16p19.phase4_arms import make_phase4_arm
from r16p19.phase4_event_broker import Phase4EventBroker


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

from r16p19.phase4_arms import make_phase4_arm
from r16p19.phase4_event_broker import Phase4EventBroker


def _arm(name="M4_ASCEL_FULL"):
    return make_phase4_arm(name, "UNIT", ("E",), {"E": ()}, {"E": []}, "episode")


def test_contradiction_revokes_verified_receipts_before_late_witness():
    arm = _arm()
    broker = Phase4EventBroker("revoke", 43)
    request, command = broker.prefix("E")
    arm.process(request)
    arm.process(command)
    arm.process(broker.positive("E", "a", request.attempt_id, command.event_id, True))
    arm.process(broker.positive("E", "b", request.attempt_id, command.event_id, True))
    late_witness = broker.witness("E", request.attempt_id, command.event_id, False)

    arm.process(broker.contradiction("E", False))
    arm.process(late_witness)

    ledger = arm.summary()["ledger"]
    assert ledger["facts"]["E"]["fact_state"] == "REVOKED"
    assert ledger["attempts"][0]["status"] == "INVALIDATED"
    assert ledger["attempts"][0]["evidence_ids"] == []
    assert ledger["counters"]["post_revocation_witness_rejected"] == 1
    assert not arm.effect_fact_verified("E")


def test_no_pre_realization_revocation_ablation_accepts_old_verified_witness():
    arm = _arm("NO_PRE_REALIZATION_REVOCATION")
    broker = Phase4EventBroker("no-revoke", 43)
    request, command = broker.prefix("E")
    arm.process(request)
    arm.process(command)
    arm.process(broker.positive("E", "a", request.attempt_id, command.event_id, True))
    arm.process(broker.positive("E", "b", request.attempt_id, command.event_id, True))
    arm.process(broker.contradiction("E", False))
    arm.process(broker.witness("E", request.attempt_id, command.event_id, False))
    assert arm.effect_fact_verified("E")

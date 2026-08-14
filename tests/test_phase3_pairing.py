from r16p19.phase3_runner import paired_unit_audit


def _row(arm, decisions):
    traces = []
    for index, decision in enumerate(decisions):
        traces.append(
            {
                "decision": decision,
                "event_prefix_sha256": "event-%d" % index,
                "simulator_state_sha256": "state-%d" % index,
                "simulator_state": [float(index)],
                "effect_id": "E",
                "action_trace_length": index + 1,
                "event_record_stop_exclusive": index + 1,
            }
        )
    return {
        "chain_id": "chain",
        "source_episode": "demo_40",
        "condition": "C1",
        "arm": arm,
        "decision_trace": traces,
        "action_trace": [
            {"executed_action_sha256": "action-%d" % index}
            for index in range(len(decisions))
        ],
        "event_records": [{"event": index} for index in range(len(decisions))],
    }


def test_pairing_finds_first_divergence_and_verifies_identical_prefix():
    rows = [
        _row("B3_MONOLITHIC", ["REOBSERVE", "ADVANCE_TO_NEXT_SUBTASK"]),
        _row("POSTCHECK_RECOVERY", ["REOBSERVE", "RETRY_CURRENT_EFFECT"]),
        _row("PERSISTENCE_RECOVERY", ["REOBSERVE", "RETRY_CURRENT_EFFECT"]),
        _row("TYPED_MATCHED_RECOVERY", ["REOBSERVE", "RETRY_CURRENT_EFFECT"]),
        _row("B6_FULL", ["REOBSERVE", "ROLLBACK_OR_REPLAN"]),
    ]
    audit = paired_unit_audit(rows)[0]
    assert audit["first_decision_divergence_index"] == 1
    assert audit["paired_prefix_event_and_action_bytes_identical"] is True
    assert audit["unique_decisions"] == [
        "ADVANCE_TO_NEXT_SUBTASK",
        "RETRY_CURRENT_EFFECT",
        "ROLLBACK_OR_REPLAN",
    ]


def test_pairing_rejects_state_drift_before_decision_divergence():
    rows = [
        _row("B3_MONOLITHIC", ["ADVANCE_TO_NEXT_SUBTASK"]),
        _row("POSTCHECK_RECOVERY", ["RETRY_CURRENT_EFFECT"]),
        _row("PERSISTENCE_RECOVERY", ["RETRY_CURRENT_EFFECT"]),
        _row("TYPED_MATCHED_RECOVERY", ["RETRY_CURRENT_EFFECT"]),
        _row("B6_FULL", ["ROLLBACK_OR_REPLAN"]),
    ]
    rows[0]["decision_trace"][0]["simulator_state_sha256"] = "drift"
    assert paired_unit_audit(rows)[0]["paired_prefix_event_and_action_bytes_identical"] is False

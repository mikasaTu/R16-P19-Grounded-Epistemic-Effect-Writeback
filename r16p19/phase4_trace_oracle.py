"""Trace oracle and exact 10,000-schedule ASCEL mechanism gate."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .phase4_arms import make_phase4_arm
from .phase4_event_broker import Phase4EventBroker
from .phase4_microenv import TASK_CONTRACTS
from .phase4_trace_generator import TraceSchedule


def _make_arm(task_id: str, unit_id: str):
    contract = TASK_CONTRACTS[task_id]
    return make_phase4_arm(
        "M4_ASCEL_FULL",
        task_id,
        contract.effects,
        contract.dependencies,
        contract.support_contract,
        unit_id,
    )


def _realize(arm, broker: Phase4EventBroker, effect_id: str) -> str:
    request = broker.request(effect_id)
    arm.process(request)
    command = broker.command(effect_id, request.attempt_id)
    arm.process(command)
    arm.process(
        broker.positive(
            effect_id,
            "sensor_a",
            request.attempt_id,
            command.event_id,
            physical_truth=True,
        )
    )
    arm.process(
        broker.positive(
            effect_id,
            "sensor_b",
            request.attempt_id,
            command.event_id,
            physical_truth=True,
        )
    )
    witness = broker.witness(
        effect_id,
        request.attempt_id,
        command.event_id,
        physical_truth=True,
    )
    arm.process(witness)
    summary = arm.summary()
    proof_ids = summary["ledger"]["facts"][effect_id]["realization_proof_ids"]
    return str(proof_ids[-1])


def _attempt_trace(schedule: TraceSchedule) -> dict:
    contract = TASK_CONTRACTS["T1_CARRY_RELEASE"]
    effect = "GRASPED"
    arm = _make_arm(contract.task_id, schedule.schedule_id)
    broker = Phase4EventBroker(schedule.schedule_id, schedule.seed)
    family = schedule.family[:2]
    result = {
        "stale_evidence_accepted": False,
        "cross_attempt_verification": False,
        "superseded_witness_realization": False,
        "post_revocation_late_witness_realization": False,
        "incidental_effect_truth_recognition": None,
        "incidental_current_skill_credit": None,
    }
    first_request = broker.request(effect)
    arm.process(first_request)
    first_command = broker.command(effect, first_request.attempt_id)
    arm.process(first_command)
    if family in ("A1", "A3", "A4"):
        for sensor in schedule.sensor_order:
            receipt = broker.positive(
                effect,
                sensor,
                first_request.attempt_id,
                first_command.event_id,
                physical_truth=True,
            )
            arm.process(receipt)
            if schedule.duplicate_first_receipt and sensor == schedule.sensor_order[0]:
                # Re-deliver the exact canonical event.  The broker stream records
                # the generated receipt once; duplicate delivery is a ledger
                # idempotence adversary, not a second physical observation.
                arm.process(receipt)
    elif family == "A2":
        receipt = broker.positive(
            effect,
            schedule.sensor_order[0],
            first_request.attempt_id,
            first_command.event_id,
            physical_truth=True,
        )
        arm.process(receipt)
        if schedule.duplicate_first_receipt:
            arm.process(receipt)
    if family == "A1":
        second_request = broker.request(effect)
        arm.process(second_request)
        second_command = broker.command(effect, second_request.attempt_id)
        arm.process(second_command)
        arm.process(
            broker.witness(
                effect,
                first_request.attempt_id,
                first_command.event_id,
                physical_truth=False,
            )
        )
        result["stale_evidence_accepted"] = arm.effect_fact_verified(effect)
    elif family == "A2":
        second_request = broker.request(effect)
        arm.process(second_request)
        second_command = broker.command(effect, second_request.attempt_id)
        arm.process(second_command)
        arm.process(
            broker.positive(
                effect,
                schedule.sensor_order[1],
                second_request.attempt_id,
                second_command.event_id,
                physical_truth=False,
            )
        )
        state = arm.summary()["ledger"]["facts"][effect]["fact_state"]
        result["cross_attempt_verification"] = state in ("VERIFIED", "REALIZED")
    elif family == "A3":
        second_request = broker.request(effect)
        arm.process(second_request)
        second_command = broker.command(effect, second_request.attempt_id)
        arm.process(second_command)
        arm.process(
            broker.witness(
                effect,
                first_request.attempt_id,
                first_command.event_id,
                physical_truth=False,
            )
        )
        result["superseded_witness_realization"] = arm.effect_fact_verified(effect)
    elif family == "A4":
        arm.process(broker.contradiction(effect, physical_truth=False))
        arm.process(
            broker.witness(
                effect,
                first_request.attempt_id,
                first_command.event_id,
                physical_truth=False,
            )
        )
        result["post_revocation_late_witness_realization"] = arm.effect_fact_verified(effect)
    elif family == "A5":
        arm.process(broker.external_realization(effect))
        result["incidental_effect_truth_recognition"] = arm.effect_fact_verified(effect)
        result["incidental_current_skill_credit"] = arm.attempt_attributed_success(effect)
    return {
        "schedule": schedule.to_dict(),
        "family": schedule.family,
        "result": result,
        "event_stream_sha256": broker.stream_hash(),
        "event_count": len(broker.records),
        "ledger_counters": arm.summary()["ledger"]["counters"],
        "pass": not any(
            bool(result[key])
            for key in (
                "stale_evidence_accepted",
                "cross_attempt_verification",
                "superseded_witness_realization",
                "post_revocation_late_witness_realization",
                "incidental_current_skill_credit",
            )
            if result[key] is not None
        ) and result["incidental_effect_truth_recognition"] is not False,
    }


def _support_trace(schedule: TraceSchedule) -> dict:
    family = schedule.family[:2]
    if family == "S1" or family == "S5":
        task_id = "T2_PERSISTENT_SUPPORT"
    elif family == "S2":
        task_id = "T1_CARRY_RELEASE"
    else:
        task_id = "T3_ALTERNATIVE_SUPPORT"
    contract = TASK_CONTRACTS[task_id]
    arm = _make_arm(task_id, schedule.schedule_id)
    broker = Phase4EventBroker(schedule.schedule_id, schedule.seed)
    initially_valid: Set[str] = set()
    expected_invalidated: Set[str] = set()
    late_dependent_realized = False
    if family == "S1":
        for effect in contract.chain_effects:
            _realize(arm, broker, effect)
            initially_valid.add(effect)
        expected_invalidated = set(contract.chain_effects)
        arm.process(broker.contradiction(contract.support_root, False))
    elif family == "S2":
        for effect in contract.chain_effects:
            _realize(arm, broker, effect)
            initially_valid.add(effect)
        expected_invalidated = {contract.support_root}
        arm.process(broker.contradiction(contract.support_root, False))
    elif family == "S3":
        for effect in contract.chain_effects:
            _realize(arm, broker, effect)
            initially_valid.add(effect)
        expected_invalidated = {"LEFT_SUPPORT"}
        arm.process(broker.contradiction("LEFT_SUPPORT", False))
    elif family == "S4":
        for effect in list(contract.chain_effects) + [contract.unrelated_effect]:
            _realize(arm, broker, effect)
            initially_valid.add(effect)
        expected_invalidated = {
            "LEFT_SUPPORT",
            "RIGHT_SUPPORT",
            "OBJECT_ELEVATED",
            "TARGET_REACHED",
        }
        arm.process(broker.contradiction("LEFT_SUPPORT", False))
        arm.process(broker.contradiction("RIGHT_SUPPORT", False))
    elif family == "S5":
        for effect in ("SUPPORT_PRESENT", "OBJECT_STABLE"):
            _realize(arm, broker, effect)
            initially_valid.add(effect)
        expected_invalidated = {"SUPPORT_PRESENT", "OBJECT_STABLE"}
        arm.process(broker.contradiction("SUPPORT_PRESENT", False))
        _realize(arm, broker, "MARKER_PLACED")
        late_dependent_realized = arm.effect_fact_verified("MARKER_PLACED")
    actual_invalidated = {
        effect for effect in initially_valid if not arm.effect_fact_verified(effect)
    }
    false_invalidations = actual_invalidated - expected_invalidated
    missed_invalidations = expected_invalidated - actual_invalidated
    result = {
        "expected_invalidated": sorted(expected_invalidated),
        "actual_invalidated": sorted(actual_invalidated),
        "false_invalidations": sorted(false_invalidations),
        "missed_invalidations": sorted(missed_invalidations),
        "late_dependent_witness_realized": late_dependent_realized,
        "unrelated_branch_valid": (
            arm.effect_fact_verified(contract.unrelated_effect)
            if contract.unrelated_effect
            else None
        ),
    }
    return {
        "schedule": schedule.to_dict(),
        "family": schedule.family,
        "result": result,
        "event_stream_sha256": broker.stream_hash(),
        "event_count": len(broker.records),
        "support_graph": arm.summary()["support_graph"],
        "pass": not false_invalidations
        and not missed_invalidations
        and not late_dependent_realized,
    }


def execute_trace_schedule(schedule: TraceSchedule) -> dict:
    if schedule.family.startswith("A"):
        return _attempt_trace(schedule)
    return _support_trace(schedule)


def summarize_trace_results(rows: Sequence[dict]) -> dict:
    by_family = Counter(row["family"] for row in rows)
    pass_by_family = Counter(row["family"] for row in rows if row["pass"])
    attempt_rows = [row for row in rows if row["family"].startswith("A")]
    support_rows = [row for row in rows if row["family"].startswith("S")]
    stale = sum(row["result"]["stale_evidence_accepted"] for row in attempt_rows)
    cross = sum(row["result"]["cross_attempt_verification"] for row in attempt_rows)
    superseded = sum(row["result"]["superseded_witness_realization"] for row in attempt_rows)
    late = sum(row["result"]["post_revocation_late_witness_realization"] for row in attempt_rows)
    incidental_rows = [
        row for row in attempt_rows if row["family"].startswith("A5")
    ]
    expected_total = sum(
        len(row["result"]["expected_invalidated"]) for row in support_rows
    )
    actual_total = sum(
        len(row["result"]["actual_invalidated"]) for row in support_rows
    )
    true_positive = sum(
        len(
            set(row["result"]["expected_invalidated"])
            & set(row["result"]["actual_invalidated"])
        )
        for row in support_rows
    )
    false_positive = sum(
        len(row["result"]["false_invalidations"]) for row in support_rows
    )
    false_negative = sum(
        len(row["result"]["missed_invalidations"]) for row in support_rows
    )
    precision = true_positive / float(true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / float(true_positive + false_negative) if true_positive + false_negative else 1.0
    metrics = {
        "stale_evidence_accepted": stale,
        "cross_attempt_verification": cross,
        "superseded_witness_realization": superseded,
        "post_revocation_late_witness_realization": late,
        "cascade_invalidation_precision": precision,
        "cascade_invalidation_recall": recall,
        "over_invalidation": false_positive,
        "under_invalidation": false_negative,
        "incidental_effect_truth_recognition": (
            sum(row["result"]["incidental_effect_truth_recognition"] for row in incidental_rows)
            / float(len(incidental_rows))
        ),
        "incidental_current_skill_credit": sum(
            row["result"]["incidental_current_skill_credit"] for row in incidental_rows
        ),
    }
    gates = {
        "stale_evidence_accepted_zero": stale == 0,
        "cross_attempt_verification_zero": cross == 0,
        "superseded_witness_realization_zero": superseded == 0,
        "post_revocation_late_witness_realization_zero": late == 0,
        "cascade_invalidation_precision_eq_1": precision == 1.0,
        "cascade_invalidation_recall_eq_1": recall == 1.0,
        "over_invalidation_zero": false_positive == 0,
        "under_invalidation_zero": false_negative == 0,
        "incidental_effect_truth_recognition_eq_1": metrics["incidental_effect_truth_recognition"] == 1.0,
        "incidental_current_skill_credit_zero": metrics["incidental_current_skill_credit"] == 0,
    }
    return {
        "schema_version": 1,
        "schedule_count": len(rows),
        "family_counts": dict(by_family),
        "family_pass_counts": dict(pass_by_family),
        "metrics": metrics,
        "confusion_counts": {
            "expected_invalidation_total": expected_total,
            "actual_invalidation_total": actual_total,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "gates": gates,
        "pass": all(gates.values()),
        "status": (
            "TRACE_GATE_PASS" if all(gates.values()) else "BLOCKED_BY_MECHANISM_IMPLEMENTATION"
        ),
    }


def run_trace_gate(schedules: Iterable[TraceSchedule]) -> Tuple[List[dict], dict]:
    rows = [execute_trace_schedule(schedule) for schedule in schedules]
    return rows, summarize_trace_results(rows)

"""Linux ``os.fork`` shared-prefix runner for Phase-4 paired experiments."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .phase4_arms import ABLATION_ARMS, MAIN_ARMS, make_phase4_arm, protected_b6_sha256
from .phase4_event_broker import Phase4EventBroker
from .phase4_executor import ExecutionMode, MacroSkillExecutor
from .phase4_microenv import Phase4MicroEnv, TASK_CONTRACTS
from .phase4_types import LedgerEvent
from .types import Decision


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


ATTEMPT_CONDITIONS = ("A1", "A2", "A3", "A4")
SUPPORT_CONDITIONS = ("S1", "S2", "S3", "S4")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rng_state_sha256(env: Phase4MicroEnv) -> str:
    payload = {
        "python": _hash_bytes(pickle.dumps(random.getstate(), protocol=4)),
        "numpy_legacy": _hash_bytes(pickle.dumps(np.random.get_state(), protocol=4)),
        "numpy_generator": env.rng.bit_generator.state,
    }
    return _hash_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


@dataclass
class PreparedPrefix:
    unit_id: str
    task_id: str
    condition: str
    seed: int
    env: Phase4MicroEnv
    executor: MacroSkillExecutor
    broker: Phase4EventBroker
    target_effect: str
    target_attempt_id: Optional[str]
    target_command_id: Optional[str]
    python_rng_state: object
    numpy_rng_state: tuple
    hashes: Dict[str, str]


def _broker_clean_realization(
    broker: Phase4EventBroker,
    effect_id: str,
) -> Tuple[str, str]:
    request = broker.request(effect_id)
    command = broker.command(effect_id, request.attempt_id)
    broker.positive(
        effect_id,
        "sensor_a",
        request.attempt_id,
        command.event_id,
        physical_truth=True,
    )
    broker.positive(
        effect_id,
        "sensor_b",
        request.attempt_id,
        command.event_id,
        physical_truth=True,
    )
    broker.witness(
        effect_id,
        request.attempt_id,
        command.event_id,
        physical_truth=True,
    )
    return str(request.attempt_id), command.event_id


def _prepare_prefix(task_id: str, condition: str, seed: int, unit_id: str) -> PreparedPrefix:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    contract = TASK_CONTRACTS[task_id]
    env = Phase4MicroEnv(task_id, int(seed))
    executor = MacroSkillExecutor(env)
    broker = Phase4EventBroker(unit_id, int(seed))
    target = contract.attempt_target
    target_attempt = None
    target_command = None
    if condition in ("C0", "A1", "A2", "A3", "A4", "A5"):
        target_index = contract.chain_effects.index(target)
        for effect_id in contract.chain_effects[:target_index]:
            executor.execute(task_id, effect_id, ExecutionMode.EXECUTE)
            _broker_clean_realization(broker, effect_id)
        request = broker.request(target)
        command = broker.command(target, request.attempt_id)
        target_attempt = request.attempt_id
        target_command = command.event_id
    elif condition in SUPPORT_CONDITIONS:
        for effect_id in contract.chain_effects:
            executor.execute(task_id, effect_id, ExecutionMode.EXECUTE)
            _broker_clean_realization(broker, effect_id)
        if condition == "S4" and contract.unrelated_effect:
            executor.execute(task_id, contract.unrelated_effect, ExecutionMode.EXECUTE)
            _broker_clean_realization(broker, contract.unrelated_effect)
    else:
        raise ValueError("unknown Phase-4 condition %s" % condition)
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    hashes = {
        "physical_state": env.state_sha256(),
        "controller_state": env.controller_sha256(),
        "rng_state": _rng_state_sha256(env),
        "event_prefix": broker.stream_hash(),
        "action_prefix": env.action_prefix_sha256(),
    }
    return PreparedPrefix(
        unit_id=unit_id,
        task_id=task_id,
        condition=condition,
        seed=int(seed),
        env=env,
        executor=executor,
        broker=broker,
        target_effect=target,
        target_attempt_id=target_attempt,
        target_command_id=target_command,
        python_rng_state=python_rng_state,
        numpy_rng_state=numpy_rng_state,
        hashes=hashes,
    )


def _process(arm, event: LedgerEvent, timings: List[int]) -> Decision:
    started = time.perf_counter_ns()
    decision = arm.process(event)
    timings.append(time.perf_counter_ns() - started)
    return decision


def _replay_prefix(arm, broker: Phase4EventBroker, timings: List[int]) -> None:
    for event in broker.records:
        _process(arm, event, timings)


def _clean_effect(
    prepared: PreparedPrefix,
    arm,
    effect_id: str,
    timings: List[int],
    mode: ExecutionMode = ExecutionMode.EXECUTE,
) -> Tuple[Decision, int]:
    request = prepared.broker.request(effect_id)
    _process(arm, request, timings)
    command = prepared.broker.command(effect_id, request.attempt_id)
    _process(arm, command, timings)
    execution = prepared.executor.execute(prepared.task_id, effect_id, mode)
    _process(
        arm,
        prepared.broker.positive(
            effect_id,
            "sensor_a",
            request.attempt_id,
            command.event_id,
            physical_truth=True,
        ),
        timings,
    )
    _process(
        arm,
        prepared.broker.positive(
            effect_id,
            "sensor_b",
            request.attempt_id,
            command.event_id,
            physical_truth=True,
        ),
        timings,
    )
    decision = _process(
        arm,
        prepared.broker.witness(
            effect_id,
            request.attempt_id,
            command.event_id,
            physical_truth=True,
        ),
        timings,
    )
    return decision, int(execution["action_steps"])


def _finish_chain(
    prepared: PreparedPrefix,
    arm,
    timings: List[int],
    skip_effects: Iterable[str] = (),
) -> Tuple[bool, int]:
    skip = set(skip_effects)
    added_steps = 0
    for effect_id in prepared.env.contract.chain_effects:
        if effect_id in skip:
            continue
        if prepared.env.effect_truth(effect_id) and arm.effect_fact_verified(effect_id):
            continue
        decision, steps = _clean_effect(
            prepared, arm, effect_id, timings, ExecutionMode.EXECUTE
        )
        added_steps += steps
        if decision != Decision.ADVANCE_TO_NEXT_SUBTASK:
            return False, added_steps
    return prepared.env.all_chain_truth(), added_steps


def _compact_arm_metrics(arm) -> dict:
    summary = arm.summary()
    if "ledger" in summary:
        ledger = summary["ledger"]
        support = summary["support_graph"]
        proof_count = len(support.get("proofs", []))
        clause_count = len(support.get("clauses", []))
        counters = ledger.get("counters", {})
        return {
            "ledger_size": int(ledger.get("ledger_size", 0)),
            "proof_graph_size": int(proof_count + clause_count),
            "attempt_leakage_count": int(
                counters.get("stale_evidence_accepted", 0)
                + counters.get("false_external_attributions", 0)
            ),
            "support_graph_invariant_violation_count": len(
                support.get("invariant_violations", [])
            ),
            "transition_violation_count": 0,
            "memory_summary_sha256": _hash_bytes(
                json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
        }
    return {
        "ledger_size": int(summary.get("ledger_event_count", 0)),
        "proof_graph_size": 0,
        "attempt_leakage_count": 0,
        "support_graph_invariant_violation_count": 0,
        "transition_violation_count": len(summary.get("transition_violations", [])),
        "memory_summary_sha256": _hash_bytes(
            json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def _attempt_scenario(prepared: PreparedPrefix, arm, timings: List[int]) -> dict:
    condition = prepared.condition
    effect = prepared.target_effect
    first_attempt = str(prepared.target_attempt_id)
    first_command = str(prepared.target_command_id)
    retry_count = 0
    reobserve_count = 0
    rollback_count = 0
    premature_advance = False
    stale_witness_accepted = False
    cross_attempt_verification = False
    superseded_command_realization = False
    late_witness_after_revocation = False
    if condition == "C0":
        prepared.executor.execute(prepared.task_id, effect, ExecutionMode.EXECUTE)
        _process(
            arm,
            prepared.broker.positive(
                effect, "sensor_a", first_attempt, first_command, True
            ),
            timings,
        )
        _process(
            arm,
            prepared.broker.positive(
                effect, "sensor_b", first_attempt, first_command, True
            ),
            timings,
        )
        natural = _process(
            arm,
            prepared.broker.witness(effect, first_attempt, first_command, True),
            timings,
        )
    elif condition in ("A1", "A3", "A4"):
        prepared.executor.execute(prepared.task_id, effect, ExecutionMode.EXECUTE)
        _process(
            arm,
            prepared.broker.positive(
                effect, "sensor_a", first_attempt, first_command, True
            ),
            timings,
        )
        _process(
            arm,
            prepared.broker.positive(
                effect, "sensor_b", first_attempt, first_command, True
            ),
            timings,
        )
        prepared.env.reverse_effect(effect, actor="fault_injector")
        if condition in ("A1", "A3"):
            second_request = prepared.broker.request(effect)
            _process(arm, second_request, timings)
            second_command = prepared.broker.command(effect, second_request.attempt_id)
            _process(arm, second_command, timings)
            natural = _process(
                arm,
                prepared.broker.witness(
                    effect, first_attempt, first_command, physical_truth=False
                ),
                timings,
            )
            if condition == "A1":
                stale_witness_accepted = natural == Decision.ADVANCE_TO_NEXT_SUBTASK
            else:
                superseded_command_realization = natural == Decision.ADVANCE_TO_NEXT_SUBTASK
        else:
            _process(arm, prepared.broker.contradiction(effect, False), timings)
            natural = _process(
                arm,
                prepared.broker.witness(
                    effect, first_attempt, first_command, physical_truth=False
                ),
                timings,
            )
            late_witness_after_revocation = natural == Decision.ADVANCE_TO_NEXT_SUBTASK
    elif condition == "A2":
        prepared.executor.execute(prepared.task_id, effect, ExecutionMode.EXECUTE)
        _process(
            arm,
            prepared.broker.positive(
                effect, "sensor_a", first_attempt, first_command, True
            ),
            timings,
        )
        prepared.env.reverse_effect(effect, actor="fault_injector")
        second_request = prepared.broker.request(effect)
        _process(arm, second_request, timings)
        second_command = prepared.broker.command(effect, second_request.attempt_id)
        _process(arm, second_command, timings)
        _process(
            arm,
            prepared.broker.positive(
                effect,
                "sensor_b",
                second_request.attempt_id,
                second_command.event_id,
                False,
                force_verification=True,
            ),
            timings,
        )
        natural = _process(
            arm,
            prepared.broker.witness(
                effect,
                second_request.attempt_id,
                second_command.event_id,
                physical_truth=False,
            ),
            timings,
        )
        cross_attempt_verification = natural == Decision.ADVANCE_TO_NEXT_SUBTASK
    elif condition == "A5":
        prepared.env.actuate_effect(effect, actor="external_event")
        natural = _process(arm, prepared.broker.external_realization(effect), timings)
    else:  # pragma: no cover
        raise ValueError("not an attempt scenario")
    target_truth_at_decision = prepared.env.effect_truth(effect)
    fact_recognition_at_decision = arm.effect_fact_verified(effect)
    attributed_at_decision = arm.attempt_attributed_success(effect)
    if natural == Decision.ADVANCE_TO_NEXT_SUBTASK and not target_truth_at_decision:
        premature_advance = True
        chain_success = False
        added_steps = 0
    elif condition == "A5":
        chain_success = bool(
            natural == Decision.ADVANCE_TO_NEXT_SUBTASK
            and target_truth_at_decision
            and arm.effect_fact_verified(effect)
        )
        added_steps = 0
        if chain_success:
            chain_success, added_steps = _finish_chain(
                prepared, arm, timings, skip_effects=(effect,)
            )
    else:
        if natural != Decision.ADVANCE_TO_NEXT_SUBTASK:
            reobserve_count += int(natural == Decision.REOBSERVE)
            rollback_count += int(natural == Decision.ROLLBACK_OR_REPLAN)
            retry_count += 1
            recovery_decision, recovery_steps = _clean_effect(
                prepared, arm, effect, timings, ExecutionMode.RETRY
            )
            added_steps = recovery_steps
            recovered = recovery_decision == Decision.ADVANCE_TO_NEXT_SUBTASK
        else:
            added_steps = 0
            recovered = target_truth_at_decision
        if recovered:
            suffix_success, suffix_steps = _finish_chain(
                prepared, arm, timings, skip_effects=(effect,)
            )
            added_steps += suffix_steps
            chain_success = suffix_success
        else:
            chain_success = False
    final_target_truth = prepared.env.effect_truth(effect)
    final_fact_recognition = arm.effect_fact_verified(effect)
    final_attributed = arm.attempt_attributed_success(effect)
    return {
        "natural_decision": natural.value,
        "chain_success": bool(chain_success),
        "target_truth_at_decision": bool(target_truth_at_decision),
        "effect_truth_recognition": bool(fact_recognition_at_decision),
        "task_advance_correctness": bool(
            (natural == Decision.ADVANCE_TO_NEXT_SUBTASK) == target_truth_at_decision
        ),
        "attempt_attributed_success": bool(attributed_at_decision),
        "final_target_truth": bool(final_target_truth),
        "final_effect_truth_recognition": bool(final_fact_recognition),
        "final_attempt_attributed_success": bool(final_attributed),
        "false_skill_credit": bool(condition == "A5" and attributed_at_decision),
        "missed_incidental_success": bool(
            condition == "A5"
            and target_truth_at_decision
            and not fact_recognition_at_decision
        ),
        "stale_witness_accepted": bool(stale_witness_accepted),
        "cross_attempt_verification": bool(cross_attempt_verification),
        "superseded_command_realization": bool(superseded_command_realization),
        "late_witness_after_revocation": bool(late_witness_after_revocation),
        "premature_advance": bool(premature_advance),
        "retry_count": int(retry_count),
        "reobserve_count": int(reobserve_count),
        "rollback_count": int(rollback_count),
        "recovery_success": bool(chain_success and retry_count > 0),
        "support_expected_invalidated": [],
        "support_actual_invalidated": [],
        "over_invalidation_count": 0,
        "under_invalidation_count": 0,
        "alternative_support_survived": None,
        "discharged_support_false_invalidation": None,
        "branch_locality_correct": None,
        "added_action_steps": int(added_steps),
    }


def _support_scenario(prepared: PreparedPrefix, arm, timings: List[int]) -> dict:
    condition = prepared.condition
    contract = prepared.env.contract
    initially_valid = {
        effect for effect in contract.effects if arm.effect_fact_verified(effect)
    }
    if condition == "S1":
        roots = (contract.support_root,)
        expected_invalidated = set(contract.chain_effects)
    elif condition == "S2":
        roots = (contract.support_root,)
        expected_invalidated = {contract.support_root}
    elif condition == "S3":
        roots = ("LEFT_SUPPORT",)
        expected_invalidated = {"LEFT_SUPPORT"}
    elif condition == "S4":
        roots = ("LEFT_SUPPORT", "RIGHT_SUPPORT")
        expected_invalidated = {
            "LEFT_SUPPORT",
            "RIGHT_SUPPORT",
            "OBJECT_ELEVATED",
            "TARGET_REACHED",
        }
    else:  # pragma: no cover
        raise ValueError("not a support scenario")
    decisions = []
    for root in roots:
        prepared.env.reverse_effect(root, actor="fault_injector")
        decisions.append(
            _process(arm, prepared.broker.contradiction(root, False), timings)
        )
    actual_invalidated = {
        effect for effect in initially_valid if not arm.effect_fact_verified(effect)
    }
    over = actual_invalidated - expected_invalidated
    under = expected_invalidated - actual_invalidated
    exact = not over and not under
    unrelated_valid = (
        arm.effect_fact_verified(contract.unrelated_effect)
        if condition == "S4" and contract.unrelated_effect
        else True
    )
    alternative_survived = (
        arm.effect_fact_verified("OBJECT_ELEVATED")
        and arm.effect_fact_verified("TARGET_REACHED")
        if condition == "S3"
        else None
    )
    discharged_false = (
        not arm.effect_fact_verified(contract.final_effect)
        if condition == "S2"
        else None
    )
    branch_locality = bool(unrelated_valid) if condition == "S4" else None
    target_truth_at_decision = prepared.env.effect_truth(contract.final_effect)
    fact_recognition_at_decision = arm.effect_fact_verified(contract.final_effect)
    attributed_at_decision = arm.attempt_attributed_success(contract.final_effect)
    added_steps = 0
    if condition in ("S1", "S4") and exact and unrelated_valid:
        recovered, added_steps = _finish_chain(prepared, arm, timings)
        chain_success = bool(recovered)
    else:
        chain_success = bool(
            exact
            and unrelated_valid
            and (alternative_survived is not False)
            and (discharged_false is not True)
        )
    final_target_truth = prepared.env.effect_truth(contract.final_effect)
    final_fact_recognition = arm.effect_fact_verified(contract.final_effect)
    final_attributed = arm.attempt_attributed_success(contract.final_effect)
    return {
        "natural_decision": decisions[-1].value if decisions else Decision.REOBSERVE.value,
        "chain_success": chain_success,
        "target_truth_at_decision": target_truth_at_decision,
        "effect_truth_recognition": fact_recognition_at_decision,
        "task_advance_correctness": exact,
        "attempt_attributed_success": attributed_at_decision,
        "final_target_truth": final_target_truth,
        "final_effect_truth_recognition": final_fact_recognition,
        "final_attempt_attributed_success": final_attributed,
        "false_skill_credit": False,
        "missed_incidental_success": False,
        "stale_witness_accepted": False,
        "cross_attempt_verification": False,
        "superseded_command_realization": False,
        "late_witness_after_revocation": False,
        "premature_advance": False,
        "retry_count": int(condition in ("S1", "S4") and exact),
        "reobserve_count": 0,
        "rollback_count": sum(
            decision == Decision.ROLLBACK_OR_REPLAN for decision in decisions
        ),
        "recovery_success": bool(condition in ("S1", "S4") and chain_success),
        "support_expected_invalidated": sorted(expected_invalidated),
        "support_actual_invalidated": sorted(actual_invalidated),
        "over_invalidation_count": len(over),
        "under_invalidation_count": len(under),
        "alternative_support_survived": alternative_survived,
        "discharged_support_false_invalidation": discharged_false,
        "branch_locality_correct": branch_locality,
        "added_action_steps": int(added_steps),
    }


def _run_child(prepared: PreparedPrefix, arm_name: str, forced_identical: bool) -> dict:
    # CPython deliberately reseeds its module-global RNG after fork.  Restore
    # the recorded canonical prefix state before the child identity check.
    random.setstate(prepared.python_rng_state)
    np.random.set_state(prepared.numpy_rng_state)
    inherited_hashes = {
        "physical_state": prepared.env.state_sha256(),
        "controller_state": prepared.env.controller_sha256(),
        "rng_state": _rng_state_sha256(prepared.env),
        "event_prefix": prepared.broker.stream_hash(),
        "action_prefix": prepared.env.action_prefix_sha256(),
    }
    identity = {
        key: inherited_hashes[key] == prepared.hashes[key]
        for key in prepared.hashes
    }
    contract = prepared.env.contract
    arm = make_phase4_arm(
        arm_name,
        prepared.task_id,
        contract.effects,
        contract.dependencies,
        contract.support_contract,
        prepared.unit_id,
    )
    timings: List[int] = []
    _replay_prefix(arm, prepared.broker, timings)
    if forced_identical:
        for effect_id in contract.effects:
            if not prepared.env.effect_truth(effect_id):
                prepared.executor.execute(
                    prepared.task_id, effect_id, ExecutionMode.EXECUTE
                )
        scenario = {
            "natural_decision": "FORCED_IDENTICAL",
            "chain_success": prepared.env.all_chain_truth(),
            "target_truth_at_decision": prepared.env.effect_truth(prepared.target_effect),
            "effect_truth_recognition": None,
            "task_advance_correctness": None,
            "attempt_attributed_success": None,
            "final_target_truth": prepared.env.effect_truth(prepared.target_effect),
            "final_effect_truth_recognition": None,
            "final_attempt_attributed_success": None,
            "false_skill_credit": False,
            "missed_incidental_success": False,
            "stale_witness_accepted": False,
            "cross_attempt_verification": False,
            "superseded_command_realization": False,
            "late_witness_after_revocation": False,
            "premature_advance": False,
            "retry_count": 0,
            "reobserve_count": 0,
            "rollback_count": 0,
            "recovery_success": False,
            "support_expected_invalidated": [],
            "support_actual_invalidated": [],
            "over_invalidation_count": 0,
            "under_invalidation_count": 0,
            "alternative_support_survived": None,
            "discharged_support_false_invalidation": None,
            "branch_locality_correct": None,
            "added_action_steps": 0,
        }
    elif prepared.condition in SUPPORT_CONDITIONS:
        scenario = _support_scenario(prepared, arm, timings)
    else:
        scenario = _attempt_scenario(prepared, arm, timings)
    arm_metrics = _compact_arm_metrics(arm)
    total_action_steps = sum(int(call["action_steps"]) for call in prepared.executor.calls)
    result = {
        "unit_id": prepared.unit_id,
        "task_id": prepared.task_id,
        "condition": prepared.condition,
        "seed": prepared.seed,
        "arm": arm_name,
        "prefix_hashes": dict(prepared.hashes),
        "inherited_prefix_hashes": inherited_hashes,
        "prefix_identity": identity,
        "pre_decision_identity_pass": all(identity.values()),
        "terminal_physical_state_sha256": prepared.env.state_sha256(),
        "terminal_controller_state_sha256": prepared.env.controller_sha256(),
        "terminal_action_trace_sha256": prepared.env.action_prefix_sha256(),
        "terminal_event_stream_sha256": prepared.broker.stream_hash(),
        "action_steps": total_action_steps,
        "event_processing_time_ns": int(sum(timings)),
        "decision_latency_ns": int(max(timings) if timings else 0),
        "event_count": len(prepared.broker.records),
        "backend_error_count": len(prepared.env.backend_errors),
        "child_process_failure": False,
        "forced_identical": bool(forced_identical),
    }
    result.update(scenario)
    result.update(arm_metrics)
    return result


def _read_all(fd: int) -> bytes:
    chunks = []
    while True:
        block = os.read(fd, 65536)
        if not block:
            break
        chunks.append(block)
    return b"".join(chunks)


def run_paired_unit(
    task_id: str,
    condition: str,
    seed: int,
    arms: Sequence[str] = MAIN_ARMS,
    forced_identical: bool = False,
) -> Tuple[List[dict], dict]:
    unit_id = "%s|%s|%d" % (task_id, condition, int(seed))
    prepared = _prepare_prefix(task_id, condition, int(seed), unit_id)
    rows = []
    child_failures = []
    for arm_name in arms:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child
            try:
                os.close(read_fd)
                row = _run_child(prepared, arm_name, forced_identical)
                payload = json.dumps(
                    {"ok": True, "row": row},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                os.write(write_fd, payload)
                os.close(write_fd)
                os._exit(0)
            except BaseException as exc:  # pragma: no cover - formal evidence path
                try:
                    payload = json.dumps(
                        {
                            "ok": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "arm": arm_name,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                    os.write(write_fd, payload)
                    os.close(write_fd)
                finally:
                    os._exit(1)
        os.close(write_fd)
        payload = _read_all(read_fd)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status):
            exit_code = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            exit_code = 128 + os.WTERMSIG(status)
        else:
            exit_code = 1
        try:
            envelope = json.loads(payload.decode("utf-8")) if payload else {}
        except Exception as exc:
            envelope = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": "invalid child envelope",
                "arm": arm_name,
            }
        if exit_code != 0 or not envelope.get("ok"):
            failure = {
                "unit_id": unit_id,
                "arm": arm_name,
                "exit_code": exit_code,
                "error_type": envelope.get("error_type", "ChildProcessError"),
                "error": envelope.get("error", "child returned no evidence"),
            }
            child_failures.append(failure)
            rows.append(
                {
                    "unit_id": unit_id,
                    "task_id": task_id,
                    "condition": condition,
                    "seed": int(seed),
                    "arm": arm_name,
                    "child_process_failure": True,
                    "failure_type": failure["error_type"],
                    "chain_success": False,
                    "prefix_hashes": dict(prepared.hashes),
                    "pre_decision_identity_pass": False,
                }
            )
        else:
            row = envelope["row"]
            row["failure_type"] = None
            rows.append(row)
    terminal_hashes = {
        row.get("terminal_physical_state_sha256")
        for row in rows
        if not row.get("child_process_failure")
    }
    audit = {
        "unit_id": unit_id,
        "task_id": task_id,
        "condition": condition,
        "seed": int(seed),
        "arms": list(arms),
        "parent_prefix_hashes": dict(prepared.hashes),
        "pre_decision_state_identity": all(
            row.get("prefix_identity", {}).get("physical_state", False) for row in rows
        ),
        "controller_state_identity": all(
            row.get("prefix_identity", {}).get("controller_state", False) for row in rows
        ),
        "rng_state_identity": all(
            row.get("prefix_identity", {}).get("rng_state", False) for row in rows
        ),
        "event_prefix_identity": all(
            row.get("prefix_identity", {}).get("event_prefix", False) for row in rows
        ),
        "action_prefix_identity": all(
            row.get("prefix_identity", {}).get("action_prefix", False) for row in rows
        ),
        "forced_identical_terminal_state_identity": (
            len(terminal_hashes) == 1 if forced_identical else None
        ),
        "terminal_physical_state_hashes": sorted(
            value for value in terminal_hashes if value is not None
        ),
        "child_process_failures": child_failures,
        "pass": not child_failures
        and all(row.get("pre_decision_identity_pass", False) for row in rows)
        and (len(terminal_hashes) == 1 if forced_identical else True),
    }
    return rows, audit


def shared_prefix_qualification(unit_count: int = 1000) -> dict:
    cells = [
        (task_id, "C0")
        for task_id in sorted(TASK_CONTRACTS)
    ]
    rows = []
    audits = []
    for ordinal in range(unit_count):
        task_id, condition = cells[ordinal % len(cells)]
        unit_rows, audit = run_paired_unit(
            task_id,
            condition,
            seed=400000 + ordinal,
            arms=MAIN_ARMS,
            forced_identical=True,
        )
        rows.extend(unit_rows)
        audits.append(audit)
    gates = {
        "pre_decision_state_identity": all(
            audit["pre_decision_state_identity"] for audit in audits
        ),
        "controller_state_identity": all(
            audit["controller_state_identity"] for audit in audits
        ),
        "rng_state_identity": all(audit["rng_state_identity"] for audit in audits),
        "event_prefix_identity": all(
            audit["event_prefix_identity"] for audit in audits
        ),
        "action_prefix_identity": all(
            audit["action_prefix_identity"] for audit in audits
        ),
        "forced_identical_terminal_state_identity": all(
            audit["forced_identical_terminal_state_identity"] for audit in audits
        ),
    }
    return {
        "schema_version": 1,
        "unit_count": unit_count,
        "arm_row_count": len(rows),
        "protected_b6_sha256": protected_b6_sha256(),
        "gates": gates,
        "child_process_failure_count": sum(
            len(audit["child_process_failures"]) for audit in audits
        ),
        "pass": all(gates.values())
        and not any(audit["child_process_failures"] for audit in audits),
        "rows": rows,
        "audits": audits,
    }


def run_matrix(
    cells: Sequence[Mapping[str, str]],
    seeds: Iterable[int],
    arms: Sequence[str] = MAIN_ARMS,
) -> Tuple[List[dict], List[dict]]:
    if "M1_B6_ORIGINAL" in arms and protected_b6_sha256() != (
        "4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5"
    ):
        raise RuntimeError("protected B6 hash mismatch before formal run")
    rows = []
    audits = []
    for seed in seeds:
        for cell in cells:
            unit_rows, audit = run_paired_unit(
                str(cell["task"]),
                str(cell["condition"]),
                int(seed),
                arms=arms,
                forced_identical=False,
            )
            rows.extend(unit_rows)
            audits.append(audit)
    return rows, audits

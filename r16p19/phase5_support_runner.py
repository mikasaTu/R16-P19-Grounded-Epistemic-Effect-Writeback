"""Run the exact 960-cell physical support-proof matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase4_arms import LedgerBackedArm
from .phase4_event_broker import Phase4EventBroker


TASKS = ("T1_CARRY_PLACE_RELEASE", "T2_ALTERNATIVE_PHYSICAL_SUPPORT")
CONDITIONS = ("C0_CLEAN", "S1_LIVE_SUPPORT_INVALIDATED", "S2_DISCHARGED_SUPPORT_REMOVED", "S3_ALTERNATIVE_SUPPORT_REMOVED")
ARMS = ("M0_TYPED_MATCHED", "M1_VERSIONED_POSTCHECK", "M3_ASCEL_CORE", "M4_ASCEL_FULL")


def _contract(task: str):
    if task == "T1_CARRY_PLACE_RELEASE":
        effects = ("GRASP_SUPPORT", "CARRIED", "RELEASED_IN_TARGET")
        deps = {"GRASP_SUPPORT": (), "CARRIED": ("GRASP_SUPPORT",), "RELEASED_IN_TARGET": ("CARRIED",)}
        support = {"GRASP_SUPPORT": [], "CARRIED": [[{"parent": "GRASP_SUPPORT", "type": "UNTIL_EFFECT_REALIZED", "until_effect": "RELEASED_IN_TARGET"}]], "RELEASED_IN_TARGET": [[{"parent": "CARRIED", "type": "UNTIL_CHILD_REALIZED"}]]}
        return effects, deps, support, "GRASP_SUPPORT", "CARRIED", "RELEASED_IN_TARGET"
    effects = ("LEFT_SUPPORT", "RIGHT_SUPPORT", "OBJECT_ELEVATED")
    deps = {"LEFT_SUPPORT": (), "RIGHT_SUPPORT": (), "OBJECT_ELEVATED": ("LEFT_SUPPORT", "RIGHT_SUPPORT")}
    support = {"LEFT_SUPPORT": [], "RIGHT_SUPPORT": [], "OBJECT_ELEVATED": [[{"parent": "LEFT_SUPPORT", "type": "PERSISTENT"}], [{"parent": "RIGHT_SUPPORT", "type": "PERSISTENT"}]]}
    return effects, deps, support, "LEFT_SUPPORT", "OBJECT_ELEVATED", "OBJECT_ELEVATED"


def _make_arm(name: str, task: str, unit: str):
    effects, deps, support, _, _, _ = _contract(task)
    if name == "M4_ASCEL_FULL":
        return LedgerBackedArm(name, effects, deps, support, True, True, True, True)
    if name == "M3_ASCEL_CORE":
        return LedgerBackedArm(name, effects, {effect: () for effect in effects}, {effect: [] for effect in effects}, True, True, True, False)
    # Both frozen non-ASCEL baselines use a static dependency graph here.
    return LedgerBackedArm(name, effects, deps, support, False, False, True, False)


def _realize(arm, broker, effect):
    request, command = broker.prefix(effect)
    for event in (request, command, broker.positive(effect, "base", request.attempt_id, command.event_id, True), broker.positive(effect, "wrist", request.attempt_id, command.event_id, True), broker.witness(effect, request.attempt_id, command.event_id, True)):
        arm.process(event)


def run_cell(task: str, condition: str, seed: int, arm_name: str) -> dict:
    from .phase5_support_env import GravitySupportEnv

    environment = GravitySupportEnv(task, seed)
    arm = _make_arm(arm_name, task, f"{task}-{condition}-{seed}-{arm_name}")
    broker = Phase4EventBroker(f"{task}-{condition}-{seed}", seed)
    effects, _, _, support_effect, child_effect, final_effect = _contract(task)
    environment.prepare_live_support()
    if task == "T2_ALTERNATIVE_PHYSICAL_SUPPORT":
        _realize(arm, broker, "LEFT_SUPPORT")
        _realize(arm, broker, "RIGHT_SUPPORT")
    else:
        _realize(arm, broker, support_effect)
    _realize(arm, broker, child_effect)
    physical_prefix = environment.snapshot()
    prefix = physical_prefix.sha256()
    recovery = False
    over_invalidation = False
    under_invalidation = False
    if condition == "C0_CLEAN":
        environment.complete()
    elif condition == "S1_LIVE_SUPPORT_INVALIDATED":
        environment.invalidate_live_support(all_supports=True)
        arm.process(broker.contradiction(support_effect, False))
        if task == "T2_ALTERNATIVE_PHYSICAL_SUPPORT":
            arm.process(broker.contradiction("RIGHT_SUPPORT", False))
        child_valid = arm.effect_fact_verified(child_effect)
        under_invalidation = bool(child_valid)
        if not child_valid:
            # Recovery starts from the recorded pre-fault physical checkpoint;
            # this is the same replay boundary used for every arm in the cell.
            environment.restore(physical_prefix)
            environment.complete()
            recovery = True
    elif condition == "S2_DISCHARGED_SUPPORT_REMOVED":
        environment.discharge()
        _realize(arm, broker, final_effect)
        arm.process(broker.contradiction(support_effect, False))
        over_invalidation = not arm.effect_fact_verified(final_effect)
    else:
        if task == "T2_ALTERNATIVE_PHYSICAL_SUPPORT":
            environment.invalidate_live_support(all_supports=False)
            arm.process(broker.contradiction("LEFT_SUPPORT", False))
            over_invalidation = not arm.effect_fact_verified(child_effect)
        else:
            environment.complete()
    success = environment.success(condition)
    return {"task_id": task, "condition": condition, "seed": seed, "formal_init": seed, "policy_seed": 0, "cluster_id": f"{task}-{seed}", "arm": arm_name, "task_success": success, "prefix_sha256": prefix, "cascade_true_positive": bool(condition == "S1_LIVE_SUPPORT_INVALIDATED" and not under_invalidation), "cascade_false_negative": bool(condition == "S1_LIVE_SUPPORT_INVALIDATED" and under_invalidation), "over_invalidation": over_invalidation, "under_invalidation": under_invalidation, "recovery_executed": recovery, "action_steps": environment.action_steps, "contact_count": environment.contact_count(), "backend_errors": len(environment.backend_errors)}


def run(output: Path) -> dict:
    rows = [run_cell(task, condition, seed, arm) for task in TASKS for condition in CONDITIONS for seed in range(30) for arm in ARMS]
    if len(rows) != 960:
        raise RuntimeError(len(rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"schema_version": 1, "rows": len(rows), "backend_errors": sum(row["backend_errors"] for row in rows), "success_rate": {arm: sum(row["task_success"] for row in rows if row["arm"] == arm) / 240 for arm in ARMS}}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()

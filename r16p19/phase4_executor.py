"""Arm-blind deterministic macro-skill executor for Phase-4 CPU tasks."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, List, Sequence

import mujoco

from .phase4_microenv import Phase4MicroEnv, TASK_CONTRACTS


class ExecutionMode(str, Enum):
    EXECUTE = "EXECUTE"
    RETRY = "RETRY"
    ROLLBACK = "ROLLBACK"


class MacroSkillExecutor:
    """Execute a physical effect without access to memory or fault identity."""

    def __init__(self, env: Phase4MicroEnv) -> None:
        self._env = env
        self.calls: List[dict] = []

    def execute(self, task_id: str, effect_id: str, mode: ExecutionMode) -> dict:
        if task_id != self._env.contract.task_id:
            raise ValueError("executor task does not match environment")
        if effect_id not in self._env.contract.effects:
            raise ValueError("executor received unknown effect")
        before = self._env.state_sha256()
        if mode in (ExecutionMode.EXECUTE, ExecutionMode.RETRY):
            action_steps = self._env.actuate_effect(effect_id, actor="executor")
        elif mode == ExecutionMode.ROLLBACK:
            action_steps = self._env.reverse_effect(effect_id, actor="executor")
        else:  # pragma: no cover
            raise ValueError("unknown execution mode")
        row = {
            "task_id": task_id,
            "effect_id": effect_id,
            "mode": mode.value,
            "action_steps": int(action_steps),
            "state_sha256_before": before,
            "state_sha256_after": self._env.state_sha256(),
            "effect_truth_after": self._env.effect_truth(effect_id),
            "backend_error_count": len(self._env.backend_errors),
        }
        self.calls.append(row)
        return row


def _prefix_for_effect(task_id: str, effect_id: str) -> Sequence[str]:
    contract = TASK_CONTRACTS[task_id]
    if task_id == "T3_ALTERNATIVE_SUPPORT" and effect_id == "OBJECT_ELEVATED":
        return ("LEFT_SUPPORT",)
    if task_id == "T3_ALTERNATIVE_SUPPORT" and effect_id == "TARGET_REACHED":
        return ("LEFT_SUPPORT", "OBJECT_ELEVATED")
    if effect_id in contract.chain_effects:
        index = contract.chain_effects.index(effect_id)
        return contract.chain_effects[:index]
    return ()


def qualify_executor(seeds: Iterable[int] = range(40, 60)) -> Dict[str, object]:
    rows = []
    for seed in seeds:
        for task_id, contract in TASK_CONTRACTS.items():
            for effect_id in contract.effects:
                env = Phase4MicroEnv(task_id, int(seed))
                executor = MacroSkillExecutor(env)
                for prefix_effect in _prefix_for_effect(task_id, effect_id):
                    executor.execute(task_id, prefix_effect, ExecutionMode.EXECUTE)
                result = executor.execute(task_id, effect_id, ExecutionMode.EXECUTE)
                rows.append(
                    {
                        "record_type": "conditional_effect",
                        "seed": int(seed),
                        "task_id": task_id,
                        "effect_id": effect_id,
                        "success": bool(result["effect_truth_after"]),
                        "action_steps": sum(call["action_steps"] for call in executor.calls),
                        "backend_error_count": len(env.backend_errors),
                        "state_sha256": env.state_sha256(),
                    }
                )
            env = Phase4MicroEnv(task_id, int(seed))
            executor = MacroSkillExecutor(env)
            for effect_id in contract.chain_effects:
                executor.execute(task_id, effect_id, ExecutionMode.EXECUTE)
            rows.append(
                {
                    "record_type": "full_chain",
                    "seed": int(seed),
                    "task_id": task_id,
                    "effect_id": None,
                    "success": env.all_chain_truth(),
                    "action_steps": sum(call["action_steps"] for call in executor.calls),
                    "backend_error_count": len(env.backend_errors),
                    "state_sha256": env.state_sha256(),
                }
            )
    conditional = [row for row in rows if row["record_type"] == "conditional_effect"]
    chains = [row for row in rows if row["record_type"] == "full_chain"]
    backend_errors = sum(row["backend_error_count"] for row in rows)
    conditional_success = sum(row["success"] for row in conditional) / float(len(conditional))
    chain_success = sum(row["success"] for row in chains) / float(len(chains))
    return {
        "schema_version": 1,
        "mujoco_version": mujoco.__version__,
        "seed_range": [min(row["seed"] for row in rows), max(row["seed"] for row in rows)],
        "conditional_effect_count": len(conditional),
        "full_chain_count": len(chains),
        "conditional_effect_success": conditional_success,
        "full_chain_success": chain_success,
        "backend_error_count": backend_errors,
        "gates": {
            "conditional_effect_success_ge_0_99": conditional_success >= 0.99,
            "full_chain_success_ge_0_98": chain_success >= 0.98,
            "backend_errors_zero": backend_errors == 0,
        },
        "pass": conditional_success >= 0.99 and chain_success >= 0.98 and backend_errors == 0,
        "rows": rows,
    }

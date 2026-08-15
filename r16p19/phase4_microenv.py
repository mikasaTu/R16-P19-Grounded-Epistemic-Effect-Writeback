"""Three deterministic CPU MuJoCo microenvironments for Phase-4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import mujoco
import numpy as np


@dataclass(frozen=True)
class TaskPhysicalContract:
    task_id: str
    effects: Tuple[str, ...]
    dependencies: Mapping[str, Tuple[str, ...]]
    support_contract: Mapping[str, Sequence[Sequence[Mapping[str, object]]]]
    chain_effects: Tuple[str, ...]
    attempt_target: str
    support_root: str
    final_effect: str
    unrelated_effect: str = ""


TASK_CONTRACTS: Dict[str, TaskPhysicalContract] = {
    "T1_CARRY_RELEASE": TaskPhysicalContract(
        task_id="T1_CARRY_RELEASE",
        effects=("GRASPED", "LIFTED", "OVER_TARGET", "RELEASED_IN_TARGET"),
        dependencies={
            "GRASPED": (),
            "LIFTED": ("GRASPED",),
            "OVER_TARGET": ("LIFTED",),
            "RELEASED_IN_TARGET": ("OVER_TARGET",),
        },
        support_contract={
            "GRASPED": [],
            "LIFTED": [[
                {
                    "parent": "GRASPED",
                    "type": "UNTIL_EFFECT_REALIZED",
                    "until_effect": "RELEASED_IN_TARGET",
                }
            ]],
            "OVER_TARGET": [[
                {
                    "parent": "GRASPED",
                    "type": "UNTIL_EFFECT_REALIZED",
                    "until_effect": "RELEASED_IN_TARGET",
                },
                {"parent": "LIFTED", "type": "UNTIL_CHILD_REALIZED"},
            ]],
            "RELEASED_IN_TARGET": [[
                {"parent": "GRASPED", "type": "UNTIL_CHILD_REALIZED"},
                {"parent": "OVER_TARGET", "type": "UNTIL_CHILD_REALIZED"},
            ]],
        },
        chain_effects=("GRASPED", "LIFTED", "OVER_TARGET", "RELEASED_IN_TARGET"),
        attempt_target="LIFTED",
        support_root="GRASPED",
        final_effect="RELEASED_IN_TARGET",
    ),
    "T2_PERSISTENT_SUPPORT": TaskPhysicalContract(
        task_id="T2_PERSISTENT_SUPPORT",
        effects=("SUPPORT_PRESENT", "OBJECT_STABLE", "MARKER_PLACED"),
        dependencies={
            "SUPPORT_PRESENT": (),
            "OBJECT_STABLE": ("SUPPORT_PRESENT",),
            "MARKER_PLACED": ("OBJECT_STABLE",),
        },
        support_contract={
            "SUPPORT_PRESENT": [],
            "OBJECT_STABLE": [[
                {"parent": "SUPPORT_PRESENT", "type": "PERSISTENT"}
            ]],
            "MARKER_PLACED": [[
                {"parent": "OBJECT_STABLE", "type": "PERSISTENT"}
            ]],
        },
        chain_effects=("SUPPORT_PRESENT", "OBJECT_STABLE", "MARKER_PLACED"),
        attempt_target="OBJECT_STABLE",
        support_root="SUPPORT_PRESENT",
        final_effect="MARKER_PLACED",
    ),
    "T3_ALTERNATIVE_SUPPORT": TaskPhysicalContract(
        task_id="T3_ALTERNATIVE_SUPPORT",
        effects=(
            "LEFT_SUPPORT",
            "RIGHT_SUPPORT",
            "OBJECT_ELEVATED",
            "TARGET_REACHED",
            "UNRELATED_BRANCH",
        ),
        dependencies={
            "LEFT_SUPPORT": (),
            "RIGHT_SUPPORT": (),
            "OBJECT_ELEVATED": ("LEFT_SUPPORT", "RIGHT_SUPPORT"),
            "TARGET_REACHED": ("OBJECT_ELEVATED",),
            "UNRELATED_BRANCH": (),
        },
        support_contract={
            "LEFT_SUPPORT": [],
            "RIGHT_SUPPORT": [],
            "OBJECT_ELEVATED": [
                [{"parent": "LEFT_SUPPORT", "type": "PERSISTENT"}],
                [{"parent": "RIGHT_SUPPORT", "type": "PERSISTENT"}],
            ],
            "TARGET_REACHED": [[
                {"parent": "OBJECT_ELEVATED", "type": "PERSISTENT"}
            ]],
            "UNRELATED_BRANCH": [],
        },
        chain_effects=(
            "LEFT_SUPPORT",
            "RIGHT_SUPPORT",
            "OBJECT_ELEVATED",
            "TARGET_REACHED",
        ),
        attempt_target="OBJECT_ELEVATED",
        support_root="LEFT_SUPPORT",
        final_effect="TARGET_REACHED",
        unrelated_effect="UNRELATED_BRANCH",
    ),
}


def _model_xml(contract: TaskPhysicalContract) -> str:
    bodies = []
    actuators = []
    for index, effect in enumerate(contract.effects):
        y = -1.2 + 0.55 * index
        bodies.append(
            """
            <body name="body_{index}" pos="-0.5 {y:.4f} 0.15">
              <joint name="joint_{index}" type="slide" axis="1 0 0"
                     limited="true" range="-0.05 1.05" damping="4"/>
              <geom name="geom_{index}" type="box" size="0.08 0.08 0.08"
                    mass="0.2" rgba="{r:.3f} {g:.3f} {b:.3f} 1"/>
            </body>
            """.format(
                index=index,
                y=y,
                r=0.2 + 0.12 * (index % 4),
                g=0.7 - 0.08 * (index % 4),
                b=0.3 + 0.1 * (index % 3),
            )
        )
        actuators.append(
            '<position name="actuator_{0}" joint="joint_{0}" kp="80" '
            'ctrllimited="true" ctrlrange="-0.02 1.02"/>'.format(index)
        )
    return """
    <mujoco model="{task}">
      <compiler angle="radian"/>
      <option timestep="0.005" gravity="0 0 0" integrator="Euler"/>
      <size njmax="128" nconmax="64"/>
      <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.1 0.1 0.1 1"/>
        {bodies}
      </worldbody>
      <actuator>
        {actuators}
      </actuator>
    </mujoco>
    """.format(task=contract.task_id, bodies="\n".join(bodies), actuators="\n".join(actuators))


class Phase4MicroEnv:
    """A small physical state machine implemented with MuJoCo actuated bodies."""

    TRUE_THRESHOLD = 0.75
    FALSE_THRESHOLD = 0.25
    CONTROL_STEPS = 100

    def __init__(self, task_id: str, seed: int) -> None:
        if task_id not in TASK_CONTRACTS:
            raise ValueError("unknown Phase-4 task %s" % task_id)
        self.contract = TASK_CONTRACTS[task_id]
        self.model_xml = _model_xml(self.contract)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self.effect_index = {
            effect: index for index, effect in enumerate(self.contract.effects)
        }
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.controller_targets = np.zeros(len(self.contract.effects), dtype=np.float64)
        self.action_trace: List[dict] = []
        self.backend_errors: List[str] = []
        self.reset(self.seed)

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        jitter = self.rng.uniform(0.0, 0.015, size=self.model.nq)
        self.data.qpos[:] = jitter
        self.data.qvel[:] = 0.0
        self.controller_targets[:] = 0.0
        self.data.ctrl[:] = self.controller_targets
        mujoco.mj_forward(self.model, self.data)
        self.action_trace = []
        self.backend_errors = []

    def _drive(self, effect_id: str, target: float, actor: str) -> int:
        if effect_id not in self.effect_index:
            raise ValueError("unknown effect %s" % effect_id)
        index = self.effect_index[effect_id]
        self.controller_targets[index] = float(target)
        self.data.ctrl[:] = self.controller_targets
        start = len(self.action_trace)
        try:
            for local_step in range(self.CONTROL_STEPS):
                mujoco.mj_step(self.model, self.data)
                if local_step in (0, self.CONTROL_STEPS - 1):
                    self.action_trace.append(
                        {
                            "actor": actor,
                            "effect_id": effect_id,
                            "local_step": local_step,
                            "target": float(target),
                            "qpos": float(self.data.qpos[index]),
                        }
                    )
        except Exception as exc:  # pragma: no cover - retained as formal evidence
            self.backend_errors.append("%s:%s" % (type(exc).__name__, str(exc)))
            raise
        return self.CONTROL_STEPS if len(self.action_trace) >= start else 0

    def actuate_effect(self, effect_id: str, actor: str = "executor") -> int:
        return self._drive(effect_id, 1.0, actor)

    def reverse_effect(self, effect_id: str, actor: str = "fault_injector") -> int:
        return self._drive(effect_id, 0.0, actor)

    def effect_truth(self, effect_id: str) -> bool:
        return bool(self.data.qpos[self.effect_index[effect_id]] >= self.TRUE_THRESHOLD)

    def all_chain_truth(self) -> bool:
        return all(self.effect_truth(effect) for effect in self.contract.chain_effects)

    def state_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.data.qpos, dtype=np.float64),
                np.asarray(self.data.qvel, dtype=np.float64),
                np.asarray(self.data.ctrl, dtype=np.float64),
                np.asarray([self.data.time], dtype=np.float64),
            ]
        )

    def state_sha256(self) -> str:
        return hashlib.sha256(
            self.state_vector().astype("<f8").tobytes(order="C")
        ).hexdigest()

    def controller_state(self) -> dict:
        return {
            "targets": [float(value) for value in self.controller_targets],
            "action_trace_length": len(self.action_trace),
        }

    def controller_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.controller_state(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def action_prefix_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.action_trace, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def snapshot(self) -> dict:
        return {
            "task_id": self.contract.task_id,
            "seed": self.seed,
            "mujoco_version": mujoco.__version__,
            "model_sha256": hashlib.sha256(self.model_xml.encode("utf-8")).hexdigest(),
            "state": [float(value) for value in self.state_vector()],
            "state_sha256": self.state_sha256(),
            "controller": self.controller_state(),
            "controller_sha256": self.controller_sha256(),
            "action_prefix_sha256": self.action_prefix_sha256(),
            "truth": {
                effect: self.effect_truth(effect) for effect in self.contract.effects
            },
            "backend_errors": list(self.backend_errors),
        }

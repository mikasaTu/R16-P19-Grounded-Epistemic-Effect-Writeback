"""Two gravity-enabled MuJoCo tasks for support-proof validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class PhysicalSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    eq_active: np.ndarray

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (self.qpos, self.qvel, self.mocap_pos, self.mocap_quat, self.eq_active):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


class GravitySupportEnv:
    def __init__(self, task: str, seed: int) -> None:
        if task not in ("T1_CARRY_PLACE_RELEASE", "T2_ALTERNATIVE_PHYSICAL_SUPPORT"):
            raise ValueError(task)
        self.task = task
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.model = mujoco.MjModel.from_xml_string(self._xml(task))
        self.data = mujoco.MjData(self.model)
        self.action_steps = 0
        self.backend_errors = []
        mujoco.mj_forward(self.model, self.data)
        self._step(300)

    @staticmethod
    def _xml(task: str) -> str:
        if task == "T1_CARRY_PLACE_RELEASE":
            return """
<mujoco model="carry_place_release">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .05" friction="1 0.01 0.001"/>
    <geom name="target_floor" type="box" pos=".5 0 .025" size=".18 .18 .025" rgba="0 .8 0 .4"/>
    <body name="gripper" mocap="true" pos="-.5 0 .12"><geom type="sphere" size=".04" contype="0" conaffinity="0" rgba="0 0 1 .3"/></body>
    <body name="object" pos="-.5 0 .08"><freejoint/><geom name="object_geom" type="box" size=".05 .05 .05" mass=".2" friction="1 0.01 0.001" rgba=".8 .2 .1 1"/></body>
  </worldbody>
  <equality><weld name="grasp" body1="gripper" body2="object" active="false" solref=".002 1"/></equality>
</mujoco>"""
        return """
<mujoco model="alternative_support">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .05"/>
    <body name="left" mocap="true" pos="-.07 0 .20"><geom name="left_geom" type="box" size=".10 .18 .04" friction="1 0.01 0.001"/></body>
    <body name="right" mocap="true" pos=".07 0 .20"><geom name="right_geom" type="box" size=".10 .18 .04" friction="1 0.01 0.001"/></body>
    <body name="object" pos="0 0 .31"><joint name="vertical" type="slide" axis="0 0 1" damping=".1"/><geom name="object_geom" type="box" size=".18 .12 .05" mass=".4" friction="1 0.01 0.001" rgba=".8 .2 .1 1"/></body>
  </worldbody>
</mujoco>"""

    def _step(self, count: int) -> None:
        try:
            for _ in range(int(count)):
                mujoco.mj_step(self.model, self.data)
                self.action_steps += 1
        except Exception as exc:
            self.backend_errors.append(f"{type(exc).__name__}:{exc}")
            raise

    def _eq_active(self) -> np.ndarray:
        # MuJoCo 2.x stores this mutable vector on MjModel; 3.x moved it to
        # MjData.  Supporting both keeps the physical task source identical
        # between the validated LIBERO runtime and the current PAI runtime.
        return self.model.eq_active if hasattr(self.model, "eq_active") else self.data.eq_active

    def snapshot(self) -> PhysicalSnapshot:
        return PhysicalSnapshot(self.data.qpos.copy(), self.data.qvel.copy(), self.data.mocap_pos.copy(), self.data.mocap_quat.copy(), self._eq_active().copy())

    def restore(self, snapshot: PhysicalSnapshot) -> None:
        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        self.data.mocap_pos[:] = snapshot.mocap_pos
        self.data.mocap_quat[:] = snapshot.mocap_quat
        self._eq_active()[:] = snapshot.eq_active
        mujoco.mj_forward(self.model, self.data)

    def _move_mocap(self, index: int, target, steps: int = 250) -> None:
        start = self.data.mocap_pos[index].copy()
        for alpha in np.linspace(0.0, 1.0, steps):
            self.data.mocap_pos[index] = (1 - alpha) * start + alpha * np.asarray(target)
            self._step(1)

    def prepare_live_support(self) -> None:
        if self.task == "T1_CARRY_PLACE_RELEASE":
            self.data.mocap_pos[0] = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")]
            self._eq_active()[0] = 1
            mujoco.mj_forward(self.model, self.data)
            self._move_mocap(0, (0.0, 0.0, 0.35))
        else:
            self._step(200)

    def invalidate_live_support(self, all_supports: bool = True) -> None:
        if self.task == "T1_CARRY_PLACE_RELEASE":
            self._eq_active()[0] = 0
            mujoco.mj_forward(self.model, self.data)
            self._step(400)
        else:
            self._move_mocap(0, (-0.8, 0.0, 0.20), 100)
            if all_supports:
                self._move_mocap(1, (0.8, 0.0, 0.20), 100)
            self._step(500)

    def complete(self) -> None:
        if self.task == "T1_CARRY_PLACE_RELEASE":
            if not self._eq_active()[0]:
                object_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
                )
                object_pos = self.data.xpos[object_id].copy()
                self.data.mocap_pos[0] = object_pos + np.asarray((0.0, 0.0, 0.04))
                self.data.qvel[:] = 0.0
                self._eq_active()[0] = 1
                mujoco.mj_forward(self.model, self.data)
                self._step(200)
            # Slow, physical transport avoids turning weld release momentum into
            # a hidden placement intervention.
            self._move_mocap(0, (0.5, 0.0, 0.35), 1000)
            self._eq_active()[0] = 0
            mujoco.mj_forward(self.model, self.data)
            self._step(600)
        else:
            # Restore both real supports and allow contact to settle.
            self._move_mocap(0, (-0.07, 0.0, 0.20), 100)
            self._move_mocap(1, (0.07, 0.0, 0.20), 100)
            self._step(400)

    def discharge(self) -> None:
        if self.task == "T1_CARRY_PLACE_RELEASE":
            self.complete()
        else:
            # Put the object on the floor: supports are no longer causal.
            self.invalidate_live_support(all_supports=True)

    def success(self, condition: str) -> bool:
        object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        pos = self.data.xpos[object_id]
        if self.task == "T1_CARRY_PLACE_RELEASE":
            return bool(abs(pos[0] - 0.5) < 0.18 and pos[2] < 0.14)
        if condition == "S2_DISCHARGED_SUPPORT_REMOVED":
            return bool(pos[2] < 0.15)
        return bool(pos[2] > 0.24)

    def contact_count(self) -> int:
        return int(self.data.ncon)

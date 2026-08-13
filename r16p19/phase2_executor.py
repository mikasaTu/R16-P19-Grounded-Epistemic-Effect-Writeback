"""Deterministic memory-independent geometric executor for LIBERO Phase-2.

The executor consumes only the frozen geometric snapshot schema below.  It
does not import the memory implementation, evaluate LIBERO predicates, inspect
fault identity, or choose high-level effects.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .config import ACTION_DIM, TASKS


ACTION_HORIZON = 8
HISTORY_LENGTH = 4
POSITION_GAIN = 8.0
ORIENTATION_GAIN = 2.0
POSITION_TOLERANCE_M = 0.010
ORIENTATION_TOLERANCE_RAD = 0.12


class ExecutionMode(str, Enum):
    EXECUTE = "EXECUTE"
    RETRY = "RETRY"


class ExecutorVariant(str, Enum):
    WORLD_FRAME_OPEN_LOOP = "world_frame_open_loop_template"
    LOCAL_FRAME_OPEN_LOOP = "add_effect_local_frame_retargeting"
    LOCAL_FRAME_CLOSED_LOOP = "add_closed_loop_cartesian_tracking"
    FROZEN_FULL = "add_receding_horizon_and_preregistered_retry_offsets"


TASK_TO_INDEX = {key: index for index, key in enumerate(TASKS)}
EFFECTS = tuple(effect for task in TASKS.values() for effect in task.effects)
EFFECT_TO_INDEX = {effect: index for index, effect in enumerate(EFFECTS)}
MODE_TO_INDEX = {mode.value: index for index, mode in enumerate(ExecutionMode)}


def _array(value, length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError("invalid geometric field shape/value: %r" % (result.shape,))
    return result


def normalize_quaternion_xyzw(value) -> np.ndarray:
    quat = _array(value, 4).astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    quat /= norm
    if quat[3] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def quaternion_conjugate_xyzw(value) -> np.ndarray:
    quat = normalize_quaternion_xyzw(value).astype(np.float64)
    return np.asarray((-quat[0], -quat[1], -quat[2], quat[3]), dtype=np.float32)


def quaternion_multiply_xyzw(left, right) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quaternion_xyzw(left).astype(np.float64)
    x2, y2, z2, w2 = normalize_quaternion_xyzw(right).astype(np.float64)
    return normalize_quaternion_xyzw(
        np.asarray(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            dtype=np.float64,
        )
    )


def quaternion_to_matrix_xyzw(value) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(value).astype(np.float64)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.asarray(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.asarray(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    return normalize_quaternion_xyzw(quat)


def quaternion_error_rotvec(target, current) -> np.ndarray:
    delta = quaternion_multiply_xyzw(target, quaternion_conjugate_xyzw(current))
    xyz = np.asarray(delta[:3], dtype=np.float64)
    vector_norm = float(np.linalg.norm(xyz))
    if vector_norm < 1e-10:
        return np.zeros((3,), dtype=np.float32)
    angle = 2.0 * math.atan2(vector_norm, max(float(delta[3]), 1e-12))
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return np.asarray(xyz / vector_norm * angle, dtype=np.float32)


def quaternion_distance(left, right) -> float:
    dot = abs(float(np.dot(normalize_quaternion_xyzw(left), normalize_quaternion_xyzw(right))))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def pose_to_local(position, quaternion, frame_position, frame_quaternion):
    frame_rotation = quaternion_to_matrix_xyzw(frame_quaternion)
    local_position = frame_rotation.T.dot(
        np.asarray(position, dtype=np.float64) - np.asarray(frame_position, dtype=np.float64)
    )
    local_quaternion = quaternion_multiply_xyzw(
        quaternion_conjugate_xyzw(frame_quaternion), quaternion
    )
    return local_position.astype(np.float32), local_quaternion


def pose_from_local(local_position, local_quaternion, frame_position, frame_quaternion):
    frame_rotation = quaternion_to_matrix_xyzw(frame_quaternion)
    position = np.asarray(frame_position, dtype=np.float64) + frame_rotation.dot(
        np.asarray(local_position, dtype=np.float64)
    )
    quaternion = quaternion_multiply_xyzw(frame_quaternion, local_quaternion)
    return position.astype(np.float32), quaternion


@dataclass(frozen=True)
class GeometricSnapshot:
    eef_position: np.ndarray
    eef_quaternion_xyzw: np.ndarray
    robot_joint_position: np.ndarray
    gripper_joint_position: np.ndarray
    effect_object_position: np.ndarray
    effect_object_quaternion_xyzw: np.ndarray
    effect_fixture_position: np.ndarray
    effect_fixture_quaternion_xyzw: np.ndarray
    effect_fixture_joint_position: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "eef_position", _array(self.eef_position, 3))
        object.__setattr__(
            self, "eef_quaternion_xyzw", normalize_quaternion_xyzw(self.eef_quaternion_xyzw)
        )
        object.__setattr__(self, "robot_joint_position", _array(self.robot_joint_position, 7))
        object.__setattr__(self, "gripper_joint_position", _array(self.gripper_joint_position, 2))
        object.__setattr__(self, "effect_object_position", _array(self.effect_object_position, 3))
        object.__setattr__(
            self,
            "effect_object_quaternion_xyzw",
            normalize_quaternion_xyzw(self.effect_object_quaternion_xyzw),
        )
        object.__setattr__(self, "effect_fixture_position", _array(self.effect_fixture_position, 3))
        object.__setattr__(
            self,
            "effect_fixture_quaternion_xyzw",
            normalize_quaternion_xyzw(self.effect_fixture_quaternion_xyzw),
        )
        object.__setattr__(
            self, "effect_fixture_joint_position", _array(self.effect_fixture_joint_position, 1)
        )

    def vector(self) -> np.ndarray:
        return np.concatenate(
            (
                self.eef_position,
                self.eef_quaternion_xyzw,
                self.robot_joint_position,
                self.gripper_joint_position,
                self.effect_object_position,
                self.effect_object_quaternion_xyzw,
                self.effect_fixture_position,
                self.effect_fixture_quaternion_xyzw,
                self.effect_fixture_joint_position,
                np.ones((3,), dtype=np.float32),
            )
        ).astype(np.float32, copy=False)


SNAPSHOT_DIM = len(
    GeometricSnapshot(
        np.zeros(3),
        np.asarray((0, 0, 0, 1)),
        np.zeros(7),
        np.zeros(2),
        np.zeros(3),
        np.asarray((0, 0, 0, 1)),
        np.zeros(3),
        np.asarray((0, 0, 0, 1)),
        np.zeros(1),
    ).vector()
)


_TASK_OBJECT = {
    "stove_moka": "moka_pot_1",
    "bowl_drawer": "akita_black_bowl_1",
}

_EFFECT_FRAME = {
    "STOVE_TURNED_ON": ("site", "flat_stove_1_default_site"),
    "MOKA_GRASPED": ("object", "moka_pot_1"),
    "MOKA_ON_STOVE": ("site", "flat_stove_1_cook_region"),
    "MOKA_RELEASED_ON_STOVE": ("site", "flat_stove_1_cook_region"),
    "BOWL_GRASPED": ("object", "akita_black_bowl_1"),
    "BOWL_IN_BOTTOM_DRAWER": ("site", "white_cabinet_1_bottom_region"),
    "BOWL_RELEASED_IN_DRAWER": ("site", "white_cabinet_1_bottom_region"),
    # The virtual handle-axis frame uses the fixed cabinet base. This keeps
    # the push trajectory expressed in the handle's slide-axis coordinates
    # instead of translating the target along with the moving drawer.
    "BOTTOM_DRAWER_CLOSED": ("site", "white_cabinet_1_default_site"),
}

_EFFECT_JOINT = {
    "STOVE_TURNED_ON": "flat_stove_1_button",
    "BOTTOM_DRAWER_CLOSED": "white_cabinet_1_bottom_level",
}


def _sim_pose(env, kind: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
    sim = env.sim
    if kind == "site":
        index = sim.model.site_name2id(name)
        position = np.asarray(sim.data.site_xpos[index], dtype=np.float32)
        matrix = np.asarray(sim.data.site_xmat[index], dtype=np.float64).reshape(3, 3)
    elif kind == "body":
        index = sim.model.body_name2id(name)
        position = np.asarray(sim.data.body_xpos[index], dtype=np.float32)
        matrix = np.asarray(sim.data.body_xmat[index], dtype=np.float64).reshape(3, 3)
    else:
        raise ValueError("unsupported pose kind %r" % kind)
    return position, matrix_to_quaternion_xyzw(matrix)


def _joint_scalar(env, joint_name: str) -> float:
    address = env.sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(address, tuple):
        address = address[0]
    return float(env.sim.data.qpos[int(address)])


def extract_geometric_snapshot(
    env, observation: Mapping[str, object], task_id: str, effect_id: str
) -> GeometricSnapshot:
    if task_id not in TASKS or effect_id not in TASKS[task_id].effects:
        raise ValueError("task/effect mismatch")
    object_name = _TASK_OBJECT[task_id]
    object_position = observation[object_name + "_pos"]
    object_quaternion = observation[object_name + "_quat"]
    frame_kind, frame_name = _EFFECT_FRAME[effect_id]
    if frame_kind == "object":
        fixture_position = object_position
        fixture_quaternion = object_quaternion
    else:
        fixture_position, fixture_quaternion = _sim_pose(env, frame_kind, frame_name)
    joint = _EFFECT_JOINT.get(effect_id)
    fixture_joint = 0.0 if joint is None else _joint_scalar(env, joint)
    return GeometricSnapshot(
        eef_position=observation["robot0_eef_pos"],
        eef_quaternion_xyzw=observation["robot0_eef_quat"],
        robot_joint_position=observation["robot0_joint_pos"],
        gripper_joint_position=observation["robot0_gripper_qpos"],
        effect_object_position=object_position,
        effect_object_quaternion_xyzw=object_quaternion,
        effect_fixture_position=fixture_position,
        effect_fixture_quaternion_xyzw=fixture_quaternion,
        effect_fixture_joint_position=np.asarray((fixture_joint,), dtype=np.float32),
    )


def canonical_state_history(state_history: Sequence[GeometricSnapshot]) -> np.ndarray:
    values = list(state_history)
    if not values:
        raise ValueError("state_history must not be empty")
    if not all(isinstance(value, GeometricSnapshot) for value in values):
        raise TypeError("Phase-2 state_history accepts GeometricSnapshot only")
    values = values[-HISTORY_LENGTH:]
    while len(values) < HISTORY_LENGTH:
        values.insert(0, values[0])
    result = np.stack([value.vector() for value in values]).astype(np.float32)
    if result.shape != (HISTORY_LENGTH, SNAPSHOT_DIM):
        raise RuntimeError("canonical Phase-2 history shape drift")
    return result


def executor_input_bytes(
    state_history: Sequence[GeometricSnapshot],
    task_id: str,
    effect_id: str,
    execution_mode,
    retry_index: int,
) -> bytes:
    try:
        mode = execution_mode if isinstance(execution_mode, ExecutionMode) else ExecutionMode(str(execution_mode))
    except ValueError as error:
        raise ValueError("execution_mode must be EXECUTE or RETRY") from error
    if task_id not in TASK_TO_INDEX or effect_id not in EFFECT_TO_INDEX:
        raise ValueError("unknown task/effect")
    retry_index = int(retry_index)
    if retry_index < 0:
        raise ValueError("retry_index must be nonnegative")
    header = np.asarray(
        (TASK_TO_INDEX[task_id], EFFECT_TO_INDEX[effect_id], MODE_TO_INDEX[mode.value], retry_index),
        dtype="<i8",
    )
    history = np.asarray(canonical_state_history(state_history), dtype="<f4")
    return header.tobytes(order="C") + history.tobytes(order="C")


def executor_input_hash(*args) -> str:
    return hashlib.sha256(executor_input_bytes(*args)).hexdigest()


@dataclass(frozen=True)
class SkillTemplate:
    template_id: str
    task_key: str
    effect_id: str
    source_episode: str
    source_frame_position: np.ndarray
    source_frame_quaternion_xyzw: np.ndarray
    local_positions: np.ndarray
    local_quaternions_xyzw: np.ndarray
    world_positions: np.ndarray
    world_quaternions_xyzw: np.ndarray
    actions: np.ndarray
    gripper: np.ndarray
    descriptor: np.ndarray
    sha256: str

    @classmethod
    def from_json(cls, path: Path) -> "SkillTemplate":
        path = Path(path)
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        expected = hashlib.sha256(raw).hexdigest()
        return cls(
            template_id=str(value["template_id"]),
            task_key=str(value["task_key"]),
            effect_id=str(value["effect_id"]),
            source_episode=str(value["source_episode"]),
            source_frame_position=np.asarray(value["source_frame_position"], dtype=np.float32),
            source_frame_quaternion_xyzw=np.asarray(
                value["source_frame_quaternion_xyzw"], dtype=np.float32
            ),
            local_positions=np.asarray(value["local_positions"], dtype=np.float32),
            local_quaternions_xyzw=np.asarray(
                value["local_quaternions_xyzw"], dtype=np.float32
            ),
            world_positions=np.asarray(value["world_positions"], dtype=np.float32),
            world_quaternions_xyzw=np.asarray(
                value["world_quaternions_xyzw"], dtype=np.float32
            ),
            actions=np.asarray(value["actions"], dtype=np.float32),
            gripper=np.asarray(value["gripper"], dtype=np.float32),
            descriptor=np.asarray(value["descriptor"], dtype=np.float32),
            sha256=expected,
        )

    def validate(self) -> None:
        count = len(self.local_positions)
        expected = {
            "local_positions": (count, 3),
            "local_quaternions_xyzw": (count, 4),
            "world_positions": (count, 3),
            "world_quaternions_xyzw": (count, 4),
            "actions": (count, ACTION_DIM),
            "gripper": (count,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("template %s has invalid %s" % (self.template_id, name))
        if count < 2 or count > 48:
            raise ValueError("template waypoint count outside frozen range")


_DEFAULT_RETRY_OFFSETS = np.asarray(
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.010), (0.010, 0.0, 0.005), (-0.010, 0.0, 0.005)),
    dtype=np.float32,
)
_STOVE_RETRY_OFFSETS = np.asarray(
    ((0.0, 0.0, 0.0), (0.0, 0.010, 0.0), (0.0, -0.010, 0.0), (0.0, 0.0, 0.010)),
    dtype=np.float32,
)
_DRAWER_RETRY_OFFSETS = np.asarray(
    ((0.0, 0.0, 0.0), (0.010, 0.0, 0.0), (-0.010, 0.0, 0.0), (0.0, 0.0, 0.010)),
    dtype=np.float32,
)


class RetargetedGeometricSkillExecutor:
    """Frozen deterministic executor with no high-level or fault input."""

    def __init__(
        self,
        template_manifest: Path,
        variant: ExecutorVariant = ExecutorVariant.FROZEN_FULL,
        position_gain: float = POSITION_GAIN,
        orientation_gain: float = ORIENTATION_GAIN,
        position_tolerance_m: float = POSITION_TOLERANCE_M,
        demonstration_feedforward: bool = True,
        monotonic_progress: bool = False,
        retry_reapproach: bool = False,
    ):
        self.template_manifest = Path(template_manifest)
        manifest_bytes = self.template_manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        self.templates: Dict[Tuple[str, str], List[SkillTemplate]] = {}
        for row in manifest["templates"]:
            path = self.template_manifest.parent / row["path"]
            template = SkillTemplate.from_json(path)
            if template.sha256 != row["sha256"]:
                raise RuntimeError("template file hash mismatch: %s" % path)
            template.validate()
            self.templates.setdefault((template.task_key, template.effect_id), []).append(template)
        for task_key, task in TASKS.items():
            for effect_id in task.effects:
                values = self.templates.get((task_key, effect_id), [])
                if not values or len(values) > 3:
                    raise RuntimeError("template count gate failed for %s/%s" % (task_key, effect_id))
                values.sort(key=lambda value: value.sha256)
        self.variant = variant if isinstance(variant, ExecutorVariant) else ExecutorVariant(str(variant))
        self.position_gain = float(position_gain)
        self.orientation_gain = float(orientation_gain)
        self.position_tolerance_m = float(position_tolerance_m)
        self.demonstration_feedforward = bool(demonstration_feedforward)
        self.monotonic_progress = bool(monotonic_progress)
        self.retry_reapproach = bool(retry_reapproach)
        self.last_trace: Dict[str, object] = {}
        self._selected_template_sha256: Dict[Tuple[str, str, str, int], str] = {}
        self._template_order_sha256: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self._waypoint_cursor: Dict[Tuple[str, str, str, int], int] = {}
        self._retry_entry_complete: Dict[Tuple[str, str, str, int], bool] = {}

    def reset_episode(self) -> None:
        """Reset only deterministic per-attempt template-selection state."""

        self._selected_template_sha256.clear()
        self._template_order_sha256.clear()
        self._waypoint_cursor.clear()
        self._retry_entry_complete.clear()
        self.last_trace = {}

    @staticmethod
    def _offsets(effect_id: str) -> np.ndarray:
        if effect_id == "STOVE_TURNED_ON":
            return _STOVE_RETRY_OFFSETS
        if effect_id == "BOTTOM_DRAWER_CLOSED":
            return _DRAWER_RETRY_OFFSETS
        return _DEFAULT_RETRY_OFFSETS

    def _ordered_templates(
        self, snapshot: GeometricSnapshot, task_id: str, effect_id: str
    ) -> List[SkillTemplate]:
        local_position, local_quaternion = pose_to_local(
            snapshot.eef_position,
            snapshot.eef_quaternion_xyzw,
            snapshot.effect_fixture_position,
            snapshot.effect_fixture_quaternion_xyzw,
        )
        scored = []
        for template in self.templates[(task_id, effect_id)]:
            position_distance = float(np.linalg.norm(local_position - template.local_positions[0]))
            orientation_distance = quaternion_distance(local_quaternion, template.local_quaternions_xyzw[0])
            scored.append((position_distance + 0.05 * orientation_distance, template.sha256, template))
        return [row[2] for row in sorted(scored, key=lambda row: (row[0], row[1]))]

    def _targets(
        self, template: SkillTemplate, snapshot: GeometricSnapshot, retry_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        positions = []
        quaternions = []
        offset = self._offsets(template.effect_id)[retry_index % 4]
        for local_position, local_quaternion in zip(
            template.local_positions, template.local_quaternions_xyzw
        ):
            target_position, target_quaternion = pose_from_local(
                local_position + offset,
                local_quaternion,
                snapshot.effect_fixture_position,
                snapshot.effect_fixture_quaternion_xyzw,
            )
            positions.append(target_position)
            quaternions.append(target_quaternion)
        return np.asarray(positions, dtype=np.float32), np.asarray(quaternions, dtype=np.float32)

    @staticmethod
    def _nearest_waypoint(
        positions: np.ndarray,
        quaternions: np.ndarray,
        snapshot: GeometricSnapshot,
        position_tolerance_m: float,
    ) -> Tuple[int, float, float]:
        distances = np.linalg.norm(positions - snapshot.eef_position[None], axis=1)
        orientation = np.asarray(
            [quaternion_distance(value, snapshot.eef_quaternion_xyzw) for value in quaternions]
        )
        score = distances + 0.02 * orientation
        nearest = int(np.argmin(score))
        position_error = float(distances[nearest])
        orientation_error = float(orientation[nearest])
        return nearest, position_error, orientation_error

    def _monotonic_target(
        self,
        context_key: Tuple[str, str, str, int],
        mode: ExecutionMode,
        positions: np.ndarray,
        quaternions: np.ndarray,
        snapshot: GeometricSnapshot,
    ) -> Tuple[int, int, int, float, float]:
        """Track one template without regressing to an earlier contact pose.

        Nearest-point projection alone is ambiguous on curved or contact-rich
        paths.  In development traces it moved the switch trajectory from
        waypoint 22 back to waypoint 18 and then repeated the old command.
        The cursor is deterministic state local to one effect attempt.  A
        retry starts at waypoint zero so that a different template/offset is
        actually re-approached instead of entering near its failed endpoint.
        """

        nearest, position_error, orientation_error = self._nearest_waypoint(
            positions, quaternions, snapshot, self.position_tolerance_m
        )
        previous = self._waypoint_cursor.get(context_key)
        if previous is None:
            cursor = 0 if mode == ExecutionMode.RETRY else nearest
        else:
            cursor = max(int(previous), nearest)
        cursor = min(cursor, len(positions) - 1)
        cursor_position_error = float(
            np.linalg.norm(positions[cursor] - snapshot.eef_position)
        )
        cursor_orientation_error = quaternion_distance(
            quaternions[cursor], snapshot.eef_quaternion_xyzw
        )
        reached = (
            cursor_position_error <= self.position_tolerance_m
            and cursor_orientation_error <= ORIENTATION_TOLERANCE_RAD
        )
        target = min(cursor + 2, len(positions) - 1) if reached else cursor
        self._waypoint_cursor[context_key] = target if reached else cursor
        return nearest, cursor, target, position_error, orientation_error

    @staticmethod
    def _grasp_retry_entry(template: SkillTemplate) -> int:
        """Return a demonstrated pre-grasp waypoint about 6 cm above contact."""

        gripper = np.asarray(template.gripper)
        close_transitions = np.flatnonzero(
            (gripper[1:] > 0.0) & (gripper[:-1] < 0.0)
        ) + 1
        if not len(close_transitions):
            return 0
        close = int(close_transitions[-1])
        close_position = template.local_positions[close]
        before = np.arange(close, dtype=np.int64)
        candidates = before[
            template.local_positions[before, 2] >= close_position[2] + 0.040
        ]
        if not len(candidates):
            return max(0, close - 5)
        deltas = template.local_positions[candidates] - close_position[None]
        score = np.linalg.norm(deltas[:, :2], axis=1) + np.abs(deltas[:, 2] - 0.060)
        return int(candidates[int(np.argmin(score))])

    def action_chunk(
        self,
        state_history: Sequence[GeometricSnapshot],
        task_id: str,
        effect_id: str,
        execution_mode,
        retry_index: int,
    ) -> np.ndarray:
        # Canonicalization is deliberately executed even when the caller has
        # already hashed the input, so extra or malformed fields fail closed.
        executor_input_bytes(state_history, task_id, effect_id, execution_mode, retry_index)
        mode = execution_mode if isinstance(execution_mode, ExecutionMode) else ExecutionMode(str(execution_mode))
        retry_index = int(retry_index)
        snapshot = list(state_history)[-1]
        order_key = (task_id, effect_id)
        frozen_order = self._template_order_sha256.get(order_key)
        if frozen_order is None:
            initial_order = self._ordered_templates(snapshot, task_id, effect_id)
            frozen_order = tuple(value.sha256 for value in initial_order)
            self._template_order_sha256[order_key] = frozen_order
        by_sha256 = {
            value.sha256: value for value in self.templates[(task_id, effect_id)]
        }
        ordered = [by_sha256[value] for value in frozen_order]
        rank = 0 if mode == ExecutionMode.EXECUTE else retry_index % len(ordered)
        context_key = (task_id, effect_id, mode.value, retry_index)
        selected_sha256 = self._selected_template_sha256.get(context_key)
        if selected_sha256 is None:
            template = ordered[rank]
            self._selected_template_sha256[context_key] = template.sha256
        else:
            template = next(
                value
                for value in self.templates[(task_id, effect_id)]
                if value.sha256 == selected_sha256
            )
            rank = ordered.index(template)

        if self.variant == ExecutorVariant.WORLD_FRAME_OPEN_LOOP:
            positions = template.world_positions
            quaternions = template.world_quaternions_xyzw
        else:
            positions, quaternions = self._targets(
                template,
                snapshot,
                retry_index if self.variant == ExecutorVariant.FROZEN_FULL else 0,
            )
        retry_entry_active = False
        if (
            self.retry_reapproach
            and mode == ExecutionMode.RETRY
            and effect_id in ("MOKA_GRASPED", "BOWL_GRASPED")
            and not self._retry_entry_complete.get(context_key, False)
        ):
            nearest, position_error, orientation_error = self._nearest_waypoint(
                positions, quaternions, snapshot, self.position_tolerance_m
            )
            cursor = self._grasp_retry_entry(template)
            entry_position_error = float(
                np.linalg.norm(positions[cursor] - snapshot.eef_position)
            )
            entry_orientation_error = quaternion_distance(
                quaternions[cursor], snapshot.eef_quaternion_xyzw
            )
            if entry_position_error <= 0.020 and entry_orientation_error <= 0.20:
                self._retry_entry_complete[context_key] = True
                target = cursor
            else:
                target = cursor
                retry_entry_active = True
        elif self.monotonic_progress:
            nearest, cursor, target, position_error, orientation_error = self._monotonic_target(
                context_key,
                mode,
                positions,
                quaternions,
                snapshot,
            )
        else:
            nearest, position_error, orientation_error = self._nearest_waypoint(
                positions, quaternions, snapshot, self.position_tolerance_m
            )
            reached = (
                position_error <= self.position_tolerance_m
                and orientation_error <= ORIENTATION_TOLERANCE_RAD
            )
            cursor = nearest
            target = min(nearest + 2, len(positions) - 1) if reached else nearest

        if self.variant in (
            ExecutorVariant.WORLD_FRAME_OPEN_LOOP,
            ExecutorVariant.LOCAL_FRAME_OPEN_LOOP,
        ):
            indices = np.minimum(
                np.arange(target, target + ACTION_HORIZON), len(template.actions) - 1
            )
            chunk = np.asarray(template.actions[indices], dtype=np.float32).copy()
            if self.variant == ExecutorVariant.LOCAL_FRAME_OPEN_LOOP:
                source_rotation = quaternion_to_matrix_xyzw(template.source_frame_quaternion_xyzw)
                current_rotation = quaternion_to_matrix_xyzw(snapshot.effect_fixture_quaternion_xyzw)
                rotation_delta = current_rotation.dot(source_rotation.T)
                chunk[:, :3] = np.asarray(
                    [rotation_delta.dot(value) for value in chunk[:, :3]], dtype=np.float32
                )
        else:
            position_delta = positions[target] - snapshot.eef_position
            orientation_delta = quaternion_error_rotvec(
                quaternions[target], snapshot.eef_quaternion_xyzw
            )
            action = np.zeros((ACTION_DIM,), dtype=np.float32)
            action[:3] = np.clip(self.position_gain * position_delta, -1.0, 1.0)
            action[3:6] = np.clip(
                self.orientation_gain * orientation_delta, -1.0, 1.0
            )
            # The demonstrated Cartesian command is the deterministic
            # feed-forward term; pose error is the feedback term. Contact-rich
            # switch and drawer motion cannot be recovered from kinematic pose
            # error alone because the constrained end effector may not reach a
            # waypoint until the corresponding force/rotation is commanded.
            if self.demonstration_feedforward and not retry_entry_active:
                action[:6] += template.actions[target, :6]
            action[:6] = np.clip(action[:6], -1.0, 1.0)
            action[6] = -1.0 if retry_entry_active else float(template.gripper[target])
            chunk = np.repeat(action[None], ACTION_HORIZON, axis=0)

        chunk = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
        if chunk.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(chunk).all():
            raise RuntimeError("executor emitted invalid action chunk")
        self.last_trace = {
            "variant": self.variant.value,
            "template_id": template.template_id,
            "template_sha256": template.sha256,
            "template_rank": rank,
            "nearest_waypoint": nearest,
            "monotonic_waypoint_cursor": cursor,
            "target_waypoint": target,
            "waypoint_count": len(positions),
            "nearest_position_error_m": position_error,
            "nearest_orientation_error_rad": orientation_error,
            "target_position_error_m": float(np.linalg.norm(positions[target] - snapshot.eef_position)),
            "target_orientation_error_rad": quaternion_distance(
                quaternions[target], snapshot.eef_quaternion_xyzw
            ),
            "retry_index": retry_index,
            "execution_mode": mode.value,
            "demonstration_feedforward": self.demonstration_feedforward,
            "monotonic_progress": self.monotonic_progress,
            "retry_reapproach": self.retry_reapproach,
            "retry_reapproach_active": retry_entry_active,
        }
        return chunk

    def frozen_manifest(self) -> dict:
        return {
            "executor": "RetargetedGeometricSkillExecutor",
            "variant": self.variant.value,
            "manifest_sha256": self.manifest_sha256,
            "position_gain": self.position_gain,
            "orientation_gain": self.orientation_gain,
            "position_tolerance_m": self.position_tolerance_m,
            "demonstration_feedforward": self.demonstration_feedforward,
            "monotonic_progress": self.monotonic_progress,
            "retry_reapproach": self.retry_reapproach,
            "action_horizon": ACTION_HORIZON,
            "executed_prefix": 4,
            "template_counts": {
                "%s/%s" % key: len(value) for key, value in sorted(self.templates.items())
            },
            "forbidden_inputs": [
                "memory",
                "epistemic_state",
                "fault_identity",
                "effect_truth",
                "reward",
                "task_success",
                "future_state",
                "init_index",
            ],
        }

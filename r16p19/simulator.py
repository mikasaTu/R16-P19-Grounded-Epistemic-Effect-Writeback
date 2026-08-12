"""LIBERO adapters used only for physical labels, oracle evaluation, and state BC."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch

from .config import MAX_STATE_DIM, TASKS, TaskSpec


def make_env(task: TaskSpec, camera_obs: bool = False):
    from libero.libero.envs.env_wrapper import ControlEnv

    cameras = ["agentview", "robot0_eye_in_hand"] if camera_obs else []
    return ControlEnv(
        bddl_file_name=str(task.bddl_path),
        has_renderer=False,
        has_offscreen_renderer=bool(camera_obs),
        use_camera_obs=bool(camera_obs),
        camera_names=cameras,
        camera_heights=128,
        camera_widths=128,
        horizon=700,
        ignore_done=True,
        hard_reset=False,
    )


def reset_to_state(env, state: np.ndarray):
    env.reset()
    observation = env.set_init_state(np.asarray(state, dtype=np.float64))
    controller = env.robots[0].controller
    controller.update(force=True)
    controller.reset_goal()
    return observation


def load_init_states(task: TaskSpec) -> np.ndarray:
    value = torch.load(str(task.init_path), map_location="cpu")
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != task.state_dim:
        raise ValueError("unexpected init-state shape %r for %s" % (value.shape, task.key))
    return value


def padded_flat_state(state: np.ndarray) -> np.ndarray:
    """Drop MuJoCo time and pad qpos/qvel to the preregistered fixed width."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    physical = state[1:]
    if len(physical) > MAX_STATE_DIM:
        raise ValueError("flattened physical state exceeds fixed actor schema")
    result = np.zeros((MAX_STATE_DIM,), dtype=np.float32)
    result[: len(physical)] = physical
    return result


def _grasp_truth(base, object_name: str) -> bool:
    obj = base.objects_dict[object_name]
    return bool(base._check_grasp(base.robots[0].gripper, obj.contact_geoms))


def effect_truths(env, task: TaskSpec) -> Dict[str, bool]:
    """Evaluate frozen effect ontology from privileged simulator state.

    This function is deliberately kept outside the memory implementation.
    """
    base = env.env
    truths: Dict[str, bool] = {}
    for effect, spec in zip(task.effects, task.predicate_specs):
        kind = spec[0]
        if kind == "predicate":
            value = bool(base._eval_predicate(tuple(spec[1:])))
        elif kind == "grasp":
            value = _grasp_truth(base, spec[1])
        elif kind in ("released_on", "released_in"):
            predicate = "on" if kind == "released_on" else "in"
            value = bool(base._eval_predicate((predicate, spec[1], spec[2]))) and not _grasp_truth(
                base, spec[1]
            )
        else:
            raise ValueError("unknown predicate spec %r" % (spec,))
        truths[effect] = value
    return truths


@dataclass(frozen=True)
class DemoLabels:
    task_key: str
    episode_id: str
    length: int
    transition_indices: Mapping[str, int]
    stable_transition_indices: Mapping[str, int]
    final_success: bool
    inferred_transition_methods: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_key": self.task_key,
            "episode_id": self.episode_id,
            "length": self.length,
            "transition_indices": dict(self.transition_indices),
            "stable_transition_indices": dict(self.stable_transition_indices),
            "final_success": self.final_success,
            "inferred_transition_methods": dict(self.inferred_transition_methods),
        }


def label_demo(env, task: TaskSpec, episode_id: str, stable_frames: int = 3) -> DemoLabels:
    transitions: Dict[str, int] = {}
    stable: Dict[str, int] = {}
    streak = {effect: 0 for effect in task.effects}
    streak_start = {effect: 0 for effect in task.effects}
    history = {effect: [] for effect in task.effects}
    inferred: Dict[str, str] = {}
    env.reset()
    with h5py.File(str(task.dataset_path), "r") as handle:
        states = handle["data"][episode_id]["states"]
        actions = np.asarray(handle["data"][episode_id]["actions"], dtype=np.float64)
        length = len(states)
        for index in range(length):
            env.set_init_state(np.asarray(states[index], dtype=np.float64))
            truth = effect_truths(env, task)
            for effect in task.effects:
                history[effect].append(bool(truth[effect]))
                if truth[effect]:
                    transitions.setdefault(effect, index)
                    if streak[effect] == 0:
                        streak_start[effect] = index
                    streak[effect] += 1
                    if streak[effect] >= stable_frames:
                        stable.setdefault(effect, streak_start[effect])
                else:
                    streak[effect] = 0
        env.set_init_state(np.asarray(states[length - 1], dtype=np.float64))
        final_success = bool(env.check_success())
    # For effects that are true at episode end, use the start of the final
    # uninterrupted run. This prevents a transient contact dropout from being
    # mislabeled as a stable release (notably the drawer task around t=159).
    for effect in task.effects:
        values = history[effect]
        if values and values[-1]:
            start = len(values) - 1
            while start > 0 and values[start - 1]:
                start -= 1
            if len(values) - start >= stable_frames:
                stable[effect] = start
    # MuJoCo state reconstruction occasionally loses a contact bit even though
    # the official demo necessarily transports the object. For a missing grasp
    # only, conservatively label the first sustained object displacement after
    # a close-action run. This is a physical transport witness, not command-as-
    # progress, and the inference method is carried into the artifact.
    for effect, spec in zip(task.effects, task.predicate_specs):
        if spec[0] != "grasp" or effect in transitions:
            continue
        address = env.sim.model.get_joint_qpos_addr(spec[1] + "_joint0")
        qpos_start = int(address[0] if isinstance(address, tuple) else address)
        with h5py.File(str(task.dataset_path), "r") as handle:
            raw = np.asarray(handle["data"][episode_id]["states"], dtype=np.float64)
        positions = raw[:, 1 + qpos_start : 1 + qpos_start + 3]
        close = actions[:, -1] > 0.5
        starts = np.flatnonzero(close & np.concatenate(([True], ~close[:-1])))
        inferred_index = None
        for start in starts:
            displacement = np.linalg.norm(positions[start:] - positions[start], axis=1)
            candidates = np.flatnonzero(displacement >= 0.005)
            if len(candidates):
                inferred_index = int(start + candidates[0])
                break
        if inferred_index is None:
            raise RuntimeError("cannot infer missing physical grasp for %s:%s" % (task.key, episode_id))
        transitions[effect] = inferred_index
        stable[effect] = inferred_index
        inferred[effect] = "first_5mm_sustained_object_transport_after_close_run"
    return DemoLabels(
        task.key,
        episode_id,
        length,
        transitions,
        stable,
        final_success,
        inferred,
    )


def label_demos(task: TaskSpec, episodes: Iterable[str]) -> List[DemoLabels]:
    env = make_env(task, camera_obs=False)
    try:
        return [label_demo(env, task, episode) for episode in episodes]
    finally:
        env.close()


def frame_bytes(task: TaskSpec, episode_id: str, index: int, sensor: str) -> bytes:
    key = {
        "agentview": "agentview_rgb",
        "robot0_eye_in_hand": "eye_in_hand_rgb",
    }[sensor]
    with h5py.File(str(task.dataset_path), "r") as handle:
        return np.asarray(handle["data"][episode_id]["obs"][key][index]).tobytes()


def frame_digest(task: TaskSpec, episode_id: str, index: int, sensor: str) -> str:
    return hashlib.sha256(frame_bytes(task, episode_id, index, sensor)).hexdigest()


def load_actor_episode(task: TaskSpec, episode_id: str, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(str(task.dataset_path), "r") as handle:
        demo = handle["data"][episode_id]
        raw_states = np.asarray(demo["states"], dtype=np.float32)
        actions = np.asarray(demo["actions"], dtype=np.float32)
    count = min(len(raw_states), len(actions))
    features = np.stack([padded_flat_state(value) for value in raw_states[:count]])
    chunks = np.empty((count, horizon, actions.shape[-1]), dtype=np.float32)
    for index in range(count):
        stop = min(count, index + horizon)
        chunks[index, : stop - index] = actions[index:stop]
        chunks[index, stop - index :] = actions[stop - 1]
    return features, chunks


def deterministic_target_effect(task_key: str, unit_id: str, condition: str) -> int:
    value = hashlib.sha256((task_key + "|" + unit_id + "|" + condition).encode()).digest()
    return int.from_bytes(value[:8], "big") % len(TASKS[task_key].effects)

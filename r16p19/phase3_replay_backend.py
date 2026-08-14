"""Frozen same-demonstration effect replay backend for Phase-3.

This module intentionally has no dependency on the memory implementation,
epistemic states, fault identities, or memory-decision history.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

import numpy as np

from .config import ACTION_DIM, TASKS
from .phase3_snapshot_bank import array_sha256, observation_sha256, sha256_file
from .simulator import effect_truths, make_env, reset_to_state


class ExecutionMode(str, enum.Enum):
    EXECUTE = "EXECUTE"
    RETRY = "RETRY"
    ROLLBACK_REPLAY = "ROLLBACK_REPLAY"
    REOBSERVE = "REOBSERVE"


@dataclass(frozen=True)
class SnapshotSegment:
    split: str
    task_key: str
    source_episode: str
    effect_id: str
    snapshot_path: Path
    snapshot_sha256: str
    entry_state_sha256: str
    action_segment_sha256: str
    action_count: int
    valid: bool


@dataclass
class EffectExecutionResult:
    task_key: str
    source_episode: str
    effect_id: str
    entry_snapshot_id: str
    execution_mode: str
    suppress_actions: bool
    action_steps: int
    source_action_sha256: str
    executed_action_sha256: str
    entry_state_sha256: str
    terminal_state_sha256: str
    entry_observation_sha256: str
    terminal_observation_sha256: str
    physical_truth_before: bool
    physical_truth_after: bool
    truth_timeline: List[bool]
    predicate_stability_duration: int
    frames: List[np.ndarray]

    def public_record(self) -> dict:
        return {
            key: value
            for key, value in vars(self).items()
            if key not in {"frames", "physical_truth_before", "physical_truth_after"}
        }


def _video_frame(observation: Mapping[str, object]) -> np.ndarray | None:
    for key in ("agentview_image", "agentview_rgb"):
        if key in observation:
            value = np.asarray(observation[key], dtype=np.uint8)
            if value.ndim == 3:
                return np.flipud(value).copy()
    return None


def _stable_suffix(values: Sequence[bool]) -> int:
    count = 0
    for value in reversed(values):
        if not value:
            break
        count += 1
    return count


class FrozenEffectReplayBackend:
    """Reset an exact entry snapshot and execute its exact frozen actions."""

    def __init__(
        self,
        manifest_path: Path,
        snapshot_root: Path,
        camera_obs: bool = False,
        frame_stride: int = 4,
    ):
        self.manifest_path = Path(manifest_path)
        self.snapshot_root = Path(snapshot_root)
        self.camera_obs = bool(camera_obs)
        self.frame_stride = max(1, int(frame_stride))
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("segments", [])
        self._segments: Dict[tuple[str, str, str, str], SnapshotSegment] = {}
        for row in rows:
            key = (
                str(row["split"]),
                str(row["task_key"]),
                str(row["source_episode"]),
                str(row["effect_id"]),
            )
            path = self.snapshot_root / str(row["snapshot_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256_file(path) != row["snapshot_sha256"]:
                raise RuntimeError("snapshot hash drift: %s" % path)
            self._segments[key] = SnapshotSegment(
                split=key[0],
                task_key=key[1],
                source_episode=key[2],
                effect_id=key[3],
                snapshot_path=path,
                snapshot_sha256=str(row["snapshot_sha256"]),
                entry_state_sha256=str(row["entry_state_sha256"]),
                action_segment_sha256=str(row["action_segment_sha256"]),
                action_count=int(row["action_count"]),
                valid=bool(row["valid"]),
            )
        self._envs: MutableMapping[str, object] = {}
        self._observations: MutableMapping[str, Mapping[str, object]] = {}
        self._active: MutableMapping[str, SnapshotSegment] = {}

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.manifest_path)

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()
        self._observations.clear()
        self._active.clear()

    def _env(self, task_key: str):
        if task_key not in self._envs:
            self._envs[task_key] = make_env(TASKS[task_key], camera_obs=self.camera_obs)
        return self._envs[task_key]

    def segment(
        self, split: str, task_key: str, source_episode: str, effect_id: str
    ) -> SnapshotSegment:
        key = (split, task_key, source_episode, effect_id)
        if key not in self._segments:
            raise KeyError("unknown snapshot segment %r" % (key,))
        segment = self._segments[key]
        if not segment.valid:
            raise RuntimeError("invalid snapshot segment %r" % (key,))
        return segment

    @staticmethod
    def snapshot_id(segment: SnapshotSegment) -> str:
        return "%s:%s:%s:%s" % (
            segment.split,
            segment.task_key,
            segment.source_episode,
            segment.effect_id,
        )

    @staticmethod
    def _load(segment: SnapshotSegment) -> dict[str, np.ndarray]:
        with np.load(segment.snapshot_path, allow_pickle=False) as handle:
            return {key: np.asarray(handle[key]).copy() for key in handle.files}

    def reset_to_segment(self, segment: SnapshotSegment, seed: int) -> Mapping[str, object]:
        arrays = self._load(segment)
        entry_state = np.asarray(arrays["entry_state"], dtype=np.float64)
        if array_sha256(entry_state, "<f8") != segment.entry_state_sha256:
            raise RuntimeError("entry-state hash drift")
        env = self._env(segment.task_key)
        observation = reset_to_state(env, entry_state, seed=int(seed))
        self._observations[segment.task_key] = observation
        self._active[segment.task_key] = segment
        return observation

    def restore_state(
        self, task_key: str, state: np.ndarray, seed: int
    ) -> Mapping[str, object]:
        observation = reset_to_state(
            self._env(task_key), np.asarray(state, dtype=np.float64), seed=int(seed)
        )
        self._observations[task_key] = observation
        return observation

    def current_state(self, task_key: str) -> np.ndarray:
        return np.asarray(
            self._env(task_key).sim.get_state().flatten(), dtype="<f8"
        ).copy()

    def current_observation(self, task_key: str) -> Mapping[str, object]:
        if task_key not in self._observations:
            raise RuntimeError("task has no active observation")
        return self._observations[task_key]

    def current_truth(self, task_key: str, effect_id: str) -> bool:
        return bool(effect_truths(self._env(task_key), TASKS[task_key])[effect_id])

    def execute_effect(
        self,
        split: str,
        task_key: str,
        source_episode: str,
        effect_id: str,
        entry_snapshot_id: str,
        execution_mode: ExecutionMode | str,
        *,
        seed: int,
        suppress_actions: bool = False,
        collect_frames: bool = False,
    ) -> EffectExecutionResult:
        mode = ExecutionMode(execution_mode)
        segment = self.segment(split, task_key, source_episode, effect_id)
        expected_id = self.snapshot_id(segment)
        if entry_snapshot_id != expected_id:
            raise ValueError("entry snapshot identity mismatch")
        arrays = self._load(segment)
        if mode == ExecutionMode.REOBSERVE:
            if task_key not in self._observations:
                raise RuntimeError("REOBSERVE requires an active physical state")
            actions = np.zeros((8, ACTION_DIM), dtype="<f4")
        else:
            self.reset_to_segment(segment, seed=seed)
            actions = np.asarray(arrays["actions"], dtype="<f4")
            if array_sha256(actions, "<f4") != segment.action_segment_sha256:
                raise RuntimeError("action-segment hash drift")

        env = self._env(task_key)
        observation = self.current_observation(task_key)
        entry_state = self.current_state(task_key)
        entry_observation_hash = observation_sha256(observation)
        physical_before = self.current_truth(task_key, effect_id)
        frames: List[np.ndarray] = []
        first_frame = _video_frame(observation) if collect_frames else None
        if first_frame is not None:
            frames.append(first_frame)
        executed = np.zeros_like(actions) if suppress_actions else actions.copy()
        timeline: List[bool] = []
        for index, action in enumerate(executed):
            observation, _, _, _ = env.step(np.asarray(action, dtype=np.float64))
            self._observations[task_key] = observation
            timeline.append(self.current_truth(task_key, effect_id))
            if collect_frames and (index + 1) % self.frame_stride == 0:
                frame = _video_frame(observation)
                if frame is not None:
                    frames.append(frame)
        terminal_state = self.current_state(task_key)
        physical_after = self.current_truth(task_key, effect_id)
        return EffectExecutionResult(
            task_key=task_key,
            source_episode=source_episode,
            effect_id=effect_id,
            entry_snapshot_id=entry_snapshot_id,
            execution_mode=mode.value,
            suppress_actions=bool(suppress_actions),
            action_steps=int(len(executed)),
            source_action_sha256=array_sha256(actions, "<f4"),
            executed_action_sha256=array_sha256(executed, "<f4"),
            entry_state_sha256=array_sha256(entry_state, "<f8"),
            terminal_state_sha256=array_sha256(terminal_state, "<f8"),
            entry_observation_sha256=entry_observation_hash,
            terminal_observation_sha256=observation_sha256(observation),
            physical_truth_before=physical_before,
            physical_truth_after=physical_after,
            truth_timeline=timeline,
            predicate_stability_duration=_stable_suffix(timeline),
            frames=frames,
        )

    def manifest_rows(self, split: str | None = None) -> List[dict]:
        rows = []
        for key, value in sorted(self._segments.items()):
            if split is not None and key[0] != split:
                continue
            rows.append(
                {
                    "split": value.split,
                    "task_key": value.task_key,
                    "source_episode": value.source_episode,
                    "effect_id": value.effect_id,
                    "entry_snapshot_id": self.snapshot_id(value),
                    "snapshot_sha256": value.snapshot_sha256,
                    "action_segment_sha256": value.action_segment_sha256,
                    "action_count": value.action_count,
                    "valid": value.valid,
                }
            )
        return rows

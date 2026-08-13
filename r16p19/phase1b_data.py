"""Frozen causal actor-data construction for LIBERO Phase-1B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from .config import ACTION_DIM, ACTION_HORIZON, MAX_STATE_DIM, TASKS
from .phase1b_actor import EFFECT_TO_INDEX, TASK_TO_INDEX, canonical_state_history


HISTORY_LENGTH = 4
EXECUTED_PREFIX = 4


@dataclass(frozen=True)
class SampleRef:
    task_key: str
    episode_id: str
    effect_id: str
    action_index: int
    segment_start: int
    segment_stop: int


@dataclass
class ActorDataset:
    state_histories: np.ndarray
    action_chunks: np.ndarray
    task_indices: np.ndarray
    effect_indices: np.ndarray
    refs: List[SampleRef]

    def __post_init__(self) -> None:
        count = len(self.state_histories)
        if self.state_histories.shape != (count, HISTORY_LENGTH, MAX_STATE_DIM):
            raise ValueError("invalid state-history array shape %r" % (self.state_histories.shape,))
        if self.action_chunks.shape != (count, ACTION_HORIZON, ACTION_DIM):
            raise ValueError("invalid action-chunk array shape %r" % (self.action_chunks.shape,))
        if self.task_indices.shape != (count,) or self.effect_indices.shape != (count,):
            raise ValueError("invalid task/effect index array shape")
        if len(self.refs) != count:
            raise ValueError("sample-reference count mismatch")
        if not (
            np.isfinite(self.state_histories).all()
            and np.isfinite(self.action_chunks).all()
        ):
            raise ValueError("actor dataset contains non-finite values")

    def __len__(self) -> int:
        return len(self.state_histories)

    def subset(self, indices: np.ndarray) -> "ActorDataset":
        indices = np.asarray(indices, dtype=np.int64)
        return ActorDataset(
            self.state_histories[indices],
            self.action_chunks[indices],
            self.task_indices[indices],
            self.effect_indices[indices],
            [self.refs[int(index)] for index in indices],
        )

    def effect_counts(self) -> Dict[str, int]:
        inverse = {value: key for key, value in EFFECT_TO_INDEX.items()}
        unique, counts = np.unique(self.effect_indices, return_counts=True)
        return {
            inverse[int(index)]: int(count)
            for index, count in sorted(zip(unique, counts), key=lambda item: int(item[0]))
        }


@dataclass(frozen=True)
class ActorNormalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    continuous_action_mean: np.ndarray
    continuous_action_std: np.ndarray
    gripper_positive_weight: float
    gripper_positive_count: int
    gripper_negative_count: int

    @classmethod
    def from_training_data(cls, dataset: ActorDataset) -> "ActorNormalization":
        states = dataset.state_histories.reshape(-1, MAX_STATE_DIM)
        continuous = dataset.action_chunks[..., : ACTION_DIM - 1].reshape(
            -1, ACTION_DIM - 1
        )
        gripper = dataset.action_chunks[..., ACTION_DIM - 1].reshape(-1)
        positive = int(np.sum(gripper > 0.0))
        negative = int(np.sum(gripper <= 0.0))
        if positive == 0 or negative == 0:
            raise ValueError("both gripper classes are required")
        return cls(
            state_mean=states.mean(axis=0).astype(np.float32),
            state_std=np.maximum(states.std(axis=0), 1e-4).astype(np.float32),
            continuous_action_mean=continuous.mean(axis=0).astype(np.float32),
            continuous_action_std=np.maximum(continuous.std(axis=0), 1e-4).astype(
                np.float32
            ),
            gripper_positive_weight=float(negative / positive),
            gripper_positive_count=positive,
            gripper_negative_count=negative,
        )

    def to_dict(self) -> dict:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "continuous_action_mean": self.continuous_action_mean.tolist(),
            "continuous_action_std": self.continuous_action_std.tolist(),
            "gripper_positive_weight": self.gripper_positive_weight,
            "gripper_positive_count": self.gripper_positive_count,
            "gripper_negative_count": self.gripper_negative_count,
        }


def read_label_rows(path: Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _transition_index(label: Mapping[str, object], effect: str) -> Optional[int]:
    stable = label.get("stable_transition_indices", {})
    transitions = label.get("transition_indices", {})
    if effect in stable:
        return int(stable[effect])
    if effect in transitions:
        return int(transitions[effect])
    return None


def effect_segment(
    label: Mapping[str, object],
    effects: Sequence[str],
    effect_index: int,
    length: int,
) -> Optional[Tuple[int, int]]:
    """Return exactly the segment frozen in the preregistered data audit."""

    current = _transition_index(label, effects[effect_index])
    if current is None:
        return None
    if effect_index == 0:
        start = 0
    else:
        previous = _transition_index(label, effects[effect_index - 1])
        start = max(0, int(previous or 0) - EXECUTED_PREFIX)
    stop = min(int(length), int(current) + 1)
    if stop <= start:
        return None
    return start, stop


def _causal_history(states: np.ndarray, index: int) -> np.ndarray:
    start = max(0, int(index) - HISTORY_LENGTH + 1)
    return canonical_state_history(states[start : int(index) + 1], HISTORY_LENGTH)


def _bounded_action_chunk(
    actions: np.ndarray, index: int, segment_stop: int
) -> np.ndarray:
    stop = min(int(segment_stop), int(index) + ACTION_HORIZON)
    values = np.asarray(actions[index:stop], dtype=np.float32)
    if not len(values):
        raise ValueError("empty action chunk")
    if len(values) < ACTION_HORIZON:
        values = np.concatenate(
            (
                values,
                np.repeat(values[-1][None], ACTION_HORIZON - len(values), axis=0),
            ),
            axis=0,
        )
    return values


def build_actor_dataset(
    label_rows: Iterable[Mapping[str, object]], episodes: Sequence[str]
) -> ActorDataset:
    labels = {
        (str(value["task_key"]), str(value["episode_id"])): value
        for value in label_rows
    }
    histories: List[np.ndarray] = []
    chunks: List[np.ndarray] = []
    task_indices: List[int] = []
    effect_indices: List[int] = []
    refs: List[SampleRef] = []
    for task_key, task in TASKS.items():
        with h5py.File(str(task.dataset_path), "r") as handle:
            for episode_id in episodes:
                label = labels[(task_key, episode_id)]
                demo = handle["data"][episode_id]
                states = np.asarray(demo["states"], dtype=np.float32)
                actions = np.asarray(demo["actions"], dtype=np.float32)
                length = min(len(states), len(actions))
                states = states[:length]
                actions = actions[:length]
                for effect_index, effect_id in enumerate(task.effects):
                    segment = effect_segment(label, task.effects, effect_index, length)
                    if segment is None:
                        continue
                    segment_start, segment_stop = segment
                    for action_index in range(segment_start, segment_stop):
                        histories.append(_causal_history(states, action_index))
                        chunks.append(
                            _bounded_action_chunk(actions, action_index, segment_stop)
                        )
                        task_indices.append(TASK_TO_INDEX[task_key])
                        effect_indices.append(EFFECT_TO_INDEX[effect_id])
                        refs.append(
                            SampleRef(
                                task_key,
                                episode_id,
                                effect_id,
                                action_index,
                                segment_start,
                                segment_stop,
                            )
                        )
    return ActorDataset(
        np.asarray(histories, dtype=np.float32),
        np.asarray(chunks, dtype=np.float32),
        np.asarray(task_indices, dtype=np.int64),
        np.asarray(effect_indices, dtype=np.int64),
        refs,
    )


class BalancedEffectSampler:
    """Draw an exactly balanced batch over the eight frozen effects."""

    def __init__(self, effect_indices: np.ndarray, batch_size: int):
        self.effect_indices = np.asarray(effect_indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.effect_count = len(EFFECT_TO_INDEX)
        if self.batch_size % self.effect_count:
            raise ValueError("batch size must be divisible by eight effects")
        self.per_effect = self.batch_size // self.effect_count
        self.pools = {
            index: np.flatnonzero(self.effect_indices == index)
            for index in range(self.effect_count)
        }
        missing = [index for index, values in self.pools.items() if not len(values)]
        if missing:
            raise ValueError("balanced sampling lacks effects %r" % missing)

    def sample(self) -> np.ndarray:
        values = [
            np.random.choice(pool, size=self.per_effect, replace=len(pool) < self.per_effect)
            for pool in self.pools.values()
        ]
        result = np.concatenate(values).astype(np.int64, copy=False)
        np.random.shuffle(result)
        return result


def dataset_manifest(dataset: ActorDataset, split: str) -> dict:
    return {
        "split": str(split),
        "sample_count": len(dataset),
        "effect_counts": dataset.effect_counts(),
        "history_length": HISTORY_LENGTH,
        "action_horizon": ACTION_HORIZON,
        "causal_history": "current_and_past_states_only_left_padded_with_episode_first",
        "chunk_boundary": "never_cross_current_effect_segment_stop",
        "missing_effect_policy": "excluded_without_fabrication",
    }

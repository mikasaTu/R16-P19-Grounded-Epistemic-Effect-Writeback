"""Tiny shared state-BC actor and deterministic retrieval augmentation."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .checkpoints import CheckpointManager
from .config import (
    ACTION_DIM,
    ACTION_HORIZON,
    MAX_EFFECTS,
    MAX_STATE_DIM,
    MEMORY_SUMMARY_DIM,
    TASKS,
    TRAIN_EPISODES,
)
from .simulator import DemoLabels, load_actor_episode, padded_flat_state
from .types import EpistemicState


STATE_ORDER = (
    EpistemicState.REQUESTED,
    EpistemicState.IMAGINED,
    EpistemicState.OBSERVED,
    EpistemicState.VERIFIED,
    EpistemicState.REALIZED,
    EpistemicState.STALLED,
    EpistemicState.INVALIDATED_REALIZATION,
)
INPUT_DIM = MAX_STATE_DIM + len(TASKS) + MAX_EFFECTS + MEMORY_SUMMARY_DIM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def one_hot(index: int, size: int) -> np.ndarray:
    result = np.zeros((size,), dtype=np.float32)
    result[int(index)] = 1.0
    return result


def memory_summary_from_states(states: Sequence[Optional[EpistemicState]]) -> np.ndarray:
    result = np.zeros((MAX_EFFECTS, len(STATE_ORDER)), dtype=np.float32)
    for effect_index, state in enumerate(states[:MAX_EFFECTS]):
        if state is not None:
            result[effect_index, STATE_ORDER.index(state)] = 1.0
    return result.reshape(-1)


def ideal_memory_summary(phase: int) -> np.ndarray:
    values: List[Optional[EpistemicState]] = []
    for index in range(MAX_EFFECTS):
        values.append(
            EpistemicState.REALIZED if index < phase else (
                EpistemicState.REQUESTED if index == phase else None
            )
        )
    return memory_summary_from_states(values)


def assemble_input(state: np.ndarray, task_index: int, phase: int, memory_summary: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            padded_flat_state(state) if np.asarray(state).size != MAX_STATE_DIM else np.asarray(state, dtype=np.float32),
            one_hot(task_index, len(TASKS)),
            one_hot(phase, MAX_EFFECTS),
            np.asarray(memory_summary, dtype=np.float32),
        )
    ).astype(np.float32)


class TinyChunkMLP(nn.Module):
    def __init__(self, input_mean, input_std, action_mean, action_std, hidden_dim=256):
        super().__init__()
        for name, value in (
            ("input_mean", input_mean),
            ("input_std", input_std),
            ("action_mean", action_mean),
            ("action_std", action_std),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        layers: List[nn.Module] = [
            nn.Linear(INPUT_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        ]
        for _ in range(3):
            layers.extend(
                (nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
            )
        layers.append(nn.Linear(hidden_dim, ACTION_HORIZON * ACTION_DIM))
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_arrays(cls, inputs: np.ndarray, chunks: np.ndarray):
        input_std = np.maximum(inputs.std(axis=0), 1e-4).astype(np.float32)
        action_std = np.maximum(chunks.reshape(-1, ACTION_DIM).std(axis=0), 1e-4).astype(np.float32)
        return cls(
            inputs.mean(axis=0).astype(np.float32),
            input_std,
            chunks.mean(axis=(0, 1)).astype(np.float32),
            action_std,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs - self.input_mean) / self.input_std
        value = self.net(normalized).reshape(-1, ACTION_HORIZON, ACTION_DIM)
        return value * self.action_std + self.action_mean

    def loss(self, inputs: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        prediction = self(inputs)
        normalized_error = (prediction - chunks) / self.action_std
        return torch.mean(normalized_error ** 2)


@dataclass
class ActorArrays:
    inputs: np.ndarray
    states: np.ndarray
    chunks: np.ndarray
    task_indices: np.ndarray
    phases: np.ndarray
    refs: List[Tuple[str, str, int]]


def _phase(index: int, labels: DemoLabels, effects: Sequence[str]) -> int:
    transitions = dict(labels.stable_transition_indices)
    transitions.update(
        {key: value for key, value in labels.transition_indices.items() if key not in transitions}
    )
    completed = sum(int(effect in transitions and index >= transitions[effect]) for effect in effects)
    return min(completed, MAX_EFFECTS - 1)


def build_actor_arrays(labels: Iterable[DemoLabels]) -> ActorArrays:
    lookup = {(label.task_key, label.episode_id): label for label in labels}
    all_inputs, all_states, all_chunks = [], [], []
    task_indices, phases, refs = [], [], []
    keys = list(TASKS)
    for task_index, task_key in enumerate(keys):
        task = TASKS[task_key]
        for episode in TRAIN_EPISODES:
            label = lookup[(task_key, episode)]
            states, chunks = load_actor_episode(task, episode, ACTION_HORIZON)
            for index in range(len(states)):
                phase = _phase(index, label, task.effects)
                summary = ideal_memory_summary(phase)
                all_states.append(states[index])
                all_chunks.append(chunks[index])
                all_inputs.append(assemble_input(states[index], task_index, phase, summary))
                task_indices.append(task_index)
                phases.append(phase)
                refs.append((task_key, episode, index))
    return ActorArrays(
        np.asarray(all_inputs, dtype=np.float32),
        np.asarray(all_states, dtype=np.float32),
        np.asarray(all_chunks, dtype=np.float32),
        np.asarray(task_indices, dtype=np.int64),
        np.asarray(phases, dtype=np.int64),
        refs,
    )


def train_actor(
    arrays: ActorArrays,
    checkpoint_root: Path,
    device: torch.device,
    steps: int = 3000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    checkpoint_interval: int = 1000,
    wandb_run=None,
) -> Tuple[TinyChunkMLP, dict]:
    model = TinyChunkMLP.from_arrays(arrays.inputs, arrays.chunks).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count >= 10_000_000:
        raise RuntimeError("tiny actor exceeds 10M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1))
    manager = CheckpointManager(Path(checkpoint_root) / "actor")
    start_step, _ = manager.load_latest(model, optimizer, scheduler, map_location=device)
    metrics_path = Path(checkpoint_root) / "actor" / "train_metrics.jsonl"
    model.train()
    for step in range(start_step + 1, steps + 1):
        indices = np.random.randint(0, len(arrays.inputs), size=batch_size)
        inputs = torch.as_tensor(arrays.inputs[indices], device=device)
        chunks = torch.as_tensor(arrays.chunks[indices], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs, chunks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == steps:
            row = {"step": step, "loss": float(loss.detach().cpu()), "time_unix": time.time()}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print("TRAIN_STEP_LOSS step=%d loss=%.8f" % (step, row["loss"]), flush=True)
            if wandb_run is not None:
                wandb_run.log({"actor/train_loss": row["loss"], "optimizer_step": step}, step=step)
        if step % checkpoint_interval == 0 or step == steps:
            manager.save(
                step,
                model,
                optimizer,
                scheduler,
                extra={"parameter_count": parameter_count, "actor_family": "retrieval_augmented_tiny_mlp"},
            )
            print("CHECKPOINT_COMPLETE component=actor step=%d" % step, flush=True)
    model.eval()
    return model, {
        "parameter_count": parameter_count,
        "optimizer_steps": steps,
        "training_samples": len(arrays.inputs),
        "checkpoint": str(manager.latest()),
    }


class RetrievalAugmentedActor:
    def __init__(self, model: TinyChunkMLP, arrays: ActorArrays, device: torch.device, model_weight=0.15, k=5):
        self.model = model
        self.arrays = arrays
        self.device = device
        self.model_weight = float(model_weight)
        self.k = int(k)
        self.state_mean = arrays.states.mean(axis=0)
        self.state_std = np.maximum(arrays.states.std(axis=0), 1e-3)
        self.last_ref: Optional[Tuple[str, str, int]] = None

    def reset(self) -> None:
        self.last_ref = None

    @torch.no_grad()
    def action_chunk(self, state, task_index: int, phase: int, memory_summary: np.ndarray) -> np.ndarray:
        state = padded_flat_state(state) if np.asarray(state).size != MAX_STATE_DIM else np.asarray(state, dtype=np.float32)
        actor_input = assemble_input(state, task_index, phase, memory_summary)
        prediction = self.model(torch.as_tensor(actor_input[None], device=self.device))[0].cpu().numpy()
        eligible = np.flatnonzero(
            (self.arrays.task_indices == task_index) & (self.arrays.phases == phase)
        )
        if self.last_ref is not None:
            last_task, last_episode, last_t = self.last_ref
            temporal = np.asarray(
                [
                    index
                    for index in eligible
                    if self.arrays.refs[index][0] == last_task
                    and self.arrays.refs[index][1] == last_episode
                    and last_t - 4 <= self.arrays.refs[index][2] <= last_t + 24
                ],
                dtype=np.int64,
            )
            if len(temporal) >= self.k:
                eligible = temporal
        delta = (self.arrays.states[eligible] - state[None]) / self.state_std[None]
        distance = np.mean(delta * delta, axis=1)
        count = min(self.k, len(eligible))
        local = np.argpartition(distance, count - 1)[:count]
        selected = eligible[local]
        weights = 1.0 / np.maximum(distance[local], 1e-6)
        weights = weights / weights.sum()
        retrieved = np.tensordot(weights, self.arrays.chunks[selected], axes=(0, 0))
        nearest = int(selected[int(np.argmin(distance[local]))])
        self.last_ref = self.arrays.refs[nearest]
        return np.clip(
            self.model_weight * prediction + (1.0 - self.model_weight) * retrieved,
            -1.0,
            1.0,
        )

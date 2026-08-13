"""Memory-independent low-dimensional ACT actor for LIBERO Phase-1B."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import ACTION_DIM, ACTION_HORIZON, MAX_STATE_DIM, TASKS
from .simulator import padded_flat_state


TASK_KEYS = tuple(TASKS)
EFFECT_KEYS = tuple(effect for task in TASKS.values() for effect in task.effects)
TASK_TO_INDEX = {value: index for index, value in enumerate(TASK_KEYS)}
EFFECT_TO_INDEX = {value: index for index, value in enumerate(EFFECT_KEYS)}


class ExecutionMode(str, Enum):
    EXECUTE = "EXECUTE"
    RETRY = "RETRY"


MODE_TO_INDEX = {mode.value: index for index, mode in enumerate(ExecutionMode)}


@dataclass(frozen=True)
class StateACTConfig:
    history_length: int = 4
    state_dim: int = MAX_STATE_DIM
    action_dim: int = ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    executed_prefix: int = 4
    hidden_dim: int = 256
    transformer_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.1
    feedforward_multiplier: int = 4

    def validate(self) -> None:
        if self.history_length != 4:
            raise ValueError("Phase-1B history length is frozen at four")
        if self.state_dim != MAX_STATE_DIM:
            raise ValueError("Phase-1B padded state dimension drifted")
        if self.action_dim != ACTION_DIM or self.action_horizon != ACTION_HORIZON:
            raise ValueError("Phase-1B action schema drifted")
        if self.executed_prefix != 4:
            raise ValueError("Phase-1B receding-horizon prefix is frozen at four")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden dimension must be divisible by attention heads")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "StateACTConfig":
        result = cls(**{key: value[key] for key in asdict(cls()).keys() if key in value})
        result.validate()
        return result

    def to_dict(self) -> dict:
        return asdict(self)


def _as_mode(execution_mode) -> ExecutionMode:
    if isinstance(execution_mode, ExecutionMode):
        return execution_mode
    try:
        return ExecutionMode(str(execution_mode))
    except ValueError as error:
        raise ValueError("execution_mode must be EXECUTE or RETRY") from error


def _task_index(task_id) -> int:
    if isinstance(task_id, (int, np.integer)):
        index = int(task_id)
        if 0 <= index < len(TASK_KEYS):
            return index
        raise ValueError("task index out of range")
    if str(task_id) in TASK_TO_INDEX:
        return TASK_TO_INDEX[str(task_id)]
    for index, task in enumerate(TASKS.values()):
        if str(task_id) == task.task_id:
            return index
    raise ValueError("unknown task_id %r" % (task_id,))


def _effect_index(effect_id) -> int:
    if isinstance(effect_id, (int, np.integer)):
        index = int(effect_id)
        if 0 <= index < len(EFFECT_KEYS):
            return index
        raise ValueError("effect index out of range")
    try:
        return EFFECT_TO_INDEX[str(effect_id)]
    except KeyError as error:
        raise ValueError("unknown effect_id %r" % (effect_id,)) from error


def canonical_state_history(state_history: Sequence[np.ndarray], history_length: int = 4) -> np.ndarray:
    values = list(state_history)
    if not values:
        raise ValueError("state_history must contain at least one state")
    if len(values) > history_length:
        values = values[-history_length:]
    first = values[0]
    while len(values) < history_length:
        values.insert(0, first)
    padded = []
    for state in values:
        flat = np.asarray(state, dtype=np.float32).reshape(-1)
        padded.append(flat if len(flat) == MAX_STATE_DIM else padded_flat_state(flat))
    result = np.stack(padded).astype(np.float32, copy=False)
    if result.shape != (history_length, MAX_STATE_DIM):
        raise ValueError("unexpected canonical state history shape %r" % (result.shape,))
    return result


def actor_input_bytes(state_history, task_id, effect_id, execution_mode) -> bytes:
    history = canonical_state_history(state_history)
    header = np.asarray(
        [_task_index(task_id), _effect_index(effect_id), MODE_TO_INDEX[_as_mode(execution_mode).value]],
        dtype="<i8",
    )
    return header.tobytes(order="C") + np.asarray(history, dtype="<f4").tobytes(order="C")


def actor_input_hash(state_history, task_id, effect_id, execution_mode) -> str:
    return hashlib.sha256(
        actor_input_bytes(state_history, task_id, effect_id, execution_mode)
    ).hexdigest()


class EffectConditionedStateACT(nn.Module):
    """Deterministic transformer action chunker with no memory-state input."""

    def __init__(
        self,
        config: StateACTConfig,
        state_mean,
        state_std,
        continuous_action_mean,
        continuous_action_std,
        gripper_positive_weight: float,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.register_buffer(
            "continuous_action_mean",
            torch.as_tensor(continuous_action_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "continuous_action_std",
            torch.as_tensor(continuous_action_std, dtype=torch.float32),
        )
        self.register_buffer(
            "gripper_positive_weight",
            torch.as_tensor(float(gripper_positive_weight), dtype=torch.float32),
        )
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.task_embedding = nn.Embedding(len(TASK_KEYS), config.hidden_dim)
        self.effect_embedding = nn.Embedding(len(EFFECT_KEYS), config.hidden_dim)
        self.mode_embedding = nn.Embedding(len(ExecutionMode), config.hidden_dim)
        self.action_queries = nn.Parameter(
            torch.empty(config.action_horizon, config.hidden_dim)
        )
        sequence_length = config.history_length + 3 + config.action_horizon
        self.position_embedding = nn.Parameter(torch.empty(sequence_length, config.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_multiplier * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.continuous_head = nn.Linear(config.hidden_dim, ACTION_DIM - 1)
        self.gripper_head = nn.Linear(config.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, state_history, task_indices, effect_indices, mode_indices):
        if state_history.ndim != 3:
            raise ValueError("state_history tensor must be [batch, history, state]")
        normalized = (state_history - self.state_mean) / self.state_std
        state_tokens = self.state_projection(normalized)
        condition_tokens = torch.stack(
            (
                self.task_embedding(task_indices),
                self.effect_embedding(effect_indices),
                self.mode_embedding(mode_indices),
            ),
            dim=1,
        )
        queries = self.action_queries.unsqueeze(0).expand(state_history.shape[0], -1, -1)
        tokens = torch.cat((state_tokens, condition_tokens, queries), dim=1)
        tokens = tokens + self.position_embedding.unsqueeze(0)
        encoded = self.transformer(tokens)
        query_output = encoded[:, -self.config.action_horizon :]
        continuous_normalized = self.continuous_head(query_output)
        gripper_logits = self.gripper_head(query_output).squeeze(-1)
        return continuous_normalized, gripper_logits

    def loss_components(
        self,
        state_history,
        task_indices,
        effect_indices,
        mode_indices,
        action_chunks,
    ) -> Dict[str, torch.Tensor]:
        continuous_prediction, gripper_logits = self(
            state_history, task_indices, effect_indices, mode_indices
        )
        continuous_target = (
            action_chunks[..., : ACTION_DIM - 1] - self.continuous_action_mean
        ) / self.continuous_action_std
        continuous_loss = torch.nn.functional.smooth_l1_loss(
            continuous_prediction, continuous_target
        )
        gripper_target = (action_chunks[..., ACTION_DIM - 1] > 0.0).to(torch.float32)
        gripper_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            gripper_logits,
            gripper_target,
            pos_weight=self.gripper_positive_weight,
        )
        return {
            "total": continuous_loss + gripper_loss,
            "continuous_smooth_l1": continuous_loss,
            "gripper_weighted_bce": gripper_loss,
        }

    @torch.no_grad()
    def predict(self, state_history, task_indices, effect_indices, mode_indices):
        continuous_normalized, gripper_logits = self(
            state_history, task_indices, effect_indices, mode_indices
        )
        continuous = (
            continuous_normalized * self.continuous_action_std
            + self.continuous_action_mean
        )
        gripper = torch.where(
            gripper_logits >= 0.0,
            torch.ones_like(gripper_logits),
            -torch.ones_like(gripper_logits),
        ).unsqueeze(-1)
        return torch.clamp(torch.cat((continuous, gripper), dim=-1), -1.0, 1.0)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def model_manifest(self) -> dict:
        return {
            "architecture": self.config.to_dict(),
            "parameter_count": self.parameter_count(),
            "task_keys": list(TASK_KEYS),
            "effect_keys": list(EFFECT_KEYS),
            "execution_modes": [mode.value for mode in ExecutionMode],
            "actor_forbidden_inputs": [
                "memory_summary",
                "epistemic_state",
                "simulator_effect_truth",
                "future_state",
            ],
        }


class SkillActor:
    """Uniform Phase-1B actor interface."""

    def action_chunk(self, state_history, task_id, effect_id, execution_mode):
        raise NotImplementedError


class TorchSkillActor(SkillActor):
    def __init__(self, model: EffectConditionedStateACT, device: torch.device, actor_seed=1619):
        self.model = model.to(device).eval()
        self.device = device
        self.actor_seed = int(actor_seed)

    @torch.no_grad()
    def action_chunk(self, state_history, task_id, effect_id, execution_mode):
        history = canonical_state_history(
            state_history, history_length=self.model.config.history_length
        )
        mode = _as_mode(execution_mode)
        states = torch.as_tensor(history[None], device=self.device)
        tasks = torch.as_tensor([_task_index(task_id)], dtype=torch.long, device=self.device)
        effects = torch.as_tensor([_effect_index(effect_id)], dtype=torch.long, device=self.device)
        modes = torch.as_tensor(
            [MODE_TO_INDEX[mode.value]], dtype=torch.long, device=self.device
        )
        return self.model.predict(states, tasks, effects, modes)[0].cpu().numpy()


class PerEffectSkillActor(SkillActor):
    def __init__(self, actors: Mapping[str, TorchSkillActor]):
        missing = sorted(set(EFFECT_KEYS) - set(actors))
        extra = sorted(set(actors) - set(EFFECT_KEYS))
        if missing or extra:
            raise ValueError("per-effect actor mapping mismatch missing=%r extra=%r" % (missing, extra))
        self.actors = dict(actors)

    def action_chunk(self, state_history, task_id, effect_id, execution_mode):
        effect = str(effect_id)
        return self.actors[effect].action_chunk(
            state_history, task_id, effect, execution_mode
        )


def checkpoint_to_actor(checkpoint: Path, device: torch.device, actor_seed=1619) -> TorchSkillActor:
    payload = torch.load(str(Path(checkpoint) / "state.pt"), map_location=device)
    extra = payload.get("extra", {})
    config = StateACTConfig.from_mapping(extra["model_config"])
    normalization = extra["normalization"]
    model = EffectConditionedStateACT(
        config,
        normalization["state_mean"],
        normalization["state_std"],
        normalization["continuous_action_mean"],
        normalization["continuous_action_std"],
        normalization["gripper_positive_weight"],
    ).to(device)
    model.load_state_dict(payload["model"])
    return TorchSkillActor(model, device, actor_seed=actor_seed)


def checkpoint_manifest(checkpoint: Path) -> dict:
    checkpoint = Path(checkpoint)
    payload = torch.load(str(checkpoint / "state.pt"), map_location="cpu")
    digest = hashlib.sha256((checkpoint / "state.pt").read_bytes()).hexdigest()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "global_step": int(payload["global_step"]),
        "model_config": payload["extra"]["model_config"],
        "model_manifest": payload["extra"]["model_manifest"],
        "normalization_sha256": hashlib.sha256(
            json.dumps(payload["extra"]["normalization"], sort_keys=True).encode()
        ).hexdigest(),
        "complete_state_keys": sorted(
            key for key in ("model", "optimizer", "scheduler", "rng", "global_step") if key in payload
        ),
    }

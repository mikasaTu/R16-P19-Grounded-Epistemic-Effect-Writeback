"""Deterministic, resumable training for preregistered Phase-1B actors."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .checkpoints import CheckpointManager
from .phase1b_actor import (
    EFFECT_KEYS,
    EffectConditionedStateACT,
    StateACTConfig,
    checkpoint_manifest,
)
from .phase1b_data import ActorDataset, ActorNormalization, BalancedEffectSampler


@dataclass(frozen=True)
class TrainingConfig:
    family: str
    seed: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    min_steps: int
    validation_interval: int
    early_stopping_patience: int
    checkpoint_interval: int
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.family not in ("primary_shared", "fallback_per_effect"):
            raise ValueError("unknown Phase-1B actor family")
        if self.max_steps < self.min_steps or self.min_steps <= 0:
            raise ValueError("invalid training step bounds")
        if self.validation_interval <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("invalid persistence interval")
        if self.checkpoint_interval % self.validation_interval:
            raise ValueError("checkpoint interval must align with validation interval")


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rewrite_metrics_through_step(path: Path, step: int) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if int(row.get("step", 0)) <= int(step)]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _batch_tensors(
    dataset: ActorDataset, indices: np.ndarray, device: torch.device, modes: np.ndarray
):
    return (
        torch.as_tensor(dataset.state_histories[indices], device=device),
        torch.as_tensor(dataset.task_indices[indices], dtype=torch.long, device=device),
        torch.as_tensor(dataset.effect_indices[indices], dtype=torch.long, device=device),
        torch.as_tensor(modes, dtype=torch.long, device=device),
        torch.as_tensor(dataset.action_chunks[indices], device=device),
    )


@torch.no_grad()
def evaluate_calibration(
    model: EffectConditionedStateACT,
    dataset: ActorDataset,
    device: torch.device,
    batch_size: int = 512,
) -> dict:
    """Evaluate both frozen execution modes and macro-average the eight effects."""

    model.eval()
    per_effect: Dict[str, dict] = {}
    present_effects = sorted(set(int(value) for value in dataset.effect_indices.tolist()))
    for effect_index in present_effects:
        effect_id = EFFECT_KEYS[effect_index]
        source = np.flatnonzero(dataset.effect_indices == effect_index)
        if not len(source):
            raise RuntimeError("calibration split lacks effect %s" % effect_id)
        accumulated = {"total": 0.0, "continuous_smooth_l1": 0.0, "gripper_weighted_bce": 0.0}
        count = 0
        for mode_index in (0, 1):
            for offset in range(0, len(source), int(batch_size)):
                indices = source[offset : offset + int(batch_size)]
                modes = np.full((len(indices),), mode_index, dtype=np.int64)
                tensors = _batch_tensors(dataset, indices, device, modes)
                losses = model.loss_components(*tensors)
                for key in accumulated:
                    accumulated[key] += float(losses[key].detach().cpu()) * len(indices)
                count += len(indices)
        per_effect[effect_id] = {
            key: value / count for key, value in accumulated.items()
        }
        per_effect[effect_id]["sample_mode_pairs"] = count
    macro = {
        key: float(np.mean([value[key] for value in per_effect.values()]))
        for key in ("total", "continuous_smooth_l1", "gripper_weighted_bce")
    }
    return {"macro": macro, "per_effect": per_effect}


def _build_model(
    model_config: StateACTConfig,
    normalization: ActorNormalization,
    device: torch.device,
) -> EffectConditionedStateACT:
    model = EffectConditionedStateACT(
        model_config,
        normalization.state_mean,
        normalization.state_std,
        normalization.continuous_action_mean,
        normalization.continuous_action_std,
        normalization.gripper_positive_weight,
    ).to(device)
    if model.parameter_count() >= 10_000_000:
        raise RuntimeError("Phase-1B actor exceeds frozen 10M parameter ceiling")
    return model


def _candidate_rows(checkpoint_root: Path, min_steps: int) -> List[dict]:
    manager = CheckpointManager(
        checkpoint_root, latest_nonmilestones=10000, milestone_interval=10**12
    )
    rows = []
    for step in manager.complete_steps():
        if step < int(min_steps):
            continue
        path = checkpoint_root / ("step_%09d" % step)
        payload = torch.load(str(path / "state.pt"), map_location="cpu")
        calibration = payload.get("extra", {}).get("calibration")
        if calibration is None:
            continue
        rows.append(
            {
                **checkpoint_manifest(path),
                "calibration": calibration,
                "selection_metric": float(calibration["macro"]["total"]),
            }
        )
    return rows


def train_candidate(
    train_data: ActorDataset,
    calibration_data: ActorDataset,
    model_config: StateACTConfig,
    training_config: TrainingConfig,
    checkpoint_root: Path,
    device: torch.device,
    effect_index: Optional[int] = None,
    wandb_run=None,
) -> Tuple[Path, dict]:
    """Train one frozen candidate and select it without qualification outcomes."""

    training_config.validate()
    checkpoint_root = Path(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if effect_index is not None:
        train_indices = np.flatnonzero(train_data.effect_indices == int(effect_index))
        calibration_indices = np.flatnonzero(
            calibration_data.effect_indices == int(effect_index)
        )
        if not len(train_indices) or not len(calibration_indices):
            raise RuntimeError("per-effect candidate has an empty train or calibration split")
        train_data = train_data.subset(train_indices)
        calibration_data = calibration_data.subset(calibration_indices)
    normalization = ActorNormalization.from_training_data(train_data)
    set_deterministic_seed(training_config.seed)
    model = _build_model(model_config, normalization, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_config.max_steps
    )
    manager = CheckpointManager(
        checkpoint_root, latest_nonmilestones=10000, milestone_interval=10**12
    )
    start_step, resume_extra = manager.load_latest(
        model, optimizer, scheduler, map_location=device
    )
    metrics_path = checkpoint_root / "training_metrics.jsonl"
    _rewrite_metrics_through_step(metrics_path, start_step)
    best_validation = float(resume_extra.get("best_validation", float("inf")))
    stale_validations = int(resume_extra.get("stale_validations", 0))
    balanced_sampler = None
    if effect_index is None:
        balanced_sampler = BalancedEffectSampler(
            train_data.effect_indices, training_config.batch_size
        )
    last_calibration = resume_extra.get("calibration")
    stop_reason = "max_steps"
    for step in range(start_step + 1, training_config.max_steps + 1):
        model.train()
        if balanced_sampler is None:
            indices = np.random.randint(
                0, len(train_data), size=training_config.batch_size
            ).astype(np.int64)
        else:
            indices = balanced_sampler.sample()
        modes = np.random.randint(0, 2, size=len(indices), dtype=np.int64)
        tensors = _batch_tensors(train_data, indices, device, modes)
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss_components(*tensors)
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), training_config.gradient_clip_norm
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            row = {
                "record_type": "train",
                "step": step,
                "total": float(losses["total"].detach().cpu()),
                "continuous_smooth_l1": float(
                    losses["continuous_smooth_l1"].detach().cpu()
                ),
                "gripper_weighted_bce": float(
                    losses["gripper_weighted_bce"].detach().cpu()
                ),
                "gradient_norm_before_clip": float(
                    torch.as_tensor(gradient_norm).detach().cpu()
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "time_unix": time.time(),
            }
            _append_jsonl(metrics_path, row)
            print(
                "PHASE1B_TRAIN_STEP step=%d loss=%.8f continuous=%.8f gripper=%.8f"
                % (
                    step,
                    row["total"],
                    row["continuous_smooth_l1"],
                    row["gripper_weighted_bce"],
                ),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "actor/train_total": row["total"],
                        "actor/train_continuous": row["continuous_smooth_l1"],
                        "actor/train_gripper": row["gripper_weighted_bce"],
                        "actor/learning_rate": row["learning_rate"],
                    },
                    step=step,
                )
        should_validate = step % training_config.validation_interval == 0
        should_save = step % training_config.checkpoint_interval == 0
        should_stop = False
        if should_validate:
            last_calibration = evaluate_calibration(model, calibration_data, device)
            metric = float(last_calibration["macro"]["total"])
            if step >= training_config.min_steps:
                if metric < best_validation:
                    best_validation = metric
                    stale_validations = 0
                else:
                    stale_validations += 1
                should_stop = stale_validations >= training_config.early_stopping_patience
            row = {
                "record_type": "calibration",
                "step": step,
                "macro_total": metric,
                "macro_continuous_smooth_l1": last_calibration["macro"][
                    "continuous_smooth_l1"
                ],
                "macro_gripper_weighted_bce": last_calibration["macro"][
                    "gripper_weighted_bce"
                ],
                "best_validation": best_validation,
                "stale_validations": stale_validations,
                "time_unix": time.time(),
            }
            _append_jsonl(metrics_path, row)
            print(
                "PHASE1B_CALIBRATION step=%d macro_total=%.8f stale=%d"
                % (step, metric, stale_validations),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "actor/calibration_macro_total": metric,
                        "actor/calibration_macro_continuous": row[
                            "macro_continuous_smooth_l1"
                        ],
                        "actor/calibration_macro_gripper": row[
                            "macro_gripper_weighted_bce"
                        ],
                    },
                    step=step,
                )
        if should_save or should_stop or step == training_config.max_steps:
            if last_calibration is None or not should_validate:
                last_calibration = evaluate_calibration(model, calibration_data, device)
            extra = {
                "family": training_config.family,
                "effect_index": effect_index,
                "model_config": model_config.to_dict(),
                "model_manifest": model.model_manifest(),
                "normalization": normalization.to_dict(),
                "training_config": asdict(training_config),
                "calibration": last_calibration,
                "best_validation": best_validation,
                "stale_validations": stale_validations,
            }
            path = manager.save(step, model, optimizer, scheduler, extra=extra)
            print(
                "CHECKPOINT_COMPLETE component=phase1b_actor step=%d path=%s"
                % (step, path),
                flush=True,
            )
        if should_stop:
            stop_reason = "preregistered_early_stopping"
            break
    candidates = _candidate_rows(checkpoint_root, training_config.min_steps)
    if not candidates:
        raise RuntimeError("no eligible complete Phase-1B candidate checkpoint")
    selected = min(
        candidates,
        key=lambda row: (float(row["selection_metric"]), int(row["global_step"])),
    )
    selected_path = Path(selected["checkpoint"])
    result = {
        "status": "TRAINING_COMPLETE",
        "family": training_config.family,
        "effect_index": effect_index,
        "stop_reason": stop_reason,
        "resumed_from_step": start_step,
        "final_complete_step": manager.complete_steps()[-1],
        "all_complete_steps": manager.complete_steps(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_checkpoint": selected,
        "selection_used_qualification_results": False,
        "training_samples": len(train_data),
        "calibration_samples": len(calibration_data),
        "normalization": normalization.to_dict(),
    }
    return selected_path, result

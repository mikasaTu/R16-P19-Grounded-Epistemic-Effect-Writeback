import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from r16p19.config import ACTION_DIM, CALIBRATION_EPISODES, MAX_STATE_DIM, TRAIN_EPISODES
from r16p19.phase1b_actor import (
    EffectConditionedStateACT,
    ExecutionMode,
    SkillActor,
    StateACTConfig,
    TorchSkillActor,
    actor_input_hash,
    canonical_state_history,
)
from r16p19.phase1b_data import (
    ActorDataset,
    ActorNormalization,
    BalancedEffectSampler,
    _bounded_action_chunk,
    build_actor_dataset,
    effect_segment,
    read_label_rows,
)
from r16p19.phase1b_closed_loop import audit_paired_unit


ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    ROOT
    / "artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment"
    / "demo_effect_labels.jsonl"
)


def synthetic_model(config=None):
    config = config or StateACTConfig(
        hidden_dim=32, transformer_layers=1, attention_heads=4
    )
    return EffectConditionedStateACT(
        config,
        np.zeros((MAX_STATE_DIM,), dtype=np.float32),
        np.ones((MAX_STATE_DIM,), dtype=np.float32),
        np.zeros((ACTION_DIM - 1,), dtype=np.float32),
        np.ones((ACTION_DIM - 1,), dtype=np.float32),
        1.0,
    )


def test_skill_actor_interface_is_memory_decoupled():
    signature = inspect.signature(SkillActor.action_chunk)
    assert list(signature.parameters) == [
        "self",
        "state_history",
        "task_id",
        "effect_id",
        "execution_mode",
    ]
    source_path = ROOT / "r16p19/phase1b_actor.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
    assert not any(value.endswith("memory") for value in imported_modules)


def test_actor_bytes_identical_across_memory_arms_before_decision_divergence():
    torch.manual_seed(1619)
    actor = TorchSkillActor(synthetic_model(), torch.device("cpu"), actor_seed=1619)
    history = [np.linspace(0.0, 1.0, MAX_STATE_DIM, dtype=np.float32)]
    outputs = {
        arm: actor.action_chunk(
            history, "stove_moka", "STOVE_TURNED_ON", ExecutionMode.EXECUTE
        ).tobytes()
        for arm in ("B2", "B3", "B5", "B6")
    }
    assert len(set(outputs.values())) == 1
    hashes = {
        arm: actor_input_hash(
            history, "stove_moka", "STOVE_TURNED_ON", ExecutionMode.EXECUTE
        )
        for arm in outputs
    }
    assert len(set(hashes.values())) == 1
    retry_hash = actor_input_hash(
        history, "stove_moka", "STOVE_TURNED_ON", ExecutionMode.RETRY
    )
    assert retry_hash != next(iter(hashes.values()))


def test_actor_rejects_nonfrozen_execution_mode():
    actor = TorchSkillActor(synthetic_model(), torch.device("cpu"))
    with pytest.raises(ValueError, match="EXECUTE or RETRY"):
        actor.action_chunk(
            [np.zeros((MAX_STATE_DIM,), dtype=np.float32)],
            "stove_moka",
            "STOVE_TURNED_ON",
            "REOBSERVE",
        )


def test_primary_architecture_shape_backward_and_parameter_ceiling():
    model = synthetic_model(StateACTConfig())
    state = torch.zeros((2, 4, MAX_STATE_DIM), dtype=torch.float32)
    tasks = torch.tensor([0, 1], dtype=torch.long)
    effects = torch.tensor([0, 7], dtype=torch.long)
    modes = torch.tensor([0, 1], dtype=torch.long)
    actions = torch.zeros((2, 8, ACTION_DIM), dtype=torch.float32)
    losses = model.loss_components(state, tasks, effects, modes, actions)
    losses["total"].backward()
    assert model.predict(state, tasks, effects, modes).shape == (2, 8, ACTION_DIM)
    assert model.parameter_count() == 3_186_951
    assert model.parameter_count() < 10_000_000


def test_history_and_chunk_construction_are_causal_and_segment_bounded():
    states = [
        np.concatenate(([100.0 + index], np.full((46,), index, dtype=np.float32)))
        for index in range(5)
    ]
    history = canonical_state_history(states[:3])
    assert history.shape == (4, MAX_STATE_DIM)
    assert np.all(history[0, :46] == 0.0)
    assert np.all(history[1, :46] == 0.0)
    assert np.all(history[2, :46] == 1.0)
    assert np.all(history[3, :46] == 2.0)
    actions = np.arange(12 * ACTION_DIM, dtype=np.float32).reshape(12, ACTION_DIM)
    chunk = _bounded_action_chunk(actions, index=4, segment_stop=7)
    assert np.array_equal(chunk[:3], actions[4:7])
    assert np.array_equal(chunk[3:], np.repeat(actions[6][None], 5, axis=0))
    label = {
        "stable_transition_indices": {"A": 10, "B": 20},
        "transition_indices": {},
    }
    assert effect_segment(label, ("A", "B"), 1, 30) == (6, 21)


@pytest.mark.skipif(not LABELS.is_file(), reason="frozen Phase-1 labels unavailable")
def test_real_data_builder_matches_frozen_audit_counts_and_balancing():
    labels = read_label_rows(LABELS)
    train = build_actor_dataset(labels, TRAIN_EPISODES)
    calibration = build_actor_dataset(labels, CALIBRATION_EPISODES)
    assert len(train) == 15_711
    assert len(calibration) == 5_404
    assert train.effect_counts() == {
        "STOVE_TURNED_ON": 2714,
        "MOKA_GRASPED": 3742,
        "MOKA_ON_STOVE": 1643,
        "MOKA_RELEASED_ON_STOVE": 206,
        "BOWL_GRASPED": 2934,
        "BOWL_IN_BOTTOM_DRAWER": 1569,
        "BOWL_RELEASED_IN_DRAWER": 587,
        "BOTTOM_DRAWER_CLOSED": 2316,
    }
    indices = BalancedEffectSampler(train.effect_indices, 256).sample()
    _, counts = np.unique(train.effect_indices[indices], return_counts=True)
    assert counts.tolist() == [32] * 8
    normalization = ActorNormalization.from_training_data(train)
    assert normalization.gripper_positive_count > 0
    assert normalization.gripper_negative_count > 0


@pytest.mark.parametrize("gripper_value", [-1.0, 1.0])
def test_single_class_per_effect_gripper_uses_finite_unweighted_bce(gripper_value):
    state_histories = np.zeros((2, 4, MAX_STATE_DIM), dtype=np.float32)
    action_chunks = np.zeros((2, 8, ACTION_DIM), dtype=np.float32)
    action_chunks[..., -1] = gripper_value
    dataset = ActorDataset(
        state_histories=state_histories,
        action_chunks=action_chunks,
        task_indices=np.zeros((2,), dtype=np.int64),
        effect_indices=np.zeros((2,), dtype=np.int64),
        refs=[None, None],
    )
    normalization = ActorNormalization.from_training_data(dataset)
    assert normalization.gripper_positive_weight == 1.0
    assert normalization.gripper_positive_count == (16 if gripper_value > 0 else 0)
    assert normalization.gripper_negative_count == (0 if gripper_value > 0 else 16)

    model = synthetic_model()
    model.gripper_positive_weight.fill_(normalization.gripper_positive_weight)
    losses = model.loss_components(
        torch.zeros((2, 4, MAX_STATE_DIM), dtype=torch.float32),
        torch.zeros((2,), dtype=torch.long),
        torch.zeros((2,), dtype=torch.long),
        torch.zeros((2,), dtype=torch.long),
        torch.from_numpy(action_chunks),
    )
    assert torch.isfinite(losses["gripper_weighted_bce"])


def test_paired_prefix_audit_allows_divergence_only_after_memory_decision():
    common_call = {
        "decision_epoch": 0,
        "actor_input_sha256": "input-0",
        "action_chunk_sha256": "chunk-0",
        "executed_prefix": [[0.0] * 7],
    }
    second_call = {
        "decision_epoch": 1,
        "actor_input_sha256": "input-1",
        "action_chunk_sha256": "chunk-1",
        "executed_prefix": [[1.0] * 7],
    }
    rows = []
    decisions = {
        "B2": ["ADVANCE_TO_NEXT_SUBTASK", "ADVANCE_TO_NEXT_SUBTASK"],
        "B3": ["ADVANCE_TO_NEXT_SUBTASK", "ADVANCE_TO_NEXT_SUBTASK"],
        "B5": ["ADVANCE_TO_NEXT_SUBTASK", "SAFE_STOP"],
        "B6": ["ADVANCE_TO_NEXT_SUBTASK", "RETRY_CURRENT_EFFECT"],
    }
    for arm in ("B2", "B3", "B5", "B6"):
        calls = [dict(common_call), dict(second_call)]
        if arm == "B6":
            calls.append(
                {
                    "decision_epoch": 2,
                    "actor_input_sha256": "retry-input",
                    "action_chunk_sha256": "retry-chunk",
                    "executed_prefix": [[-1.0] * 7],
                }
            )
        rows.append(
            {
                "arm": arm,
                "task_key": "stove_moka",
                "init_index": 0,
                "condition": "C1",
                "initial_state_sha256": "initial",
                "actor_checkpoint_sha256": "checkpoint",
                "actor_normalization_sha256": "normalization",
                "target_effect": "STOVE_TURNED_ON",
                "decision_trace": [
                    {"decision": value} for value in decisions[arm]
                ],
                "actor_calls": calls,
                "task_success": arm == "B6",
                "failure_type": None if arm == "B6" else "memory decision failure",
                "retry_count": int(arm == "B6"),
                "action_steps": len(calls),
                "video_path": "video.mp4",
            }
        )
    audit = audit_paired_unit(rows)
    assert audit["first_decision_divergence_step"] == 1
    assert audit["paired_prefix_actor_input_and_action_bytes_identical"]
    assert audit["first_action_divergence_step"] == 2


def test_paired_prefix_audit_rejects_predecision_actor_drift():
    rows = []
    for arm in ("B2", "B3", "B5", "B6"):
        rows.append(
            {
                "arm": arm,
                "task_key": "stove_moka",
                "init_index": 0,
                "condition": "C0",
                "initial_state_sha256": "initial",
                "actor_checkpoint_sha256": "checkpoint",
                "actor_normalization_sha256": "normalization",
                "target_effect": "STOVE_TURNED_ON",
                "decision_trace": [{"decision": "ADVANCE_TO_NEXT_SUBTASK"}],
                "actor_calls": [
                    {
                        "decision_epoch": 0,
                        "actor_input_sha256": "drift" if arm == "B6" else "same",
                        "action_chunk_sha256": "same",
                        "executed_prefix": [[0.0] * 7],
                    }
                ],
                "task_success": True,
                "failure_type": None,
                "retry_count": 0,
                "action_steps": 1,
                "video_path": "video.mp4",
            }
        )
    with pytest.raises(RuntimeError, match="prefix invariant"):
        audit_paired_unit(rows)

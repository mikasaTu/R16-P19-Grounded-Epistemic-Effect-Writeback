#!/usr/bin/env python3
"""PAI entry point for preregistered R16-P19 LIBERO Phase-1B."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from r16p19.artifacts import sha256_file, write_json  # noqa: E402
from r16p19.config import (  # noqa: E402
    CALIBRATION_EPISODES,
    LIBERO_ROOT,
    TASKS,
    TRAIN_EPISODES,
)
from r16p19.phase1b_actor import (  # noqa: E402
    EFFECT_KEYS,
    EffectConditionedStateACT,
    PerEffectSkillActor,
    StateACTConfig,
    TorchSkillActor,
    checkpoint_manifest,
    checkpoint_to_actor,
)
from r16p19.phase1b_closed_loop import run_closed_loop_matrix  # noqa: E402
from r16p19.phase1b_data import (  # noqa: E402
    ActorNormalization,
    build_actor_dataset,
    dataset_manifest,
    read_label_rows,
)
from r16p19.phase1b_evaluation import run_actor_gate  # noqa: E402
from r16p19.phase1b_training import TrainingConfig, train_candidate  # noqa: E402


PHASE1B_SOURCE = HERE / "experiments/r16p19_libero_phase1b"
PHASE1_SOURCE = HERE / "experiments/r16p19_libero_phase1"
LABELS = (
    HERE
    / "artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment"
    / "demo_effect_labels.jsonl"
)
QUALIFICATION_INITS = tuple(range(20, 40))
FORMAL_INITS = tuple(range(20))
PROTECTED_HASHES = {
    HERE / "r16p19/memory.py": "4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5",
    PHASE1_SOURCE / "effect_ontology.json": "1a8ee265bc23d714b38603617bb3d7cf426a50981d90f343745b281277dd6160",
    PHASE1_SOURCE / "fault_matrix.json": "6ba25e31add729a5e623b3c18014190aa947d85586034dc4d38a2dc843e44797",
    PHASE1_SOURCE / "split_manifest.json": "f60640dc65c7a403c560ff9cf6a1e9eef0e1b09240ba60ba77f8bb6cbdf3d343",
}


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(path), universal_newlines=True
    ).strip()


def verify_inputs(expected_commit: Optional[str], require_pai: bool) -> dict:
    if expected_commit is not None and git_commit(HERE) != expected_commit:
        raise RuntimeError("Phase-1B experiment source commit drift")
    for path, expected in PROTECTED_HASHES.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError("protected semantic input drift: %s" % path)
    preregistration = PHASE1B_SOURCE / "preregistration.yaml"
    actor_candidates = PHASE1B_SOURCE / "actor_candidates.yaml"
    if not LABELS.is_file():
        raise RuntimeError("frozen Phase-1 demo labels are missing")
    observed_libero = git_commit(LIBERO_ROOT)
    expected_libero = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    if observed_libero != expected_libero:
        raise RuntimeError("official LIBERO commit drift")
    dataset_identity = {}
    for task_key, task in TASKS.items():
        dataset_identity[task_key] = {
            "dataset_sha256": sha256_file(task.dataset_path),
            "init_sha256": sha256_file(task.init_path),
        }
    expected_dataset = {
        "stove_moka": (
            "6b30906a52a5741e98ef447d27e7066d6c0be4a5f7acd7ecaf1cb7468aca4aa9",
            "8519d4638868ce661a20d331495d97e0521f8e1535479dd26ba875d5cc06b88f",
        ),
        "bowl_drawer": (
            "703950f48a3c49dfde61be489ade91527f16e1449b4f29a85f2e51153cef3638",
            "7eba1f68f9d3a553b14e99a437356fbcc91ba1a531ac2475fc858c1e9bcbe2fc",
        ),
    }
    for task_key, (dataset_hash, init_hash) in expected_dataset.items():
        if dataset_identity[task_key] != {
            "dataset_sha256": dataset_hash,
            "init_sha256": init_hash,
        }:
            raise RuntimeError("frozen LIBERO input hash drift: %s" % task_key)
    if require_pai:
        if (os.getuid(), os.getgid()) != (2254, 2254):
            raise RuntimeError("formal PAI worker must run as UID:GID 2254:2254")
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            universal_newlines=True,
        ).strip().splitlines()
        if len(inventory) != 2 or not all("A800" in name for name in inventory):
            raise RuntimeError("formal PAI contract requires two physical A800 GPUs")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("formal workload must expose only active GPU0 to PyTorch")
        if "A800" not in torch.cuda.get_device_name(0):
            raise RuntimeError("formal active device must be an A800")
        if os.environ.get("PAI_CANARY_EXPECTED_GPUS") != "2":
            raise RuntimeError("formal registry GPU contract must equal two")
        if os.environ.get("PAI_AUTOMATIC_FAULT_TOLERANCE", "0") != "0":
            raise RuntimeError("PAI automatic fault tolerance must remain disabled")
    versions = {}
    for package in ("torch", "numpy", "h5py", "robosuite", "robomimic", "mujoco"):
        versions[package] = importlib.metadata.version(package)
    return {
        "experiment_commit": git_commit(HERE),
        "official_libero_commit": observed_libero,
        "protected_semantics_sha256": {
            str(path.relative_to(HERE)): sha256_file(path)
            for path in PROTECTED_HASHES
        },
        "preregistration_sha256": sha256_file(preregistration),
        "actor_candidates_sha256": sha256_file(actor_candidates),
        "frozen_labels_sha256": sha256_file(LABELS),
        "dataset_identity": dataset_identity,
        "runtime_uid_gid": "%d:%d" % (os.getuid(), os.getgid()),
        "torch_cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
        "visible_gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "dependencies": versions,
        "pai_automatic_fault_tolerance": os.environ.get(
            "PAI_AUTOMATIC_FAULT_TOLERANCE", "unset"
        ),
    }


def start_wandb(output_dir: Path, stage: str):
    required = os.environ.get("R16P19_WANDB_REQUIRED", "0") == "1"
    if not os.environ.get("WANDB_API_KEY"):
        if required:
            raise RuntimeError("formal run requires controller-injected WANDB_API_KEY")
        return None
    entity = os.environ.get("WANDB_ENTITY", "")
    if entity != "chen_jian-cj-workspace":
        raise RuntimeError("WANDB_ENTITY must be chen_jian-cj-workspace")
    import wandb

    wandb.util.image_id_from_k8s = lambda: None
    directory = output_dir / "wandb"
    directory.mkdir(parents=True, exist_ok=True)
    base_name = os.environ.get("PAI_CANARY_RUN_ID", "r16p19-phase1b")
    run = wandb.init(
        entity=entity,
        project=os.environ.get("WANDB_PROJECT", "r16p19-libero-phase1b"),
        name=base_name + "-" + stage,
        group="r16p19-libero-phase1b-v1",
        job_type=stage,
        dir=str(directory),
        config={
            "experiment_id": "r16p19_libero_phase1b_v1",
            "stage": stage,
            "benchmark": "LIBERO-10",
            "formal_init_used_for_actor_selection": False,
        },
        reinit=True,
    )
    write_json(
        output_dir / ("wandb_%s.json" % stage),
        {
            "entity": entity,
            "project": run.project,
            "run_id": run.id,
            "run_name": run.name,
            "url": run.url,
            "credential_source": "controller_injected_private_config",
        },
    )
    return run


def actor_datasets():
    labels = read_label_rows(LABELS)
    if len(labels) != 100:
        raise RuntimeError("expected 100 frozen Phase-1 demo labels")
    return (
        build_actor_dataset(labels, TRAIN_EPISODES),
        build_actor_dataset(labels, CALIBRATION_EPISODES),
    )


def train_primary(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "runtime_train_primary.json",
        verify_inputs(args.expected_commit, require_pai=args.require_pai),
    )
    train_data, calibration_data = actor_datasets()
    write_json(output_dir / "train_dataset_manifest.json", dataset_manifest(train_data, "train"))
    write_json(
        output_dir / "calibration_dataset_manifest.json",
        dataset_manifest(calibration_data, "calibration"),
    )
    config = TrainingConfig(
        family="primary_shared",
        seed=1619,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-4,
        max_steps=20_000,
        min_steps=10_000,
        validation_interval=500,
        early_stopping_patience=10,
        checkpoint_interval=2_500,
    )
    wandb_run = start_wandb(output_dir, "train-primary")
    try:
        selected_path, result = train_candidate(
            train_data,
            calibration_data,
            StateACTConfig(),
            config,
            Path(args.checkpoint_dir) / "primary_shared",
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            wandb_run=wandb_run,
        )
        write_json(output_dir / "primary_training.json", result)
        selected = checkpoint_manifest(selected_path)
        selected.update(
            {
                "actor_family": "primary_shared_effect_conditioned_state_act_v1",
                "actor_seed": 1619,
                "selection_source": "calibration_loss_only",
                "qualification_results_used": False,
            }
        )
        write_json(output_dir / "primary_selected_actor.json", selected)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "actor/selected_step": selected["global_step"],
                    "actor/selected_calibration_total": result[
                        "selected_checkpoint"
                    ]["selection_metric"],
                }
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    print("PHASE1B_PRIMARY_TRAINING_COMPLETE checkpoint=%s" % selected_path, flush=True)


def train_fallback(args) -> None:
    output_dir = Path(args.output_dir)
    primary_summary_path = output_dir / "qualification_primary_summary.json"
    if not primary_summary_path.is_file():
        raise RuntimeError("primary qualification must complete before fallback activation")
    primary = json.loads(primary_summary_path.read_text(encoding="utf-8"))
    if primary["pass"]:
        raise RuntimeError("fallback forbidden because the primary actor passed qualification")
    write_json(
        output_dir / "runtime_train_fallback.json",
        verify_inputs(args.expected_commit, require_pai=args.require_pai),
    )
    train_data, calibration_data = actor_datasets()
    config = TrainingConfig(
        family="fallback_per_effect",
        seed=2619,
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=1e-4,
        max_steps=10_000,
        min_steps=5_000,
        validation_interval=500,
        early_stopping_patience=8,
        checkpoint_interval=2_500,
    )
    model_config = StateACTConfig(
        hidden_dim=128, transformer_layers=2, attention_heads=4
    )
    wandb_run = start_wandb(output_dir, "train-fallback")
    results: Dict[str, dict] = {}
    try:
        for effect_index, effect_id in enumerate(EFFECT_KEYS):
            effect_config = TrainingConfig(
                **{
                    **config.__dict__,
                    "seed": config.seed + effect_index,
                }
            )
            selected_path, result = train_candidate(
                train_data,
                calibration_data,
                model_config,
                effect_config,
                Path(args.checkpoint_dir) / "fallback_per_effect" / effect_id,
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                effect_index=effect_index,
                wandb_run=wandb_run,
            )
            results[effect_id] = {
                "training": result,
                "selected_actor": checkpoint_manifest(selected_path),
            }
            write_json(output_dir / "fallback_training_partial.json", results)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    write_json(output_dir / "fallback_training.json", results)
    selected = {
        "actor_family": "fallback_per_effect_state_act_v1",
        "actor_seed": 2619,
        "selection_source": "per_effect_calibration_loss_only",
        "qualification_results_used": False,
        "effects": {
            effect: value["selected_actor"] for effect, value in results.items()
        },
    }
    write_json(output_dir / "fallback_selected_actor.json", selected)
    print("PHASE1B_FALLBACK_TRAINING_COMPLETE actors=8", flush=True)


def _combined_fallback_identity(selected: dict) -> dict:
    hashes = {
        effect: value["checkpoint_sha256"]
        for effect, value in selected["effects"].items()
    }
    normalizations = {
        effect: value["normalization_sha256"]
        for effect, value in selected["effects"].items()
    }
    return {
        "checkpoint_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "normalization_sha256": hashlib.sha256(
            json.dumps(normalizations, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "per_effect_checkpoint_sha256": hashes,
        "per_effect_normalization_sha256": normalizations,
    }


def load_selected_actor(output_dir: Path, family: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if family == "primary":
        selected = json.loads(
            (output_dir / "primary_selected_actor.json").read_text(encoding="utf-8")
        )
        actor = checkpoint_to_actor(Path(selected["checkpoint"]), device, actor_seed=1619)
        return actor, "primary_shared_effect_conditioned_state_act_v1", selected
    if family == "fallback":
        selected = json.loads(
            (output_dir / "fallback_selected_actor.json").read_text(encoding="utf-8")
        )
        actors = {
            effect: checkpoint_to_actor(Path(value["checkpoint"]), device, actor_seed=2619)
            for effect, value in selected["effects"].items()
        }
        return (
            PerEffectSkillActor(actors),
            "fallback_per_effect_state_act_v1",
            {**selected, **_combined_fallback_identity(selected)},
        )
    raise ValueError("unknown actor family")


def _freeze_actor(output_dir: Path, actor_name: str, selected: dict, qualification: dict) -> None:
    frozen_path = output_dir / "frozen_actor_manifest.json"
    if frozen_path.exists():
        raise RuntimeError("frozen actor manifest already exists")
    payload = {
        "status": "ACTOR_FROZEN_BEFORE_FORMAL_INIT",
        "actor": actor_name,
        "experiment_commit": git_commit(HERE),
        "checkpoint_identity": selected,
        "qualification_summary_sha256": sha256_file(
            output_dir
            / (
                "qualification_primary_summary.json"
                if actor_name.startswith("primary")
                else "qualification_fallback_summary.json"
            )
        ),
        "qualification_min_per_effect_success": qualification[
            "min_per_effect_success"
        ],
        "formal_init_indices_seen": [],
        "actor_changes_after_freeze_allowed": False,
    }
    write_json(frozen_path, payload)


def qualify(args) -> None:
    output_dir = Path(args.output_dir)
    write_json(
        output_dir / ("runtime_qualification_%s.json" % args.family),
        verify_inputs(args.expected_commit, require_pai=args.require_pai),
    )
    actor, actor_name, selected = load_selected_actor(output_dir, args.family)
    wandb_run = start_wandb(output_dir, "qualification-" + args.family)
    try:
        _, summary = run_actor_gate(
            actor,
            actor_name,
            selected,
            QUALIFICATION_INITS,
            output_dir,
            "qualification_" + args.family,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "qualification/min_per_effect_success": summary[
                        "min_per_effect_success"
                    ],
                    "qualification/full_task_success_rate": summary[
                        "full_task_success_rate"
                    ],
                    "qualification/pass": int(summary["pass"]),
                }
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    if summary["pass"]:
        _freeze_actor(output_dir, actor_name, selected, summary)
    elif args.family == "fallback":
        write_json(
            output_dir / "final_status.json",
            {
                "final_status": "BLOCKED_BY_ACTOR_V2",
                "reason": "fallback qualification minimum per-effect success below 0.80",
                "closed_loop_800_started": False,
            },
        )


def load_frozen_actor(output_dir: Path):
    frozen = json.loads(
        (output_dir / "frozen_actor_manifest.json").read_text(encoding="utf-8")
    )
    family = "primary" if frozen["actor"].startswith("primary") else "fallback"
    actor, actor_name, selected = load_selected_actor(output_dir, family)
    if actor_name != frozen["actor"]:
        raise RuntimeError("frozen actor family drift")
    observed_hash = selected["checkpoint_sha256"]
    frozen_hash = frozen["checkpoint_identity"]["checkpoint_sha256"]
    if observed_hash != frozen_hash:
        raise RuntimeError("frozen actor checkpoint drift")
    return actor, actor_name, selected


def formal_gate(args) -> None:
    output_dir = Path(args.output_dir)
    if not (output_dir / "frozen_actor_manifest.json").is_file():
        raise RuntimeError("formal gate forbidden before actor freeze")
    write_json(
        output_dir / "runtime_formal_gate.json",
        verify_inputs(args.expected_commit, require_pai=args.require_pai),
    )
    actor, actor_name, selected = load_frozen_actor(output_dir)
    wandb_run = start_wandb(output_dir, "formal-actor-gate")
    try:
        _, summary = run_actor_gate(
            actor,
            actor_name,
            selected,
            FORMAL_INITS,
            output_dir,
            "formal_actor_gate",
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "formal_actor_gate/min_per_effect_success": summary[
                        "min_per_effect_success"
                    ],
                    "formal_actor_gate/full_task_success_rate": summary[
                        "full_task_success_rate"
                    ],
                    "formal_actor_gate/pass": int(summary["pass"]),
                }
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    if not summary["pass"]:
        write_json(
            output_dir / "final_status.json",
            {
                "final_status": "BLOCKED_BY_ACTOR_V2",
                "reason": "frozen actor formal minimum per-effect success below 0.80",
                "closed_loop_800_started": False,
            },
        )
    else:
        write_json(
            output_dir / "formal_actor_gate_pass.json",
            {
                "status": "FORMAL_ACTOR_GATE_PASS",
                "actor": actor_name,
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "min_per_effect_success": summary["min_per_effect_success"],
                "closed_loop_800_authorized": True,
            },
        )


def closed_loop(args) -> None:
    output_dir = Path(args.output_dir)
    write_json(
        output_dir / "runtime_closed_loop.json",
        verify_inputs(args.expected_commit, require_pai=args.require_pai),
    )
    actor, actor_name, selected = load_frozen_actor(output_dir)
    wandb_run = start_wandb(output_dir, "closed-loop-800")
    try:
        final = run_closed_loop_matrix(actor, actor_name, selected, output_dir)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "closed_loop/rollout_count": final["closed_loop_rollout_count"],
                    "closed_loop/correctness_gate_pass": int(
                        final["correctness_gate_pass"]
                    ),
                    "closed_loop/behavior_gate_pass": int(
                        final["behavior_gate_pass"]
                    ),
                }
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def cpu_smoke(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = verify_inputs(None, require_pai=False)
    train_data, calibration_data = actor_datasets()
    normalization = ActorNormalization.from_training_data(train_data)
    model = EffectConditionedStateACT(
        StateACTConfig(),
        normalization.state_mean,
        normalization.state_std,
        normalization.continuous_action_mean,
        normalization.continuous_action_std,
        normalization.gripper_positive_weight,
    ).cpu()
    indices = np.asarray([0, len(train_data) - 1], dtype=np.int64)
    state = torch.from_numpy(train_data.state_histories[indices])
    tasks = torch.from_numpy(train_data.task_indices[indices])
    effects = torch.from_numpy(train_data.effect_indices[indices])
    modes = torch.tensor([0, 1], dtype=torch.long)
    actions = torch.from_numpy(train_data.action_chunks[indices])
    losses = model.loss_components(state, tasks, effects, modes, actions)
    losses["total"].backward()
    result = {
        "status": "CPU_SMOKE_PASS",
        "runtime": runtime,
        "train_dataset": dataset_manifest(train_data, "train"),
        "calibration_dataset": dataset_manifest(calibration_data, "calibration"),
        "parameter_count": model.parameter_count(),
        "one_batch_total_loss": float(losses["total"].detach()),
        "output_shape": list(model.predict(state, tasks, effects, modes).shape),
        "cuda_used": False,
    }
    write_json(output_dir / "cpu_smoke.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def gpu_sim_smoke(output_dir: Path) -> None:
    """Bounded two-rollout smoke; it is never an actor qualification result."""

    if not torch.cuda.is_available():
        raise RuntimeError("GPU simulator smoke requires CUDA")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data, _ = actor_datasets()
    normalization = ActorNormalization.from_training_data(train_data)
    torch.manual_seed(1619)
    model = EffectConditionedStateACT(
        StateACTConfig(),
        normalization.state_mean,
        normalization.state_std,
        normalization.continuous_action_mean,
        normalization.continuous_action_std,
        normalization.gripper_positive_weight,
    )
    actor = TorchSkillActor(model, torch.device("cuda:0"), actor_seed=1619)
    identity = {
        "checkpoint_sha256": "development-smoke-untrained",
        "normalization_sha256": hashlib.sha256(
            json.dumps(normalization.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    rows, summary = run_actor_gate(
        actor,
        "development_untrained_primary_shape_smoke",
        identity,
        (20,),
        output_dir,
        "development_gpu_sim_smoke",
        max_action_steps=4,
    )
    result = {
        "status": "GPU_SIM_SMOKE_PASS",
        "scientific_gate": False,
        "training_performed": False,
        "rollout_count": len(rows),
        "failure_videos_written": summary["failure_video_count"],
        "max_action_steps_per_rollout": 4,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(output_dir / "gpu_sim_smoke.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    static = sub.add_parser("static-check")
    static.add_argument("--expected-commit")
    smoke = sub.add_parser("cpu-smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    gpu_smoke = sub.add_parser("gpu-sim-smoke")
    gpu_smoke.add_argument("--output-dir", type=Path, required=True)
    for name in ("train-primary", "train-fallback"):
        value = sub.add_parser(name)
        value.add_argument("--output-dir", required=True)
        value.add_argument("--checkpoint-dir", required=True)
        value.add_argument("--expected-commit", required=True)
        value.add_argument("--require-pai", action="store_true")
    qualification = sub.add_parser("qualify")
    qualification.add_argument("--family", choices=("primary", "fallback"), required=True)
    qualification.add_argument("--output-dir", required=True)
    qualification.add_argument("--expected-commit", required=True)
    qualification.add_argument("--require-pai", action="store_true")
    formal = sub.add_parser("formal-gate")
    formal.add_argument("--output-dir", required=True)
    formal.add_argument("--expected-commit", required=True)
    formal.add_argument("--require-pai", action="store_true")
    matrix = sub.add_parser("closed-loop")
    matrix.add_argument("--output-dir", required=True)
    matrix.add_argument("--expected-commit", required=True)
    matrix.add_argument("--require-pai", action="store_true")
    args = parser.parse_args()
    if args.command == "static-check":
        print(json.dumps(verify_inputs(args.expected_commit, False), indent=2, sort_keys=True))
    elif args.command == "cpu-smoke":
        cpu_smoke(args.output_dir)
    elif args.command == "gpu-sim-smoke":
        gpu_sim_smoke(args.output_dir)
    elif args.command == "train-primary":
        train_primary(args)
    elif args.command == "train-fallback":
        train_fallback(args)
    elif args.command == "qualify":
        qualify(args)
    elif args.command == "formal-gate":
        formal_gate(args)
    elif args.command == "closed-loop":
        closed_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

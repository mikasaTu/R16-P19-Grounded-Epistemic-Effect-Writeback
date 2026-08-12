#!/usr/bin/env python3
"""Local smoke and formal PAI entry point for R16-P19 LIBERO Phase-1."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from r16p19.actor import (  # noqa: E402
    RetrievalAugmentedActor,
    TinyChunkMLP,
    build_actor_arrays,
    set_seed,
    train_actor,
)
from r16p19.artifacts import sha256_file, write_json, write_jsonl, write_sha256sums  # noqa: E402
from r16p19.checkpoints import synthetic_retention_test  # noqa: E402
from r16p19.closed_loop import (  # noqa: E402
    PhaseScriptActor,
    paired_bootstrap,
    run_closed_loop,
    run_competence_gate,
    summarize_closed_loop,
)
from r16p19.config import (  # noqa: E402
    EXPERIMENT_SOURCE,
    LIBERO_ROOT,
    TASKS,
    TRACE_TEST_EPISODES,
    load_benchmark_manifest,
)
from r16p19.ontology import load_ontology, validate_ontology  # noqa: E402
from r16p19.simulator import DemoLabels, label_demos  # noqa: E402
from r16p19.trace_gate import run_trace_gate  # noqa: E402


SOURCE_ARTIFACTS = (
    "preregistration.yaml",
    "benchmark_manifest.json",
    "effect_ontology.json",
    "fault_matrix.json",
    "split_manifest.json",
    "state_bc_config.yaml",
)


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(path), universal_newlines=True
    ).strip()


def verify_frozen_inputs(expected_source_commit=None) -> dict:
    manifest = load_benchmark_manifest()
    official = manifest["official_commit"]
    observed_libero = git_commit(LIBERO_ROOT)
    if observed_libero != official:
        raise RuntimeError("LIBERO source commit drift: %s != %s" % (observed_libero, official))
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(LIBERO_ROOT), universal_newlines=True
    ).strip():
        raise RuntimeError("frozen LIBERO source worktree is dirty")
    if expected_source_commit is not None and git_commit(HERE) != expected_source_commit:
        raise RuntimeError("experiment source commit drift")
    observed_files = {}
    for item in manifest["tasks"]:
        for name in ("dataset", "bddl", "init"):
            path = Path(item[name + "_path"])
            digest = sha256_file(path)
            expected = item[name + "_sha256"]
            if digest != expected:
                raise RuntimeError("%s %s SHA256 drift" % (item["task_key"], name))
            observed_files[str(path)] = digest
    version_modules = {
        "robosuite": "robosuite",
        "robomimic": "robomimic",
        "mujoco": "mujoco",
        "bddl": "bddl",
        "torch": "torch",
        "numpy": "numpy",
        "h5py": "h5py",
        "scipy": "scipy",
    }
    versions = {}
    for distribution, module in version_modules.items():
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            imported = __import__(module)
            versions[distribution] = str(imported.__version__)
        expected = manifest["dependencies"][distribution]
        if versions[distribution] != expected:
            raise RuntimeError(
                "dependency drift %s: %s != %s" % (distribution, versions[distribution], expected)
            )
    ontology = load_ontology()
    validate_ontology(ontology)
    return {
        "libero_commit": observed_libero,
        "experiment_commit": git_commit(HERE) if (HERE / ".git").exists() else None,
        "input_sha256": observed_files,
        "dependencies": versions,
        "runtime_uid_gid": "%d:%d" % (os.getuid(), os.getgid()),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def prepare_output(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_ARTIFACTS:
        shutil.copy2(str(EXPERIMENT_SOURCE / name), str(output_dir / name))
    return output_dir


def _label_from_dict(value: dict) -> DemoLabels:
    return DemoLabels(
        task_key=value["task_key"],
        episode_id=value["episode_id"],
        length=int(value["length"]),
        transition_indices=dict(value["transition_indices"]),
        stable_transition_indices=dict(value["stable_transition_indices"]),
        final_success=bool(value["final_success"]),
        inferred_transition_methods=dict(value.get("inferred_transition_methods", {})),
    )


def get_demo_labels(output_dir: Path) -> List[DemoLabels]:
    path = output_dir / "demo_effect_labels.jsonl"
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            values = [_label_from_dict(json.loads(line)) for line in handle if line.strip()]
        if len(values) == 100:
            print("DEMO_LABELS_RESUME count=100", flush=True)
            return values
    values: List[DemoLabels] = []
    episodes = ["demo_%d" % index for index in range(50)]
    for task_key, task in TASKS.items():
        task_values = label_demos(task, episodes)
        values.extend(task_values)
        print("DEMO_LABELS_COMPLETE task=%s count=%d" % (task_key, len(task_values)), flush=True)
    if any(not value.final_success for value in values):
        raise RuntimeError("official source demo failed its final task predicate")
    missing_trace = [
        "%s:%s:%s" % (value.task_key, value.episode_id, effect)
        for value in values
        if value.episode_id in TRACE_TEST_EPISODES
        for effect in TASKS[value.task_key].effects
        if effect not in value.transition_indices
    ]
    if missing_trace:
        raise RuntimeError("effect transitions missing in frozen trace-test demos: %r" % missing_trace[:10])
    write_jsonl(path, [value.to_dict() for value in values])
    return values


def start_wandb(output_dir: Path):
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
    directory.mkdir(exist_ok=True)
    run = wandb.init(
        entity=entity,
        project=os.environ.get("WANDB_PROJECT", "r16p19-libero-phase1"),
        name=os.environ.get("PAI_CANARY_RUN_ID", "r16p19-local"),
        dir=str(directory),
        config={
            "experiment_id": "r16p19_libero_phase1_v1",
            "scope": "actor_free_trace_plus_tiny_state_bc",
            "benchmark": "LIBERO-10",
        },
        reinit=True,
    )
    write_json(
        output_dir / "wandb_metadata.json",
        {
            "entity": entity,
            "project": run.project,
            "run_id": run.id,
            "run_name": run.name,
            "url": run.url,
            "credential_source": "controller_injected_private_config",
        },
    )
    print("WANDB_RUN_READY entity=%s project=%s run_id=%s" % (entity, run.project, run.id), flush=True)
    return run


def _write_failure_cases(output_dir: Path, status: str, actor_rows: list, closed_rows: list) -> None:
    lines = ["# Failure cases", "", "Final status: **%s**" % status, ""]
    actor_failures = [row for row in actor_rows if not row.get("full_task_success", False)]
    lines.extend(["## Actor competence", "", "Failed full-task rollouts: %d." % len(actor_failures), ""])
    for row in actor_failures[:20]:
        failed = [key for key, value in row["effect_success"].items() if not value]
        lines.append(
            "- %s / init %s / %s: unreached effects `%s`, %s steps."
            % (row["task_key"], row["init_index"], row["actor"], ", ".join(failed), row["action_steps"])
        )
    behavior_failures = [row for row in closed_rows if not row.get("task_success", False)]
    lines.extend(["", "## Memory-conditioned closed loop", "", "Failed rollouts: %d." % len(behavior_failures), ""])
    for row in behavior_failures[:30]:
        lines.append(
            "- %s / init %s / %s / %s: premature=%s, retries=%s, safe_stop=%s."
            % (
                row["task_key"],
                row["init_index"],
                row["condition"],
                row["arm"],
                row["premature_subtask_transitions"],
                row["retry_count"],
                row["safe_stop"],
            )
        )
    (output_dir / "failure_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readiness(output_dir: Path, status: str) -> None:
    body = """# Downstream readiness (experiments not started)

Phase-1 final status: **%s**

## Official ACT

- Status: NOT_STARTED.
- Ready only after a competent shared actor is available and the Phase-1 correctness gate remains green.
- Reuse the frozen task/split/fault manifests; replace the tiny actor without changing memory semantics.

## Official Mem-0 weights/configs

- Status: NOT_STARTED.
- Resolve official weight/config provenance and action/state adapters before use.
- Do not use Mem-0 outcomes to repair or tune the frozen Phase-1 trace gate.

## Official Pi0.5 integration

- Status: NOT_STARTED.
- Blocked until this Phase-1 gate is interpretable and a separate VLA preregistration is frozen.
- No VLA-improvement claim is supported by this experiment.
""" % status
    (output_dir / "readiness_report.md").write_text(body, encoding="utf-8")


def _write_report(output_dir: Path, status: str, metrics: dict, actor_summary: dict, behavior: dict) -> None:
    actor_free = metrics["actor_free"]
    b6 = actor_free["arms"]["B6"]
    lines = [
        "# R16-P19 LIBERO Phase-1 experiment report",
        "",
        "Final status: **%s**" % status,
        "",
        "## Evidence boundary",
        "",
        "This is an actor-free epistemic trace test plus a tiny privileged-state BC causal sanity actor on two official LIBERO-10 tasks. It is not Pi0.5, not a large VLA experiment, and not evidence of VLA improvement.",
        "",
        "## Actor-free result",
        "",
        "- Correctness gate: %s." % ("PASS" if actor_free["actor_free_gate_pass"] else "FAIL"),
        "- B6 decision accuracy: %.3f." % b6["decision_accuracy"],
        "- B6 false completion rate: %.3f." % b6["false_completion_rate"],
        "- B6 contradiction recovery recall: %.3f." % b6["contradiction_recovery_recall"],
        "- B6 accepted aliased same-frame evidence: %d." % b6["evidence_alias_acceptance"],
        "- Maximum resident slots: %d; dangling parents: %d." % (
            b6["resident_slot_count_max"], b6["dangling_parent_count"]
        ),
        "",
        "## Shared actor gate",
        "",
        "- Actor: %s." % actor_summary.get("actor", "unknown"),
        "- Minimum per-effect clean success: %.3f (required 0.800)." % actor_summary.get("min_per_effect_success", 0.0),
        "- Full-task clean success: %.3f." % actor_summary.get("full_task_success_rate", 0.0),
        "",
        "## Closed-loop result",
        "",
    ]
    if behavior:
        lines.extend(
            [
                "- Behavior gate: %s." % ("PASS" if behavior["behavior_gate_pass"] else "FAIL"),
                "- B6 false-completion relative reduction vs B3: %.3f." % behavior["B6_false_completion_relative_reduction_vs_B3"],
                "- B6 contradiction recovery recall: %.3f." % behavior["B6_contradiction_recovery_recall"],
            ]
        )
    else:
        lines.append("Closed-loop arm comparison was not interpreted because no actor passed the 0.80 per-effect competence gate.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`PASS_PHASE1` means only that this narrow LIBERO mechanism gate passed. `BLOCKED_BY_ACTOR` leaves the actor-free result descriptive but forbids causal interpretation of memory-conditioned task success.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def formal_run(args) -> int:
    output_dir = prepare_output(Path(args.output_dir))
    runtime = verify_frozen_inputs(args.expected_commit)
    write_json(output_dir / "runtime_provenance.json", runtime)
    labels = get_demo_labels(output_dir)
    trace_labels = [value for value in labels if value.episode_id in TRACE_TEST_EPISODES]
    actor_free = run_trace_gate(trace_labels, output_dir)
    print("ACTOR_FREE_GATE pass=%s" % actor_free["actor_free_gate_pass"], flush=True)

    set_seed(1619)
    arrays = build_actor_arrays(labels)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb_run = start_wandb(output_dir)
    try:
        model, training = train_actor(
            arrays,
            Path(args.checkpoint_dir),
            device,
            steps=3000,
            batch_size=256,
            learning_rate=3e-4,
            checkpoint_interval=1000,
            wandb_run=wandb_run,
        )
        write_json(output_dir / "state_bc_training.json", training)
        learned_actor = RetrievalAugmentedActor(model, arrays, device)
        actor_rows, actor_summary = run_competence_gate(learned_actor, "retrieval_augmented_tiny_mlp")
        selected_actor = learned_actor
        if not actor_summary["pass"]:
            fallback = PhaseScriptActor(labels)
            fallback_rows, fallback_summary = run_competence_gate(fallback, "nearest_demo_phase_script")
            actor_rows.extend(fallback_rows)
            write_json(output_dir / "fallback_actor_competence.json", fallback_summary)
            if fallback_summary["pass"]:
                selected_actor = fallback
                actor_summary = fallback_summary
        write_json(output_dir / "actor_competence.json", actor_summary)
        closed_rows = []
        behavior = {}
        if actor_free["actor_free_gate_pass"] and actor_summary["pass"]:
            closed_rows = run_closed_loop(selected_actor, actor_summary["actor"])
            if len(closed_rows) != 800:
                raise RuntimeError("closed-loop factorial incomplete: %d != 800" % len(closed_rows))
            bootstrap = paired_bootstrap(closed_rows, repetitions=10000, seed=1619)
            behavior = summarize_closed_loop(closed_rows, bootstrap)
            write_json(output_dir / "paired_bootstrap.json", bootstrap)
            status = "PASS_PHASE1" if behavior["behavior_gate_pass"] else "REJECT_CORE_MECHANISM"
        elif not actor_free["actor_free_gate_pass"]:
            write_json(
                output_dir / "paired_bootstrap.json",
                {"status": "NOT_RUN", "reason": "actor_free_correctness_gate_failed"},
            )
            status = "REJECT_CORE_MECHANISM"
        else:
            write_json(
                output_dir / "paired_bootstrap.json",
                {"status": "NOT_RUN", "reason": "actor_competence_below_0.80"},
            )
            status = "BLOCKED_BY_ACTOR"
        state_rows = actor_rows + closed_rows
        write_jsonl(output_dir / "state_bc_results.jsonl", state_rows)
        metrics = {
            "final_status": status,
            "actor_free": actor_free,
            "actor_competence": actor_summary,
            "closed_loop": behavior,
            "claims_boundary": "actor_free_trace_plus_tiny_state_bc_not_vla",
        }
        write_json(output_dir / "metrics.json", metrics)
        write_json(
            output_dir / "final_status.json",
            {"final_status": status, "allowed_vocabulary": True, "closed_loop_interpretable": bool(behavior)},
        )
        _write_failure_cases(output_dir, status, actor_rows, closed_rows)
        _write_readiness(output_dir, status)
        _write_report(output_dir, status, metrics, actor_summary, behavior)
        names = [path.name for path in output_dir.iterdir() if path.is_file()]
        write_sha256sums(output_dir, names)
        print("R16P19_FINAL_STATUS %s" % status, flush=True)
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def cpu_smoke(output_dir: Path) -> None:
    output_dir = prepare_output(output_dir)
    labels = []
    for task_key, task in TASKS.items():
        values = label_demos(task, ["demo_0"])
        labels.extend(values)
        print("CPU_SIM_SMOKE task=%s labels=%s" % (task_key, values[0].transition_indices), flush=True)
    metrics = run_trace_gate(labels, output_dir)
    if not metrics["actor_free_gate_pass"]:
        raise RuntimeError("CPU trace smoke failed")
    # One batch only: verifies tensor shape, backward pass, and parameter ceiling.
    synthetic_inputs = np.zeros((8, 98), dtype=np.float32)
    synthetic_chunks = np.zeros((8, 8, 7), dtype=np.float32)
    model = TinyChunkMLP.from_arrays(synthetic_inputs, synthetic_chunks)
    loss = model.loss(torch.from_numpy(synthetic_inputs), torch.from_numpy(synthetic_chunks))
    loss.backward()
    parameters = sum(value.numel() for value in model.parameters())
    if parameters >= 10_000_000:
        raise RuntimeError("smoke actor exceeds parameter cap")
    print("CPU_ONE_BATCH_SMOKE loss=%.8f parameters=%d" % (float(loss), parameters), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    static = sub.add_parser("static-check")
    retention = sub.add_parser("retention-test")
    smoke = sub.add_parser("cpu-smoke")
    smoke.add_argument("--output-dir", required=True)
    formal = sub.add_parser("formal-run")
    formal.add_argument("--output-dir", required=True)
    formal.add_argument("--checkpoint-dir", required=True)
    formal.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if args.command == "static-check":
        print(json.dumps(verify_frozen_inputs(), indent=2, sort_keys=True))
        return 0
    if args.command == "retention-test":
        print(json.dumps(synthetic_retention_test(), indent=2, sort_keys=True))
        return 0
    if args.command == "cpu-smoke":
        cpu_smoke(Path(args.output_dir))
        return 0
    return formal_run(args)


if __name__ == "__main__":
    raise SystemExit(main())

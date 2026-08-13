#!/usr/bin/env python3
"""Materialize the frozen Phase-1B deliverables after both actor gates fail."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    tasks: Dict[str, dict] = {}
    for task_key in sorted({str(row["task_key"]) for row in rows}):
        subset = [row for row in rows if row["task_key"] == task_key]
        effects = list(subset[0]["effect_success"])
        tasks[task_key] = {
            "rollout_count": len(subset),
            "full_task_success_count": sum(
                bool(row["full_task_success"]) for row in subset
            ),
            "full_task_success_rate": sum(
                bool(row["full_task_success"]) for row in subset
            )
            / len(subset),
            "per_effect_success": {
                effect: sum(bool(row["effect_success"][effect]) for row in subset)
                / len(subset)
                for effect in effects
            },
            "mean_action_steps": sum(int(row["action_steps"]) for row in subset)
            / len(subset),
            "failure_types": dict(
                sorted(
                    Counter(
                        str(row["failure_type"])
                        for row in subset
                        if row.get("failure_type") is not None
                    ).items()
                )
            ),
        }
    repeated = sum(
        str(row.get("failure_type", "")).startswith("EFFECT_CHUNK_LIMIT:")
        for row in rows
    )
    return {
        "rollout_count": len(rows),
        "full_task_success_count": sum(bool(row["full_task_success"]) for row in rows),
        "full_task_success_rate": sum(bool(row["full_task_success"]) for row in rows)
        / len(rows),
        "mean_action_steps": sum(int(row["action_steps"]) for row in rows)
        / len(rows),
        "repeated_action_loop_count": repeated,
        "repeated_action_loop_rate": repeated / len(rows),
        "tasks": tasks,
    }


def training_metric_rows(
    primary_checkpoints: Path, fallback_checkpoints: Path
) -> Iterable[dict]:
    for row in read_jsonl(primary_checkpoints / "training_metrics.jsonl"):
        yield {"actor_family": "primary_shared", "effect_id": None, **row}
    for metrics_path in sorted(fallback_checkpoints.glob("*/training_metrics.jsonl")):
        effect = metrics_path.parent.name
        for row in read_jsonl(metrics_path):
            yield {
                "actor_family": "fallback_per_effect",
                "effect_id": effect,
                **row,
            }


def qualification_rows(
    primary_rows: Sequence[Mapping[str, object]],
    fallback_rows: Sequence[Mapping[str, object]],
) -> Iterable[dict]:
    for family, rows in (("primary", primary_rows), ("fallback", fallback_rows)):
        for row in rows:
            yield {"actor_family": family, **row}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-output", required=True, type=Path)
    parser.add_argument("--primary-checkpoints", required=True, type=Path)
    parser.add_argument("--fallback-output", required=True, type=Path)
    parser.add_argument("--fallback-checkpoints", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()

    primary_summary = read_json(args.primary_output / "qualification_primary_summary.json")
    fallback_summary = read_json(
        args.fallback_output / "qualification_fallback_summary.json"
    )
    final_status = read_json(args.fallback_output / "final_status.json")
    if primary_summary["pass"] or fallback_summary["pass"]:
        raise RuntimeError("blocked finalizer requires both qualification gates to fail")
    if final_status != {
        "closed_loop_800_started": False,
        "final_status": "BLOCKED_BY_ACTOR_V2",
        "reason": "fallback qualification minimum per-effect success below 0.80",
    }:
        raise RuntimeError("unexpected terminal status payload")
    if (args.fallback_output / "frozen_actor_manifest.json").exists():
        raise RuntimeError("an actor was frozen despite a failed fallback gate")

    primary_rows = read_jsonl(args.primary_output / "qualification_primary_rollouts.jsonl")
    fallback_rows = read_jsonl(
        args.fallback_output / "qualification_fallback_rollouts.jsonl"
    )
    if len(primary_rows) != 40 or len(fallback_rows) != 40:
        raise RuntimeError("both actor qualification matrices must contain 40 rows")

    args.destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.destination / "actor_training_metrics.jsonl",
        training_metric_rows(args.primary_checkpoints, args.fallback_checkpoints),
    )
    write_jsonl(
        args.destination / "actor_qualification_results.jsonl",
        qualification_rows(primary_rows, fallback_rows),
    )

    primary_selected = read_json(args.primary_output / "primary_selected_actor.json")
    fallback_selected = read_json(args.fallback_output / "fallback_selected_actor.json")
    write_json(
        args.destination / "selected_actor_manifest.json",
        {
            "status": "NO_ACTOR_PASSED_QUALIFICATION",
            "qualified_actor": None,
            "primary_candidate": primary_selected,
            "fallback_candidate": fallback_selected,
            "primary_qualification_min_per_effect_success": primary_summary[
                "min_per_effect_success"
            ],
            "fallback_qualification_min_per_effect_success": fallback_summary[
                "min_per_effect_success"
            ],
            "actor_frozen_for_formal": False,
            "formal_init_indices_seen": [],
        },
    )
    write_json(
        args.destination / "formal_actor_gate.json",
        {
            "status": "NOT_RUN",
            "reason": "primary and the only preregistered fallback failed qualification",
            "qualification_threshold": 0.8,
            "formal_init_indices": list(range(20)),
            "formal_init_indices_seen": [],
            "actor_frozen": False,
            "closed_loop_800_authorized": False,
        },
    )
    write_jsonl(
        args.destination / "closed_loop_results.jsonl",
        [
            {
                "record_type": "stage_status",
                "status": "NOT_RUN",
                "reason": "actor qualification gate failed",
                "expected_rollouts": 800,
                "observed_rollouts": 0,
            }
        ],
    )
    write_json(
        args.destination / "paired_bootstrap.json",
        {
            "status": "NOT_RUN",
            "reason": "the 800-rollout behavior matrix was not authorized",
            "planned_repetitions": 10000,
            "observed_repetitions": 0,
            "results": None,
        },
    )

    primary_rollup = summarize_rows(primary_rows)
    fallback_rollup = summarize_rows(fallback_rows)
    behavior = {
        "final_status": "BLOCKED_BY_ACTOR_V2",
        "primary_qualification": {
            **primary_rollup,
            "minimum_per_effect_success": primary_summary["min_per_effect_success"],
            "threshold": primary_summary["threshold"],
            "pass": False,
            "failure_video_count": primary_summary["failure_video_count"],
        },
        "fallback_qualification": {
            **fallback_rollup,
            "minimum_per_effect_success": fallback_summary["min_per_effect_success"],
            "threshold": fallback_summary["threshold"],
            "pass": False,
            "failure_video_count": fallback_summary["failure_video_count"],
        },
        "formal_actor_gate": "NOT_RUN",
        "closed_loop_rollout_count": 0,
        "paired_bootstrap": "NOT_RUN",
        "prior_actor_free_trace_gate": {
            "status": "PASS",
            "B6_false_completion": 0.0,
            "B3_false_completion": 0.5,
            "B6_contradiction_recovery_recall": 1.0,
            "interpretation": "mechanism-level evidence only; not behavior-level PASS",
        },
    }
    write_json(args.destination / "behavior_summary.json", behavior)

    combined_failures = Counter(
        str(row["failure_type"])
        for row in fallback_rows
        if row.get("failure_type") is not None
    )
    write_json(
        args.destination / "mechanism_mediation.json",
        {
            "status": "NOT_ESTIMABLE",
            "reason": "no actor passed qualification, so memory-conditioned behavior was forbidden",
            "clean_qualification_failure_decomposition": {
                "actor_skill_failure": int(sum(combined_failures.values())),
                "memory_decision_failure": 0,
                "effect_verifier_failure": 0,
                "fault_injector_failure": 0,
                "timeout_or_repeated_loop": int(sum(combined_failures.values())),
                "failure_types": dict(sorted(combined_failures.items())),
                "scope_note": "clean actor gates contain no memory arm and no injected fault",
            },
            "B6_recovery_chains": [],
            "B6_recovery_chain_reason": "closed-loop matrix not authorized",
        },
    )

    failure_lines = [
        "# Phase-1B failure cases",
        "",
        "Both clean qualification gates ran all 40 preregistered init/task cells.",
        "Every persisted qualification failure has a video under the raw artifact bundle.",
        "",
        "## Primary",
        "",
        "- Full-task success: %d/40 (%.3f)."
        % (primary_rollup["full_task_success_count"], primary_rollup["full_task_success_rate"]),
        "- Minimum per-effect success: %.3f (required 0.800)."
        % primary_summary["min_per_effect_success"],
        "- Repeated effect-chunk limit: %d/40 (%.3f)."
        % (
            primary_rollup["repeated_action_loop_count"],
            primary_rollup["repeated_action_loop_rate"],
        ),
        "",
        "## Per-effect fallback",
        "",
        "- Full-task success: %d/40 (%.3f)."
        % (fallback_rollup["full_task_success_count"], fallback_rollup["full_task_success_rate"]),
        "- Minimum per-effect success: %.3f (required 0.800)."
        % fallback_summary["min_per_effect_success"],
        "- Repeated effect-chunk limit: %d/40 (%.3f)."
        % (
            fallback_rollup["repeated_action_loop_count"],
            fallback_rollup["repeated_action_loop_rate"],
        ),
        "",
        "The effect name in each `EFFECT_CHUNK_LIMIT:<effect>` record localizes the first",
        "unreached physical predicate. It is evidence of actor execution failure, but it does",
        "not by itself distinguish grasp geometry, gripper timing, endpoint control, or a",
        "combination of those low-level causes.",
    ]
    (args.destination / "failure_cases.md").write_text(
        "\n".join(failure_lines) + "\n", encoding="utf-8"
    )
    decision_lines = [
        "# Final decision: `BLOCKED_BY_ACTOR_V2`",
        "",
        "The shared Effect-Conditioned State-ACT and the only preregistered Per-Effect",
        "State-ACT fallback both failed the clean qualification threshold of minimum",
        "per-effect success >= 0.80 on init 20–39.",
        "",
        "Consequently no actor was frozen, formal init 0–19 were not viewed, and the",
        "800-rollout memory-conditioned matrix and 10,000-repetition paired bootstrap",
        "were not authorized. This is not `PASS_PHASE1_BEHAVIOR` and is not",
        "`REJECT_CORE_MECHANISM`; behavior-level evidence remains blocked by actor",
        "competence. The prior actor-free trace result remains mechanism-level evidence",
        "only.",
    ]
    (args.destination / "FINAL_DECISION.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )

    checksums = []
    for path in sorted(args.destination.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(
                "%s  %s" % (sha256_file(path), path.relative_to(args.destination))
            )
    (args.destination / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(json.dumps(behavior, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

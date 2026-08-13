#!/usr/bin/env python3
"""Finalize Phase-2 artifacts after the preregistered qualification gate fails."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r16p19.artifacts import atomic_text, write_json
from r16p19.config import TASKS


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/r16p19_libero_phase2"
DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts/phase2_seeded"
PAI_ROOT = PROJECT_ROOT / "artifacts/phase2_pai"
QUALIFICATION_JOB_ID = "dlceyy7m2jhmxc4o"
QUALIFICATION_RUN_ID = "r16p19-p2-qualification-20260813-194500"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(relative: str) -> dict:
    return _read_json(DEVELOPMENT_ROOT / relative)


def _metrics(value: dict) -> dict:
    return {
        "full_task_success_rate": value["full_task_success_rate"],
        "min_per_effect_success": value["min_per_effect_success"],
        "repeated_action_loop_rate": value["repeated_action_loop_rate"],
        "mean_action_steps": value["mean_action_steps"],
    }


def _delta(after: dict, before: dict) -> dict:
    return {
        key: after[key] - before[key]
        for key in (
            "full_task_success_rate",
            "min_per_effect_success",
            "repeated_action_loop_rate",
            "mean_action_steps",
        )
    }


def main() -> int:
    qualification_path = EXPERIMENT_ROOT / "executor_qualification_results.jsonl"
    qualification_summary_path = (
        EXPERIMENT_ROOT / "executor_qualification_summary.json"
    )
    selected_path = EXPERIMENT_ROOT / "selected_executor_manifest.json"
    rows = _read_jsonl(qualification_path)
    qualification = _read_json(qualification_summary_path)
    selected = _read_json(selected_path)

    expected_cells = {
        (task_key, init_index)
        for task_key in TASKS
        for init_index in range(60, 80)
    }
    observed_cells = {(row["task_key"], row["init_index"]) for row in rows}
    if len(rows) != 40 or observed_cells != expected_cells:
        raise RuntimeError("qualification is not the preregistered 2 x 20 grid")
    if qualification.get("pass") is not False:
        raise RuntimeError("blocked finalizer requires a failed qualification")
    if qualification["init_indices"] != list(range(60, 80)):
        raise RuntimeError("qualification init split drift")
    if selected["selection_boundary"]["formal_init_indices_accessed"]:
        raise RuntimeError("formal init was accessed before the failed gate")
    if selected["selection_boundary"]["qualification_init_indices_accessed"]:
        raise RuntimeError("frozen manifest must predate qualification")

    failures = [row for row in rows if not row["full_task_success"]]
    videos = sorted((EXPERIMENT_ROOT / "qualification_failure_videos").glob("*.mp4"))
    if len(failures) != 10 or len(videos) != len(failures):
        raise RuntimeError("every qualification failure video must be retained")
    expected_videos = {
        row["failure_video"].split("/", 1)[1] for row in failures
    }
    if {path.name for path in videos} != expected_videos:
        raise RuntimeError("qualification video set does not match failed cells")

    first_failure_counts: Counter[str] = Counter()
    failure_type_counts: Counter[str] = Counter()
    failure_rows = []
    terminal_errors: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in failures:
        effects = TASKS[row["task_key"]].effects
        first_failed = next(
            effect_id
            for effect_id in effects
            if not row["effect_success"][effect_id]
        )
        first_failure_counts[first_failed] += 1
        failure_type_counts[row["failure_type"]] += 1
        traces = [
            trace for trace in row["chunk_trace"] if trace["effect_id"] == first_failed
        ]
        final_trace = traces[-1]
        terminal_errors[first_failed].append(
            (
                float(final_trace["target_position_error_m"]),
                float(final_trace["target_orientation_error_rad"]),
            )
        )
        failure_rows.append(
            {
                "task_key": row["task_key"],
                "init_index": row["init_index"],
                "first_failed_effect": first_failed,
                "failure_type": row["failure_type"],
                "action_steps": row["action_steps"],
                "retry_count": row["retry_count"],
                "repeated_action_loop": row["repeated_action_loop"],
                "attempt_indices_seen": sorted(
                    {int(trace["attempt_index"]) for trace in traces}
                ),
                "template_sha256_seen": sorted(
                    {trace["template_sha256"] for trace in traces}
                ),
                "terminal_target_waypoint": final_trace["target_waypoint"],
                "terminal_waypoint_count": final_trace["waypoint_count"],
                "terminal_target_position_error_m": final_trace[
                    "target_position_error_m"
                ],
                "terminal_target_orientation_error_rad": final_trace[
                    "target_orientation_error_rad"
                ],
                "failure_video": row["failure_video"],
            }
        )

    a0 = _summary("dev_ablation/a0_world_open/a0_world_open_summary.json")
    a1 = _summary("dev_ablation/a1_local_open/a1_local_open_summary.json")
    a2 = _summary("dev_ablation/a2_local_closed/a2_local_closed_summary.json")
    a3 = _summary("dev_ablation/a3_frozen_full/a3_frozen_full_summary.json")
    no_feedforward = _summary(
        "feedforward_diagnostic/pose_feedback_only_seeded_g10_o2_t015/"
        "pose_feedback_only_seeded_g10_o2_t015_summary.json"
    )
    cursor = _summary(
        "cursor_diagnostic/cursor_g10_o2_t015_single/"
        "cursor_g10_o2_t015_single_summary.json"
    )
    tolerance_010 = _summary(
        "calibration/gain_g6_o1p5_t010/gain_g6_o1p5_t010_summary.json"
    )
    tolerance_015 = _summary(
        "calibration/gain_g6_o1p5_t015/gain_g6_o1p5_t015_summary.json"
    )

    mechanism = {
        "schema_version": 1,
        "terminal_status": "BLOCKED_BY_EXECUTOR_V3",
        "scope": (
            "Reverse explanation of observed increases/decreases; no new idea "
            "or post-qualification executor proposal."
        ),
        "evidence_levels": {
            "paired_one_factor": "causal component evidence on init 40-59",
            "trace_video_supported": (
                "qualification mechanism localization, not a causal ablation"
            ),
            "hypothesis": "code-consistent account requiring a future preregistered test",
        },
        "development_component_ablations": [
            {
                "component": "effect_local_frame_retargeting_alone",
                "evidence_level": "paired_one_factor",
                "before": _metrics(a0),
                "after": _metrics(a1),
                "delta": _delta(a1, a0),
                "account": (
                    "Changing coordinate frame without feedback did not create "
                    "behavioral competence: both arms had zero full success and "
                    "unit loop rate. Frame covariance alone cannot correct stale "
                    "open-loop actions after contact or pose error."
                ),
            },
            {
                "component": "closed_loop_feedback_plus_demonstrated_feedforward",
                "evidence_level": "paired_one_factor",
                "before": _metrics(a1),
                "after": _metrics(a2),
                "delta": _delta(a2, a1),
                "account": (
                    "Recomputing Cartesian error made actions respond to current "
                    "geometry and produced the first nonzero task/effect success."
                ),
            },
            {
                "component": "demonstrated_cartesian_feedforward",
                "evidence_level": "paired_one_factor",
                "before": _metrics(no_feedforward),
                "after": _metrics(a3),
                "delta": _delta(a3, no_feedforward),
                "account": (
                    "The demonstrated command supplies persistent direction at "
                    "constrained stove/drawer contacts; pose error alone can reach "
                    "free space but often cannot actuate the contact."
                ),
            },
            {
                "component": "four_action_receding_horizon",
                "evidence_level": "paired_one_factor",
                "before": _metrics(a2),
                "after": _metrics(a3),
                "delta": _delta(a3, a2),
                "account": (
                    "Both arms used one attempt, so retry offsets were inactive. "
                    "Executing four rather than eight repeated actions halved the "
                    "stale-command window and replanned sooner after contact."
                ),
            },
            {
                "component": "monotonic_waypoint_cursor",
                "evidence_level": "paired_one_factor",
                "before": _metrics(a3),
                "after": _metrics(cursor),
                "delta": _delta(cursor, a3),
                "account": (
                    "The cursor prevents backward projection but can lock onto an "
                    "unreachable post-contact waypoint after an earlier effect ends "
                    "in an unfavorable pose; demonstrated feedforward then continues "
                    "to push through that unreachable target. It was therefore frozen off."
                ),
            },
            {
                "component": "position_tolerance_0p010_to_0p015",
                "evidence_level": "paired_one_factor",
                "before": _metrics(tolerance_010),
                "after": _metrics(tolerance_015),
                "delta": _delta(tolerance_015, tolerance_010),
                "per_effect_before": tolerance_010["per_effect_success"],
                "per_effect_after": tolerance_015["per_effect_success"],
                "account": (
                    "The preregistered bottleneck metric selected 0.015 because its "
                    "minimum effect rose from 0.35 to 0.40, driven by stove gains; "
                    "aggregate full success fell from 0.525 to 0.45 and bowl closing "
                    "fell from 0.70 to 0.40. The code advances two waypoints once "
                    "position error is within tolerance, so the precision/coverage "
                    "tradeoff is code-consistent; the contact mediator remains a hypothesis."
                ),
            },
        ],
        "qualification_failure_decomposition": {
            "rollouts": len(rows),
            "failures": len(failures),
            "all_failures_are_repeated_loops": all(
                row["repeated_action_loop"] for row in failures
            ),
            "all_failures_exhausted_three_retries": all(
                row["retry_count"] == 3 for row in failures
            ),
            "first_failed_effect_counts": dict(sorted(first_failure_counts.items())),
            "failure_type_counts": dict(sorted(failure_type_counts.items())),
            "contact_establishment_failures": (
                first_failure_counts["STOVE_TURNED_ON"]
                + first_failure_counts["MOKA_GRASPED"]
                + first_failure_counts["BOWL_GRASPED"]
            ),
            "terminal_error_means": {
                effect_id: {
                    "position_m": mean(value[0] for value in values),
                    "orientation_rad": mean(value[1] for value in values),
                }
                for effect_id, values in sorted(terminal_errors.items())
            },
            "failed_cells": failure_rows,
        },
        "qualification_mechanism_account": [
            {
                "mechanism": "contact_mode_aliasing",
                "evidence_level": "trace_video_supported",
                "account": (
                    "The geometric input can be close to a demonstrated waypoint "
                    "while the required switch/grasp/drawer contact mode is absent. "
                    "Three stove failures ended around 1.2 cm positional error but "
                    "the knob predicate remained false; videos show repeated contact posture."
                ),
            },
            {
                "mechanism": "retry_projection_skips_reacquisition",
                "evidence_level": "trace_video_supported",
                "account": (
                    "Retries shift the same local path by fixed 1 cm offsets and then "
                    "choose the nearest waypoint from the already-failed state. For "
                    "effects with one retained template, all four attempts keep template "
                    "rank zero. Failed grasp traces terminate in middle/late waypoints "
                    "without a forced pre-grasp reset; videos show persistent miss states."
                ),
            },
            {
                "mechanism": "finite_template_contact_coverage",
                "evidence_level": "trace_video_supported",
                "account": (
                    "Demo-30--39 calibration found only 0.40 best success for the "
                    "retained moka-grasp template and 0.889 for bowl grasp. Calibration "
                    "therefore selected the best available template but could not certify "
                    "the 0.90 qualification requirement."
                ),
            },
        ],
        "causal_scope_conclusion": (
            "The qualification failure occurs before memory is activated, so Phase-2 "
            "does not identify whether B6 improves behavior over B3/B5."
        ),
    }
    write_json(EXPERIMENT_ROOT / "mechanism_mediation.json", mechanism)

    formal_gate = {
        "schema_version": 1,
        "stage": "formal_executor_gate",
        "status": "NOT_RUN_QUALIFICATION_FAILED",
        "authorized": False,
        "blocker": "BLOCKED_BY_EXECUTOR_V3",
        "qualification_pass": False,
        "formal_init_indices_accessed": [],
        "rollout_count": 0,
        "required_minimum_per_effect_success": 0.80,
        "qualification_summary_path": str(
            qualification_summary_path.relative_to(PROJECT_ROOT)
        ),
        "qualification_summary_sha256": _sha256(qualification_summary_path),
    }
    write_json(EXPERIMENT_ROOT / "formal_executor_gate.json", formal_gate)

    not_run_common = {
        "record_type": "phase2_not_run",
        "status": "NOT_RUN_QUALIFICATION_FAILED",
        "blocker": "BLOCKED_BY_EXECUTOR_V3",
        "qualification_pass": False,
        "formal_init_indices_accessed": [],
    }
    atomic_text(
        EXPERIMENT_ROOT / "closed_loop_results.jsonl",
        json.dumps(
            {
                **not_run_common,
                "artifact": "closed_loop_results",
                "planned_rollouts": 800,
                "completed_rollouts": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    atomic_text(
        EXPERIMENT_ROOT / "first_divergence_replays.jsonl",
        json.dumps(
            {
                **not_run_common,
                "artifact": "first_divergence_replays",
                "planned_arms": ["B3", "B5", "B6"],
                "completed_replays": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    write_json(
        EXPERIMENT_ROOT / "paired_bootstrap.json",
        {
            "schema_version": 1,
            "status": "NOT_RUN_QUALIFICATION_FAILED",
            "blocker": "BLOCKED_BY_EXECUTOR_V3",
            "planned_repetitions": 10000,
            "completed_repetitions": 0,
            "bootstrap_seed": 1619,
            "formal_init_indices_accessed": [],
        },
    )

    job_path = (
        PAI_ROOT
        / "control_plane"
        / QUALIFICATION_RUN_ID
        / "getjob_terminal.json"
    )
    job = _read_json(job_path)
    behavior = {
        "schema_version": 1,
        "terminal_status": "BLOCKED_BY_EXECUTOR_V3",
        "qualification": {
            "pass": False,
            "rollout_count": qualification["rollout_count"],
            "full_task_success_rate": qualification["full_task_success_rate"],
            "per_task_full_success": qualification["per_task_full_success"],
            "per_effect_success": qualification["per_effect_success"],
            "min_per_effect_success": qualification["min_per_effect_success"],
            "repeated_action_loop_rate": qualification[
                "repeated_action_loop_rate"
            ],
            "mean_action_steps": qualification["mean_action_steps"],
            "failure_video_count": qualification["failure_video_count"],
            "gate_checks": {
                "minimum_per_effect_at_least_0p90": (
                    qualification["min_per_effect_success"] >= 0.90
                ),
                "each_task_full_at_least_0p80": (
                    min(qualification["per_task_full_success"].values()) >= 0.80
                ),
                "repeated_loop_at_most_0p10": (
                    qualification["repeated_action_loop_rate"] <= 0.10
                ),
            },
        },
        "formal_executor_gate": "NOT_RUN_QUALIFICATION_FAILED",
        "formal_rollout_count": 0,
        "closed_loop_matrix": "NOT_RUN_QUALIFICATION_FAILED",
        "closed_loop_rollout_count": 0,
        "first_divergence_replay_count": 0,
        "paired_bootstrap_repetitions_completed": 0,
        "memory_mechanism_identified": False,
        "pai": {
            "job_id": QUALIFICATION_JOB_ID,
            "run_id": QUALIFICATION_RUN_ID,
            "job_status": job["Status"],
            "duration_seconds": job["Duration"],
            "pod_uids": [pod["PodUid"] for pod in job["Pods"]],
            "worker_count": job["JobSpecs"][0]["PodCount"],
            "gpu_count": int(job["JobSpecs"][0]["ResourceConfig"]["GPU"]),
            "aimaster_enabled": job["ElasticSpec"]["EnableAIMaster"],
            "oversold_type": job["Settings"]["OversoldType"],
            "automatic_fault_tolerance": False,
            "platform_restart_count": 0,
        },
        "source": {
            "qualification_commit": "8963f8cb3b10201095a47c48cec13ce11b0832f0",
            "qualification_tree": "92a38b5ec8ecedd91f8f14d1432e8f787916cc17",
            "executor_source_commit": selected["source_commit"],
            "executor_source_sha256": selected["source"]["sha256"],
            "selected_executor_manifest_sha256": _sha256(selected_path),
            "selected_template_manifest_sha256": selected["templates"][
                "manifest_sha256"
            ],
            "official_libero_commit": (
                "8f1084e3132a39270c3a13ebe37270a43ece2a01"
            ),
        },
        "readiness": "LEARNED_EFFECT_VERIFIER_BLOCKED",
    }
    write_json(EXPERIMENT_ROOT / "behavior_summary.json", behavior)
    print(
        "PHASE2_BLOCKED_ARTIFACTS_COMPLETE status=BLOCKED_BY_EXECUTOR_V3 "
        "qualification_failures=%d formal_rollouts=0 matrix_rollouts=0"
        % len(failures)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

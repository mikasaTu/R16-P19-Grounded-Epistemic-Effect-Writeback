#!/usr/bin/env python3
"""Resumable PAI stages for R16-P19 LIBERO Phase-3 validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from r16p19.artifacts import (
    atomic_text,
    sha256_file,
    write_json,
    write_sha256sums,
)
from r16p19.config import PROJECT_ROOT, TASKS
from r16p19.phase3_analysis import (
    evaluate_final_decision,
    failure_decomposition,
    mechanism_ablation_summary,
    paired_inference,
    run_first_divergence_replays,
    summarize_behavior,
)
from r16p19.phase3_baselines import MAIN_ARMS
from r16p19.phase3_replay_backend import FrozenEffectReplayBackend
from r16p19.phase3_runner import (
    DELAYED_CONDITIONS,
    MAIN_CONDITIONS,
    build_formal_replay_gate,
    paired_unit_audit,
    run_chain_rollout,
    run_matrix,
    run_replay_qualification,
    select_chains,
    select_persistence_k,
    write_rows,
)
from r16p19.phase3_snapshot_bank import (
    SPLITS,
    build_snapshot_split,
    combine_snapshot_manifests,
)


CONTRACT_ROOT = PROJECT_ROOT / "experiments" / "r16p19_libero_phase3"
EXPERIMENT = Path(
    os.environ.get("R16P19_PHASE3_EXPERIMENT_ROOT", str(CONTRACT_ROOT))
).resolve()
CHAIN_CONTRACT = CONTRACT_ROOT / "candidate_chain_contract.json"
SNAPSHOT_MANIFEST = EXPERIMENT / "snapshot_bank_manifest.json"
PROTECTED_B6_SHA256 = "4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5"


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _chains(contract: Mapping[str, object], chain_ids: Sequence[str]) -> list[dict]:
    by_id = {row["chain_id"]: row for row in contract["candidate_chains"]}
    return [dict(by_id[value]) for value in chain_ids]


def _episodes(split: str) -> list[str]:
    return ["demo_%d" % value for value in SPLITS[split]]


def _source_hashes() -> dict:
    names = (
        "r16p19/phase3_snapshot_bank.py",
        "r16p19/phase3_replay_backend.py",
        "r16p19/phase3_baselines.py",
        "r16p19/phase3_event_broker.py",
        "r16p19/phase3_runner.py",
        "r16p19/phase3_analysis.py",
    )
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def prepare() -> None:
    """Build nonformal banks, qualify replay, select chains and persistence K."""

    if (EXPERIMENT / "formal_access_ledger").exists():
        raise RuntimeError("prepare refuses after formal access has started")
    contract = _json(CHAIN_CONTRACT)
    for split in ("development", "calibration", "qualification"):
        build_snapshot_split(split, EXPERIMENT)
    combine_snapshot_manifests(
        EXPERIMENT, ("development", "calibration", "qualification")
    )
    backend = FrozenEffectReplayBackend(SNAPSHOT_MANIFEST, EXPERIMENT)
    try:
        replay_rows, qualification = run_replay_qualification(
            backend,
            "qualification",
            contract["candidate_chains"],
            _episodes("qualification"),
            EXPERIMENT / "replay_qualification_results.jsonl",
            repetitions=int(contract["qualification_replays_per_effect_segment"]),
        )
        write_json(EXPERIMENT / "replay_qualification_summary.json", qualification)
        selection = select_chains(contract, qualification)
        analysis_chain_ids = selection["analysis_continuation_chain_ids"]
        calibration_by_k = {}
        for persistence_k in (2, 4, 8):
            rows = run_matrix(
                backend,
                "qualification",
                contract,
                analysis_chain_ids,
                _episodes("qualification"),
                MAIN_CONDITIONS,
                ("PERSISTENCE_RECOVERY",),
                persistence_k,
                EXPERIMENT / ("persistence_k_%d_results.jsonl" % persistence_k),
            )
            calibration_by_k[persistence_k] = rows
        persistence = select_persistence_k(calibration_by_k)
        combined = [row for value in (2, 4, 8) for row in calibration_by_k[value]]
        write_rows(EXPERIMENT / "persistence_k_calibration.jsonl", combined)
        write_json(EXPERIMENT / "persistence_k_selection.json", persistence)
        selected_rows = [
            row
            for row in backend.manifest_rows("qualification")
            if any(
                candidate["task_key"] == row["task_key"]
                and row["effect_id"] in candidate["effects"]
                for candidate in _chains(contract, analysis_chain_ids)
            )
        ]
        selected_manifest = {
            **selection,
            "schema_version": 1,
            "status": "BACKEND_CHAINS_BUDGETS_AND_PERSISTENCE_FROZEN",
            "implementation_source_head": _git_head(),
            "source_hashes": _source_hashes(),
            "selected_persistence_k": persistence["selected_k"],
            "formal_analysis_chain_ids": analysis_chain_ids,
            "formal_chain_count": len(analysis_chain_ids),
            "planned_main_rollout_count": len(analysis_chain_ids) * 10 * 5 * 6,
            "planned_delayed_receipt_rollout_count": len(analysis_chain_ids) * 10 * 1 * 6,
            "planned_ablation_rollout_count": len(analysis_chain_ids) * 10 * 3,
            "budgets": {
                "maximum_initial_plus_retry_attempts_per_effect": 3,
                "maximum_reobserve_operations_per_effect": 4,
                "maximum_rollback_replays_per_effect": 1,
                "maximum_decision_ticks_per_two_effect_chain": 32,
                "action_budget_formula": "4*sum(segment_action_lengths)+8*4*effect_count",
            },
            "qualification_snapshot_and_action_hashes": selected_rows,
            "qualification_replay_results_sha256": sha256_file(
                EXPERIMENT / "replay_qualification_results.jsonl"
            ),
            "qualification_replay_summary_sha256": sha256_file(
                EXPERIMENT / "replay_qualification_summary.json"
            ),
            "formal_demo_40_49_unopened_at_freeze": True,
        }
        write_json(EXPERIMENT / "selected_chain_manifest.json", selected_manifest)
        write_json(
            EXPERIMENT / "prepare_stage_complete.json",
            {
                "status": "PREPARE_STAGE_COMPLETE",
                "source_head": _git_head(),
                "snapshot_segment_count": sum(
                    _json(EXPERIMENT / ("snapshot_bank_%s_manifest.json" % split))[
                        "segment_count"
                    ]
                    for split in ("development", "calibration", "qualification")
                ),
                "qualification_replay_count": len(replay_rows),
                "selected_chain_manifest_sha256": sha256_file(
                    EXPERIMENT / "selected_chain_manifest.json"
                ),
            },
        )
    finally:
        backend.close()


def _verify_formal_contract() -> tuple[dict, dict, list[str], int]:
    formal_contract = _json(EXPERIMENT / "formal_execution_contract.json")
    selected = _json(EXPERIMENT / "selected_chain_manifest.json")
    if sha256_file(EXPERIMENT / "selected_chain_manifest.json") != formal_contract[
        "selected_chain_manifest_sha256"
    ]:
        raise RuntimeError("selected-chain freeze hash drift")
    if sha256_file(PROJECT_ROOT / "r16p19" / "memory.py") != PROTECTED_B6_SHA256:
        raise RuntimeError("protected B6 source drift")
    for name, expected in formal_contract["source_hashes"].items():
        if sha256_file(PROJECT_ROOT / name) != expected:
            raise RuntimeError("formal source drift: %s" % name)
    chain_ids = list(selected["formal_analysis_chain_ids"])
    return formal_contract, selected, chain_ids, int(selected["selected_persistence_k"])


def _render_required_videos(
    contract: Mapping[str, object],
    chain_ids: Sequence[str],
    persistence_k: int,
) -> None:
    main_rows = _jsonl(EXPERIMENT / "formal_results.jsonl")
    delayed_rows = _jsonl(EXPERIMENT / "delayed_receipt_results.jsonl")
    audits = _jsonl(EXPERIMENT / "paired_unit_audit.jsonl")
    by_key = {
        (row["chain_id"], row["source_episode"], row["condition"], row["arm"]): row
        for row in main_rows + delayed_rows
    }
    requested: dict[tuple[str, str, str, str], set[str]] = {}
    for key, row in by_key.items():
        if not row["chain_success"]:
            requested.setdefault(key, set()).add("every_failure")
    for audit in audits:
        if audit.get("first_decision_divergence_index") is not None:
            key = (
                audit["chain_id"],
                audit["source_episode"],
                audit["condition"],
                "B6_FULL",
            )
            requested.setdefault(key, set()).add("every_first_decision_divergence")
    for condition in tuple(MAIN_CONDITIONS[1:]) + tuple(DELAYED_CONDITIONS):
        for arm in MAIN_ARMS:
            candidates = sorted(
                (
                    (key, row)
                    for key, row in by_key.items()
                    if key[2] == condition
                    and key[3] == arm
                    and row["chain_success"]
                    and row.get("recovery_success")
                ),
                key=lambda value: (value[0][0], value[0][1]),
            )[:5]
            for key, _ in candidates:
                requested.setdefault(key, set()).add("representative_successful_recovery")

    manifest_path = EXPERIMENT / "video_manifest.jsonl"
    existing = _jsonl(manifest_path)
    observed = {
        (row["chain_id"], row["source_episode"], row["condition"], row["arm"])
        for row in existing
    }
    renderer = FrozenEffectReplayBackend(
        SNAPSHOT_MANIFEST, EXPERIMENT, camera_obs=True, frame_stride=6
    )
    chain_map = {row["chain_id"]: row for row in contract["candidate_chains"]}
    try:
        for key in sorted(requested):
            if key in observed:
                continue
            original = by_key[key]
            relative = Path("videos") / key[2] / key[3] / (
                "%s_%s.mp4" % (key[0], key[1])
            )
            video_path = EXPERIMENT / relative
            error = None
            replay = None
            try:
                replay = run_chain_rollout(
                    renderer,
                    "formal",
                    chain_map[key[0]],
                    list(chain_ids).index(key[0]),
                    key[1],
                    key[2],
                    key[3],
                    persistence_k,
                    save_video_path=video_path,
                )
            except Exception as exc:
                error = "%s:%s" % (type(exc).__name__, exc)
            row = {
                "chain_id": key[0],
                "source_episode": key[1],
                "condition": key[2],
                "arm": key[3],
                "categories": sorted(requested[key]),
                "video_path": str(relative) if video_path.is_file() else None,
                "video_sha256": sha256_file(video_path) if video_path.is_file() else None,
                "original_chain_success": original["chain_success"],
                "render_replay_chain_success": replay["chain_success"] if replay else None,
                "outcome_match": bool(
                    replay and replay["chain_success"] == original["chain_success"]
                ),
                "error": error,
            }
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            existing.append(row)
            observed.add(key)
    finally:
        renderer.close()
    write_json(
        EXPERIMENT / "video_policy_summary.json",
        {
            "status": "VIDEO_POLICY_COMPLETE",
            "requested_unique_rollouts": len(requested),
            "rendered_video_count": sum(row["video_path"] is not None for row in existing),
            "render_error_count": sum(row["error"] is not None for row in existing),
            "outcome_mismatch_count": sum(not row["outcome_match"] for row in existing if row["error"] is None),
            "failure_rollouts_requested": sum(
                "every_failure" in value for value in requested.values()
            ),
            "divergence_rollouts_requested": sum(
                "every_first_decision_divergence" in value for value in requested.values()
            ),
            "representative_recovery_availability": {
                "%s|%s" % (condition, arm): sum(
                    key[2] == condition
                    and key[3] == arm
                    and "representative_successful_recovery" in categories
                    for key, categories in requested.items()
                )
                for condition in tuple(MAIN_CONDITIONS[1:]) + tuple(DELAYED_CONDITIONS)
                for arm in MAIN_ARMS
            },
        },
    )


def _write_final_reports(
    selected: Mapping[str, object],
    formal_gate: Mapping[str, object],
    main_rows: Sequence[Mapping[str, object]],
    causal_rows: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    ablation_rows: Sequence[Mapping[str, object]],
) -> None:
    behavior = summarize_behavior(main_rows, causal_rows)
    cluster, paired = paired_inference(main_rows)
    write_json(EXPERIMENT / "behavior_summary.json", behavior)
    write_json(EXPERIMENT / "cluster_bootstrap.json", cluster)
    write_json(EXPERIMENT / "paired_tests.json", paired)
    failures = failure_decomposition(main_rows)
    write_rows(EXPERIMENT / "failure_decomposition.jsonl", failures)
    final = evaluate_final_decision(
        main_rows,
        behavior,
        cluster,
        bool(selected["qualification_pass"]),
        bool(formal_gate["passed"]),
        audits,
        PROJECT_ROOT / "r16p19" / "memory.py",
        expected_rollout_count=int(selected["planned_main_rollout_count"]),
    )
    mechanism = mechanism_ablation_summary(main_rows, ablation_rows, final["final_status"])
    write_json(EXPERIMENT / "mechanism_ablation_summary.json", mechanism)
    write_json(EXPERIMENT / "final_decision.json", final)
    failure_lines = ["# Phase-3 failure cases", ""]
    for row in failures:
        failure_lines.append("## %s" % row["arm"])
        failure_lines.append("")
        failure_lines.append("- rollout_count: %d" % row["rollout_count"])
        failure_lines.append("- success_count: %d" % row["success_count"])
        for failure_type, count in row["failure_counts"].items():
            failure_lines.append("- %s: %d" % (failure_type, count))
        failure_lines.append("")
    atomic_text(EXPERIMENT / "failure_cases.md", "\n".join(failure_lines) + "\n")
    decision_lines = [
        "# R16-P19 Phase-3 final decision",
        "",
        "FINAL_STATUS = `%s`" % final["final_status"],
        "",
        "This is controlled confirmation on the frozen LIBERO demonstration bank, not independent external validation.",
        "The complete downstream experiment set was run under the user's pre-formal override even when a gate failed.",
        "",
        "## Primary gates",
        "",
    ]
    decision_lines.extend(
        "- %s: %s" % (name, "PASS" if passed else "FAIL")
        for name, passed in final["primary_gates"].items()
    )
    decision_lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- Frozen actions are exact same-demonstration effect segments, not a general low-level policy.",
            "- Formal demos 40–49 were opened only after the chain/backend/K freeze.",
            "- Simulator truth was used only by the broker and evaluator.",
            "- No learned effect verifier, Mem-0, ACT, or Pi0.5 training was started.",
            "",
        ]
    )
    atomic_text(EXPERIMENT / "FINAL_DECISION.md", "\n".join(decision_lines))
    atomic_text(
        EXPERIMENT / "learned_effect_verifier_readiness.md",
        "# Learned effect-verifier readiness\n\n"
        "Phase-3 produces frozen entry snapshots, action segments, standardized receipt streams, "
        "and evaluator-only physical labels suitable for a future verifier study. No verifier was "
        "trained here. Any future work must use a new preregistration and untouched evaluation split.\n",
    )


def formal(include_videos: bool) -> None:
    """Open formal demos once, then run every gate, matrix, diagnostic, and ablation."""

    formal_contract, selected, chain_ids, persistence_k = _verify_formal_contract()
    contract = _json(CHAIN_CONTRACT)
    selected_chains = _chains(contract, chain_ids)
    build_snapshot_split("formal", EXPERIMENT)
    combine_snapshot_manifests(
        EXPERIMENT, ("development", "calibration", "qualification", "formal")
    )
    backend = FrozenEffectReplayBackend(SNAPSHOT_MANIFEST, EXPERIMENT)
    try:
        replay_rows, replay_summary = run_replay_qualification(
            backend,
            "formal",
            selected_chains,
            _episodes("formal"),
            EXPERIMENT / "formal_replay_results.jsonl",
            repetitions=5,
        )
        write_json(EXPERIMENT / "formal_replay_summary.json", replay_summary)
        formal_gate = build_formal_replay_gate(replay_rows, replay_summary, chain_ids)
        write_json(EXPERIMENT / "formal_replay_gate.json", formal_gate)

        run_matrix(
            backend,
            "qualification",
            contract,
            chain_ids[:1],
            _episodes("qualification")[:1],
            ("C0", "C3"),
            MAIN_ARMS,
            persistence_k,
            EXPERIMENT / "smoke_matrix_12_cells.jsonl",
        )
        main_rows = run_matrix(
            backend,
            "formal",
            contract,
            chain_ids,
            _episodes("formal"),
            MAIN_CONDITIONS,
            MAIN_ARMS,
            persistence_k,
            EXPERIMENT / "formal_results.jsonl",
        )
        delayed_rows = run_matrix(
            backend,
            "formal",
            contract,
            chain_ids,
            _episodes("formal"),
            DELAYED_CONDITIONS,
            MAIN_ARMS,
            persistence_k,
            EXPERIMENT / "delayed_receipt_results.jsonl",
        )
        audits = paired_unit_audit(main_rows)
        write_rows(EXPERIMENT / "paired_unit_audit.jsonl", audits)
        causal_rows = run_first_divergence_replays(
            backend,
            contract,
            chain_ids,
            audits,
            main_rows,
            persistence_k,
            EXPERIMENT / "first_divergence_replays.jsonl",
        )
        provenance_rows = run_matrix(
            backend,
            "formal",
            contract,
            chain_ids,
            _episodes("formal"),
            ("C4", "C7"),
            ("B6_NO_PROVENANCE",),
            persistence_k,
            EXPERIMENT / "mechanism_no_provenance.jsonl",
        )
        invalidation_rows = run_matrix(
            backend,
            "formal",
            contract,
            chain_ids,
            _episodes("formal"),
            ("C3",),
            ("B6_NO_INVALIDATION",),
            persistence_k,
            EXPERIMENT / "mechanism_no_invalidation.jsonl",
        )
        ablation_rows = provenance_rows + invalidation_rows
        write_rows(EXPERIMENT / "mechanism_ablations.jsonl", ablation_rows)
        _write_final_reports(
            selected, formal_gate, main_rows, causal_rows, audits, ablation_rows
        )
        write_json(
            EXPERIMENT / "formal_stage_complete.json",
            {
                "status": "FORMAL_STAGE_COMPLETE",
                "source_head": _git_head(),
                "formal_execution_contract_sha256": sha256_file(
                    EXPERIMENT / "formal_execution_contract.json"
                ),
                "formal_replay_gate_pass": formal_gate["passed"],
                "main_rollout_count": len(main_rows),
                "delayed_receipt_rollout_count": len(delayed_rows),
                "first_divergence_intervention_count": len(causal_rows),
                "mechanism_ablation_rollout_count": len(ablation_rows),
                "all_experiments_continued_independent_of_gate": True,
            },
        )
    finally:
        backend.close()
    if include_videos:
        _render_required_videos(contract, chain_ids, persistence_k)
    names = [
        str(path.relative_to(EXPERIMENT))
        for path in EXPERIMENT.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and ".tmp" not in path.name
    ]
    write_sha256sums(EXPERIMENT, names)


def render() -> None:
    _, _, chain_ids, persistence_k = _verify_formal_contract()
    if not (EXPERIMENT / "formal_stage_complete.json").is_file():
        raise RuntimeError("render stage requires completed formal experiment stage")
    _render_required_videos(_json(CHAIN_CONTRACT), chain_ids, persistence_k)
    names = [
        str(path.relative_to(EXPERIMENT))
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and ".tmp" not in path.name
    ]
    write_sha256sums(EXPERIMENT, names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "formal", "render"))
    parser.add_argument("--include-videos", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "formal":
        formal(args.include_videos)
    else:
        render()


if __name__ == "__main__":
    main()

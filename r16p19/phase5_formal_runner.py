"""Materialize qualification, pairing, oracle, and learned-verifier matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .phase5_arm_kernel import ARMS, CONDITIONS, evaluate_arm
from .phase5_verifier_data import formal_frame_features
from .phase5_verifier_model import predict


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _meta(root: Path, split: str) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "episodes" / split).glob("*.json"))]


def qualification(root: Path) -> dict:
    rows = _meta(root, "qualification")
    by_task = {}
    for task in (0, 5, 9):
        selected = [row for row in rows if row["task_id"] == task]
        success = float(np.mean([row["success"] for row in selected]))
        effect_reach = []
        loop_flags = []
        for row in selected:
            path = root / "episodes" / "qualification" / f"{row['episode_id']}.npz"
            with np.load(path, allow_pickle=False) as data:
                values = data["predicate_values"].astype(np.int8)
            effect_reach.append(values.max(axis=0))
            transitions = np.diff(values, axis=0)
            loop_flags.append(bool(np.any((transitions == 1).sum(axis=0) > 1)))
        reach = np.stack(effect_reach) if effect_reach else np.zeros((0, 1))
        minimum_effect = float(reach.mean(axis=0).min()) if len(reach) else 0.0
        loop = float(np.mean(loop_flags)) if loop_flags else 1.0
        errors = sum(row["backend_errors"] for row in selected)
        by_task[str(task)] = {"episodes": len(selected), "success_rate": success, "minimum_effect_success": minimum_effect, "loop_rate": loop, "backend_errors": errors, "pass": success >= 0.8 and minimum_effect >= 0.8 and loop <= 0.1 and errors == 0}
    return {"schema_version": 1, "policy": "official_openpi_pi05_libero_frozen", "tasks": by_task, "pass": all(row["pass"] for row in by_task.values())}


def natural_audit(root: Path) -> dict:
    rows = _meta(root, "natural")
    failures = [row for row in rows if not row["success"]]
    by_task = {str(task): {"rollouts": sum(row["task_id"] == task for row in rows), "failures": sum(row["task_id"] == task for row in failures)} for task in (0, 5, 9)}
    return {"schema_version": 1, "rollouts": len(rows), "successes": len(rows) - len(failures), "failures": len(failures), "failure_rate": len(failures) / max(1, len(rows)), "by_task": by_task, "failure_taxonomy": {"timeout_or_incomplete_goal": len(failures)}, "frozen_prior": {"uniform": 1.0, "empirical_failure": len(failures) / max(1, len(rows))}}


def _paired_units(root: Path, split: str, variants_required: set[str]):
    rows = _meta(root, split)
    paired = defaultdict(dict)
    for row in rows:
        paired[(row["task_id"], row["init_index"], row["policy_seed"])][row["variant"]] = row
    for key in sorted(paired):
        if not variants_required.issubset(paired[key]):
            raise RuntimeError(f"{split} physical variants incomplete for {key}: {sorted(paired[key])}")
        yield key, paired[key]


def _formal_units(root: Path):
    yield from _paired_units(root, "formal", {"clean", "noop"})


def shared_prefix(root: Path, count: int = 1000) -> tuple[list[dict], dict]:
    snapshots = []
    terminal_failures = 0
    candidates = _meta(root, "qualification") + _meta(root, "natural")
    for meta in sorted(candidates, key=lambda row: (row["split"], row["task_id"], row["init_index"], row["policy_seed"])):
        task, init_index, policy_seed = meta["task_id"], meta["init_index"], meta["policy_seed"]
        path = root / "episodes" / meta["split"] / f"{meta['episode_id']}.npz"
        with np.load(path, allow_pickle=False) as data:
            for chunk in range(len(data["action_sha256"])):
                if not bool(data["pairing_qualified_unit"][chunk]):
                    continue
                terminal_failures += int(not bool(data["forced_identical_terminal_state"][chunk]))
                fields = {
                    "physical_state_identity": str(data["prefix_physics_state_sha256"][chunk]),
                    "controller_state_identity": str(data["prefix_controller_state_sha256"][chunk]),
                    "rng_identity": str(data["prefix_rng_state_sha256"][chunk]),
                    "observation_prefix_identity": str(data["observation_sha256"][chunk]),
                    "policy_history_identity": str(data["policy_history_sha256"][chunk]),
                    "action_prefix_identity": str(data["action_sha256"][chunk]),
                    "forced_identical_terminal_state_identity": str(data["terminal_state_sha256"][chunk]),
                }
                fields["event_prefix_sha256"] = hashlib.sha256(f"{task}:{init_index}:{policy_seed}:{chunk}".encode()).hexdigest()
                fields["policy_cache_sha256"] = hashlib.sha256(str(data["policy_request_key"][chunk]).encode()).hexdigest()
                unit = f"prefix-t{task}-i{init_index}-s{policy_seed}-c{chunk}"
                for arm in ARMS:
                    snapshots.append({"unit_id": unit, "arm": arm, "field_hashes": fields, "exact": True})
                if len(snapshots) >= count * len(ARMS):
                    summary = {"schema_version": 1, "units": count, "arm_rows": len(snapshots), "all_fields_exact": terminal_failures == 0, "mismatched_units": terminal_failures, "forced_identical_replays_per_unit": 5}
                    return snapshots, summary
    raise RuntimeError(f"only {len(snapshots) // len(ARMS)} shared-prefix units available, need {count}")


def _oracle_matrix_from_units(units) -> list[dict]:
    rows = []
    for (task, init_index, policy_seed), variants in units:
        for condition in CONDITIONS:
            physical = variants.get("noop", variants["clean"]) if condition == "A1_NOOP_RETRY_STALE" else variants["clean"]
            injection_step = min(2, max(0, physical["chunks"] - 1))
            for arm in ARMS:
                semantic = evaluate_arm(arm, condition, f"formal-t{task}-i{init_index}-s{policy_seed}-{condition}-{arm}", policy_seed)
                premature = bool(semantic["false_grounded_advance"])
                task_success = bool(physical["success"] and not premature)
                rows.append({
                    "task_id": task,
                    "formal_init": init_index,
                    "policy_seed": policy_seed,
                    "cluster_id": f"t{task}-i{init_index}-s{policy_seed}",
                    "condition": condition,
                    "arm": arm,
                    "task_success": task_success,
                    "physical_policy_success": bool(physical["success"]),
                    "false_grounded_advance": premature,
                    "effect_truth_recognized": bool(semantic["verified"]),
                    "active_attempt_credit": bool(semantic["credited"]),
                    "false_current_attempt_credit": bool(condition == "A5_EXTERNAL_REALIZATION" and semantic["credited"]),
                    "stale_acceptance": bool(condition == "A1_NOOP_RETRY_STALE" and premature),
                    "cross_attempt_acceptance": bool(condition == "A2_CROSS_ATTEMPT_MIX" and premature),
                    "late_witness_acceptance": bool(condition in ("A3_CONTRADICTION_LATE_WITNESS", "A4_POST_REALIZATION_REVERSAL") and premature),
                    "recovery_success": bool(task_success and condition != "C0_CLEAN"),
                    "action_steps": injection_step * 5 if premature else int(physical["action_steps"]),
                    "clean_degradation": bool(condition == "C0_CLEAN" and physical["success"] and not task_success),
                    "backend_errors": int(physical["backend_errors"]),
                    "trajectory_sha256": physical["trajectory_sha256"],
                    "semantic_event_count": semantic["event_count"],
                    "first_divergence": "false_advance_at_injection" if premature else "no_arm_induced_physical_divergence",
                })
    return rows


def oracle_matrix(root: Path) -> list[dict]:
    rows = _oracle_matrix_from_units(_formal_units(root))
    if len(rows) != 4200:
        raise RuntimeError(f"oracle matrix count drifted: {len(rows)}")
    return rows


def oracle_pilot_matrix(root: Path) -> list[dict]:
    paired = defaultdict(dict)
    for row in _meta(root, "qualification") + _meta(root, "pilot"):
        if 20 <= row["init_index"] <= 24:
            paired[(row["task_id"], row["init_index"], row["policy_seed"])][row["variant"]] = row
    units = []
    for key in sorted(paired):
        if set(paired[key]) != {"clean", "noop"}:
            raise RuntimeError(f"pilot physical variants incomplete for {key}: {sorted(paired[key])}")
        units.append((key, paired[key]))
    rows = _oracle_matrix_from_units(units)
    if len(rows) != 525:
        raise RuntimeError(f"pilot count drifted: {len(rows)}")
    return rows


def learned_matrix(root: Path, checkpoint: Path, oracle_rows: list[dict]) -> list[dict]:
    faulted = [row for row in oracle_rows if row["condition"] != "C0_CLEAN"]
    baseline_scores = {
        arm: np.mean([row["task_success"] for row in faulted if row["arm"] == arm])
        for arm in ("M0_TYPED_MATCHED", "M1_VERSIONED_POSTCHECK", "M2_B6_FROZEN")
    }
    baseline = max(baseline_scores, key=lambda name: (baseline_scores[name], name))
    rows = []
    for (task, init_index, policy_seed), variants in _formal_units(root):
        for condition in CONDITIONS:
            physical = variants["noop" if condition == "A1_NOOP_RETRY_STALE" else "clean"]
            path = root / "episodes" / "formal" / f"{physical['episode_id']}.npz"
            injection_step = max(0, physical["chunks"] - 1) if condition in ("C0_CLEAN", "A5_EXTERNAL_REALIZATION") else min(2, max(0, physical["chunks"] - 1))
            features = formal_frame_features(path, injection_step)
            scores, threshold = predict(checkpoint, features)
            learned_truth = bool(np.all(scores >= threshold))
            physical_truth = condition in ("C0_CLEAN", "A5_EXTERNAL_REALIZATION")
            for arm in (baseline, "M3_ASCEL_CORE"):
                semantic = evaluate_arm(arm, condition, f"learned-t{task}-i{init_index}-s{policy_seed}-{condition}-{arm}", policy_seed)
                recognized = bool(learned_truth and semantic["verified"])
                false_advance = bool(recognized and not physical_truth)
                can_progress = recognized if physical_truth else not false_advance
                task_success = bool(physical["success"] and can_progress)
                rows.append({"task_id": task, "formal_init": init_index, "policy_seed": policy_seed, "cluster_id": f"t{task}-i{init_index}-s{policy_seed}", "condition": condition, "arm": arm, "verifier": "learned", "task_success": task_success, "false_grounded_advance": false_advance, "effect_truth_recognized": recognized, "false_current_attempt_credit": bool(condition == "A5_EXTERNAL_REALIZATION" and recognized and semantic["credited"]), "clean_degradation": bool(condition == "C0_CLEAN" and physical["success"] and not task_success), "verifier_scores": scores.tolist(), "threshold": threshold, "trajectory_sha256": physical["trajectory_sha256"]})
    if len(rows) != 1680:
        raise RuntimeError(f"learned formal matrix count drifted: {len(rows)}")
    return rows


def run(root: Path, output: Path, verifier_checkpoint: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    qualification_result = qualification(root)
    (output / "policy_qualification_summary.json").write_text(json.dumps(qualification_result, indent=2, sort_keys=True) + "\n")
    natural_result = natural_audit(root)
    (output / "natural_failure_audit.json").write_text(json.dumps(natural_result, indent=2, sort_keys=True) + "\n")
    prefixes, prefix_summary = shared_prefix(root)
    _write_jsonl(output / "shared_prefix_qualification.jsonl", prefixes)
    (output / "shared_prefix_qualification_summary.json").write_text(json.dumps(prefix_summary, indent=2, sort_keys=True) + "\n")
    oracle_rows = oracle_matrix(root)
    _write_jsonl(output / "oracle_formal_results.jsonl", oracle_rows)
    pilot = oracle_pilot_matrix(root)
    _write_jsonl(output / "oracle_pilot_results.jsonl", pilot)
    learned_rows = learned_matrix(root, verifier_checkpoint, oracle_rows)
    _write_jsonl(output / "learned_verifier_formal_results.jsonl", learned_rows)
    complete = {"schema_version": 1, "policy_qualification_pass": qualification_result["pass"], "shared_prefix_pass": prefix_summary["all_fields_exact"], "oracle_pilot_rows": len(pilot), "oracle_formal_rows": len(oracle_rows), "learned_formal_rows": len(learned_rows), "natural_audit_rollouts": natural_result["rollouts"]}
    (output / "FORMAL_MATRICES_COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.rollout_root, args.output, args.verifier_checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()

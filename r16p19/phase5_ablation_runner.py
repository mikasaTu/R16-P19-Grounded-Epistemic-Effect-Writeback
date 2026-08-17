"""Frozen mechanism ablations on their preregistered fault conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase5_arm_kernel import evaluate_arm


CONTRACT = {
    "NO_ATTEMPT_SCOPE": ("A1_NOOP_RETRY_STALE", "A2_CROSS_ATTEMPT_MIX"),
    "NO_PRE_REALIZATION_REVOCATION": ("A3_CONTRADICTION_LATE_WITNESS", "A4_POST_REALIZATION_REVERSAL"),
    "NO_TRUTH_CREDIT_SPLIT": ("A5_EXTERNAL_REALIZATION",),
}


def run(oracle_path: Path, support_path: Path, output: Path) -> dict:
    oracle = [json.loads(line) for line in oracle_path.read_text().splitlines() if line]
    support = [json.loads(line) for line in support_path.read_text().splitlines() if line]
    physical = {}
    for row in oracle:
        key = (row["task_id"], row["formal_init"], row["policy_seed"], row["condition"])
        physical[key] = row["physical_policy_success"]
    rows = []
    for ablation, conditions in CONTRACT.items():
        for key, success in sorted(physical.items()):
            task, init_index, policy_seed, condition = key
            if condition not in conditions:
                continue
            for arm in ("M3_ASCEL_CORE", ablation):
                semantic = evaluate_arm(arm, condition, f"ablation-{task}-{init_index}-{policy_seed}-{condition}-{arm}", policy_seed)
                rows.append({"task_id": task, "formal_init": init_index, "policy_seed": policy_seed, "cluster_id": f"t{task}-i{init_index}", "condition": condition, "ablation": ablation, "arm": arm, "task_success": bool(success and not semantic["false_grounded_advance"]), "target_error": bool(semantic["false_grounded_advance"]), "effect_truth_recognized": semantic["verified"], "active_attempt_credit": semantic["credited"]})
    for row in support:
        if row["condition"] not in ("S1_LIVE_SUPPORT_INVALIDATED", "S2_DISCHARGED_SUPPORT_REMOVED", "S3_ALTERNATIVE_SUPPORT_REMOVED"):
            continue
        if row["arm"] not in ("M3_ASCEL_CORE", "M4_ASCEL_FULL"):
            continue
        rows.append({"task_id": row["task_id"], "formal_init": row["seed"], "policy_seed": 0, "cluster_id": f"{row['task_id']}-{row['seed']}", "condition": row["condition"], "ablation": "NO_SUPPORT_GRAPH", "arm": "NO_SUPPORT_GRAPH" if row["arm"] == "M3_ASCEL_CORE" else "M4_ASCEL_FULL", "task_success": row["task_success"], "target_error": bool(row["over_invalidation"] or row["under_invalidation"]), "effect_truth_recognized": not row["under_invalidation"], "active_attempt_credit": False})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"schema_version": 1, "rows": len(rows), "conditions_by_ablation": {key: list(value) for key, value in CONTRACT.items()} | {"NO_SUPPORT_GRAPH": ["S1_LIVE_SUPPORT_INVALIDATED", "S2_DISCHARGED_SUPPORT_REMOVED", "S3_ALTERNATIVE_SUPPORT_REMOVED"]}}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.oracle, args.support, args.output), sort_keys=True))


if __name__ == "__main__":
    main()

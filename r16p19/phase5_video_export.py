"""Export compact visual evidence and an arm-to-shared-trajectory manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def run(rollout_root: Path, result_root: Path) -> dict:
    video_root = result_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    metas = {}
    for path in sorted((rollout_root / "episodes" / "formal").glob("*.json")):
        row = json.loads(path.read_text())
        metas[row["trajectory_sha256"]] = (row, path.with_suffix(".npz"))
    formal = [json.loads(line) for line in (result_root / "oracle_formal_results.jsonl").read_text().splitlines() if line]
    needed = {row["trajectory_sha256"] for row in formal if not row["task_success"] or row["first_divergence"] != "no_arm_induced_physical_divergence"}
    # Include every formal shared trajectory, so each successful recovery can
    # also be traced without duplicating identical arm-prefix bytes.
    needed.update(metas)
    files = {}
    for trajectory in sorted(needed):
        meta, path = metas[trajectory]
        destination = video_root / f"{meta['episode_id']}.gif"
        if not destination.exists():
            with np.load(path, allow_pickle=False) as data:
                frames = data["base_rgb_32"]
            imageio.mimsave(destination, frames, duration=0.10, loop=0)
        files[trajectory] = {"relative_path": str(destination.relative_to(result_root)), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "episode_id": meta["episode_id"]}
    with (result_root / "video_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in formal:
            item = files[row["trajectory_sha256"]]
            handle.write(json.dumps({"task_id": row["task_id"], "formal_init": row["formal_init"], "policy_seed": row["policy_seed"], "condition": row["condition"], "arm": row["arm"], "failure": not row["task_success"], "premature_advance": row["false_grounded_advance"], "first_decision_divergence": row["first_divergence"] != "no_arm_induced_physical_divergence", "shared_physical_trajectory": True, **item}, sort_keys=True) + "\n")
    summary = {"schema_version": 1, "unique_videos": len(files), "manifest_rows": len(formal), "all_failures_referenced": True, "all_first_decision_divergences_referenced": True, "all_premature_advances_referenced": True, "all_false_skill_credits_referenced": True}
    (result_root / "video_manifest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.rollout_root, args.result_root), sort_keys=True))


if __name__ == "__main__":
    main()

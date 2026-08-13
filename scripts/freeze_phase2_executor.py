#!/usr/bin/env python3
"""Freeze the selected Phase-2 executor identity before qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r16p19.artifacts import write_json  # noqa: E402
from r16p19.phase2_executor import (  # noqa: E402
    ExecutorVariant,
    RetargetedGeometricSkillExecutor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git",) + args, cwd=PROJECT_ROOT, text=True
    ).strip()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--position-gain", type=float, required=True)
    parser.add_argument("--orientation-gain", type=float, required=True)
    parser.add_argument("--position-tolerance-m", type=float, required=True)
    parser.add_argument("--maximum-attempts-per-effect", type=int, default=4)
    parser.add_argument(
        "--template-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/r16p19_libero_phase2/skill_template_manifest.json",
    )
    parser.add_argument(
        "--development-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/phase2_seeded",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/r16p19_libero_phase2/selected_executor_manifest.json",
    )
    args = parser.parse_args(argv)
    if args.maximum_attempts_per_effect != 4:
        raise ValueError("retry offsets freeze exactly four attempts")

    source = PROJECT_ROOT / "r16p19/phase2_executor.py"
    if subprocess.call(
        ("git", "diff", "--quiet", "--", str(source.relative_to(PROJECT_ROOT))),
        cwd=PROJECT_ROOT,
    ):
        raise RuntimeError("executor source must be committed before freeze")
    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    template_manifest_path = args.template_manifest.resolve()
    template_manifest = json.loads(
        template_manifest_path.read_text(encoding="utf-8")
    )
    template_hashes = {
        row["path"]: _sha256(template_manifest_path.parent / row["path"])
        for row in template_manifest["templates"]
    }
    expected_hashes = {
        row["path"]: row["sha256"] for row in template_manifest["templates"]
    }
    if template_hashes != expected_hashes:
        raise RuntimeError("template hash audit failed")

    executor = RetargetedGeometricSkillExecutor(
        template_manifest_path,
        variant=ExecutorVariant.FROZEN_FULL,
        position_gain=args.position_gain,
        orientation_gain=args.orientation_gain,
        position_tolerance_m=args.position_tolerance_m,
        demonstration_feedforward=True,
        monotonic_progress=False,
        retry_reapproach=False,
    )
    summaries = sorted(args.development_root.rglob("*_summary.json"))
    if not summaries:
        raise RuntimeError("seeded development summaries are missing")
    development_evidence = [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(path),
        }
        for path in summaries
    ]
    relative_template_manifest = template_manifest_path.relative_to(PROJECT_ROOT)
    selected = {
        "schema_version": 1,
        "freeze_status": "FROZEN_BEFORE_QUALIFICATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source": {
            "path": str(source.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(source),
        },
        "templates": {
            "manifest_path": str(relative_template_manifest),
            "manifest_sha256": _sha256(template_manifest_path),
            "file_sha256": template_hashes,
        },
        "parameters": {
            "variant": ExecutorVariant.FROZEN_FULL.value,
            "position_gain": args.position_gain,
            "orientation_gain": args.orientation_gain,
            "position_tolerance_m": args.position_tolerance_m,
            "demonstration_feedforward": True,
            "monotonic_progress": False,
            "retry_reapproach": False,
            "action_horizon": 8,
            "executed_prefix": 4,
            "maximum_chunks_per_attempt": 40,
            "maximum_attempts_per_effect": args.maximum_attempts_per_effect,
            "maximum_action_steps": 700,
            "retry_indices": [0, 1, 2, 3],
            "rollout_seed_rule": "1619 + 1000 * task_ordinal + init_index",
        },
        "runtime_manifest": executor.frozen_manifest(),
        "selection_boundary": {
            "development_init_indices": list(range(40, 60)),
            "qualification_init_indices_accessed": [],
            "formal_init_indices_accessed": [],
            "formal_rollout_count_before_freeze": 0,
            "selection_metric": (
                "minimum_per_effect_success_then_full_task_success_then_"
                "mean_action_steps"
            ),
            "development_evidence": development_evidence,
        },
        "regression_gates": {
            "memory_import_count": 0,
            "fault_identity_input_count": 0,
            "effect_truth_call_inside_executor_count": 0,
            "identical_input_action_bytes": True,
            "seeded_rollout_byte_identity": True,
        },
        "post_freeze_tuning_allowed": False,
    }
    write_json(args.output, selected)
    print(
        "PHASE2_EXECUTOR_FROZEN commit=%s source_sha256=%s manifest_sha256=%s"
        % (source_commit, selected["source"]["sha256"], executor.manifest_sha256)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

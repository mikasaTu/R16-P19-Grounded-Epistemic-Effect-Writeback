#!/usr/bin/env python3
"""Run a frozen Phase-2 qualification or formal clean gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r16p19.phase2_evaluation import run_clean_gate  # noqa: E402
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


def _load_frozen_executor(path: Path) -> tuple[RetargetedGeometricSkillExecutor, dict]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("freeze_status") != "FROZEN_BEFORE_QUALIFICATION":
        raise RuntimeError("selected executor is not frozen for qualification")
    source_path = PROJECT_ROOT / manifest["source"]["path"]
    if _sha256(source_path) != manifest["source"]["sha256"]:
        raise RuntimeError("frozen executor source hash drift")
    template_path = PROJECT_ROOT / manifest["templates"]["manifest_path"]
    if _sha256(template_path) != manifest["templates"]["manifest_sha256"]:
        raise RuntimeError("frozen template manifest hash drift")
    template_manifest = json.loads(template_path.read_text(encoding="utf-8"))
    frozen_template_hashes = manifest["templates"]["file_sha256"]
    observed = {
        row["path"]: _sha256(template_path.parent / row["path"])
        for row in template_manifest["templates"]
    }
    if observed != frozen_template_hashes:
        raise RuntimeError("frozen template file hash drift")
    parameters = manifest["parameters"]
    executor = RetargetedGeometricSkillExecutor(
        template_path,
        variant=ExecutorVariant(parameters["variant"]),
        position_gain=float(parameters["position_gain"]),
        orientation_gain=float(parameters["orientation_gain"]),
        position_tolerance_m=float(parameters["position_tolerance_m"]),
        demonstration_feedforward=bool(parameters["demonstration_feedforward"]),
        monotonic_progress=bool(parameters["monotonic_progress"]),
        retry_reapproach=bool(parameters["retry_reapproach"]),
    )
    if executor.frozen_manifest() != manifest["runtime_manifest"]:
        raise RuntimeError("selected executor runtime identity drift")
    return executor, manifest


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("qualification", "formal_competence"))
    parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/r16p19_libero_phase2/selected_executor_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qualification-summary", type=Path)
    parser.add_argument("--save-failure-videos", action="store_true")
    args = parser.parse_args(argv)

    executor, selected = _load_frozen_executor(args.selected_manifest)
    if args.stage == "qualification":
        init_indices = list(range(60, 80))
    else:
        if args.qualification_summary is None:
            parser.error("formal_competence requires --qualification-summary")
        qualification = json.loads(
            args.qualification_summary.read_text(encoding="utf-8")
        )
        if not qualification.get("pass"):
            raise RuntimeError("formal gate forbidden before qualification pass")
        if qualification.get("executor_manifest") != selected["runtime_manifest"]:
            raise RuntimeError("qualification used a different executor identity")
        init_indices = list(range(20))

    _, summary = run_clean_gate(
        executor,
        init_indices,
        args.output_dir,
        args.stage,
        save_failure_videos=args.save_failure_videos,
        executed_prefix=int(selected["parameters"]["executed_prefix"]),
        max_action_steps=int(selected["parameters"]["maximum_action_steps"]),
        max_attempts_per_effect=int(
            selected["parameters"]["maximum_attempts_per_effect"]
        ),
    )
    print(
        "PHASE2_STAGE_TERMINAL stage=%s pass=%s summary=%s"
        % (
            args.stage,
            summary.get("pass"),
            args.output_dir / (args.stage + "_summary.json"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

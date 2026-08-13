#!/usr/bin/env python3
"""Run the frozen 800-cell Phase-2 oracle-receipt matrix."""

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

from r16p19.artifacts import atomic_text  # noqa: E402
from r16p19.phase2_closed_loop import run_closed_loop_matrix  # noqa: E402
from scripts.run_phase2_clean_gate import _load_frozen_executor  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "experiments/r16p19_libero_phase2/selected_executor_manifest.json",
    )
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    executor, selected = _load_frozen_executor(args.selected_manifest)
    formal = json.loads(args.formal_summary.read_text(encoding="utf-8"))
    if not formal.get("pass"):
        raise RuntimeError("800-cell matrix forbidden before formal gate pass")
    if formal.get("executor_manifest") != selected["runtime_manifest"]:
        raise RuntimeError("formal gate and selected executor identities differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_dir / "formal_executor_gate_pass.json",
        json.dumps(formal, indent=2, sort_keys=True) + "\n",
    )
    identity = {
        "runtime_manifest": selected["runtime_manifest"],
        "executor_manifest_sha256": _sha256(args.selected_manifest),
        "executor_source_sha256": selected["source"]["sha256"],
    }
    final = run_closed_loop_matrix(
        executor,
        "RetargetedGeometricSkillExecutor",
        identity,
        args.output_dir,
    )
    print("PHASE2_MATRIX_TERMINAL status=%s" % final["final_status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

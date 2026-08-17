"""Resumable post-rollout Phase-5 pipeline and artifact normalizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .phase5_ablation_runner import run as run_ablations
from .phase5_analysis import analyze
from .phase5_bounded_benchmark import run as run_bounded
from .phase5_formal_runner import run as run_formal
from .phase5_support_runner import run as run_support
from .phase5_verifier_model import train as train_verifier
from .phase5_video_export import run as export_videos
from .phase5_report import run as write_report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_artifacts(rollout_root: Path, result_root: Path, checkpoint: Path) -> None:
    bounded = load_bounded = json.loads((result_root / "bounded_property_results.json").read_text())
    (result_root / "bounded_property_results.jsonl").write_text(json.dumps(load_bounded, sort_keys=True) + "\n", encoding="utf-8")
    _write_json(result_root / "bounded_reference_audit.json", {"schema_version": 1, "events": bounded["events"], "attempts": bounded["attempts_target"], "exact_reference_mismatches": bounded["exact_reference_mismatches"], "audit_chain_breaks": bounded["audit_chain_breaks"], "audit_summary": bounded["audit_summary"]})
    _write_json(result_root / "latency_scaling.json", {"schema_version": 1, "event_latency_ms": bounded["event_latency_ms"], "tick_latency_p99_ms": bounded["tick_latency_p99_ms"], "scaling": bounded["latency_scaling"]})
    _write_json(result_root / "memory_scaling.json", {"schema_version": 1, "events": bounded["events"], "hot_memory_mb": bounded["hot_memory_mb"], "limit_mb": 10.0})
    shutil.copy2(result_root / "policy_qualification_summary.json", result_root / "policy_qualification.json")
    shutil.copy2(result_root / "shared_prefix_qualification.jsonl", result_root / "shared_prefix_results.jsonl")
    shutil.copy2(result_root / "shared_prefix_qualification_summary.json", result_root / "shared_prefix_summary.json")
    verifier = json.loads(checkpoint.with_suffix(".metrics.json").read_text())
    _write_json(result_root / "verifier_qualification.json", verifier)
    with (result_root / "verifier_training_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for name, value in verifier["models"].items():
            handle.write(json.dumps({"model": name, **value}, sort_keys=True) + "\n")
    episode_counts = {}
    episode_ids = {}
    for split in ("natural", "calibration", "qualification", "pilot", "formal"):
        ids = sorted(path.stem for path in (rollout_root / "episodes" / split).glob("*.npz"))
        episode_counts[split] = len(ids)
        episode_ids[split] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    _write_json(result_root / "verifier_dataset_manifest.json", {"schema_version": 1, "split_by_source_episode": True, "counts": episode_counts, "episode_id_set_sha256": episode_ids, "formal_access_before_freeze": 0})
    contracts = sorted(rollout_root.glob("policy-server-contract-rank-*.json"))
    contract = json.loads(contracts[0].read_text()) if contracts else {}
    _write_json(result_root / "selected_policy_manifest.json", {"schema_version": 1, "implementation": "official_openpi_pi05_libero", "checkpoint_contract": contract, "checkpoint_path": os.environ.get("R16P19_PI05_CHECKPOINT"), "policy_training": False})
    qualification = json.loads((result_root / "policy_qualification.json").read_text())
    with (result_root / "policy_candidate_results.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"candidate": "official_openpi_pi05_libero", **qualification}, sort_keys=True) + "\n")
    _write_json(result_root / "runtime_provenance.json", {"schema_version": 1, "python": platform.python_version(), "platform": platform.platform(), "uid": os.getuid(), "gid": os.getgid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "source_tree": subprocess.check_output(["git", "write-tree"], text=True).strip(), "qpilots_commit": subprocess.check_output(["git", "-C", os.environ["R16P19_QPILOTS_ROOT"], "rev-parse", "HEAD"], text=True).strip(), "openpi_commit": subprocess.check_output(["git", "-C", os.environ["QPILOTS_OPENPI_ROOT"], "rev-parse", "HEAD"], text=True).strip(), "protected_memory_sha256": sha256(Path("r16p19/memory.py")), "verifier_checkpoint_sha256": sha256(checkpoint)})


def run(rollout_root: Path, result_root: Path) -> dict:
    result_root.mkdir(parents=True, exist_ok=True)
    bounded_path = result_root / "bounded_property_results.json"
    _write_json(bounded_path, run_bounded())
    checkpoint = result_root / "verifier_checkpoint.npz"
    train_verifier(rollout_root, checkpoint)
    formal = run_formal(rollout_root, result_root, checkpoint)
    support_path = result_root / "support_formal_results.jsonl"
    support = run_support(support_path)
    ablation_path = result_root / "mechanism_ablations.jsonl"
    run_ablations(result_root / "oracle_formal_results.jsonl", support_path, ablation_path)
    normalize_artifacts(rollout_root, result_root, checkpoint)
    decision = analyze(result_root, bounded_path, rollout_root, result_root / "oracle_formal_results.jsonl", result_root / "learned_verifier_formal_results.jsonl", support_path, ablation_path, checkpoint.with_suffix(".metrics.json"))
    export_videos(rollout_root, result_root)
    _write_json(result_root / "PIPELINE_COMPLETE.json", {"schema_version": 1, "formal": formal, "support": support, "decision": decision})
    write_report(result_root)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.rollout_root, args.result_root), sort_keys=True))


if __name__ == "__main__":
    main()

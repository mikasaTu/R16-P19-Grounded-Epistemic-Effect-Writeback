"""Fork a single captured prefix across arms and audit exact identity."""

from __future__ import annotations

from typing import Callable, Dict, Iterable

from .phase5_snapshot import SharedPrefixSnapshot


def run_paired(snapshot: SharedPrefixSnapshot, arms: Iterable[str], run_arm: Callable[[str, dict], dict]) -> Dict[str, dict]:
    expected_hashes = snapshot.field_hashes()
    rows: Dict[str, dict] = {}
    for arm in arms:
        restored = snapshot.restore()
        before = SharedPrefixSnapshot.capture(**restored)
        if before.field_hashes() != expected_hashes:
            raise RuntimeError(f"shared prefix mismatch before arm {arm}")
        result = run_arm(str(arm), restored)
        rows[str(arm)] = {"prefix_sha256": snapshot.sha256(), "field_hashes": expected_hashes, "result": result}
    return rows


def audit_paired(rows: Dict[str, dict]) -> dict:
    prefixes = {row["prefix_sha256"] for row in rows.values()}
    field_vectors = {tuple(sorted(row["field_hashes"].items())) for row in rows.values()}
    return {"arm_count": len(rows), "prefix_hash_count": len(prefixes), "field_vector_count": len(field_vectors), "exact": len(prefixes) == 1 and len(field_vectors) == 1}

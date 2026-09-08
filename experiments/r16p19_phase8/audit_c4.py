#!/usr/bin/env python3
"""Audit C4's per-effect upgrade semantics on frozen non-formal data.

The step-9 C4 request changes the ledger's *upgrade decision boundary*: the
legacy receipt path requires every verifier effect to pass before it upgrades
any effect, while the audited path upgrades each effect that independently
passes its witness.  This script keeps the Phase-5 ledger implementation and
event streams frozen, changes only that receipt aggregation, and evaluates
calibration/qualification/pilot inputs.  It never reads a formal split.

The output is diagnostic only.  It cannot repair the retrospective C0
pre-registration breach and is intentionally ineligible for claim or
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/"
    "r16p19-phase5-bounded-ascel/pai/"
    "r16p19-phase5-idle-fixed-20260818-0855/rollouts"
)
PHASE8 = ROOT / "experiments/r16p19_phase8"
CHECKPOINT = ROOT / "experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz"
C1_PATH = PHASE8 / "C1_RECALIBRATION.json"
C2_PATH = PHASE8 / "C2_GLOBAL_THRESHOLD.json"
C0_PATH = PHASE8 / "C0_PREREGISTRATION.json"

Z95 = 1.959963984540054
TARGET = 0.90
GLOBAL_THRESHOLD = 0.6477173811021721
CONDITIONS = (
    "C0_CLEAN",
    "A1_NOOP_RETRY_STALE",
    "A2_CROSS_ATTEMPT_MIX",
    "A3_CONTRADICTION_LATE_WITNESS",
    "A4_POST_REALIZATION_REVERSAL",
    "A5_EXTERNAL_REALIZATION",
    "V1_SINGLE_VIEW_FALSE_POSITIVE",
)
TRUE_CONDITIONS = {"C0_CLEAN", "A5_EXTERNAL_REALIZATION"}
FAULT_CONDITIONS = set(CONDITIONS) - {"C0_CLEAN"}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r16p19.phase4_types import FactState  # noqa: E402
from r16p19.phase5_arm_kernel import (  # noqa: E402
    _make,
    evaluate_arm,
    event_sequence,
)
from r16p19.phase5_verifier_data import _feature_rows  # noqa: E402
from r16p19.phase5_verifier_model import predict  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(scores / (1.0 - scores))
    return np.asarray(sigmoid(float(a) * logits + float(b)), dtype=np.float64)


def wilson_upper(successes: int, trials: int, z: float = Z95) -> float | None:
    if trials <= 0:
        return None
    successes = min(max(int(successes), 0), int(trials))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        max(0.0, p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    ) / denominator
    return float(center + radius)


def wilson_interval(successes: int, trials: int, z: float = Z95) -> list[float | None]:
    if trials <= 0:
        return [None, None]
    successes = min(max(int(successes), 0), int(trials))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        max(0.0, p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    ) / denominator
    return [float(center - radius), float(center + radius)]


def effect_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    greater = np.greater.outer(positive, negative).sum()
    equal = np.equal.outer(positive, negative).sum()
    return float((greater + 0.5 * equal) / (len(positive) * len(negative)))


def ensure_no_formal(paths: Iterable[Path]) -> None:
    offenders = [str(path) for path in paths if "formal" in path.parts]
    if offenders:
        raise RuntimeError(f"C4 formal access is forbidden: {offenders}")


def load_maps() -> dict[str, dict[str, float]]:
    payload = json.loads(C1_PATH.read_text(encoding="utf-8"))
    maps = {}
    for key, row in payload.get("effects", {}).items():
        a = row.get("a")
        b = row.get("b")
        if a is None or b is None:
            raise RuntimeError(f"missing C1 Platt mapping for {key}")
        if float(a) < 0.0:
            raise RuntimeError(f"non-monotone C1 mapping for {key}")
        maps[str(key)] = {"a": float(a), "b": float(b)}
    if len(maps) != 5:
        raise RuntimeError(f"expected five C1 effect mappings, found {len(maps)}")
    return maps


@dataclass(frozen=True)
class Unit:
    unit_id: str
    split: str
    task_id: int
    init_index: int
    policy_seed: int
    clean_path: Path
    noop_path: Path | None


@dataclass
class Episode:
    path: Path
    meta: dict[str, Any]
    keys: list[str]
    labels: np.ndarray
    raw_scores: np.ndarray
    calibrated_scores: np.ndarray


def split_index(split: str) -> dict[tuple[int, int, int], Path]:
    directory = RAW_ROOT / "episodes" / split
    paths = sorted(directory.glob("*.npz"))
    ensure_no_formal(paths)
    indexed: dict[tuple[int, int, int], Path] = {}
    for path in paths:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        key = (int(meta["task_id"]), int(meta["init_index"]), int(meta["policy_seed"]))
        if key in indexed:
            raise RuntimeError(f"duplicate {split} unit {key}")
        indexed[key] = path
    return indexed


def discover_units() -> tuple[list[Unit], dict[str, Any]]:
    calibration = split_index("calibration")
    qualification = split_index("qualification")
    pilot = split_index("pilot")
    units: list[Unit] = []
    for key, path in sorted(calibration.items()):
        task_id, init_index, seed = key
        units.append(
            Unit(
                unit_id=f"calibration-t{task_id}-i{init_index}-s{seed}",
                split="calibration",
                task_id=task_id,
                init_index=init_index,
                policy_seed=seed,
                clean_path=path,
                noop_path=None,
            )
        )
    for key, path in sorted(qualification.items()):
        task_id, init_index, seed = key
        units.append(
            Unit(
                unit_id=f"qualification-t{task_id}-i{init_index}-s{seed}",
                split="qualification",
                task_id=task_id,
                init_index=init_index,
                policy_seed=seed,
                clean_path=path,
                noop_path=pilot.get(key),
            )
        )
    if len(units) != 60:
        raise RuntimeError(f"expected 60 nonformal units, found {len(units)}")
    return units, {
        "calibration_episodes": len(calibration),
        "qualification_episodes": len(qualification),
        "pilot_episodes": len(pilot),
        "qualification_units_with_pilot_noop": sum(unit.noop_path is not None for unit in units),
        "formal_paths_read": [],
    }


def load_episode(path: Path, maps: dict[str, dict[str, float]]) -> Episode:
    ensure_no_formal([path])
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as data:
        features, _, effect_indices = _feature_rows(data, int(meta["task_id"]))
        flat_scores, _ = predict(CHECKPOINT, features)
        labels = np.asarray(data["predicate_values"], dtype=bool)
        names = [str(value) for value in data["predicate_labels"].tolist()]
    chunks = len(labels)
    raw_scores = np.stack(
        [flat_scores[index * chunks : (index + 1) * chunks] for index in range(len(names))],
        axis=1,
    )
    keys = [
        f"task{int(meta['task_id'])}:effect{index}:{name}"
        for index, name in enumerate(names)
    ]
    if any(key not in maps for key in keys):
        raise RuntimeError(f"effect mapping/schema mismatch for {path}: {keys}")
    calibrated_scores = np.column_stack(
        [platt(raw_scores[:, index], maps[key]["a"], maps[key]["b"]) for index, key in enumerate(keys)]
    )
    if effect_indices.shape[0] != flat_scores.shape[0]:
        raise RuntimeError(f"feature/effect index mismatch for {path}")
    return Episode(path, meta, keys, labels, raw_scores, calibrated_scores)


def injection_index(episode: Episode, condition: str) -> int:
    chunks = len(episode.labels)
    if condition in TRUE_CONDITIONS:
        return max(0, chunks - 1)
    return min(2, max(0, chunks - 1))


def replay_ledger(unit_id: str, effect_key_value: str, condition: str, seed: int) -> dict[str, Any]:
    """Replay the frozen Phase-5 core arm and retain state *and* decision."""

    stream_id = f"{unit_id}|{effect_key_value}"
    events = event_sequence(condition, stream_id, int(seed))
    arm = _make("M3_ASCEL_CORE", stream_id)
    decisions: list[str] = []
    for event in events:
        decisions.append(arm.process(event).value)
    final_state = arm.ledger.facts["TASK_GOAL"].fact_state.value
    final_decision = arm.decide("TASK_GOAL").value
    summary = arm.summary()
    reference = evaluate_arm("M3_ASCEL_CORE", condition, stream_id, int(seed))
    if bool(arm.effect_fact_verified("TASK_GOAL")) != bool(reference["verified"]):
        raise AssertionError(f"ledger verified mismatch for {condition}")
    if bool(arm.attempt_attributed_success("TASK_GOAL")) != bool(reference["credited"]):
        raise AssertionError(f"ledger credit mismatch for {condition}")
    return {
        "arm": "M3_ASCEL_CORE",
        "stream_id": stream_id,
        "stream_sha256": canonical_sha([event.to_dict() for event in events]),
        "event_count": len(events),
        "final_fact_state": final_state,
        "effect_fact_verified": bool(arm.effect_fact_verified("TASK_GOAL")),
        "attempt_attributed_success": bool(arm.attempt_attributed_success("TASK_GOAL")),
        "final_decision": final_decision,
        "decision_sequence": decisions,
        "ledger_counters": summary["ledger"]["counters"],
        "reference_semantics": {
            "verified": bool(reference["verified"]),
            "credited": bool(reference["credited"]),
            "false_grounded_advance": bool(reference["false_grounded_advance"]),
            "final_decision": str(reference["final_decision"]),
            "event_count": int(reference["event_count"]),
        },
    }


def physical_path(unit: Unit, condition: str) -> tuple[Path, str]:
    if condition == "A1_NOOP_RETRY_STALE" and unit.noop_path is not None:
        return unit.noop_path, "pilot_noop"
    return unit.clean_path, "clean"


def legacy_receipt_and(detector_pass: np.ndarray, ledger_verified: bool) -> np.ndarray:
    """Legacy receipt gate: every effect must pass before any effect upgrades."""

    accepted = bool(ledger_verified and bool(np.all(detector_pass)))
    return np.full(len(detector_pass), accepted, dtype=bool)


def independent_effect_upgrade(detector_pass: np.ndarray, ledger_verified: bool) -> np.ndarray:
    """C4 semantic change: each effect keeps its own detector/witness decision."""

    return np.asarray(detector_pass, dtype=bool) & bool(ledger_verified)


def run_property_tests() -> dict[str, Any]:
    """Small deterministic tests that distinguish the two semantics."""

    all_pass = np.asarray([True, True], dtype=bool)
    partial = np.asarray([True, False], dtype=bool)
    assert np.array_equal(legacy_receipt_and(all_pass, True), np.array([True, True]))
    assert np.array_equal(independent_effect_upgrade(all_pass, True), np.array([True, True]))
    assert np.array_equal(legacy_receipt_and(partial, True), np.array([False, False]))
    assert np.array_equal(independent_effect_upgrade(partial, True), np.array([True, False]))
    assert np.array_equal(legacy_receipt_and(partial, False), np.array([False, False]))
    assert np.array_equal(independent_effect_upgrade(partial, False), np.array([False, False]))
    checks: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        events = event_sequence(condition, f"property|{condition}", 0)
        arm = _make("M3_ASCEL_CORE", f"property|{condition}")
        for event in events:
            arm.process(event)
        reference = evaluate_arm("M3_ASCEL_CORE", condition, f"property|{condition}", 0)
        checks.append(
            {
                "condition": condition,
                "event_count": len(events),
                "fact_state": arm.ledger.facts["TASK_GOAL"].fact_state.value,
                "final_decision": arm.decide("TASK_GOAL").value,
                "reference_verified": bool(reference["verified"]),
                "reference_credited": bool(reference["credited"]),
            }
        )
        assert bool(arm.effect_fact_verified("TASK_GOAL")) == bool(reference["verified"])
        assert bool(arm.attempt_attributed_success("TASK_GOAL")) == bool(reference["credited"])
    return {
        "passed": True,
        "semantic_distinction": {
            "detector_pass": [True, False],
            "ledger_verified": True,
            "legacy_receipt_and_effects": [False, False],
            "independent_effect_upgrade_effects": [True, False],
        },
        "event_replay_checks": checks,
    }


def stored_nonformal_oracle_audit(units: list[Unit]) -> dict[str, Any]:
    """Cross-check the actual replay against the stored pilot oracle matrix."""

    path = ROOT / "experiments/r16p19_phase5/artifacts/results/oracle_pilot_results.jsonl"
    ensure_no_formal([path])
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {
        (row["task_id"], row["formal_init"], row["policy_seed"], row["condition"]): row
        for row in rows
        if row["arm"] == "M3_ASCEL_CORE"
    }
    comparisons = []
    for unit in units:
        if unit.noop_path is None:
            continue
        for condition in CONDITIONS:
            key = (unit.task_id, unit.init_index, unit.policy_seed, condition)
            row = selected.get(key)
            if row is None:
                continue
            replay = replay_ledger(unit.unit_id, "TASK_GOAL", condition, unit.policy_seed)
            comparisons.append(
                {
                    "unit_id": unit.unit_id,
                    "condition": condition,
                    "verified_match": replay["effect_fact_verified"] == bool(row["effect_truth_recognized"]),
                    "credit_match": replay["attempt_attributed_success"] == bool(row["active_attempt_credit"]),
                    "event_count_match": replay["event_count"] == int(row["semantic_event_count"]),
                }
            )
    mismatches = [
        row
        for row in comparisons
        if not (row["verified_match"] and row["credit_match"] and row["event_count_match"])
    ]
    return {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "rows_available": len(rows),
        "pilot_overlap_rows_compared": len(comparisons),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "formal_paths_read": [],
    }


def aggregate_variant(cells: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if variant not in {"legacy_receipt_and", "per_effect_independent"}:
        raise ValueError(variant)
    pred_key = "legacy_pred" if variant == "legacy_receipt_and" else "independent_pred"
    positives = sum(int(row["label"]) for row in cells)
    negatives = len(cells) - positives
    true_positive = sum(int(row[pred_key] and row["label"]) for row in cells)
    false_positive = sum(int(row[pred_key] and not row["label"]) for row in cells)
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        units[row["unit_id"]].append(row)
    concordant_ids = sorted(
        unit_id
        for unit_id, rows in units.items()
        if all(bool(row[pred_key]) == bool(row["label"]) for row in rows)
    )
    divergent_ids = sorted(set(units) - set(concordant_ids))
    fault_cells = [row for row in cells if row["condition"] in FAULT_CONDITIONS]
    fault_tp = sum(int(row[pred_key] and row["label"]) for row in fault_cells)
    fault_old_tp = sum(int(row["legacy_pred"] and row["label"]) for row in fault_cells)
    known_rows = [
        row
        for row in fault_cells
        if row["unit_id"] in concordant_ids
    ]
    unknown_rows = [
        row
        for row in fault_cells
        if row["unit_id"] in divergent_ids
    ]
    known_delta = sum(
        int(row[pred_key] and row["label"]) - int(row["legacy_pred"] and row["label"])
        for row in known_rows
    )
    unknown_cells = len(unknown_rows)
    oracle_unknown_delta = sum(
        int(row["label"]) - int(row["legacy_pred"] and row["label"])
        for row in unknown_rows
    )
    fault_denominator = len(fault_cells)
    lower = (
        float((known_delta - unknown_cells) / fault_denominator)
        if fault_denominator
        else None
    )
    upper = (
        float((known_delta + oracle_unknown_delta) / fault_denominator)
        if fault_denominator
        else None
    )
    known_cells = len(known_rows)
    neutral = None
    if fault_denominator and known_cells:
        neutral_rate = known_delta / known_cells
        neutral = float((known_delta + unknown_cells * neutral_rate) / fault_denominator)
    return {
        "effect_cells": len(cells),
        "positive_effect_cells": positives,
        "negative_effect_cells": negatives,
        "true_positive_effect_cells": true_positive,
        "false_upgrade_count": false_positive,
        "false_upgrade_wilson_95_upper": wilson_upper(false_positive, negatives),
        "false_upgrade_denominator": negatives,
        "effect_tpr": float(true_positive / positives) if positives else None,
        "effect_fpr": float(false_positive / negatives) if negatives else None,
        "unit_count": len(units),
        "concordant_units": len(concordant_ids),
        "divergent_units": len(divergent_ids),
        "divergent_unit_ids": divergent_ids,
        "faulted_effect_cells": fault_denominator,
        "faulted_true_positive_effect_cells": fault_tp,
        "gain_intervals": {
            "lower_adversarial": lower,
            "lower_rule": (
                "on divergent units, every faulted effect cell is assigned the "
                "worst possible delta (-1); concordant units inherit observed labels"
            ),
            "upper_inherited_oracle": upper,
            "upper_rule": (
                "conditional descriptive assumption: divergent units inherit their "
                "stored predicate labels as the oracle outcome"
            ),
            "neutral_assumption": neutral,
            "neutral_rule": (
                "conditional descriptive assumption: divergent-unit gain rate equals "
                "the concordant-unit observed gain rate; never a point estimate"
            ),
            "point_estimate_forbidden": True,
            "same_faulted_effect_denominator_before_after": fault_denominator,
        },
    }


def per_split_metrics(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "legacy_receipt_and": aggregate_variant(cells, "legacy_receipt_and"),
        "per_effect_independent": aggregate_variant(cells, "per_effect_independent"),
    }


def run_audit() -> dict[str, Any]:
    maps = load_maps()
    c2 = json.loads(C2_PATH.read_text(encoding="utf-8"))
    threshold = float(c2.get("global_threshold"))
    if not math.isclose(threshold, GLOBAL_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"C2 global threshold drift: {threshold}")
    units, inventory = discover_units()
    episode_cache: dict[Path, Episode] = {}
    cells: list[dict[str, Any]] = []
    event_cache: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for unit in units:
        for condition in CONDITIONS:
            path, physical_source = physical_path(unit, condition)
            episode = episode_cache.setdefault(path, load_episode(path, maps))
            index = injection_index(episode, condition)
            labels = episode.labels[index].astype(bool)
            scores = episode.calibrated_scores[index]
            detector_pass = scores >= threshold
            for effect_index, key in enumerate(episode.keys):
                cache_key = (unit.unit_id, key, condition, unit.policy_seed)
                if cache_key not in event_cache:
                    event_cache[cache_key] = replay_ledger(
                        unit.unit_id, key, condition, unit.policy_seed
                    )
                ledger = event_cache[cache_key]
                ledger_verified = bool(ledger["effect_fact_verified"])
                old_pred = legacy_receipt_and(detector_pass, ledger_verified)
                new_pred = independent_effect_upgrade(detector_pass, ledger_verified)
                cells.append(
                    {
                        "unit_id": unit.unit_id,
                        "split": unit.split,
                        "task_id": unit.task_id,
                        "init_index": unit.init_index,
                        "policy_seed": unit.policy_seed,
                        "condition": condition,
                        "effect_key": key,
                        "physical_source": physical_source,
                        "episode_path": str(path),
                        "injection_index": index,
                        "label": bool(labels[effect_index]),
                        "raw_score": float(episode.raw_scores[index, effect_index]),
                        "calibrated_score": float(scores[effect_index]),
                        "detector_pass": bool(detector_pass[effect_index]),
                        "receipt_all_detector_pass": bool(np.all(detector_pass)),
                        "ledger_final_fact_state": ledger["final_fact_state"],
                        "ledger_effect_fact_verified": ledger_verified,
                        "ledger_attempt_attributed_success": ledger[
                            "attempt_attributed_success"
                        ],
                        "ledger_final_decision": ledger["final_decision"],
                        "ledger_event_count": ledger["event_count"],
                        "ledger_stream_sha256": ledger["stream_sha256"],
                        "legacy_pred": bool(old_pred[effect_index]),
                        "independent_pred": bool(new_pred[effect_index]),
                    }
                )
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_split[cell["split"]].append(cell)
    pooled = per_split_metrics(cells)
    split_metrics = {
        split: per_split_metrics(rows) for split, rows in sorted(by_split.items())
    }
    replay_audit = stored_nonformal_oracle_audit(units)
    property_tests = run_property_tests()
    ledger_semantics_by_condition = {}
    for condition in CONDITIONS:
        replay = replay_ledger("C4-ledger-summary", "TASK_GOAL", condition, 0)
        ledger_semantics_by_condition[condition] = {
            "event_count": replay["event_count"],
            "stream_sha256": replay["stream_sha256"],
            "final_fact_state": replay["final_fact_state"],
            "effect_fact_verified": replay["effect_fact_verified"],
            "attempt_attributed_success": replay["attempt_attributed_success"],
            "final_decision": replay["final_decision"],
            "decision_sequence": replay["decision_sequence"],
            "ledger_counters": replay["ledger_counters"],
        }
    source_paths = [
        ROOT / "r16p19/phase4_event_broker.py",
        ROOT / "r16p19/phase4_types.py",
        ROOT / "r16p19/phase4_attempt_ledger.py",
        ROOT / "r16p19/phase5_arm_kernel.py",
        ROOT / "experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz",
        C0_PATH,
        C1_PATH,
        C2_PATH,
    ]
    ensure_no_formal(source_paths)
    evidence = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for path in source_paths
    ]
    result = {
        "schema_version": 2,
        "status": "COMPLETED_CPU_NONFORMAL_DIAGNOSTIC",
        "ledger_semantics_changed": True,
        "semantic_change": {
            "legacy": (
                "receipt-level AND: upgrade every effect only when the same "
                "receipt's detector vector is all true and the ledger witness is verified"
            ),
            "audited": (
                "per-effect independent upgrade: each effect upgrades when its own "
                "detector passes and its own ledger witness is verified"
            ),
            "ledger_arm_held_fixed": "M3_ASCEL_CORE",
            "receipt_and_is_outside_frozen_ledger": True,
        },
        "event_stream_materialization": {
            "serialized_phase5_6_7_event_log_paths": [],
            "serialized_event_log_available": False,
            "materialized_from_frozen_source": (
                "r16p19/phase5_arm_kernel.py:event_sequence plus the actual "
                "M3_ASCEL_CORE ledger; deterministic replay is cross-checked "
                "against oracle_pilot_results.jsonl"
            ),
            "event_log_search_scope": [
                "experiments/r16p19_phase5",
                "experiments/r16p19_phase6",
                "experiments/r16p19_phase7",
            ],
            "formal_paths_read": [],
        },
        "threshold": {
            "value": threshold,
            "source": "experiments/r16p19_phase8/C2_GLOBAL_THRESHOLD.json",
            "platt_mapping_source": "experiments/r16p19_phase8/C1_RECALIBRATION.json",
            "formal_access_count": 0,
        },
        "data_scope": {
            "splits": ["calibration", "qualification", "pilot"],
            "formal_paths_read": [],
            "formal_evaluation_performed": False,
            "new_data_collected": False,
            "inventory": inventory,
            "units": len(units),
            "effect_cells": len(cells),
            "faulted_effect_cells": sum(
                cell["condition"] in FAULT_CONDITIONS for cell in cells
            ),
        },
        "pooled_metrics": pooled,
        "split_metrics": split_metrics,
        "ledger_semantics_by_condition": ledger_semantics_by_condition,
        "stored_nonformal_oracle_replay": replay_audit,
        "property_tests": property_tests,
        "input_evidence": evidence,
        "c0_retrospective": {
            "pre_registration_sha256": sha256(C0_PATH),
            "audit_retrospective": True,
            "note": (
                "C0 was not independently sealed before this audit; this output "
                "records the breach and does not repair it."
            ),
        },
        "claim_eligible": False,
        "selection_eligible": False,
        "formal_claim_boundary": (
            "No formal labels or formal workpoint evaluation were read in C4; "
            "all metrics are diagnostic on nonformal frozen data."
        ),
    }
    write_json(PHASE8 / "C4_AUDITED.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    global PHASE8
    if args.output is not None:
        PHASE8 = args.output.parent
    result = run_audit()
    print(
        json.dumps(
            {
                "status": result["status"],
                "effect_cells": result["data_scope"]["effect_cells"],
                "formal_access_count": result["threshold"]["formal_access_count"],
                "pooled": {
                    key: {
                        "concordant_units": value["concordant_units"],
                        "divergent_units": value["divergent_units"],
                        "false_upgrade_count": value["false_upgrade_count"],
                        "false_upgrade_denominator": value["false_upgrade_denominator"],
                    }
                    for key, value in result["pooled_metrics"].items()
                },
                "oracle_overlap_mismatches": result[
                    "stored_nonformal_oracle_replay"
                ]["mismatch_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

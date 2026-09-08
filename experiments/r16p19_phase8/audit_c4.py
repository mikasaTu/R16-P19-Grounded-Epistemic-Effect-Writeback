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
from collections import Counter, defaultdict
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

Z95 = 1.959963984540054
TARGET = 0.90
# Retrospective diagnostic snapshot.  These values are copied from the
# non-formal C1/C2 artifacts at the frozen source commit below so this audit
# does not depend on a later/replaced canonical C1 or C2 file.
LEGACY_SOURCE_COMMIT = "585814e6645a1d42b4ea5742d302bea041b3b49c"
LEGACY_PLATT_SOURCE_SHA256 = "683bc2ad1acb35dd8322d1841703d87318cf7af84016524a11cdc8a1d3ebb251"
LEGACY_THRESHOLD_SOURCE_SHA256 = "ed46b20203134b1938c708d7212492762982e678f76d50abc29a317a466493d3"
LEGACY_C0_PREREGISTRATION_SHA256 = "de415c807142cecfba8469b3f3a589bc704784a378d1b54251e03f50bdeb6c6a"
GLOBAL_THRESHOLD = 0.6477173811021721
LEGACY_PLATT_MAPS = {
    "task0:effect0:in alphabet_soup_1 basket_1_contain_region": {
        "a": 1.6947934621244543,
        "b": -1.5731023652446074,
    },
    "task0:effect1:in tomato_sauce_1 basket_1_contain_region": {
        "a": 2.9801937660683024,
        "b": 1.8796757736654242,
    },
    "task5:effect0:in black_book_1 desk_caddy_1_back_contain_region": {
        "a": 3.9222267214660373,
        "b": 4.8056220553709945,
    },
    "task9:effect0:in white_yellow_mug_1 microwave_1_heating_region": {
        "a": 2.3269717128471488,
        "b": -1.7488385582958972,
    },
    "task9:effect1:close microwave_1": {
        "a": 0.9046587892921005,
        "b": -0.43846295399387,
    },
}
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

# Executable read/write boundary for the nonformal replay.
sys.dont_write_bytecode = True
def access_audit(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).absolute()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing:
        if (str(path).startswith(str(ROOT)) and not str(path).startswith(str(PHASE8))) or str(path).startswith(str(RAW_ROOT)):
            raise RuntimeError("protected source write: " + str(path))
    elif "formal" in path.parts or "formal_results" in path.name:
        raise RuntimeError("formal read forbidden in C4: " + str(path))
sys.addaudithook(access_audit)

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
    maps = {
        key: {"a": float(row["a"]), "b": float(row["b"])}
        for key, row in LEGACY_PLATT_MAPS.items()
    }
    for key, row in maps.items():
        if row["a"] < 0.0:
            raise RuntimeError(f"non-monotone frozen Platt mapping for {key}")
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


class PersistentPerEffectUpgradeLedger:
    """Persist detector-gated upgrade state around the frozen core ledger.

    The inner arm consumes the exact same Phase-5 events in both modes.  Its
    fact state, proof ids, validity, attribution, and decision are retained as
    witness provenance.  The outer persistent upgrade record is the C4
    semantic boundary: receipt-AND gates every effect together, whereas the
    independent mode gates each effect separately.
    """

    def __init__(self, effect_keys: list[str], unit_id: str, mode: str) -> None:
        if mode not in {"receipt_and", "per_effect_independent"}:
            raise ValueError(mode)
        self.effect_keys = list(effect_keys)
        self.unit_id = unit_id
        self.mode = mode
        self.arms = {
            key: _make("M3_ASCEL_CORE", f"{unit_id}|{key}")
            for key in self.effect_keys
        }
        self.records: dict[str, dict[str, Any]] = {}
        self.receipt_gate = False

    def process(
        self,
        events_by_effect: dict[str, list[Any]],
        detector_pass: np.ndarray,
    ) -> dict[str, Any]:
        detector_pass = np.asarray(detector_pass, dtype=bool)
        if len(detector_pass) != len(self.effect_keys):
            raise ValueError("detector/effect length mismatch")
        for key in self.effect_keys:
            if key not in events_by_effect:
                raise ValueError(f"missing event stream for {key}")
            for event in events_by_effect[key]:
                self.arms[key].process(event)
        witness_verified = np.asarray(
            [self.arms[key].effect_fact_verified("TASK_GOAL") for key in self.effect_keys],
            dtype=bool,
        )
        self.receipt_gate = bool(np.all(detector_pass) and np.all(witness_verified))
        records: dict[str, dict[str, Any]] = {}
        for index, key in enumerate(self.effect_keys):
            arm = self.arms[key]
            fact = arm.ledger.facts["TASK_GOAL"]
            witness_state = fact.fact_state.value
            witness_decision = arm.decide("TASK_GOAL").value
            proof_ids = list(fact.realization_proof_ids)
            proof_validity = {
                proof_id: arm.ledger.proofs[proof_id].validity_status.value
                for proof_id in proof_ids
            }
            if self.mode == "receipt_and":
                accepted = self.receipt_gate
                rejection_reason = "receipt_and_veto" if not accepted else None
            else:
                accepted = bool(detector_pass[index] and witness_verified[index])
                rejection_reason = (
                    "effect_detector_miss"
                    if not detector_pass[index]
                    else "ledger_witness_not_verified"
                ) if not accepted else None
            if bool(witness_verified[index]) and accepted:
                upgrade_state = "REALIZED"
                upgrade_decision = "ADVANCE_TO_NEXT_SUBTASK"
            elif bool(witness_verified[index]) and not accepted:
                # The witness and proof remain present above; only the C4
                # upgrade record is held back by the detector gate.
                upgrade_state = "UNKNOWN"
                upgrade_decision = "REOBSERVE"
            else:
                upgrade_state = witness_state
                upgrade_decision = witness_decision
            records[key] = {
                "detector_pass": bool(detector_pass[index]),
                "witness_fact_state": witness_state,
                "witness_effect_fact_verified": bool(witness_verified[index]),
                "witness_attempt_attributed_success": bool(
                    arm.attempt_attributed_success("TASK_GOAL")
                ),
                "witness_decision": witness_decision,
                "proof_ids": proof_ids,
                "proof_validity": proof_validity,
                "upgrade_fact_state": upgrade_state,
                "upgrade_decision": upgrade_decision,
                "upgrade_accepted": bool(accepted),
                "upgrade_rejection_reason": rejection_reason,
            }
        self.records = records
        return {
            "mode": self.mode,
            "receipt_gate": self.receipt_gate,
            "records": records,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "unit_id": self.unit_id,
            "receipt_gate": self.receipt_gate,
            "records": self.records,
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
    """Small deterministic tests that distinguish persistent ledger states."""

    all_pass = np.asarray([True, True], dtype=bool)
    partial = np.asarray([True, False], dtype=bool)
    assert np.array_equal(legacy_receipt_and(all_pass, True), np.array([True, True]))
    assert np.array_equal(independent_effect_upgrade(all_pass, True), np.array([True, True]))
    assert np.array_equal(legacy_receipt_and(partial, True), np.array([False, False]))
    assert np.array_equal(independent_effect_upgrade(partial, True), np.array([True, False]))
    assert np.array_equal(legacy_receipt_and(partial, False), np.array([False, False]))
    assert np.array_equal(independent_effect_upgrade(partial, False), np.array([False, False]))
    effect_keys = ["effect_a", "effect_b"]
    events_by_effect = {
        key: event_sequence("C0_CLEAN", f"property|{key}", 0)
        for key in effect_keys
    }
    legacy_ledger = PersistentPerEffectUpgradeLedger(
        effect_keys, "property", "receipt_and"
    )
    independent_ledger = PersistentPerEffectUpgradeLedger(
        effect_keys, "property", "per_effect_independent"
    )
    legacy_state = legacy_ledger.process(events_by_effect, partial)
    independent_state = independent_ledger.process(events_by_effect, partial)
    assert legacy_state["records"]["effect_a"]["upgrade_fact_state"] == "UNKNOWN"
    assert legacy_state["records"]["effect_a"]["upgrade_decision"] == "REOBSERVE"
    assert independent_state["records"]["effect_a"]["upgrade_fact_state"] == "REALIZED"
    assert independent_state["records"]["effect_a"]["upgrade_decision"] == "ADVANCE_TO_NEXT_SUBTASK"
    assert legacy_state["records"]["effect_b"]["upgrade_fact_state"] == "UNKNOWN"
    assert independent_state["records"]["effect_b"]["upgrade_fact_state"] == "UNKNOWN"
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
        "persistent_ledger_distinction": {
            "same_event_stream": True,
            "legacy": legacy_state,
            "independent": independent_state,
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


def pilot_task_success_gain(
    units: list[Unit], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute the frozen pilot task-success gain envelope separately.

    The per-effect detector output is not a task-success outcome.  For the
    fifteen qualification units paired with a stored pilot NOOP episode, a
    variant is therefore called concordant only when its complete upgraded
    effect vector equals the legacy oracle receipt-AND vector in every one of
    the seven conditions.  Only those concordant units inherit the stored
    Phase-5 M0/M3 task outcomes.  Divergent units remain unknown, yielding the
    requested adversarial, inherited-oracle, and neutral (zero divergent gain)
    envelopes.
    """

    path = ROOT / "experiments/r16p19_phase5/artifacts/results/oracle_pilot_results.jsonl"
    ensure_no_formal([path])
    source_rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    oracle_by = {
        (
            int(row["task_id"]),
            int(row["formal_init"]),
            int(row["policy_seed"]),
            str(row["condition"]),
            str(row["arm"]),
        ): row
        for row in source_rows
        if row["arm"] in {"M0_TYPED_MATCHED", "M3_ASCEL_CORE"}
    }
    pilot_units = [unit for unit in units if unit.noop_path is not None]
    by_unit_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if cell["unit_id"] in {unit.unit_id for unit in pilot_units}:
            by_unit_condition[(cell["unit_id"], cell["condition"])].append(cell)
    expected_rows = len(pilot_units) * len(CONDITIONS) * 2
    selected_rows = sum(
        1
        for unit in pilot_units
        for condition in CONDITIONS
        for arm in ("M0_TYPED_MATCHED", "M3_ASCEL_CORE")
        if (
            unit.task_id,
            unit.init_index,
            unit.policy_seed,
            condition,
            arm,
        ) in oracle_by
    )
    if selected_rows != expected_rows:
        raise RuntimeError(
            f"pilot oracle task rows incomplete: {selected_rows} != {expected_rows}"
        )

    variants = {
        "legacy_receipt_and": "legacy_pred",
        "per_effect_independent": "independent_pred",
    }
    variant_results: dict[str, Any] = {}
    for variant, pred_key in variants.items():
        concordant: list[str] = []
        divergent: list[str] = []
        mismatch_examples: list[dict[str, Any]] = []
        for unit in pilot_units:
            unit_match = True
            for condition in CONDITIONS:
                rows = sorted(
                    by_unit_condition[(unit.unit_id, condition)],
                    key=lambda row: row["effect_key"],
                )
                if len(rows) not in {1, 2}:
                    raise RuntimeError(
                        f"pilot cell vector has unexpected required-effect count for "
                        f"{unit.unit_id}/{condition}: {len(rows)}"
                    )
                oracle_receipt = bool(all(bool(row["label"]) for row in rows))
                predicted = [bool(row[pred_key]) for row in rows]
                expected = [oracle_receipt] * len(rows)
                if predicted != expected:
                    unit_match = False
                    if len(mismatch_examples) < 12:
                        mismatch_examples.append(
                            {
                                "unit_id": unit.unit_id,
                                "condition": condition,
                                "oracle_receipt_and_vector": expected,
                                "variant_effect_vector": predicted,
                            }
                        )
            (concordant if unit_match else divergent).append(unit.unit_id)

        concordant_set = set(concordant)
        known_diff = 0
        inherited_oracle_diff = 0
        for unit in pilot_units:
            for condition in sorted(FAULT_CONDITIONS):
                baseline_key = (
                    unit.task_id,
                    unit.init_index,
                    unit.policy_seed,
                    condition,
                    "M0_TYPED_MATCHED",
                )
                core_key = (
                    unit.task_id,
                    unit.init_index,
                    unit.policy_seed,
                    condition,
                    "M3_ASCEL_CORE",
                )
                baseline_success = int(bool(oracle_by[baseline_key]["task_success"]))
                core_success = int(bool(oracle_by[core_key]["task_success"]))
                diff = core_success - baseline_success
                inherited_oracle_diff += diff
                if unit.unit_id in concordant_set:
                    known_diff += diff
        denominator = len(pilot_units) * len(FAULT_CONDITIONS)
        divergent_faulted_cells = len(divergent) * len(FAULT_CONDITIONS)
        variant_results[variant] = {
            "unit_count": len(pilot_units),
            "concordant_units": len(concordant),
            "concordant_unit_ids": sorted(concordant),
            "divergent_units": len(divergent),
            "divergent_unit_ids": sorted(divergent),
            "condition_vector_mismatch_examples": mismatch_examples,
            "oracle_vector_definition": (
                "at each condition, the one or two required-effect predicate labels "
                "at the injection frame are reduced by legacy receipt AND; the expected "
                "vector repeats that receipt value for every required effect"
            ),
            "faulted_conditions": sorted(FAULT_CONDITIONS),
            "faulted_task_cell_denominator": denominator,
            "known_concordant_core_minus_baseline_sum": known_diff,
            "all_pilot_inherited_oracle_core_minus_baseline_sum": inherited_oracle_diff,
            "divergent_faulted_cells": divergent_faulted_cells,
            "task_success_gain_interval": {
                "lower_adversarial": float(
                    (known_diff - divergent_faulted_cells) / denominator
                ),
                "upper_inherited_oracle": float(inherited_oracle_diff / denominator),
                "neutral_divergence_zero": float(known_diff / denominator),
                "lower_rule": (
                    "concordant cells inherit stored M3 minus M0 task_success; every "
                    "divergent faulted cell is assigned worst-case gain -1"
                ),
                "upper_rule": (
                    "conditional descriptive assumption: every divergent unit inherits "
                    "the stored pilot M3 minus M0 task_success"
                ),
                "neutral_rule": (
                    "conditional descriptive assumption: every divergent faulted cell "
                    "has zero gain"
                ),
                "point_estimate_forbidden": True,
            },
        }
    return {
        "scope": "pilot_only_nonformal",
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "arms": ["M0_TYPED_MATCHED", "M3_ASCEL_CORE"],
        "rows_available": len(source_rows),
        "task_rows_used": selected_rows,
        "formal_paths_read": [],
        "variants": variant_results,
        "nonpilot_task_success_gain": None,
        "nonpilot_reason": (
            "No frozen task-level M0/M3 outcomes are available for calibration or "
            "qualification outside the paired pilot; per-effect predictions cannot "
            "be substituted for task_success."
        ),
    }


def aggregate_variant(cells: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if variant not in {"legacy_receipt_and", "per_effect_independent"}:
        raise ValueError(variant)
    pred_key = "legacy_pred" if variant == "legacy_receipt_and" else "independent_pred"
    state_key = (
        "legacy_upgrade_fact_state"
        if variant == "legacy_receipt_and"
        else "independent_upgrade_fact_state"
    )
    decision_key = (
        "legacy_upgrade_decision"
        if variant == "legacy_receipt_and"
        else "independent_upgrade_decision"
    )
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
        "persistent_upgrade_fact_state_counts": dict(sorted(Counter(
            row[state_key] for row in cells
        ).items())),
        "persistent_upgrade_decision_counts": dict(sorted(Counter(
            row[decision_key] for row in cells
        ).items())),
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
    # This is a fixed retrospective diagnostic snapshot.  C1/C2 files are
    # deliberately not read because the canonical step-9 artifacts are being
    # replaced by the parent audit.
    threshold = GLOBAL_THRESHOLD
    units, inventory = discover_units()
    episode_cache: dict[Path, Episode] = {}
    cells: list[dict[str, Any]] = []
    for unit in units:
        for condition in CONDITIONS:
            path, physical_source = physical_path(unit, condition)
            episode = episode_cache.setdefault(path, load_episode(path, maps))
            index = injection_index(episode, condition)
            labels = episode.labels[index].astype(bool)
            scores = episode.calibrated_scores[index]
            detector_pass = scores >= threshold
            ledger_unit_id = f"{unit.unit_id}|{condition}"
            events_by_effect = {
                key: event_sequence(
                    condition,
                    f"{ledger_unit_id}|{key}",
                    unit.policy_seed,
                )
                for key in episode.keys
            }
            event_hashes = {
                key: canonical_sha([event.to_dict() for event in events])
                for key, events in events_by_effect.items()
            }
            legacy_ledger = PersistentPerEffectUpgradeLedger(
                episode.keys, ledger_unit_id, "receipt_and"
            )
            independent_ledger = PersistentPerEffectUpgradeLedger(
                episode.keys, ledger_unit_id, "per_effect_independent"
            )
            legacy_result = legacy_ledger.process(events_by_effect, detector_pass)
            independent_result = independent_ledger.process(
                events_by_effect, detector_pass
            )
            for effect_index, key in enumerate(episode.keys):
                legacy_record = legacy_result["records"][key]
                independent_record = independent_result["records"][key]
                if (
                    legacy_record["witness_fact_state"]
                    != independent_record["witness_fact_state"]
                    or legacy_record["witness_effect_fact_verified"]
                    != independent_record["witness_effect_fact_verified"]
                    or legacy_record["proof_ids"] != independent_record["proof_ids"]
                    or legacy_record["proof_validity"]
                    != independent_record["proof_validity"]
                ):
                    raise AssertionError(
                        f"ledger witness provenance diverged for {ledger_unit_id}/{key}"
                    )
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
                        "ledger_final_fact_state": legacy_record["witness_fact_state"],
                        "ledger_effect_fact_verified": legacy_record[
                            "witness_effect_fact_verified"
                        ],
                        "ledger_attempt_attributed_success": legacy_record[
                            "witness_attempt_attributed_success"
                        ],
                        "ledger_final_decision": legacy_record["witness_decision"],
                        "ledger_event_count": len(events_by_effect[key]),
                        "ledger_stream_sha256": event_hashes[key],
                        "ledger_proof_ids": legacy_record["proof_ids"],
                        "ledger_proof_validity": legacy_record["proof_validity"],
                        "legacy_upgrade_fact_state": legacy_record[
                            "upgrade_fact_state"
                        ],
                        "legacy_upgrade_decision": legacy_record["upgrade_decision"],
                        "legacy_upgrade_accepted": legacy_record["upgrade_accepted"],
                        "legacy_upgrade_rejection_reason": legacy_record[
                            "upgrade_rejection_reason"
                        ],
                        "independent_upgrade_fact_state": independent_record[
                            "upgrade_fact_state"
                        ],
                        "independent_upgrade_decision": independent_record[
                            "upgrade_decision"
                        ],
                        "independent_upgrade_accepted": independent_record[
                            "upgrade_accepted"
                        ],
                        "independent_upgrade_rejection_reason": independent_record[
                            "upgrade_rejection_reason"
                        ],
                        "legacy_pred": bool(legacy_record["upgrade_accepted"]),
                        "independent_pred": bool(
                            independent_record["upgrade_accepted"]
                        ),
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
    task_gain_audit = pilot_task_success_gain(units, cells)
    property_tests = run_property_tests()
    upgrade_examples = []
    for cell in cells:
        if (
            cell["legacy_upgrade_fact_state"]
            != cell["independent_upgrade_fact_state"]
            or cell["legacy_upgrade_decision"]
            != cell["independent_upgrade_decision"]
            or cell["legacy_upgrade_accepted"]
            != cell["independent_upgrade_accepted"]
        ):
            upgrade_examples.append(
                {
                    "unit_id": cell["unit_id"],
                    "condition": cell["condition"],
                    "effect_key": cell["effect_key"],
                    "detector_pass": cell["detector_pass"],
                    "witness_fact_state": cell["ledger_final_fact_state"],
                    "witness_decision": cell["ledger_final_decision"],
                    "proof_ids": cell["ledger_proof_ids"],
                    "proof_validity": cell["ledger_proof_validity"],
                    "legacy_upgrade_fact_state": cell[
                        "legacy_upgrade_fact_state"
                    ],
                    "legacy_upgrade_decision": cell["legacy_upgrade_decision"],
                    "legacy_upgrade_accepted": cell["legacy_upgrade_accepted"],
                    "legacy_upgrade_rejection_reason": cell[
                        "legacy_upgrade_rejection_reason"
                    ],
                    "independent_upgrade_fact_state": cell[
                        "independent_upgrade_fact_state"
                    ],
                    "independent_upgrade_decision": cell[
                        "independent_upgrade_decision"
                    ],
                    "independent_upgrade_accepted": cell[
                        "independent_upgrade_accepted"
                    ],
                    "independent_upgrade_rejection_reason": cell[
                        "independent_upgrade_rejection_reason"
                    ],
                }
            )
            if len(upgrade_examples) >= 12:
                break
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
    per_effect_ledger_records = [
        {
            "unit_id": cell["unit_id"],
            "split": cell["split"],
            "condition": cell["condition"],
            "effect_key": cell["effect_key"],
            "detector_pass": cell["detector_pass"],
            "event_count": cell["ledger_event_count"],
            "event_stream_sha256": cell["ledger_stream_sha256"],
            "witness_fact_state": cell["ledger_final_fact_state"],
            "witness_effect_fact_verified": cell["ledger_effect_fact_verified"],
            "witness_attempt_attributed_success": cell[
                "ledger_attempt_attributed_success"
            ],
            "witness_decision": cell["ledger_final_decision"],
            "proof_ids": cell["ledger_proof_ids"],
            "proof_validity": cell["ledger_proof_validity"],
            "legacy_upgrade_fact_state": cell["legacy_upgrade_fact_state"],
            "legacy_upgrade_decision": cell["legacy_upgrade_decision"],
            "legacy_upgrade_accepted": cell["legacy_upgrade_accepted"],
            "legacy_upgrade_rejection_reason": cell[
                "legacy_upgrade_rejection_reason"
            ],
            "independent_upgrade_fact_state": cell[
                "independent_upgrade_fact_state"
            ],
            "independent_upgrade_decision": cell["independent_upgrade_decision"],
            "independent_upgrade_accepted": cell["independent_upgrade_accepted"],
            "independent_upgrade_rejection_reason": cell[
                "independent_upgrade_rejection_reason"
            ],
        }
        for cell in cells
    ]
    source_paths = [
        ROOT / "r16p19/phase4_event_broker.py",
        ROOT / "r16p19/phase4_types.py",
        ROOT / "r16p19/phase4_attempt_ledger.py",
        ROOT / "r16p19/phase5_arm_kernel.py",
        ROOT / "experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz",
        ROOT / "experiments/r16p19_phase5/artifacts/results/oracle_pilot_results.jsonl",
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
            "source": "retrospective diagnostic snapshot",
            "platt_mapping_source": "retrospective diagnostic snapshot",
            "source_commit": LEGACY_SOURCE_COMMIT,
            "platt_map_snapshot_sha256": LEGACY_PLATT_SOURCE_SHA256,
            "threshold_snapshot_sha256": LEGACY_THRESHOLD_SOURCE_SHA256,
            "maps": maps,
            "canonical_selection": False,
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
        "pilot_task_success_gain": task_gain_audit,
        "persistent_upgrade_examples": upgrade_examples,
        "per_effect_ledger_records": per_effect_ledger_records,
        "property_tests": property_tests,
        "input_evidence": evidence,
        "c0_retrospective": {
            "pre_registration_sha256": LEGACY_C0_PREREGISTRATION_SHA256,
            "audit_retrospective": True,
            "note": (
                "C0 was not independently sealed before this audit; the source "
                "digest is recorded from the frozen retrospective snapshot and "
                "does not repair the breach."
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

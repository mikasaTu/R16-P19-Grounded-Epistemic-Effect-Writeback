from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "run_s1.py"
SPEC = importlib.util.spec_from_file_location("phase6_s1", MODULE_PATH)
S1 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(S1)


def test_effect_curve_selects_highest_threshold_reaching_target():
    labels = np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93])
    curve = S1._effect_curve(labels, scores)
    assert curve["threshold_at_tpr_0_90"] == 0.4
    assert curve["tpr_at_selected_threshold"] == 0.9


def test_soft_rule_uses_no_formal_input_and_zero_calibration_false_upgrades():
    receipts = [
        {"task_id": 0, "effect_keys": ["a", "b"], "scores": np.asarray([0.9, 0.9]), "labels": np.asarray([1, 1])},
        {"task_id": 0, "effect_keys": ["a", "b"], "scores": np.asarray([0.9, 0.1]), "labels": np.asarray([1, 0])},
        {"task_id": 5, "effect_keys": ["c"], "scores": np.asarray([0.8]), "labels": np.asarray([1])},
        {"task_id": 5, "effect_keys": ["c"], "scores": np.asarray([0.2]), "labels": np.asarray([0])},
    ]
    rule = S1._fit_soft_rule(receipts, ["a", "b", "c"])
    assert rule["calibration_false_upgrades"] == 0
    assert set(rule["weights"]) == {"a", "b", "c"}


def test_formal_index_matches_frozen_phase5_logic():
    meta = {"chunks": 9}
    assert S1._formal_index(meta, "C0_CLEAN") == 8
    assert S1._formal_index(meta, "A5_EXTERNAL_REALIZATION") == 8
    assert S1._formal_index(meta, "A2_CROSS_ATTEMPT_MIX") == 2


def test_calibration_guard_rejects_formal_path(tmp_path):
    formal = tmp_path / "episodes" / "formal"
    formal.mkdir(parents=True)
    try:
        S1._calibration_dataset(formal, tmp_path / "checkpoint.npz")
    except RuntimeError as error:
        assert "formal" in str(error)
    else:
        raise AssertionError("formal selection input was not rejected")

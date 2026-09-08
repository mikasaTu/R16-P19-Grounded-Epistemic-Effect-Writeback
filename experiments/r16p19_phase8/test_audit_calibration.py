import json
from pathlib import Path

import numpy as np

import audit_calibration as audit


ROOT = Path(__file__).resolve().parent


def test_temperature_and_platt_are_strictly_increasing_and_auc_invariant():
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=bool)
    scores = np.asarray([0.05, 0.25, 0.35, 0.65, 0.8, 0.9], dtype=float)
    z = audit.logit(scores)
    temperature, _ = audit.fit_temperature(z, labels)
    a, b, _ = audit.fit_platt_positive(z, labels)
    assert temperature > 0.0
    assert a > 0.0
    mapped_temperature = z / temperature
    mapped_platt = a * z + b
    assert np.all(np.diff(mapped_temperature) > 0.0)
    assert np.all(np.diff(mapped_platt) > 0.0)
    assert audit.auc_rank(labels, z) == audit.auc_rank(labels, mapped_temperature)
    assert audit.auc_rank(labels, z) == audit.auc_rank(labels, mapped_platt)


def test_nonformal_map_artifact_has_positive_slopes_and_no_formal_selection():
    maps = json.loads((ROOT / "C2_EXTRA_MAPS_NONFORMAL.json").read_text())
    assert set(maps) == {"temperature", "platt"}
    for family in maps.values():
        assert len(family) == 5
        for params in family.values():
            assert params["a"] > 0.0
            assert set(params) == {"a", "b"}

    artifact = json.loads((ROOT / "C1_MODELS_NONFORMAL.json").read_text())
    assert artifact["formal_access_count"] == 0
    assert artifact["formal_read_paths"] == []
    assert artifact["selection_seal_written"] is False
    for row in artifact["models"].values():
        assert row["monotonicity"]["temperature_non_decreasing"] is True
        assert row["monotonicity"]["platt_non_decreasing"] is True
        before = row["auc_decision_score_before"]
        assert row["auc_decision_score_after"]["temperature"] == before
        assert row["auc_decision_score_after"]["platt"] == before


def test_c1_formal_diagnostic_is_ineligible_and_reports_both_maps():
    result = json.loads((ROOT / "C1_AUDITED.json").read_text())
    formal = result["formal_diagnostic"]
    assert formal["formal_labels_used"] is True
    assert formal["claim_eligible"] is False
    assert formal["selection_eligible"] is False
    assert formal["selection_seal_written"] is False
    assert set(result["operating_points"]) == {"0.02", "0.05"}
    for cap in result["operating_points"].values():
        assert set(["raw_phase7_reference", "raw_recomputed", "temperature", "platt"]).issubset(cap)
        for name in ("raw_recomputed", "temperature", "platt"):
            assert cap[name]["claim_eligible"] is False
            assert cap[name]["selection_eligible"] is False


def test_c3_reproduces_phase7_and_has_receipt_and_effect_curves():
    result = json.loads((ROOT / "C3_AUDITED.json").read_text())
    assert result["formal_access_count"] == 0
    assert result["sum_log_p"]["status"] == "EXCLUDED"
    for split in ("estimation", "qualification"):
        row = result["splits"][split]
        assert row["baseline_phase7_b4_reproduction"]["auc_match"] is True
        assert row["baseline_phase7_b4_reproduction"]["point_count_match"] is True
        assert row["raw"]["receipt_count"] == row["raw"]["positives"] + row["raw"]["negatives"]
        for model_name in ("raw", "temperature", "platt"):
            curve = row[model_name]
            assert curve["produces_effect_decisions"] is True
            assert curve["points"]
            first = curve["points"][0]
            assert "fpr" in first and "fp" in first and "per_effect_tpr" in first

#!/usr/bin/env python3
"""D2 correctness checks: maps, sampling, and diagnostics boundary."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("phase9_baserate", HERE / "phase9_baserate.py")
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_maps_are_positive_monotone_and_auc_invariant() -> None:
    effect_rows, _, _, _, _ = mod.load_snapshot()
    for key, values in sorted(effect_rows.items()):
        _, diagnostic = mod.fit_one_effect(values)
        assert diagnostic["positive_slope"] is True
        assert diagnostic["monotonic"] is True
        assert diagnostic["auc_invariant"] is True
        assert abs(diagnostic["auc_before"] - diagnostic["auc_after"]) <= 1e-12


def test_resampling_counts_are_exact() -> None:
    effect_rows, _, _, _, _ = mod.load_snapshot()
    for target in mod.TARGET_LEVELS:
        for seed in mod.SEEDS:
            for offset, key in enumerate(sorted(effect_rows)):
                _, diagnostic = mod.fit_one_effect(
                    effect_rows[key], target, seed + offset * 1000003
                )
                positive = diagnostic["positive_count"]
                expected_negative = round(positive * (1.0 - target) / target)
                assert diagnostic["negative_count_used"] == expected_negative
                assert abs(diagnostic["realized_positive_fraction"] - target) <= 1e-12


def test_result_has_all_replicates_and_flags_formal_diagnostic() -> None:
    result = json.loads((HERE / "D2_BASERATE.json").read_text(encoding="utf-8"))
    assert result["formal_source_reopened"] is False
    assert result["formal_labels_used"] is True
    assert result["claim_eligible"] is False
    assert result["selection_eligible"] is False
    assert result["fit"]["verifier_weights_changed"] is False
    assert result["fit"]["l2"] == 1e-6
    assert result["replicates_per_level"] == 10
    assert result["seeds"] == list(range(19001, 19011))
    assert set(result["levels"]) == {"0.02", "0.05", "0.10", "0.25", "0.50"}
    for level in result["levels"].values():
        assert len(level["replicates"]) == 10
        for replicate in level["replicates"]:
            assert set(replicate["operating_points"]) == {"0.02", "0.05"}
            assert all(d["positive_slope"] and d["monotonic"] and d["auc_invariant"] for d in replicate["fit_diagnostics"].values())


if __name__ == "__main__":
    test_maps_are_positive_monotone_and_auc_invariant()
    test_resampling_counts_are_exact()
    test_result_has_all_replicates_and_flags_formal_diagnostic()
    print("D2 base-rate tests: 3 passed")

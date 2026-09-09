#!/usr/bin/env python3
"""Protocol assertions for the D3 negative-only selector."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("phase9_nonformal_d3", HERE / "phase9_nonformal.py")
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_positive_path_rejected() -> None:
    try:
        mod.assert_selector_read_path(HERE / "positive_labels.json")
    except RuntimeError:
        return
    raise AssertionError("positive-label path was accepted by D3 selector hook")


def test_threshold_rank_and_null() -> None:
    threshold, rank, reject_all = mod.order_threshold([0.1, 0.2, 0.3], 0.05)
    assert threshold is None and rank == 4 and reject_all
    threshold, rank, reject_all = mod.order_threshold([0.1, 0.2, 0.3], 0.5)
    assert threshold == 0.2 and rank == 2 and not reject_all


def test_d3_artifacts_are_negative_only() -> None:
    export = json.loads((HERE / "D3_NEGATIVE_EXPORT.json").read_text(encoding="utf-8"))
    seal = json.loads((HERE / "D3_OPERATING_POINT_SEAL.json").read_text(encoding="utf-8"))
    assert export["label_fields_present"] is False
    assert export["positive_label_values_present"] is False
    assert export["formal_access_count"] == 0
    assert export["split_episode_id_overlap_count"] == 0
    assert seal["formal_access_count"] == 0
    assert seal["positive_label_access_count"] == 0
    assert seal["read_hook"]["positive_path_test_passed"] is True
    assert seal["certified_point"] is None
    assert seal["certified"] is False
    assert seal["claim_eligible"] is False
    assert seal["selection_eligible"] is False


if __name__ == "__main__":
    test_positive_path_rejected()
    test_threshold_rank_and_null()
    test_d3_artifacts_are_negative_only()
    print("D3 protocol tests: 3 passed")

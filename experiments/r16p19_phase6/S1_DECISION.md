# S1 G1 decision

**Decision: `FAIL_G1`.** This is the end of S1. S2 was not started and this is not a Phase-6 final report.

## Gates

- `g1_min_per_effect_tpr_ge_0_90`: `false`
- `g2_same_variant_false_upgrade_count_zero`: `false`
- `g3_calibration_selection_never_accessed_formal_receipts`: `true`
- `g4_reproduction_mismatch_rows_zero`: `true`
- Qualifying variant(s): `none`

## Leakage and reproduction evidence

- Phase-5 replay: 1680 rows, 0 mismatches, max score error 0.
- Calibration selection read 30 calibration episodes and exactly 0 formal receipts.
- Calibration input-evidence digest: `43776fb09e6b05e51664048d6565523657a83cda38656cec13202abc5e24568b`.
- Formal evaluation began only after `S1_CALIBRATION_SEAL.json` was written and subsequently hash-verified.

## Mechanism reverse-engineering (observed, not a new idea)

At the Phase-5 global threshold, B has min per-effect TPR 0.0000, oracle agreement 0.7333, and 0 false upgrades. Per-effect recalibration changes these to 0.8421, 0.9571, and 8; weighted soft aggregation gives 0.0000, 0.7833, and 4.

The frozen model ranks examples well, but effect prevalences and score scales differ sharply. Rare late effects (second-object placement or closure) sit below 0.9395 despite being separable from negatives. Receipt-level `np.all` then turns one low-scale effect into a hard veto. C isolates the threshold change while retaining AND: it recovers most oracle agreement, confirming that threshold scale mismatch and AND amplification caused a large part of the Phase-5 collapse. However C still misses the 0.90 formal TPR gate and introduces false upgrades; D also fails. Therefore the collapse cannot be attributed exclusively to aggregation: calibration-to-formal transfer/detector reliability remains insufficient on this frozen substrate.

## S2 handoff

- Union disagreement units: 93.
- Exact IDs: `S1_DISAGREEMENT_UNITS.txt`.
- No rollout, GPU job, PAI job, or S2 execution was started.

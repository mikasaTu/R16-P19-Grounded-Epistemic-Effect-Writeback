# R16-P19 Phase-11 / step12

CPU-only F1–F5 migration diagnostics. `F_DECISION.md` is the stopping gate. No GPU, PAI, rollout, S2, new data, or real-robot experiment was run.

`run_phase11.py` trains new small-MLP diagnostics under `models/` from natural policy seeds 0–4 and calibration, then evaluates qualification. `phase11_formal.py` loads formal once after model freeze into `F1_FORMAL_CACHE.npz`; an earlier schema probe and Phase-9 mixed-cache preview are disclosed in `PROTOCOL_DEVIATIONS.json`, so every formal result is descriptive and `protocol_valid=false`.

`phase11_finalize.py` writes F1–F5 reports, named gates, mechanism reverse engineering, and decision files. `test_phase11.py` has 11 unit tests, including negative gate inputs, tie-safe FPR thresholds, episode bootstrap, model weight/hash checks, split counts, and formal non-eligibility. `SHA256SUMS` excludes itself.

The 35-dimensional source feature layout is documented in `F2_SENSOR_PARITY.md`; the 207-dimensional spatial transfer model is conditional on unverified proprio parity. QPILOTS is readable but `arrange_3_flowers` takeover data was not found, so F3 provides requirements with null inventory values.

# Phase-9 / step9a stage diagnostics

Source plan title is Phase-9 / step10; the requested Feishu hierarchy is step9a.

All D0–D6 diagnostic computations were completed. **The original protocol was not fully satisfied:** formal was loaded in two attempts, and D1 compares different sampling populations. The terminal decision is `NOT_READY_FOR_S2`. These are diagnostic results, not a valid single-read formal certification experiment.

- D0 independent commit: `d4c81be6149068ab56cda8350c99b86ddf1c7fa8`
- D3 independent seal commit: `21acdc25331cb221fbeab2c34a1b9044580cadc2`
- `USER_PLAN.md` and `PROTOCOL_AUTHORIZATION.md`: original plan and approved order change.
- `D3_SELECTION.json`: all four alpha levels, both confidence conventions and both counting units.
- `D1_EXCHANGEABILITY.json`: all requested quantiles/KS/FPR ratios, with sampling-population limitation.
- `D2_BASERATE.json`: 5 base rates × 10 resamples × 2 FPR caps, positive-slope regularized Platt.
- `D4_FORMAL.json`: corrected effect-negative episode clustering and receipt metrics.
- `D5_REPLAY.json`: actual persistent receipt-AND replay, 1400 event streams/8000 events, unit lists and gain bounds.
- `D5_S2_UNITS.txt`: 114 diagnostic candidate units; S2 was not started.
- `D6_FAILURE_MODES.json`: empty false-upgrade set; attribution mechanism unidentifiable.
- `D_DECISION.md` / `.json`: four named gates, numerical gaps and protocol limitations.
- `MECHANISM_REVERSE_ENGINEERING.md`: code-based explanations, without new ideas.
- `ORIGINAL_EXECUTION_FAILURES.md` / `PROTOCOL_DEVIATIONS.json`: unrepaired protocol breach and access-audit limits.

`phase9_formal.py` is a retired failed implementation and its entry point is disabled. Do not rerun D0/D3 or open the old formal sources. Corrected programs consume only `RECOVERED_LOADED_EVIDENCE.json`, extracted from the already running process before termination. The read-only memory salvage program is retained for provenance, not intended to run after that process has exited.

Tests (CPU only, no original evidence reads):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -B -m pytest -q -p no:cacheprovider experiments/r16p19_phase9/test_phase9_d3.py experiments/r16p19_phase9/test_phase9_baserate.py experiments/r16p19_phase9/test_phase9_cached_eval.py experiments/r16p19_phase9/test_phase9_ledger_cache.py
sha256sum -c experiments/r16p19_phase9/SHA256SUMS
```

The one-sided Clopper–Pearson episode FPR is probability of any false acceptance among negative observations in an episode, not a mean frame FPR. The conformal marginal guarantee and a binomial bound computed after adaptive calibration selection are distinct. The latter is not an independent validation guarantee. Probability-space Platt outputs can saturate numerically; diagnostic ceiling decisions use monotone logit coordinates.

No GPU/PAI/rollout/data collection/verifier retraining/S2 occurred. Phase5–8 were not modified. This is a stage execution record, not a Phase-9 final report.

# Phase-5 artifact archive

This directory archives the complete compact result bundle for the bounded ASCEL embodied bridge validation.

- `results/`: all formal matrices, analyses, learned-verifier checkpoint and metrics, 240 GIFs, reports, decision, and the 283-entry result checksum manifest.
- `rollout_summaries/`: all 525 trajectory JSON summaries, eight policy-server contracts, eight worker completion markers, and the first-work marker.
- `raw_rollout_sha256.txt`: SHA256 identities for the 525 raw NPZ trajectories.
- `logs/pai_tests.log`: test output from the successful PAI worker.
- `pai_execution.json`: sanitized job and recovery provenance; no credentials are included.

The raw NPZ trajectories total about 609 MiB and remain at the immutable CPFS location recorded in `pai_execution.json`. They are not duplicated into Git because the compact summaries, per-trajectory SHA256 identities, formal matrices, and visual evidence are sufficient to audit this repository without inflating it by hundreds of MiB.

Verify the compact formal bundle with:

```bash
cd experiments/r16p19_phase5/artifacts/results
sha256sum -c SHA256SUMS
```

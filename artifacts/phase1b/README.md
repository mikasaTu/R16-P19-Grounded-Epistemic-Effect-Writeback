# Phase-1B raw evidence bundle

This directory preserves the raw evidence behind
`experiments/r16p19_libero_phase1b/`. It deliberately includes failed and
successful execution attempts so that the terminal status can be audited
without relying on a best-run-only narrative.

## Contents

- `checkpoints/primary_shared/`: every complete primary checkpoint retained
  at steps 2,500, 5,000, 7,500, 10,000, 12,500, and 15,000.
- `checkpoints/fallback_per_effect/`: every complete checkpoint retained for
  all eight corrected per-effect fallback actors.
- `experiment/`: primary/fallback training manifests, selected actor
  identities, all 80 qualification rows, 38 qualification failure videos,
  runtime identities, W&B run receipts, terminal status, and the implementation
  fix record. Raw W&B caches are intentionally excluded; immutable run IDs and
  URLs are retained.
- `pai/control-plane/`: redacted PAI submission/readback evidence for the four
  successful training/qualification jobs and the local fail-closed preflight.
- `pai/runtime/`: launch contracts, full run logs, completion sentinels, GPU-1
  idle telemetry, and simulator cache evidence written by the successful jobs.
- `failed/dlc9iq5myqo4j49t/`: the complete ineligible first fallback attempt,
  including its five partial actors, PAI evidence, logs, and failure receipt.
- `local-validation/`: pytest JUnit XML, CPU smoke records, and the bounded
  development GPU simulator smoke result.
- `SHA256SUMS`: checksums for all 466 raw evidence files in this directory.

The first fallback attempt failed because one effect contained only positive
gripper targets and the normalization constructor required both classes. The
corrected run uses ordinary unweighted BCE for a single-class target, creates no
synthetic samples, and reruns all eight actors from step zero in an isolated
`v2` namespace. The partial failed-run actors are retained but are ineligible
for selection or qualification.

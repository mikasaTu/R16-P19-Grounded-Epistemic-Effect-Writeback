# R16-P19 Phase-2 final decision

## Terminal status

`BLOCKED_BY_EXECUTOR_V3`

The frozen `RetargetedGeometricSkillExecutor` failed the preregistered clean
qualification on LIBERO init 60--79. The PAI workload completed normally; this
is a scientific gate failure, not an infrastructure failure.

| Frozen gate | Requirement | Observed | Pass |
|---|---:|---:|---|
| Minimum per-effect success | >= 0.90 | 0.70 | No |
| Full success, stove_moka | >= 0.80 | 0.80 | Yes |
| Full success, bowl_drawer | >= 0.80 | 0.70 | No |
| Repeated-effect-loop rate | <= 0.10 | 0.25 | No |

Overall full-task success was 30/40 (0.75), with ten failures and ten retained
failure videos. All ten failures exhausted the retry budget and became repeated
loops.

## Stop-rule consequences

- Formal init 0--19: `NOT_RUN_QUALIFICATION_FAILED` (zero rollouts).
- Formal executor gate: not activated.
- C0/C1/C2/C3/C7 x B2/B3/B5/B6 matrix: not activated (0/800).
- First-divergence causal replay: not activated.
- 10,000-repetition paired bootstrap and McNemar tests: not activated.
- No post-qualification tuning and no alternate executor were attempted.

The Phase-2 run therefore does not identify the behavioral effect of B6 versus
B3/B5. It neither passes nor rejects the core memory mechanism.

## Frozen identities and runtime

- Qualification source commit: `8963f8cb3b10201095a47c48cec13ce11b0832f0`
- Qualification source tree: `92a38b5ec8ecedd91f8f14d1432e8f787916cc17`
- Executor implementation source commit: `136e8923829c0436ca27755078a609f91bcf75a5`
- Selected executor manifest SHA-256: `384957cae10f96b6a53645a555e53d38876573a06a452fc93af0d95b4f254b6b`
- Selected template manifest SHA-256: `6a123a334bb901da880baf3b14e72576015bc013ed1f3224e9dd2c6d3c49d431`
- Official LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- PAI qualification JobId: `dlceyy7m2jhmxc4o`
- PAI run ID: `r16p19-p2-qualification-20260813-194500`
- Runtime: one RTX 4090, one worker, UID/GID 2254:2254, no AIMaster,
  no automatic fault tolerance, zero platform restarts.

The resource differs from the optional A800 only in GPU model: the workload is
non-neural and used a verified single-GPU rendering carrier. A second idle GPU
was not reserved, as required by the preregistration.

## Next-phase readiness

`LEARNED_EFFECT_VERIFIER_BLOCKED`

No next phase was started automatically.

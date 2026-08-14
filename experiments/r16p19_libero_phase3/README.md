# R16-P19 Phase-3 Effect-Boundary Replay validation

This directory freezes the Effect-Boundary Replay Causal Validation before any
Phase-3 access to formal demonstrations 40--49. The base repository state is
`981ad1a64936b4e970e1f934be2d354497b5fc8e`; the protected B6 implementation
hash is `4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5`.

The backend resets an exact same-demonstration effect-entry simulator snapshot
and executes that demonstration's frozen action segment with normal
`env.step`. It is intentionally not an actor or general-purpose executor.
Simulator truth is available only to the shared broker and evaluation labels,
never directly to a memory arm.

The user explicitly overrode the plan's early-stop clauses before formal
access. Every gate remains frozen and is reported, but the full matrix, D1,
first-divergence replay, statistics, and both B6 mechanism ablations will run
even if replay or primary gates fail. Such downstream results are diagnostic
and cannot turn a failed replay gate into a Phase-3 pass.

Splits are demo 0--19 development, 20--29 boundary/action calibration, 30--39
qualification, and 40--49 formal confirmation. Simulator init 0--19 remain
reserved and forbidden. The formal bank was used by earlier trace-level work,
so this phase is controlled confirmation rather than independent external
benchmark validation.

Feishu planning nodes:

- step4: https://icnbwz7kd1ui.feishu.cn/wiki/SqxXwnTIFiistlk1bOrcprPWnmb
- experiment report: https://icnbwz7kd1ui.feishu.cn/wiki/DvI3wxPQ7ixoSdkA9kncspgdnUc

## Terminal result

The full downstream matrix and paired analysis are finished. The frozen
machine-readable terminal status is `BLOCKED_BY_IMPLEMENTATION`, with
independent qualification and formal replay-only gate failures. B6 and the
strongest `TYPED_MATCHED_RECOVERY` baseline tied exactly on the 92
backend-valid primary paired units: success `0.913043` versus `0.913043`,
paired difference `0`, clustered 95% interval `[0, 0]`, and McNemar `p=1`.
The first-divergence causal win rate was `54/88 = 0.613636`, below the frozen
0.70 threshold. In addition, 27/150 paired units failed the byte-identical
pre-decision simulator/event-prefix requirement.

The result therefore does not validate an incremental B6 advantage. The
mechanism diagnostics show that invalidation is necessary for B6's C3
recovery, but command-parent provenance has no observed incremental success
effect under the frozen C4/C7 interventions.

The video policy enumerated all 748 required/available requests. It produced
496 videos and 252 deterministic invalid-snapshot errors; four of the rendered
replays changed a frozen failure into success. These videos remain diagnostic
and do not replace the frozen main-matrix outcomes.

Read:

- [`EXPERIMENT_REPORT_ZH.md`](EXPERIMENT_REPORT_ZH.md) for the complete report;
- [`MECHANISM_REVERSE_ENGINEERING.md`](MECHANISM_REVERSE_ENGINEERING.md) for the
  code-path explanation;
- [`PAI_EXECUTION_AUDIT.md`](PAI_EXECUTION_AUDIT.md) for the immutable PAI job
  and evidence boundary;
- [`LEARNED_EFFECT_VERIFIER_READINESS.md`](LEARNED_EFFECT_VERIFIER_READINESS.md)
  for the frozen `NOT READY` decision (no model was launched);
- [`VALIDATION.md`](VALIDATION.md) for test results, exact row counts, hashes,
  and structural checks;
- `final_decision.json`, `behavior_summary.json`, `cluster_bootstrap.json`, and
  `paired_tests.json` for machine-readable conclusions.

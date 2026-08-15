# R16-P19 Phase-4: Attempt-Scoped Causal Effect Ledger

This directory freezes the Phase-4 confirmatory design before ASCEL method
implementation. The scientific target is narrower than the original B6: separate
physical effect truth, active-attempt attribution, and live/discharged support proofs.

The previous evidence boundary remains unchanged. Phase-3 ended
`BLOCKED_BY_IMPLEMENTATION`; 27/150 paired units differed before the first memory
decision, and backend-valid B6 tied the strongest typed matched-recovery baseline at
0.913043 with paired difference 0 and 95% CI [0, 0]. Nothing in Phase-4 overwrites or
reinterprets that result.

The original `r16p19/memory.py` is protected by SHA256 and remains the frozen M1
comparison arm. ASCEL is implemented only in new `phase4_*` modules.

All scientific execution is Linux CPU-only. The latest user instruction requires the
complete planned matrix even if an intermediate gate fails. Such continuation is
diagnostic only: a failed mechanism, microenvironment, or shared-prefix gate still
blocks a PASS status and cannot be repaired by threshold changes or downstream results.

The frozen files in this commit are:

- `preregistration.yaml`
- `current_evidence_manifest.json`
- `method_contract.yaml`
- `attempt_schema.json`
- `support_graph_schema.json`
- `task_condition_contract.json`
- `metric_contract.json`
- `statistical_analysis_plan.yaml`

No learned actor, verifier, VLA, world model, DINO-WM, Pi0.5, Mem-0, RMBench, or
external benchmark is part of this phase.

## Completed result

The complete frozen matrix ran on CPU from source commit `1e30ff4055ab2681abbc376cfeaf9272fb22f442`:

- trace gate: 10,000/10,000 schedules passed;
- executor: conditional effect success 1.0, full-chain success 1.0, zero backend errors;
- true shared-prefix qualification: 1,000/1,000 paired units passed, zero child failures;
- pilot: 1,000 rollouts, all authorization gates passed;
- formal: 5,000 rollouts and 1,000 paired-unit audits completed;
- mechanism ablations: 4,000 rollouts and 1,000 paired-unit audits completed.

ASCEL M4 reached 1.0 on clean, A1--A4, S1--S4, and A5 physical-truth recognition,
with zero A5 false skill credit. Relative to the strongest frozen baseline, the
attempt-family and support-family absolute gains were both +0.50; their clustered
95% CIs were `[0.50, 0.50]` and `[0.45, 0.548223]`. All three mechanism components
passed. The overall result is nevertheless `BLOCKED_BY_IMPLEMENTATION`: C0 event
processing was 816,234 ns for M4 versus 395,884 ns for M0 (+106.18%), exceeding the
frozen +10% clean-overhead gate. Clean success and physical action-step overhead did
not degrade.

See [FINAL_DECISION.md](FINAL_DECISION.md) for the complete result and
[MECHANISM_REVERSE_ENGINEERING.md](MECHANISM_REVERSE_ENGINEERING.md) for the
code-and-ablation mechanism explanation. Reproduce all stages with:

```bash
/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python \
  scripts/run_phase4_pipeline.py all
```

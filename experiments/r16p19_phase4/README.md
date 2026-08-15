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

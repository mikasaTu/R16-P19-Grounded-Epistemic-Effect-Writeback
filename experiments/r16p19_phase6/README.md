# R16-P19 Phase-6 / step7 — S1 offline recalibration and decision replay

This directory contains the CPU-only S1 analysis. Phase-5 is a frozen,
read-only substrate. No GPU/PAI job, simulator rollout, or S2 execution is
permitted by this step.

Run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python experiments/r16p19_phase6/run_s1.py \
  --repo . \
  --raw-root /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/pai/r16p19-phase5-idle-fixed-20260818-0855/rollouts \
  --output experiments/r16p19_phase6
```

The terminal S1 artifact is `S1_DECISION.md`. It is not a Phase-6 final
report. `S1_DISAGREEMENT_UNITS.txt` is the frozen S2 re-execution candidate
set; S2 remains unstarted pending human confirmation.

# R16-P19 LIBERO Phase-1B

Phase-1B upgrades only the low-level actor and removes actor–memory input
confounding. It does not change the R16-P19 memory state machine, ontology,
fault definitions, dataset/demo split, formal initial states, behavior metrics,
or pass/fail thresholds.

The preregistration parent is commit
`80825fc10ba0e57bb14edb3c54837ce95d8eabf1`. The actor candidates and gates in
this directory are frozen before the Phase-1B data audit, implementation, or
training begins.

Execution order is fail closed:

1. audit the frozen data and init splits;
2. implement the memory-independent actor contract and regression tests;
3. train and qualify the shared primary actor on init 20–39;
4. train the single preregistered fallback only if primary qualification fails;
5. freeze source, normalization, and checkpoint hashes;
6. evaluate formal init 0–19 exactly once;
7. run the 800-rollout matrix only if formal minimum per-effect success is at
   least 0.80.

Allowed final statuses are `PASS_PHASE1_BEHAVIOR`, `REJECT_CORE_MECHANISM`,
`BLOCKED_BY_ACTOR_V2`, and `BLOCKED_BY_IMPLEMENTATION`.

## Implementation entry point

`phase1b_pipeline.py` is the only Phase-1B stage entry point. The stages are
deliberately separate so the fallback and 800-rollout matrix remain fail
closed:

```text
static-check
cpu-smoke
gpu-sim-smoke                 # bounded untrained shape/EGL/video smoke only
train-primary
qualify --family primary
train-fallback                  # primary qualification failure only
qualify --family fallback       # primary qualification failure only
formal-gate                     # frozen actor only, init 0--19 once
closed-loop                     # formal actor gate pass only
```

The low-level contract lives in `r16p19/phase1b_actor.py`. Its public method is
exactly `SkillActor.action_chunk(state_history, task_id, effect_id,
execution_mode)`, where the mode is only `EXECUTE` or `RETRY`. The module does
not import memory state. `r16p19/phase1b_closed_loop.py` separately maps frozen
memory decisions to actor behavior and fails if actor inputs or action bytes
diverge before the first paired memory-decision divergence.

## Terminal result

Phase-1B terminated as `BLOCKED_BY_ACTOR_V2`.

- The shared primary actor completed 40 qualification rollouts on init 20--39,
  with 18/40 full-task success and minimum per-effect success 0.40.
- The only preregistered per-effect fallback completed a separate 40-rollout
  qualification matrix on the same frozen init range, with 24/40 full-task
  success and minimum per-effect success 0.50.
- Both minima are below the frozen 0.80 threshold. No actor was frozen.
- Formal init 0--19 received zero actor rollouts. The 800-rollout
  memory-conditioned matrix and 10,000-repetition paired bootstrap were not
  authorized and remain `NOT_RUN`.

The machine-readable result is in `behavior_summary.json`; the exact gate
decision is in `FINAL_DECISION.md`. `SHA256SUMS` covers every final
deliverable in this directory.

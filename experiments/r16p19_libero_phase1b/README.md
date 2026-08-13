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

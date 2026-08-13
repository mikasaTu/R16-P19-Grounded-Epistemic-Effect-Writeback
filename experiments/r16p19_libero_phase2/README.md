# R16-P19 LIBERO Phase-2 preregistration

Phase-2 tests the frozen R16-P19 memory mechanism under exactly one competent,
deterministic, non-neural, memory-independent executor. The entry status is
`BLOCKED_BY_ACTOR_V2`; this phase is not another learned-actor search.

## Terminal outcome

The frozen executor completed qualification but failed the preregistered gates:
minimum per-effect success 0.70 (<0.90), per-task full success 0.80/0.70
(both required >=0.80), and repeated-loop rate 0.25 (>0.10). The exact terminal
status is `BLOCKED_BY_EXECUTOR_V3`. Formal init 0--19, the 800-cell matrix,
causal replay, and bootstrap were not run. See `FINAL_DECISION.md`,
`behavior_summary.json`, and `failure_cases.md`.

This directory was committed before template extraction, executor
implementation, access to init 40--79, or any Phase-2 access to formal init
0--19. The six files here freeze the executor family and input boundary, demo
and init splits, qualification/formal gates, fault timing, causal-replay
contract, and resource policy.

## Fail-closed order

1. Extract no more than three local-frame templates per effect from successful
   demo 0--29 segments and calibrate only with demo 30--39/init 40--59.
2. Seal source, template, controller, retry, seed, and schema hashes.
3. Evaluate the frozen executor once on init 60--79. If any qualification gate
   fails, stop as `BLOCKED_BY_EXECUTOR_V3`; do not try another executor.
4. Only after qualification passes, run the clean formal competence gate once
   on init 0--19. A minimum per-effect success below 0.80 terminates as
   `BLOCKED_BY_EXECUTOR_V3` and forbids the matrix.
5. Only after the formal competence gate passes, run the frozen 800-cell
   C0/C1/C2/C3/C7 × B2/B3/B5/B6 matrix, first-divergence causal replay,
   10,000-repetition paired bootstrap, McNemar tests, and mediation analysis.

## Scientific boundary

`r16p19/memory.py`, the effect ontology, receipt/provenance semantics, fault
meanings, arm definitions, original behavior gates, formal init set, and
bootstrap rule are immutable. The executor receives geometry needed for the
current effect but never receives memory state, fault identity, simulator
effect truth, reward, task success, init index, future state, or the next
effect.

## Compute boundary

The executor has no training checkpoint. Evaluation resumes by atomic
completed-rollout markers. Local work is limited to CPU validation or a bounded
GPU smoke. PAI may use at most one A800 only when headless rendering requires
it; reserving a second idle GPU for this non-neural experiment is forbidden.

Allowed terminal statuses are `PASS_PHASE2_ORACLE_BEHAVIOR`,
`REJECT_CORE_MECHANISM`, `BLOCKED_BY_EXECUTOR_V3`, and
`BLOCKED_BY_IMPLEMENTATION`.

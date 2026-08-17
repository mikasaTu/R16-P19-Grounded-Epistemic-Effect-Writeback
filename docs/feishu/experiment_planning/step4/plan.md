# step4

Task:
R16-P19 Phase-3 Effect-Boundary Replay Causal Validation

Repository:
mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback

Current terminal status:
BLOCKED_BY_EXECUTOR_V3

Current evidence boundary:

- Phase-1 actor-free trace gate passed.
- Phase-1B learned actors failed competence gates.
- Phase-2 RetargetedGeometricSkillExecutor reached 30/40 clean full-task
successes, minimum reported effect reach 0.70, and repeated-loop rate 0.25.
- Formal init 0–19 remain unobserved.
- The 800-rollout B2/B3/B5/B6 matrix, causal replay, and bootstrap remain NOT_RUN.
- Therefore B6 has not yet been behaviorally validated or rejected.

Scientific objective:
Perform the first controlled behavior-level causal test of the original
R16-P19 Grounded Epistemic Effect Writeback mechanism.

The experiment must isolate memory decisions from general low-level policy
competence by using frozen effect-boundary simulator snapshots and exact
successful demonstration action segments.

This phase does not train a new actor, VLA, world model, or effect verifier.

==================================================

1. HARD SCOPE BOUNDARIES
==================================================

Do not modify the semantics or bytes of the existing B6 implementation in:

r16p19/memory.py

Do not modify:

- REQUESTED / IMAGINED / OBSERVED / VERIFIED / REALIZED /
STALLED / INVALIDATED_REALIZATION;
- the existing effect ontology;
- evidence receipt and provenance rules;
- resident-memory capacity;
- existing Phase-1 or Phase-2 artifacts;
- formal init 0–19.

Do not:

- train another MLP, ACT, BC, diffusion policy, Pi0.5, Mem-0, or world model;
- implement another general-purpose geometric executor;
- retarget action templates;
- tune waypoint gains;
- use formal init 0–19;
- claim VLA improvement, benchmark generalization, paper acceptance, or N3
validation from this phase;
- build formal-activation, encryption, inode, WORM, or large mutation
infrastructure;
- change thresholds after seeing formal results.

Use ordinary Git commits, JSON/JSONL, SHA256, unit tests, and videos.

Use CPU simulation where possible.
If EGL rendering requires a GPU, use at most one rendering GPU.

# ==================================================
2. CREATE AND FREEZE PHASE-3 PREREGISTRATION

Create an isolated branch:

phase3-effect-boundary-replay

Before accessing formal demo 40–49, create and commit:

experiments/r16p19_libero_phase3/
preregistration.yaml
data_split_manifest.json
candidate_chain_contract.json
replay_backend_contract.yaml
baseline_contracts.yaml
fault_contracts.yaml
metric_contract.json
statistical_analysis_plan.yaml
README.md

Freeze:

Development demonstrations:

- demo 0–19

Boundary/action calibration:

- demo 20–29

Replay qualification:

- demo 30–39

Formal effect-boundary evaluation:

- demo 40–49

Reserved and untouched:

- simulator init 0–19

The formal demo bank has previously been used by trace-level work.
Therefore this phase is controlled confirmation, not independent external
benchmark validation.

# ==================================================
3. BUILD A REAL EFFECT-BOUNDARY SNAPSHOT BANK

Implement:

r16p19/phase3_snapshot_bank.py

For every official demonstration and every effect:

1. Reset the official LIBERO environment to the demonstration trajectory.
2. Independently recompute effect predicates from the simulator.
3. Treat existing demo_effect_labels.jsonl only as a candidate boundary index,
never as unquestioned ground truth.
4. Find:

   - entry state before the effect begins;
   - first action index belonging to the effect;
   - first state where the effect becomes true;
   - first point where it remains true for five consecutive simulator steps;
   - exact action segment;
   - stable post-effect state;
   - next-effect entry state.
5. Persist:

   - full simulator entry state;
   - action segment as float32;
   - source episode and frame indices;
   - observation hashes;
   - effect-truth timeline;
   - precondition truth;
   - stable completion truth;
   - source dataset and LIBERO commit;
   - all SHA256 values.

Use normal env.step execution.
Do not set success or object state directly during segment extraction.

Candidate chains:

1. STOVE_TURNED_ON -> MOKA_GRASPED
2. MOKA_GRASPED -> MOKA_ON_STOVE
3. MOKA_ON_STOVE -> MOKA_RELEASED_ON_STOVE
4. BOWL_GRASPED -> BOWL_IN_BOTTOM_DRAWER
5. BOWL_IN_BOTTOM_DRAWER -> BOWL_RELEASED_IN_DRAWER
6. BOWL_RELEASED_IN_DRAWER -> BOTTOM_DRAWER_CLOSED

# ==================================================
4. IMPLEMENT A FROZEN EFFECT REPLAY BACKEND

Implement:

r16p19/phase3_replay_backend.py

Name:

FrozenEffectReplayBackend

This backend is not a general robot policy.

Its only action operation is:

- reset to a frozen effect-entry simulator snapshot;
- execute the frozen action segment from the same source demonstration through
normal env.step;
- report observations and simulator effect truth to the event broker.

Interface:

execute_effect(
task_key,
source_episode,
effect_id,
entry_snapshot_id,
execution_mode
)

Allowed execution modes:

- EXECUTE
- RETRY
- ROLLBACK_REPLAY
- REOBSERVE

Decision mapping:

ADVANCE_TO_NEXT_SUBTASK:

- allowed only when physical current-effect truth is true;
- if false, mark premature advance and terminate the chain as failure;
- if true, move to the frozen next-effect entry snapshot.

RETRY_CURRENT_EFFECT:

- replay the current frozen action segment.

REOBSERVE:

- execute exactly eight zero-action simulator steps;
- deliver new observations and receipts.

ROLLBACK_OR_REPLAN:

- restore the current effect-entry snapshot;
- replay the current frozen action segment.

SAFE_STOP:

- terminate the chain as failure.

All memory arms must use the same backend, states, actions, reset contract,
budgets, and fault schedule.

The backend must not import:

- r16p19.memory;
- epistemic states;
- fault identity;
- memory decision history.

# ==================================================
5. REPLAY QUALIFICATION

Use demo 30–39 only.

For each candidate effect segment, run at least five deterministic replays from
its exact entry snapshot.

Report separately:

- conditional_effect_success_given_entry;
- cumulative_chain_reach;
- chain_success;
- action steps;
- predicate-stability duration.

Selection rule, frozen before formal access:

- every segment in a selected chain:
conditional replay success >= 0.95;
- selected chain success >= 0.90;
- at least three chains must pass;
- both task families must be represented;
- each selected chain must have at least eight valid demo 40–49 units.

If fewer than three chains pass:

FINAL_STATUS = BLOCKED_BY_REPLAY_BACKEND

Do not build a replacement executor.

Freeze:

- selected chain list;
- snapshot/action hashes;
- replay implementation;
- budgets;
- all parameters.

Then access demo 40–49 exactly once for a formal replay-only gate.

Formal replay-only gate:

- each selected segment conditional success >= 0.90;
- each selected chain success >= 0.85.

Failure:

FINAL_STATUS = BLOCKED_BY_REPLAY_BACKEND

Do not modify chain selection after formal access.

# ==================================================
6. IMPLEMENT SIX MEMORY ARMS

Reuse the current B2 and B3 implementations unchanged.

Implement new strong baselines in:

r16p19/phase3_baselines.py

Arms:

B2_COMMAND_PROGRESS

- existing B2 semantics.

B3_MONOLITHIC

- existing B3 semantics.

POSTCHECK_RECOVERY

- state only UNKNOWN / TRUE / FALSE;
- one current positive postcondition receipt marks TRUE;
- negative or contradiction marks FALSE;
- no REQUESTED/IMAGINED/VERIFIED/REALIZED distinction;
- receives the same retry, reobserve, and rollback operations as B6.

PERSISTENCE_RECOVERY

- same binary state;
- requires K consecutive positive decision ticks before TRUE;
- K selected only on demo 30–39 from {2, 4, 8};
- same recovery operations and budgets as B6.

TYPED_MATCHED_RECOVERY

- typed REQUESTED / IMAGINED / OBSERVED / VERIFIED / REALIZED states;
- two-source verification;
- no command-parent provenance requirement;
- no invalidated-realization lineage;
- generic contradiction -> rollback;
- same recovery operations and budgets as B6.

B6_FULL

- existing unmodified R16-P19 implementation.

All strong arms:

POSTCHECK_RECOVERY
PERSISTENCE_RECOVERY
TYPED_MATCHED_RECOVERY
B6_FULL

must have exactly the same:

- retry limit;
- reobserve limit;
- rollback permission;
- replay backend;
- action budget;
- decision budget;
- fault schedule.

# ==================================================
7. EVENT BROKER AND INFORMATION FAIRNESS

Implement one shared event broker.

Simulator physical truth may only:

- generate standardized receipts;
- determine evaluation labels;
- implement declared faults.

It may not be passed directly into any memory arm.

Before the first memory-decision divergence, every arm must receive
byte-identical events.

Persist:

- event bytes;
- event-stream hash;
- receipt sensor identity;
- frame digest;
- parent IDs;
- effect ID;
- decision index.

# ==================================================
8. FORMAL FAULT CONDITIONS

C0 CLEAN

Normal replay and normal receipts.

C1 COMMAND_NOOP

- first attempt only;
- command is emitted;
- all actions of the first attempt become zero actions;
- physical effect remains false;
- the fault is removed after a legitimate retry.

C3 POST_REALIZATION_REVERSAL

- permit effect execution and positive verification;
- after realization, reverse the physical effect;
- send a contradiction receipt;
- require recovery before advancing.

C4 SINGLE_VIEW_FALSE_POSITIVE

- agentview supplies a positive receipt;
- wrist view does not confirm it;
- physical effect remains false;
- do not emit a valid realization witness.

C7 IMAGINED_SUCCESS_OBSERVED_FAILURE

- emit the same high-confidence imagined-success event to every arm;
- suppress the first physical attempt;
- later emit negative observed evidence.

Secondary condition:

D1 DELAYED_RECEIPT

- physical effect happens normally;
- positive receipts are withheld for three decision cycles;
- memory decisions during those cycles must be physically executed;
- this is a cost/calibration diagnostic and is not part of the main success
gate.

All faults must be tied to effect attempt index and physical events, not
absolute wall-clock time after arms diverge.

# ==================================================
9. RUN THE FORMAL MATRIX

For N selected chains, run:

N chains
x 10 formal source episodes
x 5 main conditions
x 6 arms

Expected:

- three chains: 900 rollouts;
- four chains: 1,200 rollouts.

Run D1 separately:

N chains
x 10 episodes
x 1 condition
x 6 arms.

Every paired unit must share:

- source episode;
- chain;
- simulator entry snapshot;
- action segment;
- target effect;
- fault schedule;
- event broker;
- reset policy;
- retry budget;
- reobserve duration.

Persist every rollout immediately in resumable JSONL.

Save videos for:

- every failure;
- every first-decision divergence;
- at least five representative successful recoveries per condition and arm.

# ==================================================
10. FIRST-DIVERGENCE CAUSAL REPLAY

For each paired unit:

1. Find the first high-level decision divergence among:
B3_MONOLITHIC,
POSTCHECK_RECOVERY,
PERSISTENCE_RECOVERY,
TYPED_MATCHED_RECOVERY,
B6_FULL.
2. Save the exact simulator snapshot and event prefix.
3. From that same snapshot, replay every unique decision with:

   - the same backend;
   - the same fault state;
   - the same action budget.
4. Measure:

   - immediate effect completion;
   - eventual chain completion;
   - irreversible failure;
   - extra steps;
   - recovery cost.

Compute decision_causal_win_rate for B6.

# ==================================================
11. REQUIRED METRICS

Backend metrics:

- conditional_effect_success_given_entry
- cumulative_chain_reach
- replay_chain_success
- action_steps_per_effect

Behavior metrics:

- chain_success
- current_effect_success
- next_effect_success
- faulted_chain_success
- clean_chain_success
- recovery_success
- repeated_loop_rate

Memory metrics:

- grounded_advance_precision
- grounded_advance_recall
- false_completion_rate
- premature_advance_rate
- contradiction_detection_recall
- contradiction_recovery_recall
- single_view_false_positive_advance_rate
- imagined_as_realized_rate
- invalidated_realization_accuracy

Cost metrics:

- retry_count
- reobserve_count
- rollback_count
- unnecessary_retry_rate
- unnecessary_recovery_rate
- safe_stop_rate
- action_steps
- completion_latency

Causal metrics:

- decision_causal_win_rate
- effect_completion_after_first_divergence
- chain_completion_after_first_divergence

Failure decomposition:

- MEMORY_DECISION_ERROR
- REPLAY_BACKEND_FAILURE
- RECEIPT_BROKER_ERROR
- FAULT_INJECTOR_ERROR
- TIMEOUT
- PREMATURE_ADVANCE
- OVERCONSERVATIVE_STOP

# ==================================================
12. STATISTICAL ANALYSIS

Primary endpoint:

Mean chain success over C1, C3, C4, and C7.

Primary comparison:

B6_FULL versus the best of:

- POSTCHECK_RECOVERY
- PERSISTENCE_RECOVERY
- TYPED_MATCHED_RECOVERY

Use source demonstration episode as the bootstrap cluster.

Run:

- 10,000-repetition cluster bootstrap;
- paired absolute success differences;
- exact paired McNemar tests;
- Holm correction for B6 versus the three strong baselines.

Report per-condition and per-chain effect sizes.

Do not treat multiple chains from one source episode as independent samples.

# ==================================================
13. FROZEN PHASE-3 GATES

PASS_PHASE3_STRONG_CONTROLLED requires all:

1. B6 faulted chain success exceeds the best strong baseline by at least 0.10
absolute.
2. Cluster-bootstrap 95% CI lower bound for B6 minus best strong baseline is
greater than zero.
3. grounded_advance_precision >= 0.95.
4. C3 contradiction_recovery_recall >= 0.80.
5. C4 false-positive advance rate <= 0.05.
6. Clean success degradation versus the best strong baseline <= 0.02.
7. Clean action-step overhead versus the best strong baseline <= 15%.
8. decision_causal_win_rate >= 0.70.
9. All provenance and resident-memory correctness gates pass.

WEAK_BASELINE_ONLY if:

- B6 clearly outperforms B2 and B3;
- but does not outperform POSTCHECK_RECOVERY, PERSISTENCE_RECOVERY, or
TYPED_MATCHED_RECOVERY.

REJECT_PHASE3_INCREMENTAL_VALUE if:

- replay backend gates pass;
- formal matrix completes;
- B6 does not improve over strong baselines or is excessively conservative.

BLOCKED_BY_REPLAY_BACKEND if:

- qualification or formal replay-only competence gates fail.

BLOCKED_BY_IMPLEMENTATION if:

- code or simulator faults prevent a valid matrix.

# ==================================================
14. POST-PASS MECHANISM ABLATIONS

Run only if B6 passes the primary behavioral gate.

Implement separate ablation wrappers without editing the frozen B6 source:

B6_NO_PROVENANCE

- remove command/receipt parent-link requirement.

Run on C4 and C7 only.

B6_NO_INVALIDATION

- contradiction cannot invalidate an existing realization.

Run on C3 only.

A paper-level mechanism claim requires the relevant B6 advantage to decrease
by at least 50% under at least one corresponding ablation.

Do not use ablation results to retune B6.

# ==================================================
15. DELIVERABLES

Code:

r16p19/phase3_snapshot_bank.py
r16p19/phase3_replay_backend.py
r16p19/phase3_baselines.py
r16p19/phase3_event_broker.py
r16p19/phase3_runner.py
r16p19/phase3_analysis.py

Tests:

tests/test_phase3_snapshot_bank.py
tests/test_phase3_replay_backend.py
tests/test_phase3_baselines.py
tests/test_phase3_pairing.py
tests/test_phase3_metrics.py

Artifacts:

experiments/r16p19_libero_phase3/
preregistration.yaml
data_split_manifest.json
candidate_chain_contract.json
selected_chain_manifest.json
snapshot_bank_manifest.json
replay_qualification_results.jsonl
replay_qualification_summary.json
formal_replay_gate.json
baseline_contracts.yaml
fault_contracts.yaml
formal_results.jsonl
delayed_receipt_results.jsonl
paired_unit_audit.jsonl
first_divergence_replays.jsonl
behavior_summary.json
cluster_bootstrap.json
paired_tests.json
mechanism_ablations.jsonl
failure_decomposition.jsonl
failure_cases.md
FINAL_DECISION.md
SHA256SUMS

Update the repository root README with:

- Phase-1 trace-level PASS;
- Phase-1B BLOCKED_BY_ACTOR;
- Phase-2 BLOCKED_BY_EXECUTOR_V3;
- Phase-3 exact status and evidence boundary.

# ==================================================
16. EXECUTION ORDER

1. Read and audit the current repository state.
2. Create Phase-3 preregistration and commit it.
3. Build the snapshot bank from development/calibration demonstrations.
4. Qualify exact replay on demo 30–39.
5. Freeze selected chains and backend.
6. Access demo 40–49 once for formal replay gate.
7. Stop immediately if replay gate fails.
8. Implement and test all six memory arms.
9. Run a 12-cell smoke matrix on non-formal data.
10. Run the complete formal matrix.
11. Run first-divergence causal replays.
12. Run cluster bootstrap and paired tests.
13. Make the primary Phase-3 decision.
14. Only after a primary pass, run the two mechanism ablations.
15. Produce a readiness note for a future learned effect verifier.
16. Do not start the learned verifier, Mem-0, ACT, or Pi0.5 automatically.

The next progress report must state:

- exact source HEAD;
- preregistration commit;
- number of extracted effect segments;
- qualification conditional success per effect;
- selected chains;
- whether demo 40–49 remain unopened;
- whether the formal replay gate passed;
- exact planned rollout count;
- no claim about B6 before the formal matrix completes.

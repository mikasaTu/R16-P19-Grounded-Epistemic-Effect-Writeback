# step5

Task:
R16-P19 Phase-4 Attempt-Scoped Causal Effect Ledger Validation

Repository:
mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback

Current repository status:
BLOCKED_BY_IMPLEMENTATION

Current evidence boundary:

- Phase-1 trace-level logic passed.
- Phase-1B and Phase-2 were blocked by learned and geometric actor competence.
- Phase-3 completed its downstream diagnostic matrix under a preregistered
override, but replay qualification and formal replay gates failed.
- 27/150 Phase-3 paired units differed in simulator/event prefix before the
first memory decision.
- B6_FULL tied TYPED_MATCHED_RECOVERY exactly on backend-valid primary units:
0.913043 versus 0.913043, paired difference 0, 95% CI [0,0].
- Therefore the current full B6 has no established incremental behavioral
value over the strongest typed matched-recovery baseline.
- No learned verifier, ACT, Mem-0, Pi0.5, VLA, or external benchmark validation
is authorized.

This phase must not merely repair the Phase-3 replay backend and rerun the same
fault matrix.

The scientific goal is to refine and test a narrower mechanism:

Attempt-Scoped Causal Effect Ledger, ASCEL

ASCEL must separate:

1. physical effect truth;
2. attribution to the current command attempt;
3. live support proofs between realized effects.

The phase must test failure modes that cannot be solved by ordinary typed
state plus generic rollback.

# ==================================================
0. HARD SCOPE BOUNDARIES

Create a new isolated branch:

phase4-attempt-scoped-causal-ledger

Do not modify the bytes or semantics of:

r16p19/memory.py

The original B6 must remain a frozen comparison arm.

Record and verify its current SHA256 before every formal run.

Do not:

- reuse Phase-3 formal demo 40–49 as unseen evidence;
- use simulator init 0–19;
- train another actor, BC, ACT, diffusion policy, VLA, world model, or verifier;
- use Pi0.5, Mem-0, DINO-WM, or RMBench in this phase;
- use the Phase-3 independent-replay-per-arm design;
- continue formal experiments after a shared-prefix or executor gate fails;
- use a user override to continue a failed confirmatory gate;
- change thresholds after pilot or formal results;
- claim N3, paper readiness, VLA improvement, or benchmark generalization;
- create formal-activation, encryption, inode mutation, or excessive
publication-security infrastructure.

Use normal Git commits, JSON/JSONL, SHA256, unit tests, and CPU simulation.

Do not use a GPU for the scientific experiment.
Rendering is optional and may be done after primary results.

==================================================

1. FREEZE PHASE-4 PREREGISTRATION
==================================================

Before implementing ASCEL, create:

experiments/r16p19_phase4/
preregistration.yaml
current_evidence_manifest.json
method_contract.yaml
attempt_schema.json
support_graph_schema.json
task_condition_contract.json
metric_contract.json
statistical_analysis_plan.yaml
README.md

Record:

- current repository HEAD and tree;
- protected r16p19/memory.py SHA256;
- current ontology SHA256;
- Phase-3 final decision and report hashes;
- exact method hypotheses;
- exact baselines;
- seed splits;
- pilot and formal gates;
- final status definitions.

Commit preregistration before method implementation.

# ==================================================
2. IMPLEMENT ASCEL IN SEPARATE MODULES

Create:

r16p19/phase4_types.py
r16p19/phase4_attempt_ledger.py
r16p19/phase4_support_graph.py

Do not retrofit the frozen Phase-1 EffectRecord.

Implement:

AttemptRecord:

- attempt_id
- effect_id
- generation
- command_event_id
- start_epoch
- end_epoch
- ACTIVE / SUCCEEDED / FAILED / SUPERSEDED / INVALIDATED
- attempt-scoped evidence IDs and digests
- attribution witness IDs

EffectFactRecord:

- effect_id
- fact state
- fact epoch
- active attempt ID
- verified fact observations
- realization proofs
- invalidation event
- current physical truth confidence

RealizationProof:

- proof_id
- effect_id
- fact_epoch
- attributed_attempt_id or null
- evidence IDs
- witness ID
- support clause ID
- validity status

Every REQUEST after the first must:

1. supersede the previous active attempt;
2. create a new deterministic attempt ID;
3. create a new evidence scope;
4. retain old evidence only in append-only historical storage;
5. prevent old evidence from participating in the new attempt.

Every command must have one command ID bound to one attempt.

Positive observations may update current physical fact truth only when their
observation epoch is current.

Attempt attribution requires:

- matching effect ID;
- matching active attempt ID;
- current command ID;
- valid observation epoch;
- non-superseded attempt.

A stale or mismatched receipt may be retained for audit but must not contribute
to current verification or attribution.

# ==================================================
3. SEPARATE EFFECT TRUTH FROM ATTEMPT CREDIT

An effect may be physically true without being caused by the active attempt.

Represent separately:

effect_fact_verified
attempt_attributed_success

Decision semantics:

- verified physical truth may permit task advance;
- attempt attribution controls skill credit and capability update;
- incidental or external realization must not credit the current skill;
- lack of attribution must not force a retry when the task effect is already
physically verified.

Persist both values in every decision record.

# ==================================================
4. IMPLEMENT PRE-REALIZATION REVOCATION

A contradiction or valid negative receipt must revoke the active evidence set
at any of:

OBSERVED
VERIFIED
REALIZED

For OBSERVED or VERIFIED:

- mark the active fact proof revoked or stalled;
- clear the active attempt verification set;
- close or invalidate the current attempt as appropriate;
- prevent any witness issued before the contradiction epoch from later
realizing the effect.

For REALIZED:

- invalidate the realization proof;
- invoke support-graph propagation.

A witness is accepted only if:

- its attempt ID is active;
- its command ID is current;
- its evidence epoch is after the latest revocation epoch;
- its required positive evidence belongs to the same attempt scope.

# ==================================================
5. IMPLEMENT A DISCHARGE-AWARE SUPPORT PROOF GRAPH

Do not use static transitive descendant invalidation as the primary mechanism.

Represent each effect using one or more support clauses.

An effect remains supported if at least one clause is valid.

Each clause is a conjunction of support references.

Support-reference validity types:

PERSISTENT

- parent realization must remain valid.

UNTIL_CHILD_REALIZED

- parent is required only until this child is realized.

UNTIL_EFFECT_REALIZED:<effect_id>

- parent support is discharged when the named effect is realized.

Support invalidation procedure:

1. invalidate the directly contradicted realization proof;
2. recompute all clauses that reference that proof;
3. invalidate a dependent only if all of its support clauses become invalid;
4. propagate recursively only through newly invalidated proofs;
5. do not invalidate unrelated branches;
6. do not invalidate effects whose support requirement was legitimately
discharged;
7. preserve effects with a valid alternative support clause.

Persist:

- proof IDs;
- support clauses;
- invalidation path;
- discharge event;
- remaining alternative proofs.

# ==================================================
6. IMPLEMENT FIVE PRIMARY ARMS

M0_TYPED_MATCHED

- reproduce the Phase-3 TYPED_MATCHED_RECOVERY semantics.

M1_B6_ORIGINAL

- adapter around frozen original B6.

M2_ATTEMPT_ONLY

- ASCEL attempt scoping and pre-realization revocation;
- use simple static dependency behavior.

M3_SUPPORT_ONLY

- discharge-aware support graph;
- no attempt-scoped evidence isolation.

M4_ASCEL_FULL

- attempt scoping;
- fact/attribution separation;
- pre-realization revocation;
- discharge-aware support graph.

All arms must have identical:

- retry limit;
- reobserve limit;
- rollback limit;
- action budget;
- event access;
- fault schedule;
- physical executor.

# ==================================================
7. DETERMINISTIC ADVERSARIAL TRACE GATE

Create:

r16p19/phase4_trace_generator.py
r16p19/phase4_trace_oracle.py

Generate at least 10,000 deterministic schedules covering:

A1 STALE_WITNESS_AFTER_RETRY

- attempt 1 fails;
- attempt 2 starts;
- a positive attempt-1 witness arrives during attempt 2.

A2 CROSS_ATTEMPT_SENSOR_MIX

- sensor A positive belongs to attempt 1;
- attempt 1 is superseded;
- sensor B positive belongs to attempt 2;
- the two receipts must not create verification.

A3 SUPERSEDED_COMMAND_WITNESS

- command A is superseded by command B;
- a witness parented by command A arrives during command B.

A4 VERIFIED_CONTRADICTION_LATE_WITNESS

- two current positives produce VERIFIED;
- contradiction arrives before realization witness;
- the older witness arrives later.

A5 INCIDENTAL_EFFECT

- the physical effect becomes true due to an external event;
- the active command did not cause it.

S1 LIVE_SUPPORT_REVERSAL

- a persistent parent is invalidated while a dependent proof remains live.

S2 DISCHARGED_SUPPORT_REVERSAL

- a support was legitimately discharged;
- later parent false evidence must not invalidate the child.

S3 ALTERNATIVE_SUPPORT

- one support clause fails while another remains valid.

S4 BRANCH_LOCAL_INVALIDATION

- one branch fails;
- an unrelated branch must remain valid.

S5 LATE_DEPENDENT_WITNESS

- parent support is invalidated;
- a previously generated child witness arrives afterward.

Trace gate requirements for M4:

- stale evidence accepted = 0;
- cross-attempt verification = 0;
- superseded witness realization = 0;
- post-revocation late witness realization = 0;
- cascade invalidation precision = 1.0;
- cascade invalidation recall = 1.0;
- over-invalidation = 0;
- under-invalidation = 0;
- incidental effect truth recognition = 1.0;
- incidental current-skill credit = 0.

If any trace gate fails:

FINAL_STATUS = BLOCKED_BY_MECHANISM_IMPLEMENTATION

Do not build the physical benchmark.

# ==================================================
8. BUILD CPU MUJOCO MICROBENCHMARKS

Create:

r16p19/phase4_microenv.py
r16p19/phase4_executor.py

Implement three short, deterministic physical tasks.

Task T1 CARRY_RELEASE:

GRASPED
-> LIFTED
-> OVER_TARGET
-> RELEASED_IN_TARGET

Support semantics:

- GRASPED persistently supports LIFTED and OVER_TARGET;
- GRASPED is discharged by RELEASED_IN_TARGET;
- a later intentional release must not invalidate RELEASED_IN_TARGET.

Task T2 PERSISTENT_SUPPORT:

SUPPORT_PRESENT
-> OBJECT_STABLE
-> MARKER_PLACED

Support semantics:

- SUPPORT_PRESENT remains persistent;
- removing support invalidates OBJECT_STABLE and MARKER_PLACED.

Task T3 ALTERNATIVE_SUPPORT:

LEFT_SUPPORT or RIGHT_SUPPORT
-> OBJECT_ELEVATED
-> TARGET_REACHED

Support semantics:

- either support clause is sufficient;
- removing only one support must not invalidate the dependent;
- removing both must invalidate it;
- unrelated task branches must remain valid.

The MacroSkillExecutor:

- receives only task ID, effect ID, and EXECUTE/RETRY/ROLLBACK mode;
- must not read memory arm, condition, attempt state, or fault identity;
- must use ordinary MuJoCo stepping;
- must not directly write a success flag;
- must not teleport the controlled object during ordinary execution;
- must be deterministic under a fixed process state.

Qualification seeds:

development: 0–19
calibration: 20–39
qualification: 40–59
formal: 1000–1049
reserved OOD: 2000–2029

Executor qualification:

- minimum conditional effect success >= 0.99;
- full-chain success >= 0.98;
- backend errors = 0.

Failure:

FINAL_STATUS = BLOCKED_BY_MICROENV

Do not continue.

# ==================================================
9. IMPLEMENT TRUE SHARED-PREFIX FORKING

Create:

r16p19/phase4_fork_runner.py

Run on Linux CPU only.

For every paired unit:

1. create one environment in one parent process;
2. execute the common physical and event prefix exactly once;
3. freeze the first memory-decision boundary;
4. record:

   - MuJoCo data state;
   - controller state;
   - environment task state;
   - Python RNG;
   - NumPy RNG;
   - event-prefix bytes;
   - action-prefix bytes;
5. call os.fork once per arm;
6. each child inherits the same physical process state;
7. each child instantiates its arm from the same canonical event prefix;
8. each child verifies inherited hashes before the first decision;
9. each child executes only its arm-specific decisions after the fork.

Do not independently replay the pre-decision physics per arm.

Use:

OMP_NUM_THREADS=1
MKL_NUM_THREADS=1

Do not initialize a GPU or offscreen renderer before fork.

Shared-prefix qualification:

- 1,000 paired units;
- force every arm to take the same decision;
- require identical terminal physical-state hashes.

Required exact gate:

- pre-decision state identity: 100%;
- controller-state identity: 100%;
- RNG-state identity: 100%;
- event-prefix identity: 100%;
- action-prefix identity: 100%;
- forced-identical terminal-state identity: 100%.

Any mismatch:

FINAL_STATUS = BLOCKED_BY_SHARED_PREFIX

Stop immediately.

# ==================================================
10. FORMAL TASK-CONDITION CONTRACT

Freeze exactly 20 task-condition cells before pilot.

All three tasks:

C0 CLEAN
A1 STALE_WITNESS_AFTER_RETRY
A2 CROSS_ATTEMPT_SENSOR_MIX
A3 SUPERSEDED_COMMAND_WITNESS
A4 VERIFIED_CONTRADICTION_LATE_WITNESS

Additional T1 cells:

A5 INCIDENTAL_EFFECT_WITHOUT_SKILL_CAUSATION
S2 DISCHARGED_SUPPORT_REVERSAL

Additional T2 cell:

S1 LIVE_SUPPORT_REVERSAL

Additional T3 cells:

S3 ALTERNATIVE_SUPPORT_SURVIVES
S4 BRANCH_LOCAL_INVALIDATION

Pilot:

20 cells
x 10 pilot seeds
x 5 arms
= 1,000 rollouts

Pilot authorization gates:

- shared-prefix gate remains exact;
- executor conditional success >= 0.99;
- no broker or fault-injector errors;
- M4 clean success >= 0.98;
- no M4 attempt leakage;
- no M4 support-graph invariant violation.

Do not tune thresholds or semantics from pilot outcomes.

Formal:

20 cells
x 50 formal seeds
x 5 arms
= 5,000 rollouts

# ==================================================
11. REQUIRED METRICS

Attempt metrics:

- stale_witness_acceptance_rate
- cross_attempt_verification_rate
- superseded_command_realization_rate
- late_witness_after_revocation_rate
- current_attempt_realization_precision
- attempt_attribution_precision
- attempt_attribution_recall

Fact/attribution metrics:

- effect_truth_recognition
- task_advance_correctness
- false_skill_credit_rate
- missed_incidental_success_rate

Support metrics:

- cascade_invalidation_precision
- cascade_invalidation_recall
- over_invalidation_rate
- under_invalidation_rate
- alternative_support_survival_rate
- discharged_support_false_invalidation_rate
- branch_locality_accuracy

Behavior metrics:

- clean_chain_success
- faulted_chain_success
- grounded_advance_precision
- recovery_success
- premature_advance_rate
- timeout_rate
- action_steps
- retry_count
- reobserve_count
- rollback_count

System metrics:

- decision latency
- ledger size
- proof-graph size
- event-processing time
- child-process failures

# ==================================================
12. STATISTICAL ANALYSIS

Primary comparison:

M4_ASCEL_FULL versus the best of:

M0_TYPED_MATCHED
M1_B6_ORIGINAL

Report separate primary endpoints:

Attempt-family endpoint:
mean chain success over A1–A4.

Support-family endpoint:
mean chain success over S1–S4.

Truth-attribution endpoint:
A5 fact recognition and false skill credit.

Use:

- paired design;
- task + formal seed as the cluster;
- 10,000-repetition paired cluster bootstrap;
- exact paired McNemar tests;
- absolute success differences;
- Holm correction where multiple strong comparisons are reported.

Do not average attempt and support families into one number before reporting the
family-specific outcomes.

# ==================================================
13. FROZEN PASS GATES

PASS_ATTEMPT_SCOPE requires:

- M4 attempt-family success exceeds the best strong baseline by >= 0.15;
- paired 95% CI lower bound > 0;
- stale witness acceptance = 0;
- cross-attempt verification = 0;
- superseded witness realization = 0;
- late-witness-after-revocation realization = 0.

PASS_SUPPORT_PROOF requires:

- cascade precision >= 0.95;
- cascade recall >= 0.95;
- over-invalidation <= 0.05;
- under-invalidation <= 0.05;
- M4 support-family success exceeds the best strong baseline by >= 0.15;
- paired 95% CI lower bound > 0.

PASS_TRUTH_ATTRIBUTION requires:

- effect truth recognition >= 0.95;
- task advance correctness >= 0.95;
- false current-skill credit <= 0.05.

Clean requirements:

- M4 clean success degradation <= 0.02;
- action-step overhead <= 0.15;
- event-processing latency overhead <= 0.10.

# ==================================================
14. MECHANISM ABLATIONS

Run only after the complete formal matrix.

NO_ATTEMPT_SCOPE:

- pool evidence at effect level;
- preserve all other M4 behavior.

NO_SUPPORT_VALIDITY:

- replace support clauses with static transitive descendant invalidation.

NO_ATTRIBUTION_SPLIT:

- merge physical fact truth and attempt success.

NO_PRE_REALIZATION_REVOCATION:

- contradiction acts only after REALIZED.

Required interpretation:

- NO_ATTEMPT_SCOPE must remove at least 50% of M4's A1–A4 advantage;
- NO_SUPPORT_VALIDITY must substantially increase S2/S3/S4
over-invalidation or remove at least 50% of support-family advantage;
- NO_ATTRIBUTION_SPLIT must materially increase A5 false skill credit;
- NO_PRE_REALIZATION_REVOCATION must materially increase A4 false realization.

Do not use ablations to retune M4.

# ==================================================
15. FINAL DECISION TREE

Report component statuses:

ATTEMPT_SCOPE_PASS / FAIL
SUPPORT_PROOF_PASS / FAIL
TRUTH_ATTRIBUTION_PASS / FAIL

Use one overall status:

PASS_PHASE4_ASC_EFFECT_LEDGER

- attempt and support gates pass;
- clean and correctness gates pass.

NARROW_TO_ATTEMPT_SCOPED_LEDGER

- attempt gate passes;
- support gate fails.

NARROW_TO_SUPPORT_PROOF_LEDGER

- support gate passes;
- attempt gate fails.

NARROW_TO_EFFECT_ATTRIBUTION_LEDGER

- only truth-attribution gate passes.

REJECT_R16P19_V2

- valid platform and full matrix;
- no component shows incremental value over strong baselines.

BLOCKED_BY_MECHANISM_IMPLEMENTATION
BLOCKED_BY_SHARED_PREFIX
BLOCKED_BY_MICROENV
BLOCKED_BY_IMPLEMENTATION

Do not report PASS when a platform gate is blocked.

# ==================================================
16. DELIVERABLES

Code:

r16p19/phase4_types.py
r16p19/phase4_attempt_ledger.py
r16p19/phase4_support_graph.py
r16p19/phase4_trace_generator.py
r16p19/phase4_trace_oracle.py
r16p19/phase4_microenv.py
r16p19/phase4_executor.py
r16p19/phase4_event_broker.py
r16p19/phase4_fork_runner.py
r16p19/phase4_analysis.py

Tests:

tests/test_phase4_attempt_scope.py
tests/test_phase4_revocation.py
tests/test_phase4_support_graph.py
tests/test_phase4_truth_attribution.py
tests/test_phase4_shared_prefix.py
tests/test_phase4_microenv.py
tests/test_phase4_metrics.py

Artifacts:

experiments/r16p19_phase4/
preregistration.yaml
current_evidence_manifest.json
method_contract.yaml
attempt_schema.json
support_graph_schema.json
task_condition_contract.json
trace_results.jsonl
trace_summary.json
executor_qualification.jsonl
shared_prefix_qualification.jsonl
pilot_results.jsonl
pilot_summary.json
formal_results.jsonl
paired_unit_audit.jsonl
component_metrics.json
cluster_bootstrap.json
paired_tests.json
mechanism_ablations.jsonl
failure_decomposition.jsonl
FINAL_DECISION.md
SHA256SUMS

Retain:

- every failed qualification;
- every shared-prefix mismatch;
- every formal failure;
- representative videos generated only after primary metrics;
- exact source commit and tree;
- exact environment and MuJoCo versions.

# ==================================================
17. EXECUTION ORDER

1. Audit current repository.
2. Freeze Phase-4 preregistration.
3. Implement ASCEL in separate modules.
4. Run deterministic 10,000-schedule trace gate.
5. Stop if the trace gate fails.
6. Implement the three CPU MuJoCo microenvironments.
7. Qualify the macro executor.
8. Stop if executor qualification fails.
9. Implement os.fork shared-prefix runner.
10. Run 1,000 exact prefix qualification units.
11. Stop on any prefix mismatch.
12. Freeze the 20-cell contract and all method parameters.
13. Run the 1,000-rollout pilot.
14. Do not tune from pilot outcomes.
15. Run the 5,000-rollout formal matrix.
16. Run paired statistics.
17. Run four mechanism ablations.
18. Produce component-specific and overall decisions.
19. Update the repository root README.
20. Do not start a learned verifier, ACT, Mem-0, Pi0.5, or novelty claim
automatically.

The next progress report must contain:

- exact branch, HEAD, and tree;
- preregistration commit;
- protected original B6 SHA256;
- finalized AttemptRecord and RealizationProof schemas;
- deterministic trace-gate results;
- executor qualification results;
- shared-prefix exact-identity results;
- whether the pilot and formal matrices are authorized;
- no performance or novelty claim before the full formal matrix completes.

# step6

# R16-P19 Phase-5：Bounded ASCEL Embodied Bridge Validation

## 0. Repository and current evidence

Repository:

`mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback`

Current terminal status:

`BLOCKED_BY_IMPLEMENTATION`

Current evidence boundary:

- Phase-1 actor-free trace semantics passed.
- Phase-1B learned actors failed competence gates.
- Phase-2 geometric executor failed qualification.
- Phase-3 completed a diagnostic matrix, but replay qualification, formal replay competence and pair-prefix identity failed.
- Phase-3 B6 tied `TYPED_MATCHED_RECOVERY` exactly on backend-valid primary units.
- Phase-4 introduced Attempt-Scoped Causal Effect Ledger, ASCEL.
- Phase-4 attempt scope, pre-realization revocation, truth–credit separation and discharge-aware support proofs passed their synthetic mechanism tests.
- Phase-4 clean event-processing latency violated the frozen clean-overhead gate.
- No VLA, learned-verifier, LIBERO-generalization or real-robot improvement claim is currently established.

The original B6 implementation in:

`r16p19/memory.py`

must remain byte-protected and may only be used as a historical baseline.

The goal of Phase-5 is not to rescue B6.

The goal is to determine whether a bounded ASCEL implementation provides genuine embodied behavioral value over strong matched baselines.

---

# 1. Scientific questions

Phase-5 must answer four separate questions.

## Q1. ASCEL-Core embodied value

Under an oracle effect verifier and a qualified frozen policy, does attempt-scoped evidence isolation, pre-realization revocation and physical-truth/skill-credit separation improve faulted task success over strong matched baselines?

## Q2. Learned-verifier robustness

If Q1 passes, does the improvement remain under a learned multi-view effect verifier that does not access simulator truth?

## Q3. Support-proof incremental value

Does discharge-aware support proof reasoning improve behavior beyond ASCEL-Core on tasks that genuinely require persistent, discharged or alternative support?

## Q4. Systems viability

Can the ledger remain bounded in hot memory and meet preregistered latency requirements over long event streams?

These questions must be evaluated separately.

Do not combine a failure in one component with the result of another component.

---

# 2. New method decomposition

Implement and evaluate two method levels.

## ASCEL-Core

Includes:

- attempt generation and active attempt identity;
- command identity bound to attempt identity;
- attempt-scoped evidence isolation;
- rejection of stale and cross-attempt evidence;
- pre-realization revocation at OBSERVED and VERIFIED;
- realization revocation;
- physical fact truth separated from active-attempt skill credit;
- bounded recovery decisions.

Does not include discharge-aware support proof graphs.

## ASCEL-Full

Includes all ASCEL-Core functionality plus:

- realization proof graph;
- conjunctive references inside one support clause;
- disjunctive alternative support clauses;
- persistent support;
- support discharged when a declared endpoint is realized;
- recursive invalidation only through newly unsupported proofs;
- unrelated branch preservation.

Support reasoning must be optional and task-contract dependent.

Do not enable support graphs for tasks without a meaningful frozen support contract.

---

# 3. Hard prohibitions

Do not:

- modify `r16p19/memory.py`;
- reinterpret Phase-3 diagnostic results as confirmatory evidence;
- reinterpret Phase-4 microbenchmark results as VLA evidence;
- train another tiny BC, ACT or geometric executor;
- construct another independent-slider microbenchmark as the main result;
- pass simulator effect truth directly to any memory arm;
- pass fault identity or condition name to the policy, verifier or memory;
- use formal seeds for policy selection, verifier training or threshold calibration;
- replay the pre-decision physical prefix independently for each arm;
- change gates after seeing pilot or formal results;
- continue to learned-verifier experiments if oracle ASCEL-Core fails;
- claim support-proof value if ASCEL-Full does not beat ASCEL-Core;
- claim visual/VLA improvement from oracle-verifier results alone;
- build excessive cryptographic or formal-activation infrastructure.

Use ordinary Git commits, manifests, JSON/JSONL, SHA256, unit tests and Markdown reports.

---

# 4. Branch and preregistration

Create branch:

`phase5-bounded-ascel-embodied`

Before implementing Phase-5 code or accessing formal tasks/seeds, create and commit:

```text
experiments/r16p19_phase5/
  preregistration.yaml
  prior_evidence_manifest.json
  protected_source_manifest.json
  bounded_ledger_contract.yaml
  task_selection_contract.yaml
  policy_contract.yaml
  verifier_contract.yaml
  arm_contracts.yaml
  fault_contracts.yaml
  pairing_contract.yaml
  metric_contract.yaml
  statistical_analysis_plan.yaml
  final_status_contract.yaml
  README.md
```

Record:

- repository HEAD and tree;
- protected B6 SHA256;
- Phase-3 final decision/report hashes;
- Phase-4 final decision/report hashes;
- exact method decomposition;
- exact arm definitions;
- task and seed splits;
- policy selection procedure;
- verifier training and calibration procedure;
- fault conditions;
- sample counts;
- all gates;
- final statuses.

Commit preregistration before implementation.

---

# 5. Stage A：bounded live ledger

Create:

```text
r16p19/phase5_live_ledger.py
r16p19/phase5_audit_store.py
r16p19/phase5_compaction.py
r16p19/phase5_reference_ledger.py
r16p19/phase5_property_generator.py
r16p19/phase5_benchmark.py
```

## 5.1 Hot live state

The hot state may contain only:

- one active attempt per effect;
- at most four recent closed attempts per effect;
- bounded current receipt buckets;
- current fact state;
- latest revocation epoch;
- currently valid realization proofs;
- currently live support clauses;
- reverse dependency indices;
- current recovery route.

Freeze exact capacity values in `bounded_ledger_contract.yaml`.

Recommended defaults:

```text
recent_closed_attempts_per_effect: 4
evidence_receipts_per_source_per_active_attempt: 4
active_proofs_per_effect: 4
active_support_clauses_total: 256
```

## 5.2 Cold audit storage

Superseded history must be compacted into append-only summaries containing:

```text
attempt_id
generation
command_id
start_epoch
end_epoch
terminal_status
accepted_evidence_digest_set_hash
attributed_success
invalidation_reason
parent_audit_hash
```

Late historical evidence must still be rejectable after compaction.

## 5.3 Receipt schema

Every receipt must include:

```text
effect_id
attempt_id
command_id
event_epoch
capture_id
sensor_source
source_group
correlation_group
frame_digest
timestamp
verifier_model_version
calibration_version
parent_ids
```

Two receipts count as independent only when:

- `correlation_group` differs;
- `capture_id` differs;
- raw frame digest differs;
- both match the active attempt;
- both match the current command;
- both occur after the latest revocation epoch.

Different sensor names alone are not sufficient.

## 5.4 Reference equivalence tests

Maintain an unbounded reference implementation.

Generate randomized event streams containing:

- duplicate events;
- reused event IDs with changed bytes;
- delayed receipts;
- out-of-order receipts;
- repeated attempts;
- stale witnesses;
- pre-realization contradiction;
- post-realization contradiction;
- external realization;
- support removal;
- support discharge;
- alternative support;
- unrelated branches;
- compaction followed by stale evidence arrival.

Run at least:

```text
100,000 events
10,000 attempts
multiple effects and branches
```

Require exact agreement between bounded and unbounded implementations on:

- physical fact state;
- attributed success;
- accepted/rejected evidence;
- high-level decision;
- valid proof set;
- invalidated effect set;
- recovery route.

## 5.5 Systems gates

Require:

```text
single-event process P99 <= 1 ms
single-control-tick ledger P99 <= 2 ms
ledger time <= 5% of measured policy control cycle
P99 latency at 100k events <= 1.2 × P99 latency at 1k events
hot-state memory after 100k events <= 10 MB
bounded/reference decision mismatch = 0
audit-chain break = 0
```

Failure:

`BLOCKED_BY_LEDGER_IMPLEMENTATION`

Do not continue to policy experiments.

---

# 6. Stage B：formal task and policy selection

The previous two LIBERO tasks are development/regression only:

```text
stove_moka
bowl_drawer
```

Select exactly three new official benchmark tasks not previously used by R16-P19.

Selection criteria:

- at least two task families;
- at least one reversible articulated effect;
- at least one grasp–place–release chain;
- at least one containment, closure or long-dependency chain;
- physical retry and reversal are possible;
- enough mutually exclusive initial states exist.

Freeze for each task:

```text
development:   init 0–19
calibration:   init 20–39
qualification: init 40–59
formal:        init 60–99
```

If the benchmark exposes a different number of initial states, create an equivalent mutually exclusive split and preregister it before use.

## 6.1 Frozen policy candidates

Audit already available policy checkpoints.

Allowed candidates include existing:

- π0.5/OpenPI policies;
- official ACT;
- already trained LIBERO policies;
- another already available frozen policy.

Allow at most:

```text
one primary policy
one preregistered fallback policy
```

Do not train another policy family in this phase.

## 6.2 Policy interface

Implement:

```text
FrozenSkillPolicy.action_chunk(
    observation_history,
    task_id,
    effect_id,
    execution_mode,
    policy_seed
)
```

Allowed execution modes:

```text
EXECUTE
RETRY
ROLLBACK_REEXECUTE
REOBSERVE
```

Policy inputs must exclude:

- memory arm;
- epistemic state;
- fault condition;
- simulator truth;
- reward;
- task success;
- future state.

## 6.3 Policy qualification

On qualification initial states require:

```text
per-task clean full success >= 0.80
minimum per-effect success/reach >= 0.80
repeated-effect-loop rate <= 0.10
backend error count = 0
```

Failure of both primary and fallback:

`BLOCKED_BY_POLICY`

Do not run formal memory experiments.

---

# 7. Deterministic Policy Broker

Create:

`r16p19/phase5_policy_broker.py`

Policy request key:

```text
observation_hash
history_hash
task_id
effect_id
execution_mode
policy_checkpoint_sha256
policy_config_sha256
policy_seed
```

For an identical key:

- execute real policy inference once;
- cache the exact action chunk bytes;
- return byte-identical actions to every arm;
- retain policy latency and model-state hash;
- retain flow/diffusion noise or RNG state;
- retain executed-prefix metadata.

Identical policy inputs and high-level decisions must produce identical action bytes.

If CUDA is used, do not initialize CUDA inside a process that will later call `os.fork`.

Use a separate deterministic policy server or cache process.

---

# 8. Stage C：true shared-prefix pairing

Create:

```text
r16p19/phase5_snapshot.py
r16p19/phase5_pair_runner.py
r16p19/phase5_pair_audit.py
```

For every paired unit:

1. run the physical and observation prefix once;
2. stop at the first memory-decision boundary;
3. record:

   - complete simulator state;
   - controller state;
   - observation history;
   - policy history and chunk buffer;
   - policy broker cache;
   - Python, NumPy and Torch RNG;
   - event-prefix bytes;
   - action-prefix bytes;
4. clone/fork each memory arm from this exact state;
5. verify all inherited hashes;
6. allow divergence only after different high-level memory decisions.

Run 1,000 qualification units with forced-identical decisions.

Exact required gates:

```text
physical_state_identity = 1.0
controller_state_identity = 1.0
policy_history_identity = 1.0
rng_identity = 1.0
observation_prefix_identity = 1.0
event_prefix_identity = 1.0
action_prefix_identity = 1.0
forced_identical_terminal_state_identity = 1.0
```

Any mismatch:

`BLOCKED_BY_PAIRING`

Stop immediately.

---

# 9. Formal memory arms

Implement or adapt five arms.

## M0_TYPED_MATCHED_RECOVERY

- typed evidence states;
- two correlation-group verification;
- generic contradiction rollback;
- same retry/reobserve/rollback budget as ASCEL;
- no attempt-scoped evidence isolation;
- no truth–credit split;
- no support graph.

## M1_VERSIONED_POSTCHECK

- per-attempt UNKNOWN/TRUE/FALSE;
- K=2 distinct correlation-group positives before TRUE;
- contradiction or negative to FALSE;
- TTL/hysteresis;
- same recovery operations and budgets;
- no causal witness requirement;
- no support graph.

Select K and TTL using calibration data only.

## M2_B6_ORIGINAL

Adapter around frozen original B6.

Verify protected B6 SHA before every formal run.

## M3_ASCEL_CORE

- bounded attempt scope;
- active command binding;
- evidence isolation;
- stale/cross-attempt rejection;
- pre-realization revocation;
- realization revocation;
- physical fact / skill credit split;
- static dependencies only.

## M4_ASCEL_FULL

M3 plus discharge-aware support proof graph.

All arms must receive the same:

- policy checkpoint;
- policy requests;
- policy seeds;
- environment state;
- observations;
- event bytes before divergence;
- retry/reobserve/rollback limits;
- action budget;
- fault schedule.

---

# 10. Oracle effect-verifier broker

Create:

`r16p19/phase5_oracle_verifier.py`

Simulator truth may only:

- create standardized receipts;
- create contradiction/negative events;
- provide evaluation labels;
- implement declared physical perturbations.

Simulator truth may not be passed directly to a memory arm or policy.

---

# 11. Formal Core conditions

Freeze seven conditions.

## C0 CLEAN

Normal execution and normal receipts.

## A1 NOOP_THEN_RETRY_WITH_STALE_RECEIPT

- suppress the first attempt’s physical effect;
- start a second attempt;
- deliver a delayed positive or witness from attempt 1 during attempt 2;
- the delayed event must not verify or credit attempt 2.

## A2 CROSS_ATTEMPT_SENSOR_MIX

- source group A positive belongs to attempt 1;
- attempt 1 is superseded;
- source group B positive belongs to attempt 2;
- the two receipts must not create verification.

## A3 VERIFIED_CONTRADICTION_LATE_WITNESS

- two current positives create VERIFIED;
- contradiction or valid negative arrives before realization witness;
- old witness arrives later;
- the effect must remain revoked.

## A4 POST_REALIZATION_REVERSAL

- effect is physically realized and verified;
- before the dependent effect, physically reverse it;
- deliver contradiction;
- require recovery before progress.

## A5 INCIDENTAL_EXTERNAL_REALIZATION

- the active policy attempt does not cause the effect;
- an external perturbation or controller realizes the effect;
- task progress may continue;
- active skill must not receive success credit.

## V1 SINGLE_VIEW_CORRELATED_FALSE_POSITIVE

- one source group reports positive;
- another source does not confirm;
- physical effect remains false;
- no valid realization witness is emitted.

Faults must be tied to attempt identity and physical events, not wall-clock time after arms diverge.

---

# 12. Core pilot and formal counts

Pilot:

```text
3 tasks × 7 conditions × 5 seeds × 5 arms
= 525 rollouts
```

Pilot gates:

- shared-prefix gate remains exact;
- no policy/backend/broker errors;
- M3 clean success >= 0.80;
- M3 stale/cross-attempt leakage = 0;
- no bounded/reference mismatch;
- no protected-source mismatch.

Do not tune method semantics using pilot outcomes.

Formal:

```text
3 tasks × 7 conditions × 40 formal seeds × 5 arms
= 4,200 rollouts
```

Persist every row immediately in resumable JSONL.

Save videos for:

- every failure;
- every first-decision divergence;
- every premature advance;
- every false skill credit;
- at least five representative successful recoveries per condition and arm.

---

# 13. Core metrics

Behavior:

```text
full_task_success
faulted_task_success
clean_task_success
per_effect_success
recovery_success
repeated_loop_rate
action_steps
completion_latency
```

Attempt/evidence:

```text
stale_evidence_acceptance_rate
cross_attempt_verification_rate
superseded_command_realization_rate
late_witness_after_revocation_rate
current_attempt_realization_precision
attempt_attribution_precision
attempt_attribution_recall
```

Truth/credit:

```text
effect_truth_recognition
task_advance_correctness
false_skill_credit_rate
missed_incidental_success_rate
```

Safety:

```text
premature_advance_rate
false_grounded_advance_rate
unsafe_continuation_rate
safe_stop_rate
```

Cost/system:

```text
retry_count
reobserve_count
rollback_count
clean_extra_steps
event_processing_P50_P95_P99
decision_latency_P50_P95_P99
hot_state_memory
audit_storage_growth
policy_cycle_fraction
```

Failure decomposition:

```text
MEMORY_DECISION_ERROR
POLICY_EXECUTION_FAILURE
VERIFIER_ERROR
BROKER_ERROR
FAULT_INJECTOR_ERROR
PAIRING_ERROR
TIMEOUT
```

---

# 14. Statistical analysis

Primary Core comparison:

```text
M3_ASCEL_CORE
vs
best of:
  M0_TYPED_MATCHED_RECOVERY
  M1_VERSIONED_POSTCHECK
  M2_B6_ORIGINAL
```

Cluster:

```text
task_id + formal_init
```

Run:

- 10,000-repetition paired cluster bootstrap;
- exact paired McNemar tests;
- Holm correction;
- per-condition effects;
- per-task effects;
- absolute risk difference;
- relative false-advance reduction.

Primary Core pass requires all:

```text
faulted success margin >= 0.08
bootstrap 95% CI lower bound > 0
positive improvement on at least 2 of 3 tasks
remaining task degradation <= 0.02
false grounded advance relative reduction >= 0.50
stale evidence acceptance = 0
cross-attempt verification = 0
late witness realization = 0
A5 truth recognition >= 0.95
A5 false skill credit <= 0.05
clean success degradation <= 0.02
clean extra action steps <= 0.10
all latency, memory and scaling gates pass
```

If Core fails:

`REJECT_ASCEL_EMBODIED_VALUE`

Do not train or run a learned verifier.

---

# 15. Natural-failure audit

Before formal access, run at least 200 unmodified base-policy rollouts on development tasks/seeds.

Using oracle labels only for offline audit, classify:

- no-op;
- premature transition;
- failed retry;
- reversal;
- delayed observation;
- false visual completion;
- incidental realization;
- repeated loop.

Freeze a development-only fault prior.

Report both:

- equal-weight stress results;
- prior-weighted expected results.

Do not use formal outcomes to modify the prior.

The prior-weighted endpoint is secondary and may not replace primary gates.

---

# 16. Stage D：learned effect verifier

Only begin if oracle ASCEL-Core passes.

Create:

```text
r16p19/phase5_verifier_data.py
r16p19/phase5_verifier_model.py
r16p19/phase5_verifier_training.py
r16p19/phase5_learned_verifier.py
r16p19/phase5_verifier_evaluation.py
```

## 16.1 Inputs

Allowed:

- multi-view RGB;
- short temporal frame window;
- proprioception;
- gripper/contact/wrench when available;
- task ID;
- effect ID.

Forbidden:

- simulator truth;
- reward;
- success predicate;
- future state;
- fault condition;
- memory arm.

## 16.2 Dataset split

Split by source episode and environment variation.

The same episode or rollout may not appear in multiple splits.

Use:

```text
train
calibration
qualification
formal rollout
```

Formal rollout data must not participate in training, checkpoint selection, temperature scaling or threshold selection.

## 16.3 Outputs

For each source group emit:

```text
p_effect_true
p_effect_false_or_contradicted
capture_id
correlation_group
frame_digest
model_version
calibration_version
```

## 16.4 Qualification

Require:

```text
macro AUROC >= 0.90
macro ECE <= 0.05
minimum per-effect TPR >= 0.80
maximum per-effect FPR <= 0.10
formal-data access = 0
```

Also report performance under:

- occlusion;
- lighting shift;
- camera corruption;
- stale frames;
- sensor dropout.

Failure:

`BLOCKED_BY_VERIFIER`

Do not use oracle results to claim learned visual improvement.

---

# 17. Learned-verifier formal matrix

Compare only:

```text
strongest oracle-stage baseline
M3_ASCEL_CORE
```

Run:

```text
3 tasks × 7 conditions × 40 formal seeds × 2 arms
= 1,680 rollouts
```

Require:

```text
faulted success margin >= 0.05
bootstrap 95% CI lower bound > 0
improvement direction consistent with oracle
at least 50% of oracle absolute gain retained
false grounded advance <= 0.05
false skill credit <= 0.05
clean success degradation <= 0.02
latency and bounded-memory gates pass
```

Passing this stage authorizes an embodied visual/VLA claim.

---

# 18. Stage E：support-proof extension

Do not force support semantics into every LIBERO task.

First search selected formal tasks for genuine:

- persistent support;
- discharged support;
- alternative support;
- unrelated branch.

If insufficient, implement exactly two contact-rich gravity-enabled MuJoCo/robosuite tasks:

## T1 Carry–Place–Release

- actual movable object;
- actual gripper/object interaction or constraint;
- grasp supports lift and transport;
- support is discharged after valid release in target;
- later gripper opening must not invalidate placement.

## T2 Alternative Physical Support

- object rests on two physical supports;
- removing one preserves stability;
- removing both invalidates elevation;
- unrelated branch remains valid.

Do not implement these tasks as independent logical sliders.

Support conditions:

```text
C0 CLEAN
S1 LIVE_SUPPORT_REMOVAL
S2 DISCHARGED_SUPPORT_REMOVAL
S3 ALTERNATIVE_SUPPORT_ONE_BRANCH_REMOVED
```

Relevant arms:

```text
M0_TYPED_MATCHED_RECOVERY
M1_VERSIONED_POSTCHECK
M3_ASCEL_CORE
M4_ASCEL_FULL
```

Formal:

```text
2 tasks × 4 conditions × 30 seeds × 4 arms
= 960 rollouts
```

M4 support pass requires:

```text
success margin over M3 >= 0.05
bootstrap 95% CI lower bound > 0
cascade precision >= 0.95
cascade recall >= 0.95
over-invalidation <= 0.05
under-invalidation <= 0.05
clean degradation <= 0.02
systems gates pass
```

If Core passes but Full fails:

`NARROW_TO_ASCEL_CORE`

---

# 19. Mechanism ablations

Run only after oracle Core passes.

Ablations:

```text
NO_ATTEMPT_SCOPE
NO_PRE_REALIZATION_REVOCATION
NO_TRUTH_CREDIT_SPLIT
NO_SUPPORT_GRAPH
```

Run each only on corresponding conditions.

A mechanism claim requires either:

```text
at least 50% of the corresponding advantage removed
```

or:

```text
target error increases by at least 0.10 absolute
```

Do not retune the ablation.

---

# 20. Deliverables

Code:

```text
r16p19/phase5_live_ledger.py
r16p19/phase5_audit_store.py
r16p19/phase5_compaction.py
r16p19/phase5_reference_ledger.py
r16p19/phase5_property_generator.py
r16p19/phase5_policy_broker.py
r16p19/phase5_oracle_verifier.py
r16p19/phase5_snapshot.py
r16p19/phase5_pair_runner.py
r16p19/phase5_pair_audit.py
r16p19/phase5_runner.py
r16p19/phase5_analysis.py
r16p19/phase5_verifier_data.py
r16p19/phase5_verifier_model.py
r16p19/phase5_verifier_training.py
r16p19/phase5_learned_verifier.py
```

Tests:

```text
tests/test_phase5_bounded_equivalence.py
tests/test_phase5_compaction.py
tests/test_phase5_attempt_scope.py
tests/test_phase5_truth_credit.py
tests/test_phase5_support_graph.py
tests/test_phase5_policy_broker.py
tests/test_phase5_shared_prefix.py
tests/test_phase5_latency_scaling.py
tests/test_phase5_statistics.py
tests/test_phase5_verifier_contract.py
```

Results:

```text
experiments/r16p19_phase5/
  bounded_property_results.jsonl
  bounded_reference_audit.json
  latency_scaling.json
  memory_scaling.json
  task_selection_manifest.json
  policy_candidate_results.jsonl
  selected_policy_manifest.json
  policy_qualification.json
  shared_prefix_results.jsonl
  shared_prefix_summary.json
  natural_failure_audit.json
  oracle_pilot_results.jsonl
  oracle_formal_results.jsonl
  oracle_analysis.json
  oracle_cluster_bootstrap.json
  oracle_mcnemar_holm.json
  verifier_dataset_manifest.json
  verifier_training_metrics.jsonl
  verifier_qualification.json
  learned_verifier_formal_results.jsonl
  learned_verifier_analysis.json
  support_formal_results.jsonl
  support_analysis.json
  mechanism_ablations.jsonl
  mechanism_attribution.json
  failure_cases.md
  EXPERIMENT_REPORT_ZH.md
  FINAL_DECISION.md
  SHA256SUMS
```

Save:

- exact source commits;
- exact policy checkpoint SHA256;
- exact verifier checkpoint SHA256;
- normalization and calibration hashes;
- all failed qualification videos;
- all formal failures;
- all first-decision divergences;
- all premature advances;
- all false skill credits;
- runtime and hardware contracts.

---

# 21. Final status vocabulary

Use exactly one final status:

```text
PASS_PHASE5_ASCEL_FULL_EMBODIED
PASS_PHASE5_ASCEL_CORE_EMBODIED
NARROW_TO_ASCEL_CORE
BLOCKED_BY_LEDGER_IMPLEMENTATION
BLOCKED_BY_POLICY
BLOCKED_BY_PAIRING
BLOCKED_BY_VERIFIER
BLOCKED_BY_IMPLEMENTATION
REJECT_ASCEL_EMBODIED_VALUE
```

Decision precedence:

1. Invalid preregistration or formal contamination  
-> `BLOCKED_BY_IMPLEMENTATION`
2. Bounded/reference or systems gate failure  
-> `BLOCKED_BY_LEDGER_IMPLEMENTATION`
3. No qualified policy  
-> `BLOCKED_BY_POLICY`
4. Shared-prefix gate failure  
-> `BLOCKED_BY_PAIRING`
5. Oracle ASCEL-Core failure  
-> `REJECT_ASCEL_EMBODIED_VALUE`
6. Oracle Core passes but learned verifier fails qualification  
-> `BLOCKED_BY_VERIFIER`
7. Learned Core passes, Full support fails  
-> `PASS_PHASE5_ASCEL_CORE_EMBODIED`
8. Learned Core and support extension both pass  
-> `PASS_PHASE5_ASCEL_FULL_EMBODIED`

Do not weaken a gate after observing pilot or formal outcomes.

Update the repository root README with:

- the new terminal status;
- the complete Phase-1 through Phase-5 lineage;
- explicit separation between B6, ASCEL-Core and ASCEL-Full;
- exact claims that are and are not supported.

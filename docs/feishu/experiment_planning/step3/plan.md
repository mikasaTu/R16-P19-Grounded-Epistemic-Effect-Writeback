# step3

Task:
R16-P19 Phase-2 Competent-Executor Causal Behavior Validation

Repository:
mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback

Current terminal status:
BLOCKED_BY_ACTOR_V2

Current scientific evidence:

- actor-free trace mechanism gate passed;
- B6 false completion was 0.0 while B3 was 0.5;
- B6 contradiction recovery recall was 1.0;
- actor–memory input confounding has been removed;
- shared State-ACT reached 18/40 full-task success and minimum per-effect 0.40;
- per-effect State-ACT reached 24/40 full-task success and minimum per-effect 0.50;
- formal init 0–19, the 800-rollout matrix, and paired bootstrap remain NOT_RUN.

Scientific objective:
Test the original R16-P19 mechanism at behavior level under a competent,
memory-independent low-level executor.

This phase is not another learned-actor search.
Do not train a third neural BC/ACT/VLA actor.

==================================================

1. HARD SCIENTIFIC BOUNDARIES
==================================================

Do not change:

- r16p19/memory.py semantics;
- REQUESTED / IMAGINED / OBSERVED / VERIFIED / REALIZED /
STALLED / INVALIDATED_REALIZATION;
- effect ontology;
- evidence receipt and provenance rules;
- C0–C7 fault definitions;
- B2/B3/B5/B6 memory definitions;
- demo 0–29 / 30–39 / 40–49 split;
- formal init 0–19;
- existing behavior gates;
- 10,000-repetition paired bootstrap protocol.

Do not:

- train Pi0.5, Mem-0, a world model, or another neural actor;
- allow the executor to read memory summary or epistemic state;
- allow the executor to read fault identity;
- allow the executor to call full task-success predicates to advance effects;
- use simulator effect truth as a high-level decision input;
- inspect formal init 0–19 before the executor is frozen;
- lower the original minimum per-effect threshold of 0.80;
- tune after seeing formal results;
- reserve two GPUs for a non-neural experiment;
- claim VLA or publication-level success.

Use ordinary Git, JSONL, SHA256 and Markdown.
Do not create new formal-activation or excessive integrity infrastructure.

# ==================================================
2. NEW PHASE SPLITS

Freeze:

- demo 0–29:
skill-template extraction only;
- demo 30–39:
template/controller calibration only;
- demo 40–49:
untouched diagnostic/receipt test;
- init 20–39:
archival only; already used in Phase-1B;
- init 40–59:
executor development and smoke only;
- init 60–79:
frozen executor qualification;
- init 0–19:
formal competence and memory evaluation, viewed once;
- init 80–99:
reserved for future stress/OOD.

Create and commit preregistration before implementation:

experiments/r16p19_libero_phase2/
preregistration.yaml
executor_contract.yaml
split_manifest.json
behavior_gates.json
fault_schedule.yaml
README.md

# ==================================================
3. IMPLEMENT A RETARGETED GEOMETRIC SKILL EXECUTOR

Name:
RetargetedGeometricSkillExecutor

Public interface:

SkillExecutor.action_chunk(
state_history,
task_id,
effect_id,
execution_mode,
retry_index
)

Allowed inputs:

- current and recent low-dimensional simulator geometry;
- robot joint/end-effector state;
- object and fixture poses needed for local control;
- task ID;
- effect ID;
- EXECUTE or RETRY;
- retry index.

Forbidden inputs:

- memory summary;
- epistemic state;
- REQUESTED/REALIZED/STALLED flags;
- fault condition;
- effect truth boolean;
- current task stage derived from the memory manager;
- future simulator state;
- oracle next-effect identity.

The executor must never advance the effect by itself.
Only the memory manager may choose ADVANCE, RETRY, REOBSERVE,
ROLLBACK_OR_REPLAN, or SAFE_STOP.

# ==================================================
4. BUILD EFFECT-RELATIVE TRAJECTORY TEMPLATES

Effects:

stove_moka:

- STOVE_TURNED_ON
- MOKA_GRASPED
- MOKA_ON_STOVE
- MOKA_RELEASED_ON_STOVE

bowl_drawer:

- BOWL_GRASPED
- BOWL_IN_BOTTOM_DRAWER
- BOWL_RELEASED_IN_DRAWER
- BOTTOM_DRAWER_CLOSED

For each effect:

1. extract successful segments from demo 0–29;
2. express end-effector trajectories in the relevant local task frame;
3. preserve gripper open/close schedule;
4. cluster into at most three representative templates;
5. choose a template using only current geometric state;
6. retarget it to the current object/fixture frame;
7. track it with deterministic closed-loop Cartesian delta control;
8. execute in receding-horizon chunks;
9. use a preregistered retry order over templates or fixed small offsets.

Suggested local frames:

- stove control frame for STOVE_TURNED_ON;
- moka object frame for MOKA_GRASPED;
- stove placement frame for MOKA_ON_STOVE and release;
- bowl object frame for BOWL_GRASPED;
- drawer interior frame for BOWL_IN_BOTTOM_DRAWER and release;
- drawer closing-axis frame for BOTTOM_DRAWER_CLOSED.

The executor may use local geometric stopping tolerances for waypoint
tracking, but it must not call the task effect predicate to decide that the
memory should advance.

# ==================================================
5. EXECUTOR DEVELOPMENT AND QUALIFICATION

Development:

- use init 40–59 only;
- inspect videos and tune templates/controller gains only here.

Before qualification:

- freeze source commit;
- freeze template files and hashes;
- freeze controller gains;
- freeze retry order;
- freeze maximum chunks and action steps;
- freeze deterministic seeds.

Qualification:

- run 2 tasks × init 60–79 = 40 clean rollouts.

Qualification requirements:

- minimum per-effect success >= 0.90;
- each task full-task success >= 0.80;
- repeated-effect-loop rate <= 0.10;
- no memory or fault leakage;
- identical executor bytes for identical executor inputs.

If qualification fails:

- final status BLOCKED_BY_EXECUTOR_V3;
- do not implement another executor family;
- do not inspect formal init 0–19;
- do not run the memory matrix.

If qualification passes:

- freeze one executor manifest and proceed.

# ==================================================
6. FORMAL EXECUTOR GATE

Run formal init 0–19 exactly once with the frozen executor.

Original authorization condition:

minimum per-effect success >= 0.80

If it fails:

FINAL_STATUS = BLOCKED_BY_EXECUTOR_V3

Do not run the 800-rollout matrix.

# ==================================================
7. FORMAL ORACLE-RECEIPT MEMORY MATRIX

If the formal executor gate passes, run exactly:

2 tasks
× 20 formal init
× 5 conditions
× 4 memory arms
= 800 rollouts

Conditions:

C0 clean
C1 command no-op
C2 delayed realization
C3 post-realization reversal
C7 imagined success followed by observed failure

Arms:

B2 command-as-progress
B3 monolithic writeback
B5 typed + verification without contradiction recovery
B6 full R16-P19

All arms must use:

- the same executor;
- the same initial-state bytes;
- the same template selection seed;
- the same fault target effect;
- the same effect-relative fault schedule;
- the same max steps;
- the same observation stream;
- the same receipt broker;
- the same physical fault seed.

The only allowed difference is the memory semantics and resulting
high-level decision.

# ==================================================
8. PHYSICAL FAULT CONTRACTS

C1 command no-op:

- apply only on the first attempt of the target effect;
- record that the command was issued;
- prevent the requested physical effect;
- disable the fault on a legitimate retry.

C2 delayed realization:

- let the physical effect occur;
- withhold the verification receipt for a frozen number of steps;
- do not modify other physical dynamics.

C3 post-realization reversal:

- allow effect verification and realization;
- before the dependent next effect, apply one deterministic reversal;
- send later contradictory receipts.

C7 imagined-success failure:

- issue the same high-confidence IMAGINED-success event to all arms;
- do not realize the physical effect;
- later provide observed failure evidence.

Faults must be tied to effect attempts or physical events,
not absolute wall-clock time after arms diverge.

# ==================================================
9. FIRST-DIVERGENCE CAUSAL REPLAY

For every paired unit:

1. find the first time B3/B5/B6 choose different high-level decisions;
2. save the simulator snapshot immediately before decision execution;
3. from that exact snapshot, replay each distinct decision using:

   - the same frozen executor;
   - the same fault seed;
   - the same execution horizon;
4. evaluate immediate effect completion and recoverability.

Report:

- decision_causal_win_rate;
- effect_completion_after_decision;
- irreversible_failure_after_decision;
- extra_steps_after_decision.

This branch replay is secondary evidence but mandatory for mechanism
interpretation.

# ==================================================
10. REQUIRED METRICS

Behavior:

- full_task_success
- clean_success
- faulted_success
- per_effect_success
- recovery_success
- repeated_effect_loop_rate
- total_action_steps
- completion_time

Memory:

- false_completion_rate
- premature_advance_rate
- realized_precision
- realized_recall
- contradiction_detection_recall
- contradiction_recovery_recall
- invalidated_realization_accuracy
- recovery_routing_accuracy
- imagined_as_realized_error

Cost/conservatism:

- retry_count
- reobserve_count
- rollback_count
- safe_stop_rate
- unnecessary_recovery_rate
- clean_extra_steps

Causal decomposition:

- memory_decision_error
- executor_failure_after_correct_decision
- receipt_error
- fault_injector_error
- timeout
- decision_causal_win_rate

# ==================================================
11. FROZEN BEHAVIOR GATES

Keep all original gates:

- B6 false completion relative reduction vs B3 >= 0.50;
- B6 contradiction recovery recall >= 0.80;
- B6 target-fault success > B3;
- B6 target-fault success > B5;
- paired bootstrap 95% CI lower bound for B6-B3 > 0;
- clean success degradation vs B3 <= 0.03;
- B6 statistically distinguishable from B2.

Add preregistered Phase-2 gates:

- B6 faulted-task success exceeds max(B3, B5) by >= 0.10 absolute;
- decision_causal_win_rate >= 0.70;
- clean unnecessary_recovery_rate <= 0.10;
- clean SAFE_STOP rate = 0.

Run the frozen 10,000-repetition paired bootstrap over formal init units.
Also report paired McNemar tests and absolute effect sizes,
but do not replace the bootstrap gates.

# ==================================================
12. REGRESSION AND FAIRNESS TESTS

Required regression:

- before the first memory-decision divergence, all paired arms must have
identical executor-input hashes and identical action bytes;
- the executor must not import or access memory-state classes;
- fault identity must not appear in executor input;
- formal init 0–19 must have zero access before executor freeze;
- all failed qualification videos must be retained;
- B6 may not obtain low false completion by always REOBSERVE or SAFE_STOP.

# ==================================================
13. FINAL STATUS

Use exactly one:

PASS_PHASE2_ORACLE_BEHAVIOR
REJECT_CORE_MECHANISM
BLOCKED_BY_EXECUTOR_V3
BLOCKED_BY_IMPLEMENTATION

PASS_PHASE2_ORACLE_BEHAVIOR requires:

- executor qualification pass;
- formal executor gate pass;
- all 800 rollouts completed;
- all correctness gates pass;
- all frozen behavior gates pass.

REJECT_CORE_MECHANISM requires:

- executor qualification and formal gate pass;
- complete 800-rollout matrix;
- B6 fails the frozen behavior gates.

Do not modify memory thresholds or fault definitions after a rejection.

# ==================================================
14. DELIVERABLES

Produce:

experiments/r16p19_libero_phase2/
preregistration.yaml
executor_contract.yaml
split_manifest.json
skill_template_manifest.json
skill_templates/
executor_development_log.md
executor_qualification_results.jsonl
selected_executor_manifest.json
formal_executor_gate.json
closed_loop_results.jsonl
first_divergence_replays.jsonl
paired_bootstrap.json
behavior_summary.json
mechanism_mediation.json
failure_cases.md
FINAL_DECISION.md
SHA256SUMS

Save:

- all qualification failures;
- all formal failures;
- at least five representative successful and failed recoveries per condition;
- exact source commit;
- exact LIBERO commit;
- executor/template hashes;
- job IDs and runtime contracts.

Update the repository root README so that the latest scientific status is
visible and the older BLOCKED_BY_ACTOR / BLOCKED_BY_ACTOR_V2 results are
clearly described as predecessor phases.

# ==================================================
15. RESOURCE POLICY

The executor is non-neural.

Prefer CPU + OSMesa/EGL.
If hardware rendering requires a GPU, request at most one GPU.
Do not reserve a second idle A800.
Do not start Pi0.5, Mem-0, RMBench, or a learned effect verifier in this phase.

At the end, provide only a readiness note for the next phase:

LEARNED_EFFECT_VERIFIER_READY
or
LEARNED_EFFECT_VERIFIER_BLOCKED

Do not start that next phase automatically.

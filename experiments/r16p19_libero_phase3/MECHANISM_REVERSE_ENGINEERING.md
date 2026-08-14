# Phase-3 mechanism reverse engineering

## Evidence boundary

This document explains the behavior of the already implemented mechanisms. It
does not propose a new idea. The analysis is diagnostic because replay
qualification failed, the formal replay-only gate failed, and 27/150 formal
paired units violated the preregistered byte-identical pre-divergence prefix
requirement. The implemented terminal precedence therefore reports
`BLOCKED_BY_IMPLEMENTATION`; the independent replay-backend blockers remain
present as well.

The frozen B6 implementation in `r16p19/memory.py` was not changed. Its SHA256
is `4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5`.

## Code-path map used for reverse engineering

The explanation below was traced from the concrete transition code rather
than inferred from aggregate scores alone:

| Mechanism | Source path | Decisive transition |
|---|---|---|
| B2 command-as-progress | `MemoryArm._process_baseline` in `r16p19/memory.py` | `COMMAND -> REALIZED` |
| B3 monolithic progress | `MemoryArm._process_baseline` | request/imagination/command/positive receipt/witness all `-> REALIZED` |
| POSTCHECK | `StrongRecoveryBaseline._process_postcheck` in `r16p19/phase3_baselines.py` | any one positive receipt `-> TRUE` |
| Persistence | `StrongRecoveryBaseline._process_persistence` | unique decision-tick positives increment `positive_streak`; `K` positives `-> TRUE` |
| Typed matched | `StrongRecoveryBaseline._process_typed` | two sensor identities `-> VERIFIED`; any later witness `-> REALIZED`; contradiction `-> FALSE` |
| B6 realization | `MemoryArm._process_typed` | two independent receipts `-> VERIFIED`; only a witness parented by the current command `-> REALIZED` |
| B6 invalidation | `MemoryArm._process_typed` and `_block_dependents` | contradiction after realization `-> INVALIDATED_REALIZATION`, blocks realized dependents and installs rollback/retry |
| No-provenance ablation | `B6NoProvenanceAdapter.process` | appends the current command ID to an otherwise unlinked witness |
| No-invalidation ablation | `B6NoInvalidationAdapter.process` | rewrites contradiction into an effect-free `IRRELEVANT` event |
| Fault/event exposure | `Phase3EventBroker` in `r16p19/phase3_event_broker.py` | standardized receipts expose no `physical_truth`, fault identity, or condition to an arm |
| Pair-prefix implementation | `run_matrix` and `run_chain_rollout` in `r16p19/phase3_runner.py` | the arm loop calls a new rollout, and each rollout constructs its own broker and independently replays the simulator prefix |

This mapping is also why the ablations have asymmetric interpretability:
`B6_NO_INVALIDATION` deletes a transition that C3 necessarily exercises,
whereas `B6_NO_PROVENANCE` relaxes a transition that neither C4 nor the
successful C7 retry actually exercises with an unlinked witness.

## 1. Why B2 and B3 fail under faults

`B2_COMMAND_PROGRESS` maps `COMMAND` directly to `REALIZED`. Consequently, a
suppressed first attempt under C1 is advanced as if its physical effect had
happened. C3 reversal and C4 single-view false positives cannot repair that
state, and the high-confidence imagination in C7 is not distinguished from
execution progress by the weak monolithic semantics.

`B3_MONOLITHIC` is even broader: `REQUEST`, `IMAGINE`, `COMMAND`, positive
observation, verification, or witness can all mark the effect `REALIZED`.
There is no typed separation between intent, prediction, evidence, and
physical realization.

On the 23 backend-valid formal units per condition, B2/B3 chain success was
zero on C1, C3, and C4 and 0.0435 on C7. Across the complete 150-row arm grids,
B2 and B3 produced 95 and 94 premature advances respectively. Their grounded
advance precision was only 0.469 and 0.492. This is the direct consequence of
command/imagination-as-progress, rather than a low-level action difference:
all arms were assigned the same frozen segments and budgets.

## 2. Why POSTCHECK improves, and exactly where it still fails

`POSTCHECK_RECOVERY` uses a binary state. One positive postcondition receipt
marks `TRUE`; negative evidence, contradiction, or timeout marks `FALSE`. This
is sufficient for C1 retry, C3 rollback, and C7 negative-evidence recovery.
On backend-valid units its success exactly matched B6 on C0, C1, C3, and C7:
0.9130, 0.9130, 0.8696, and 0.9130.

Its weakness is isolated to C4. A single agent-view false positive marks
`TRUE`, so all 23 backend-valid C4 units prematurely advanced and chain success
was zero. B6 succeeded on 22/23 (0.9565). Thus the B6-versus-POSTCHECK paired
advantage comes entirely from rejecting single-view evidence, not from a
general recovery advantage. Over the preregistered valid primary units, the
paired difference was 0.2391; exact McNemar-Holm adjusted `p` was
`9.5367431640625e-07`.

## 3. Why persistence K=2 is safe but inefficient

`PERSISTENCE_RECOVERY` rejects repeated frame digests and increments its
positive streak at most once per unique decision tick. It needs K positive
ticks before `TRUE`. Calibration correctly selected K=2: K=2/K=4/K=8 chain
success was 0.22/0.1267/0.0, so stricter persistence monotonically reduced
completion in this backend.

The formal mechanism is overconservative. It repeatedly reobserves or retries
to obtain a second positive tick; contact-sensitive predicates can disappear
during those operations, and the fixed retry/decision budgets are then
exhausted. On backend-valid formal units its C0/C1/C3/C4/C7 success was
0.7391/0.2609/0.0/0.2174/0.3043. Across the full grid it had an action cost of
206.1 steps, a repeated-loop rate of 0.533, and 80 timeouts. Persistence removes
false completion (precision 1.0) but pays for it with missed completions and
loops.

## 4. Why TYPED_MATCHED and B6 have identical task success

`TYPED_MATCHED_RECOVERY` retains typed intent/evidence states and requires two
sensor identities before `VERIFIED`. A realization witness then permits
`REALIZED`, but it does not require the witness to name the originating command
as a parent. Contradiction maps to generic `FALSE` and rollback.

B6 additionally requires:

- two independent receipt sources;
- a realization witness whose parent IDs contain the current command event;
- contradiction-driven `INVALIDATED_REALIZATION`;
- invalidation of realized dependents plus a rollback/retry route.

Despite these representational differences, TYPED_MATCHED and B6 had identical
success for every condition and every chain. Their valid-primary success was
0.9130; the paired difference and its 10,000-draw clustered bootstrap interval
were exactly 0.0 and `[0.0, 0.0]`; McNemar `p=1.0`. TYPED_MATCHED also used
fewer steps in every main condition. On the complete grid B6 used 164.44 steps
versus 158.59, and clean steps were 124.70 versus 120.97 (+3.09%). Therefore
the formal data show no incremental behavioral value for command-parent
provenance over the strong typed baseline.

## 5. What the two ablations actually identify

### B6_NO_PROVENANCE

The wrapper turns an unlinked realization witness into a command-linked one.
It did not change C4/C7 success: B6 and the ablation were both 0.7167 on all
rows and 0.9348 after excluding replay-backend failures.

The reason is visible in the broker schedule. C4 emits no realization witness,
so relaxing a parent check cannot make the missing witness appear. C7's
suppressed attempt produces negative evidence, while the later successful
retry produces an already linked witness. The tested conditions therefore did
not activate a behaviorally decisive unlinked-witness path. The zero ablation
effect cannot support a provenance mechanism claim.

### B6_NO_INVALIDATION

This wrapper transforms contradiction into an irrelevant event. Under C3 it
left the already realized effect advanceable after physical reversal. Among 23
backend-valid units it produced zero successes: 21 premature advances and two
timeouts. Full B6 succeeded on 20/23 (0.8696); including the seven invalid
backend units, the rates were 0.6667 versus 0.0.

Realization invalidation is therefore causally necessary for B6's C3 recovery.
It is not, however, incrementally superior to the simpler typed baseline:
TYPED_MATCHED's generic contradiction-to-rollback path also achieved 0.8696.
Because B6 had zero advantage over the strongest baseline before ablation, the
preregistered “50% advantage drop” paper claim is undefined and disallowed.

## 6. First-divergence intervention evidence

There were 115 observed decision-divergence units, but only 88 passed the
byte-identical prefix audit and were eligible for intervention. They generated
176 forced-decision replays. B6's natural decision tied the best forced
decision in 54/88 units, for a causal win rate of 0.6136, below the frozen 0.70
gate.

The forced-decision outcomes were:

| Forced decision | Rows | Immediate effect completion | Eventual chain completion | Irreversible failure |
|---|---:|---:|---:|---:|
| ADVANCE_TO_NEXT_SUBTASK | 88 | 0.2159 | 0.5682 | 0.4091 |
| REOBSERVE | 60 | 0.8833 | 0.9667 | 0.0000 |
| RETRY_CURRENT_EFFECT | 23 | 0.7826 | 0.6957 | 0.0000 |
| ROLLBACK_OR_REPLAN | 5 | 1.0000 | 1.0000 | 0.0000 |

These subsets are selected by which decisions occurred at a divergence and
must not be compared as randomized groups. They do show the local failure
mode: premature advance is often irreversible, while conservative recovery
avoids that error at additional action cost.

## 7. D1 delayed-receipt behavior

On the 23 backend-valid D1 units, B6, POSTCHECK, and TYPED_MATCHED all achieved
0.9130 chain success. B6 used 184.13 steps, versus 182.78 and 181.35. Its mean
reobserve/retry counts were 2.478/0.261, compared with 2.609/0.217 for
POSTCHECK and 2.739/0.174 for TYPED_MATCHED. Delayed receipt therefore exposes
a small calibration/cost difference, not a B6 success advantage.

## 8. Why the formal pairing correctness gate failed

The plan required one shared broker and byte-identical events before the first
memory-decision divergence. The implementation creates a deterministic broker
inside each arm rollout and independently re-executes the same simulator
segment for each arm. Event identifiers are deterministic, but the
contact-sensitive MuJoCo trajectory is not guaranteed to reproduce
bit-for-bit across those independent executions.

All first divergences occurred at decision index 0. In 27/150 paired units,
the executed-action hashes were identical but simulator-state and/or event
prefix hashes differed across arms before that decision. The failures were
concentrated in contact-sensitive units (14 in `B2_BOWL_IN_TO_RELEASED`, 11 in
`S2_MOKA_GRASP_TO_ON_STOVE`, and two in `B1_BOWL_GRASP_TO_DRAWER`). This is a
real information-fairness violation, not a presentation-only audit error.

The snapshot files contain flattened MuJoCo state, but that does not make the
separately executed controller/contact trajectories a synchronized paired
experiment. Fixing this would require a newly frozen experiment version with a
truly synchronous shared prefix/branch protocol. It is not valid to repair the
formal result after observing demo 40--49.

The separate video pass provides a secondary reproducibility check. Of 496
successfully rendered requests, four independently re-executed rollouts changed
the frozen scientific outcome from failure to success. All four involved B2 or
B3 and contact-sensitive B2/S2 chain units. This does not alter the main JSONL
or its statistics, but it independently confirms that a later renderer replay
cannot be treated as a byte-identical copy of the formal rollout.

## 9. Scientific conclusion

The downstream diagnostic matrix reveals useful mechanisms but does not
validate the original incremental B6 claim:

- typed evidence prevents the catastrophic weak-baseline premature advances;
- two-view confirmation specifically fixes POSTCHECK's C4 vulnerability;
- invalidation is necessary for C3 recovery;
- persistence trades false-completion safety for severe timeout cost;
- command-parent provenance adds no observed success over TYPED_MATCHED;
- replay competence and pre-divergence pairing both fail their frozen gates.

No VLA, actor, learned verifier, benchmark-generalization, or paper-level
mechanism claim follows from these results.

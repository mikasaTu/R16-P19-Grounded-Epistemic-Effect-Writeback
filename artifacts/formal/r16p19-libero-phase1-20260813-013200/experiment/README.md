# R16-P19 LIBERO Phase-1 experiment report

Final status: **BLOCKED_BY_ACTOR**

## Evidence boundary

This is an actor-free epistemic trace test plus a tiny privileged-state BC causal sanity actor on two official LIBERO-10 tasks. It is not Pi0.5, not a large VLA experiment, and not evidence of VLA improvement.

## Actor-free result

- Correctness gate: PASS.
- B6 decision accuracy: 1.000.
- B6 false completion rate: 0.000.
- B6 contradiction recovery recall: 1.000.
- B6 accepted aliased same-frame evidence: 0.
- Maximum resident slots: 32; dangling parents: 0.

## Shared actor gate

- Actor: retrieval_augmented_tiny_mlp.
- Minimum per-effect clean success: 0.450 (required 0.800).
- Full-task clean success: 0.600.

## Closed-loop result

Closed-loop arm comparison was not interpreted because no actor passed the 0.80 per-effect competence gate.

## Interpretation

`PASS_PHASE1` means only that this narrow LIBERO mechanism gate passed. `BLOCKED_BY_ACTOR` leaves the actor-free result descriptive but forbids causal interpretation of memory-conditioned task success.

# Future learned effect verifier readiness note

## Decision

**NOT READY. Do not start a learned effect verifier from this Phase-3 result.**

This note satisfies the plan's readiness deliverable; it is not a proposal for
a new mechanism and no verifier, actor, Mem-0, ACT, Pi0.5, or world model was
trained or launched.

## Blocking evidence

1. The replay qualification gate failed: none of the six candidate chains met
   the frozen segment/chain thresholds.
2. The formal replay-only gate also failed, with chain replay success
   `0.50/0.48/0.44`.
3. Seven of 30 chain-demo units had invalid replay segments, yielding 35
   `REPLAY_BACKEND_FAILURE` rows per main arm.
4. The formal implementation did not provide a truly synchronous shared
   physical prefix: 27/150 paired units differed in simulator/event prefix
   before the first memory decision.
5. B6 had zero incremental success over `TYPED_MATCHED_RECOVERY`, with paired
   difference `0` and clustered 95% interval `[0, 0]`.

These blockers mean a learned verifier would currently be trained/evaluated
against an incompetent replay carrier and a violated paired-information
contract. Its result would not identify verifier quality separately from those
errors.

## Preconditions for any future readiness review

The present phase does not authorize or implement these actions. A future,
newly preregistered phase would first need independently demonstrated replay
competence, a truly shared byte-identical pre-decision physical/event prefix,
an untouched evaluation split, and a frozen verifier-specific label and
calibration contract. The current formal bank cannot be reused as unseen data.

Until those prerequisites are met, the correct action is to preserve this
negative/blocked evidence rather than add model capacity.

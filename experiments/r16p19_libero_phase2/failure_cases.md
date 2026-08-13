# Phase-2 qualification failure cases

All ten failed qualification rollouts are retained below. Each failure used the
frozen executor, exhausted the preregistered four attempts (initial attempt plus
three retries), and was counted as a repeated-effect loop. The table reports the
first unmet effect; downstream effects were not attempted after that blocker.

| Task | Init | First failed effect | Terminal reason | Steps | Final position error (m) | Final orientation error (rad) | Video |
|---|---:|---|---|---:|---:|---:|---|
| stove_moka | 63 | MOKA_GRASPED | MAX_ACTION_STEPS | 700 | 0.0762 | 0.0219 | [video](qualification_failure_videos/qualification_stove_moka_init_63.mp4) |
| stove_moka | 72 | STOVE_TURNED_ON | EFFECT_CHUNK_LIMIT | 640 | 0.0115 | 0.1536 | [video](qualification_failure_videos/qualification_stove_moka_init_72.mp4) |
| stove_moka | 77 | STOVE_TURNED_ON | EFFECT_CHUNK_LIMIT | 640 | 0.0123 | 0.1550 | [video](qualification_failure_videos/qualification_stove_moka_init_77.mp4) |
| stove_moka | 79 | STOVE_TURNED_ON | EFFECT_CHUNK_LIMIT | 640 | 0.0124 | 0.1557 | [video](qualification_failure_videos/qualification_stove_moka_init_79.mp4) |
| bowl_drawer | 66 | BOWL_IN_BOTTOM_DRAWER | EFFECT_CHUNK_LIMIT | 699 | 0.0476 | 0.0333 | [video](qualification_failure_videos/qualification_bowl_drawer_init_66.mp4) |
| bowl_drawer | 68 | BOWL_GRASPED | EFFECT_CHUNK_LIMIT | 640 | 0.0350 | 0.0590 | [video](qualification_failure_videos/qualification_bowl_drawer_init_68.mp4) |
| bowl_drawer | 71 | BOTTOM_DRAWER_CLOSED | MAX_ACTION_STEPS | 700 | 0.0300 | 0.0471 | [video](qualification_failure_videos/qualification_bowl_drawer_init_71.mp4) |
| bowl_drawer | 72 | BOWL_GRASPED | EFFECT_CHUNK_LIMIT | 640 | 0.0917 | 0.2458 | [video](qualification_failure_videos/qualification_bowl_drawer_init_72.mp4) |
| bowl_drawer | 74 | BOWL_GRASPED | EFFECT_CHUNK_LIMIT | 640 | 0.1193 | 0.2560 | [video](qualification_failure_videos/qualification_bowl_drawer_init_74.mp4) |
| bowl_drawer | 77 | BOWL_GRASPED | EFFECT_CHUNK_LIMIT | 640 | 0.1590 | 0.3386 | [video](qualification_failure_videos/qualification_bowl_drawer_init_77.mp4) |

## Reverse explanation

Eight failures occur while establishing a switch or grasp contact. The three
stove-switch failures end within about 1.2 cm of a demonstrated waypoint, but
with the switch still off. This is a contact-mode alias: pose proximity does not
encode whether the end effector is on the effective side of the control or is
applying a useful constrained motion. The videos show the arm repeating near
the same switch posture.

The grasp failures expose a retry-state problem. `action_chunk` applies the
fixed 1 cm retry offset and then projects the current, already-failed end-effector
state onto the nearest waypoint. Because template calibration retained one
template for each grasp effect, all four attempts keep the same template rank.
There is no forced pre-grasp reacquisition in the frozen executor. Failed traces
therefore finish at waypoint 47/48 for the moka or waypoints 18--27/28 for the
bowl while the grasp predicate remains false. This account is supported by
code, trace, and video; it is not a new causal ablation.

The remaining two failures localize the same limitation to placement and drawer
contact. Init 66 transports the bowl but stalls before the inside-drawer
predicate. Init 71 satisfies grasp, placement, and release, then repeatedly
fails to close the drawer. The executor has geometry and demonstrated
feed-forward action, but no explicit representation of achieved contact mode or
a deterministic reset to a recoverable contact-entry state.

## Scope

These failures happen in a clean executor-only qualification before any memory
arm is activated. They block, but do not reject or validate, the proposed B6
memory mechanism. The frozen protocol forbids post-qualification tuning or a
replacement executor in this phase.

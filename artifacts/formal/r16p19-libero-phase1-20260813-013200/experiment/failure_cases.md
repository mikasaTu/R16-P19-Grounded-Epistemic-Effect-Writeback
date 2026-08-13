# Failure cases

Final status: **BLOCKED_BY_ACTOR**

## Actor competence

Failed full-task rollouts: 56.

- stove_moka / init 0 / retrieval_augmented_tiny_mlp: unreached effects `MOKA_GRASPED, MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.
- stove_moka / init 2 / retrieval_augmented_tiny_mlp: unreached effects `MOKA_GRASPED, MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.
- stove_moka / init 3 / retrieval_augmented_tiny_mlp: unreached effects `MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.
- stove_moka / init 8 / retrieval_augmented_tiny_mlp: unreached effects `MOKA_GRASPED, MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.
- stove_moka / init 14 / retrieval_augmented_tiny_mlp: unreached effects `MOKA_GRASPED, MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.
- bowl_drawer / init 0 / retrieval_augmented_tiny_mlp: unreached effects `BOWL_IN_BOTTOM_DRAWER, BOWL_RELEASED_IN_DRAWER, BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 3 / retrieval_augmented_tiny_mlp: unreached effects `BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 5 / retrieval_augmented_tiny_mlp: unreached effects `BOWL_IN_BOTTOM_DRAWER, BOWL_RELEASED_IN_DRAWER, BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 10 / retrieval_augmented_tiny_mlp: unreached effects `BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 11 / retrieval_augmented_tiny_mlp: unreached effects `BOWL_IN_BOTTOM_DRAWER, BOWL_RELEASED_IN_DRAWER, BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 13 / retrieval_augmented_tiny_mlp: unreached effects `BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 14 / retrieval_augmented_tiny_mlp: unreached effects `BOWL_GRASPED, BOWL_IN_BOTTOM_DRAWER, BOWL_RELEASED_IN_DRAWER, BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 16 / retrieval_augmented_tiny_mlp: unreached effects `BOWL_GRASPED, BOWL_IN_BOTTOM_DRAWER, BOWL_RELEASED_IN_DRAWER, BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 17 / retrieval_augmented_tiny_mlp: unreached effects `BOWL_IN_BOTTOM_DRAWER, BOWL_RELEASED_IN_DRAWER, BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 18 / retrieval_augmented_tiny_mlp: unreached effects `BOWL_RELEASED_IN_DRAWER, BOTTOM_DRAWER_CLOSED`, 600 steps.
- bowl_drawer / init 19 / retrieval_augmented_tiny_mlp: unreached effects `BOWL_IN_BOTTOM_DRAWER, BOWL_RELEASED_IN_DRAWER, BOTTOM_DRAWER_CLOSED`, 600 steps.
- stove_moka / init 0 / nearest_demo_phase_script: unreached effects `STOVE_TURNED_ON, MOKA_GRASPED, MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.
- stove_moka / init 1 / nearest_demo_phase_script: unreached effects `STOVE_TURNED_ON, MOKA_GRASPED, MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.
- stove_moka / init 2 / nearest_demo_phase_script: unreached effects `STOVE_TURNED_ON, MOKA_GRASPED, MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.
- stove_moka / init 3 / nearest_demo_phase_script: unreached effects `STOVE_TURNED_ON, MOKA_GRASPED, MOKA_ON_STOVE, MOKA_RELEASED_ON_STOVE`, 600 steps.

## Memory-conditioned closed loop

Failed rollouts: 0.


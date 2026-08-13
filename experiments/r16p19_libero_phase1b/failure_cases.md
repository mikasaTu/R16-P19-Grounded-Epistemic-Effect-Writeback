# Phase-1B failure cases

Both clean qualification gates ran all 40 preregistered init/task cells.
Every persisted qualification failure has a video under the raw artifact bundle.

## Primary

- Full-task success: 18/40 (0.450).
- Minimum per-effect success: 0.400 (required 0.800).
- Repeated effect-chunk limit: 22/40 (0.550).

## Per-effect fallback

- Full-task success: 24/40 (0.600).
- Minimum per-effect success: 0.500 (required 0.800).
- Repeated effect-chunk limit: 16/40 (0.400).

The effect name in each `EFFECT_CHUNK_LIMIT:<effect>` record localizes the first
unreached physical predicate. It is evidence of actor execution failure, but it does
not by itself distinguish grasp geometry, gripper timing, endpoint control, or a
combination of those low-level causes.

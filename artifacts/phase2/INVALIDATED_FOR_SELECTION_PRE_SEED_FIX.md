# Pre-seed Phase-2 development results

Status: **`INVALIDATED_FOR_SELECTION_PRE_SEED_FIX`**.

Every file below this directory is retained as raw development history, but no
metric from this directory is used to select or freeze the Phase-2 executor.
The first development implementation reset official LIBERO states without
resetting NumPy and environment RNG state.  Because LIBERO uses soft resets,
the same task/init result could depend on which rollouts had preceded it in the
process.  This invalidated cross-run comparisons even though task state bytes
were identical.

The correction was to seed NumPy and the environment before every reset with
`1619 + 1000 * task_ordinal + init_index`.  Two independent post-fix executions
of init 40 were then byte-identical after removing only the diagnostic gate
name:

- `stove_moka`: `400d42db19705a82d7d0373decfc728971cbde38f37a95ff71e4f4b6253bfd1e`
- `bowl_drawer`: `05d91939dc985679921b8370c9f0a32d4049241d546340339f8a6a7ae60b49e8`

Selection-valid development evidence is under `artifacts/phase2_seeded/`.

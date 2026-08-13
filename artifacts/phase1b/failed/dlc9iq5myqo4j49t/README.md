# Failed fallback training run `dlc9iq5myqo4j49t`

This PAI job failed after completing five of eight per-effect actors.  The
sixth effect, `BOWL_IN_BOTTOM_DRAWER`, contains 12,552 positive and zero
negative gripper targets in the frozen train split.  The original generic
normalization implementation rejected any single-class gripper dataset and
raised `ValueError: both gripper classes are required` before training that
effect.

The files here are retained as failed-run evidence.  None of these partial
checkpoints is eligible for fallback qualification, formal evaluation, or the
800-rollout matrix.  The scientifically neutral fix and restart policy are
recorded in
`experiments/r16p19_libero_phase1b/implementation_fix_log.json`.  The corrected
run starts all eight actors from step 0 in the isolated `v2` namespace.


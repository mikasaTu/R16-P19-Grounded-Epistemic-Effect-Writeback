# Phase-6 S1 test report

## S1 focused tests

Command:

```bash
PYTHONPATH=. pytest -q experiments/r16p19_phase6/tests/test_s1.py
```

Result: `4 passed`.

Coverage includes the TPR-target threshold rule, calibration-only weighted-soft fitting with zero calibration false upgrades, exact frozen Phase-5 injection-frame indexing, and a fail-closed guard that rejects any `formal` path as a calibration input.

## Full repository tests

Command in the repository's qualified LIBERO environment:

```bash
PYTHONPATH=. /mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python -m pytest -q
```

Result: `74 passed in 150.02s`.

The system Python was also attempted first and failed during historical-test collection because it lacks `torch`, `imageio`, and `mujoco`. No test assertion ran or failed in that environment; the qualified environment above is the authoritative result.

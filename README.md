# R16-P19 LIBERO Phase-1 validation

This repository implements the actor-decoupled Phase-1 gate for Grounded
Epistemic Effect Writeback on two frozen official LIBERO-10 tasks. It contains
an actor-free trace test and a retrieval-augmented tiny state behavior-cloning
actor. It does not train or evaluate a VLA.

The formal experiment is designed for PAI DLC. Local execution is limited to
unit tests, static validation, a CPU trace smoke, and an optional one-batch GPU
smoke.

```bash
export PYTHONPATH="$PWD:/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/LIBERO-r16p19-official-8f1084e"
/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python -m pytest -q
/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python pipeline.py static-check
/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python pipeline.py retention-test
```

Formal outputs are written below
`experiments/r16p19_libero_phase1/` in the PAI artifact directory and include
the preregistered deliverable set plus runtime provenance and readiness notes.


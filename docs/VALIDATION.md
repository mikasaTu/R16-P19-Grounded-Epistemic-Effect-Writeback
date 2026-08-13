# Validation record

This file separates scientific run evidence from later publication-bundle
checks. A failed environment setup command is not treated as a failed
scientific assertion.

## Formal PAI execution

The authoritative formal job was `dlc6sr1fu466f1g9`:

- PAI control-plane status: `Succeeded`;
- run ID: `r16p19-libero-phase1-20260813-013200`;
- duration: 764 seconds;
- worker pod: `Succeeded`;
- runtime: NVIDIA A800-SXM4-80GB, CUDA available;
- complete optimizer steps: 3,000;
- complete checkpoints: 1,000, 2,000 and 3,000;
- actor-free gate: pass;
- terminal scientific status: `BLOCKED_BY_ACTOR`;
- `RUN_COMPLETE.json`: `run_complete=true`;
- all 24 files in the formal output `SHA256SUMS`: OK.

The formal workload completed 40 learned-actor and 40 fallback clean simulator
rollouts. It did not run the 800 memory-conditioned rollouts because neither
actor passed the preregistered competence gate.

Machine-readable evidence:

- `pai/control-plane/r16p19-libero-phase1-20260813-013200/getjob-latest.json`
- `artifacts/formal/r16p19-libero-phase1-20260813-013200/RUN_COMPLETE.json`
- `artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment/metrics.json`
- `artifacts/formal/r16p19-libero-phase1-20260813-013200/run.log`

## Development-time checks before formal submission

The source checkout was tested with:

- six unit tests: passed;
- frozen source/data/hash and dependency validation: passed;
- synthetic checkpoint retention/resume test: passed;
- one-demo-per-task simulator trace and one-batch actor smoke: passed during
  development;
- 100-demo physical effect-label audit: completed;
- checkpoint model/optimizer/scheduler/RNG/global-step round trip: passed.

The formal run independently repeated the frozen-input validation, retention
test, physical label extraction, actor-free trace gate, training, checkpointing
and simulator competence rollouts.

## Publication-checkout revalidation

On 2026-08-13, after assembling this GitHub bundle:

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python \
  -m pytest -q --junitxml=artifacts/local-validation/pytest.xml
...... [100%]
6 passed in 0.59s
```

The JUnit XML is committed at `artifacts/local-validation/pytest.xml`.
An earlier invocation with the DSW system Python stopped during collection
because that interpreter does not contain PyTorch; no project test executed in
that invocation. Re-running with the exact Python pinned by the formal launcher
produced the passing record above.

The synthetic retention test returned:

```json
{
  "incomplete_preserved": true,
  "retained_steps": [10000, 13000, 14000, 16000],
  "rng_resume_exact": true,
  "status": "CHECKPOINT_RETENTION_OK"
}
```

`pipeline.py static-check` initially could not report an experiment commit
because the destination GitHub repository was empty and therefore had no
`HEAD`. After restoring the exact formal source history, it passed and reported
experiment commit `ae362efeba68643ab4dd2a99cfd295c72a9cbdcc`, LIBERO commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01`, the expected dependency versions,
all six frozen input hashes, UID:GID `2254:2254`, and an available A800. Its
machine-readable result is stored in
`artifacts/local-validation/static-check.json`.

An additional simulator smoke from the publication checkout could not create a
local GL context in the current DSW container:

- OSMesa: no `libOSMesa` implementation was available to PyOpenGL;
- EGL: the container exposed four A800s through `nvidia-smi`, but had no GLVND
  EGL vendor descriptor, so MuJoCo discovered zero EGL devices.

This is recorded as an environment limitation, not a passing smoke and not a
scientific failure. The authoritative PAI container had working EGL and
successfully completed all 80 competence rollouts.

## First failed PAI attempt

Job `dlc1rycl56e4nvac` failed before any optimizer step because official LIBERO
attempted an interactive first-run configuration prompt on a clean batch
worker. Batch stdin returned EOF. The fix added a deterministic checked-in
LIBERO config and exported `LIBERO_CONFIG_PATH` from the launcher. The
replacement job then passed the same point and completed.

Evidence is retained under `artifacts/failed/dlc1rycl56e4nvac/` and
`pai/control-plane/r16p19-libero-phase1-20260813-012000/`.

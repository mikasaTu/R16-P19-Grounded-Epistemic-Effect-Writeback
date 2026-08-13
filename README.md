# R16-P19 Grounded Epistemic Effect Writeback

This repository is the complete publication bundle for the actor-decoupled
Phase-1 validation of Grounded Epistemic Effect Writeback on two frozen
official LIBERO-10 tasks. It contains the exact experiment code, frozen
protocols, tests, PAI submission contracts, control-plane evidence, raw
trace-level outputs, checkpoints, failure diagnosis, and the final report.

The scientific terminal status is **`BLOCKED_BY_ACTOR`**:

- the actor-free correctness gate passed;
- the learned tiny state-BC actor did not reach the preregistered clean
  per-effect competence threshold of 0.80;
- the preregistered nearest-demo fallback also failed;
- therefore the 800 memory-conditioned closed-loop rollouts and paired
  bootstrap were correctly **not run**.

This is a mechanism-level positive result and a behavior-level blocked result.
It is not evidence that R16-P19 improves end-to-end LIBERO task success, Pi0.5,
or any large VLA.

## Headline results

| Gate | Result | Key evidence |
|---|---:|---|
| Actor-free trace gate | PASS | 1,120 cases; all correctness gates true |
| B6 false completion | 0.000 | B3 was 0.500 |
| B6 contradiction recovery recall | 1.000 | B4 and B5 were 0.000 |
| Evidence alias acceptance | 0 | 20 alias challenges were rejected |
| Dangling parents / transition violations | 0 / 0 | Maximum resident slots was exactly 32 |
| Learned actor clean full-task success | 24/40 = 0.600 | Minimum per-effect success was 0.450 |
| Fallback clean full-task success | 0/40 | Minimum per-effect success was 0 |
| 800 closed-loop factorial rollouts | NOT_RUN | Competence gate failed, as preregistered |
| Formal PAI job | Succeeded | `dlc6sr1fu466f1g9`, 764 seconds |

The complete interpretation is in
[docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md). Machine-readable metrics
are in
[metrics.json](artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment/metrics.json).

## Experiment scope

Selected official LIBERO-10 tasks:

1. `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
2. `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`

Each task uses 50 official demonstrations, frozen as:

- train: `demo_0`–`demo_29`;
- calibration: `demo_30`–`demo_39`;
- trace-test: `demo_40`–`demo_49`;
- simulator evaluation: init indices 0–19.

The eight fault conditions are C0 clean, C1 command no-op, C2 delayed effect,
C3 post-realization reversal, C4 single-camera false positive, C5 evidence-ID
alias, C6 32-slot resident-memory pressure, and C7 imagined success followed by
observed failure.

The seven memory arms are B1 sliding history, B2 command-as-progress, B3
monolithic writeback, B4 typed states without recovery, B5 typed states plus
verification without recovery, B6 full R16-P19, and B7 oracle effect ledger.

## Repository contents

```text
r16p19/                     Core epistemic memory, trace gate, actor and rollout code
tests/                      Unit tests for transitions, provenance and checkpoints
experiments/                Frozen ontology, manifests, split, faults and preregistration
launch/                     Exact formal PAI workload launcher
artifacts/formal/           Successful formal run, including raw JSONL outputs and logs
artifacts/failed/           First failed PAI attempt and root-cause evidence
artifacts/checkpoints/      Complete step 1000/2000/3000 model checkpoints
artifacts/local-validation/ Publication-checkout validation records
pai/control-plane/          Requested/resolved contracts and PAI GetJob readbacks
pai/controller-patches/     Exact patches applied to the external PAI registry controller
pai/registry/               Pinned profile and payload used by the controller
docs/                       Protocol, report, provenance and validation notes
```

The formal scientific output directory has its own `SHA256SUMS`; the repository
also includes `artifacts/BUNDLE_SHA256SUMS` for all uploaded run artifacts and
checkpoints.

## Exact provenance

- Formal experiment source commit: `ae362efeba68643ab4dd2a99cfd295c72a9cbdcc`
- Initial implementation commit: `773a02e653e9b0ecf4d040e6b31891528f4825f0`
- Frozen official LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Formal run ID: `r16p19-libero-phase1-20260813-013200`
- PAI job ID: `dlc6sr1fu466f1g9`
- W&B run: [63azf17y](https://wandb.ai/chen_jian-cj-workspace/r16p19-libero-phase1/runs/63azf17y)
- Runtime identity: `2254:2254`
- Hardware contract: 2×A800, GPU0 active and GPU1 reserved idle

The parent commits preserve the exact code used by the formal run. Later
publication commits add only uploaded evidence and documentation unless stated
otherwise.

## Validation

The core checks use the same pinned LIBERO Python as the formal job:

```bash
LIBERO_PYTHON=/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python
"$LIBERO_PYTHON" -m pytest -q
"$LIBERO_PYTHON" pipeline.py static-check
"$LIBERO_PYTHON" pipeline.py retention-test
```

The formal simulator validation requires the pinned LIBERO source, official
datasets, MuJoCo/robosuite dependencies, and a working EGL or OSMesa backend.
The original experiment intentionally freezes absolute CPFS paths. Reproduce
the same mount layout, or adapt the paths and treat that as a new experiment.

The publication checkout was re-tested with six unit tests passing. The
synthetic checkpoint retention test passed. See
[docs/VALIDATION.md](docs/VALIDATION.md) for exact commands, formal PAI evidence,
and the headless-GL limitation encountered during the additional checkout
smoke.

## Integrity and secrets

No dataset binaries or official LIBERO source are vendored; their exact commits
and SHA-256 digests are recorded in the manifests. No credentials are included.
Control-plane exports retain only the literal `<redacted>` placeholder for the
W&B API key. W&B internal cache files were intentionally excluded; the public
run metadata and URL are preserved.

Verify the uploaded artifacts with (the bundle manifest covers every artifact
other than itself):

```bash
(cd artifacts && sha256sum -c BUNDLE_SHA256SUMS)
(cd artifacts/formal/r16p19-libero-phase1-20260813-013200/experiment \
  && sha256sum -c SHA256SUMS)
```

## Related records

- Original idea and planning document:
  [Feishu wiki](https://icnbwz7kd1ui.feishu.cn/wiki/AfN7wFfFWi7dBOkBBtucoroanff)
- Final Feishu experiment report:
  [实验报告](https://icnbwz7kd1ui.feishu.cn/wiki/Wr28wjd1aivlpLkU0ovcDTWdnWf)
- Original Step-1 protocol:
  [docs/original-phase1-protocol.txt](docs/original-phase1-protocol.txt)

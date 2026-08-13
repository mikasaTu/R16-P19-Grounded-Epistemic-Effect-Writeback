# R16-P19 Grounded Epistemic Effect Writeback

This repository contains the complete Phase-1, Phase-1B, and Phase-2 validation
bundle for Grounded Epistemic Effect Writeback on two frozen official
LIBERO-10 tasks: frozen protocols, implementation, tests, PAI submission and
control-plane evidence, raw JSONL, failure videos, mechanism diagnosis, and
reports.

The latest scientific terminal status is **`BLOCKED_BY_EXECUTOR_V3`**. The
frozen deterministic, non-neural `RetargetedGeometricSkillExecutor` completed
its 2×20 clean qualification but missed every combined competence gate: minimum
per-effect success was 0.70 (required 0.90), bowl full-task success was 0.70
(required 0.80), and repeated-loop rate was 0.25 (maximum 0.10). The PAI job
succeeded; this is a scientific executor failure, not an infrastructure error.

The stop rule therefore kept formal init 0--19, the 800-rollout memory matrix,
causal replay, and paired bootstrap unobserved. Phase-2 neither validates nor
rejects the B6 memory mechanism. Earlier `BLOCKED_BY_ACTOR` and
`BLOCKED_BY_ACTOR_V2` results remain predecessor phases, not the current status.

## Headline results

| Phase-2 gate | Result | Key evidence |
|---|---:|---|
| Frozen qualification | FAIL | 30/40 full-task successes; minimum effect 0.70 |
| stove_moka full success | 16/20 = 0.80 | Meets the per-task 0.80 threshold |
| bowl_drawer full success | 14/20 = 0.70 | Below the per-task 0.80 threshold |
| Repeated-effect loops | 10/40 = 0.25 | Maximum allowed was 0.10 |
| Qualification failure videos | 10/10 retained | Every failed rollout has an MP4 |
| Formal clean gate | NOT_RUN | Qualification failed; formal init access is zero |
| Closed-loop matrix | 0/800 | Preregistered stop rule applied |
| PAI qualification job | Succeeded | `dlceyy7m2jhmxc4o`, 231 seconds, one RTX 4090 |

The complete interpretation is in
[docs/PHASE2_STEP3_EXPERIMENT_REPORT.md](docs/PHASE2_STEP3_EXPERIMENT_REPORT.md).
Machine-readable results are in
[behavior_summary.json](experiments/r16p19_libero_phase2/behavior_summary.json)
and [mechanism_mediation.json](experiments/r16p19_libero_phase2/mechanism_mediation.json).

## Phase-2 scope

Phase-2 uses demo 0--29 for template extraction, demo 30--39 for calibration,
init 40--59 for development, and untouched init 60--79 for qualification. The
executor is deterministic and memory-independent. Its frozen input explicitly
excludes memory state, fault identity, effect truth, reward, task success,
future state, and init index.

The complete Phase-2 protocol and terminal evidence are under
`experiments/r16p19_libero_phase2/`; PAI artifacts and all ten failure videos
are under `artifacts/phase2_pai/` and the experiment directory respectively.

## Phase-1 predecessor scope

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

## Phase-2 exact provenance

- Qualification source commit: `8963f8cb3b10201095a47c48cec13ce11b0832f0`
- Frozen executor source commit: `136e8923829c0436ca27755078a609f91bcf75a5`
- Selected executor manifest SHA-256: `384957cae10f96b6a53645a555e53d38876573a06a452fc93af0d95b4f254b6b`
- Selected template manifest SHA-256: `6a123a334bb901da880baf3b14e72576015bc013ed1f3224e9dd2c6d3c49d431`
- Official LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Qualification run / JobId: `r16p19-p2-qualification-20260813-194500` / `dlceyy7m2jhmxc4o`
- Runtime identity: `2254:2254`
- Hardware contract: one rendering GPU; observed NVIDIA GeForce RTX 4090
- Platform restarts / automatic fault tolerance: 0 / disabled

## Phase-1 exact provenance

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

The latest checkout was re-tested with 23 unit tests passing. The
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
- Phase-2 step3 plan:
  [step3](https://icnbwz7kd1ui.feishu.cn/wiki/CP8qwoVcFiyPrMkLS7KcYNx4ngd)
- Phase-2 step3 experiment report:
  [实验报告](https://icnbwz7kd1ui.feishu.cn/wiki/IBc9wR2Mai8Cv1ks6VecHuqTnCk)
- Phase-1 Feishu experiment report:
  [实验报告](https://icnbwz7kd1ui.feishu.cn/wiki/Wr28wjd1aivlpLkU0ovcDTWdnWf)
- Original Step-1 protocol:
  [docs/original-phase1-protocol.txt](docs/original-phase1-protocol.txt)

# R16-P19 Grounded Epistemic Effect Writeback

This repository contains the complete Phase-1, Phase-1B, Phase-2, and Phase-3 validation
bundle for Grounded Epistemic Effect Writeback on two frozen official
LIBERO-10 tasks: frozen protocols, implementation, tests, PAI submission and
control-plane evidence, raw JSONL, failure videos, mechanism diagnosis, and
reports.

The latest machine-readable terminal status is **`BLOCKED_BY_IMPLEMENTATION`**.
Phase-3 independently failed both the qualification replay and formal
replay-only competence gates. Under the user's preregistered override, all
downstream diagnostics nevertheless completed: 900 main rollouts, 180 delayed-
receipt rollouts, 176 first-divergence interventions, 10,000-draw clustered
bootstrap, exact paired tests, and 90 mechanism-ablation rollouts.

B6 tied the strongest `TYPED_MATCHED_RECOVERY` baseline exactly: valid-primary
chain success 0.9130 versus 0.9130, paired difference 0.0, clustered 95% CI
`[0.0, 0.0]`, and McNemar `p=1.0`. In addition, 27/150 paired units violated
the byte-identical pre-decision event/state-prefix requirement even though
their frozen action hashes matched. The experiment therefore supports no B6
incremental-value, VLA, benchmark-generalization, or paper-level mechanism
claim. Phase-2 `BLOCKED_BY_EXECUTOR_V3`, Phase-1B `BLOCKED_BY_ACTOR`, and
Phase-1 trace-level PASS remain predecessor evidence.

## Headline results

| Phase-3 item | Result | Key evidence |
|---|---:|---|
| Qualification replay | FAIL | 0/6 candidate chains eligible |
| Formal replay-only gate | FAIL | chain success 0.50 / 0.48 / 0.44 |
| Full downstream execution | COMPLETE | 900 main + 180 D1 + 176 causal + 90 ablation |
| B6 vs best strong baseline | NO ADVANTAGE | paired difference 0.0; 95% CI `[0,0]` |
| B6 vs POSTCHECK | diagnostic improvement | +0.2391 valid-primary; isolated to C4 |
| C3 no-invalidation ablation | diagnostic collapse | raw success 0.667 → 0.000 |
| First-divergence causal gate | FAIL | 54/88 = 0.6136, required 0.70 |
| Pair-prefix information fairness | FAIL | 27/150 units differed before decision |
| Formal PAI execution | Succeeded | `dlc1o37keilxr3sr`; 18,485 s; 2×A800 carrier, GPU0 visible |
| Video policy | diagnostic complete | 748 requests; 496 videos; 252 invalid-snapshot errors; 4 outcome mismatches |
| Protected B6 source | PASS | frozen SHA256 `4992462e...` |

The complete Phase-3 interpretation is in
[EXPERIMENT_REPORT_ZH.md](experiments/r16p19_libero_phase3/EXPERIMENT_REPORT_ZH.md),
with code-path causality in
[MECHANISM_REVERSE_ENGINEERING.md](experiments/r16p19_libero_phase3/MECHANISM_REVERSE_ENGINEERING.md).

## Phase-2 predecessor results

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

The latest checkout was re-tested with the complete 35-test suite passing;
the five specified Phase-3 test files contributed 12 passing tests. The
synthetic checkpoint retention test also passed. See
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
(cd experiments/r16p19_libero_phase3 && sha256sum -c SHA256SUMS)
(cd experiments/r16p19_libero_phase3 && sha256sum -c BUNDLE_SHA256SUMS)
```

## Related records

- Original idea and planning document:
  [Feishu wiki](https://icnbwz7kd1ui.feishu.cn/wiki/AfN7wFfFWi7dBOkBBtucoroanff)
- Phase-2 step3 plan:
  [step3](https://icnbwz7kd1ui.feishu.cn/wiki/CP8qwoVcFiyPrMkLS7KcYNx4ngd)
- Phase-2 step3 experiment report:
  [实验报告](https://icnbwz7kd1ui.feishu.cn/wiki/IBc9wR2Mai8Cv1ks6VecHuqTnCk)
- Phase-3 step4 plan:
  [step4](https://icnbwz7kd1ui.feishu.cn/wiki/SqxXwnTIFiistlk1bOrcprPWnmb)
- Phase-3 step4 experiment report:
  [实验报告](https://icnbwz7kd1ui.feishu.cn/wiki/DvI3wxPQ7ixoSdkA9kncspgdnUc)
- Phase-1 Feishu experiment report:
  [实验报告](https://icnbwz7kd1ui.feishu.cn/wiki/Wr28wjd1aivlpLkU0ovcDTWdnWf)
- Original Step-1 protocol:
  [docs/original-phase1-protocol.txt](docs/original-phase1-protocol.txt)

# Audited CPU execution and reproduction

Working directory: dev14 `/mnt/cpfs/zbl-cpfs-new/share/leon/codex-archives/Explore-claude-local-worktrees/r16p19-step8-s1b-20260907`.

Original `run_step9.py` is retired. The original C0 file is unchanged; these commands are retrospective repairs and do not restore preregistration. No GPU/PAI/new rollout or verifier weight fitting occurs.

```bash
PYTHONPATH=. python -B experiments/r16p19_phase8/audit_calibration.py
PYTHONPATH=. python -B experiments/r16p19_phase8/audit_c0_c2.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -B experiments/r16p19_phase8/audit_c4.py
PYTHONPATH=. python -B experiments/r16p19_phase8/audit_mechanism.py
PYTHONPATH=. python -B experiments/r16p19_phase8/audit_reproduce.py
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider experiments/r16p19_phase8/test_audit_c0_c2.py experiments/r16p19_phase8/test_audit_calibration.py experiments/r16p19_phase8/test_audit_c4.py
python -B experiments/r16p19_phase8/assemble_checkpoint.py
python -B experiments/r16p19_phase8/verify_release.py
```

`audit_calibration.py` writes the nonformal mappings BEFORE opening any formal inputs. It always recomputes C3; no unbound cache is reused. Its Python audit hook rejects formal reads until the explicit ceiling diagnostic stage and rejects writes to protected phases. `audit_c0_c2.py` runs in a separate process and rejects all formal inputs including formal-bearing C1 reports. C4 also rejects formal reads and protects source writes. The only formal uses are C1 ceilings and exact reproduction of the old Phase-5 classifier, not a step9 formal workpoint evaluation.

`C2_SELECTION_SEAL_AUDITED.json` is an execution audit, explicitly not a valid retrospective preregistration. It contains no selected deployable point. Its input and output hashes are verified by `verify_release.py`.

`SHA256SUMS` hashes all phase8 artifacts except itself and runtime caches; a file cannot stably contain its own SHA256. Run `sha256sum --quiet -c experiments/r16p19_phase8/SHA256SUMS` from repo root. Regenerate the manifest after adding publication/readback artifacts.

C4 deterministically materializes the frozen Phase-5 event templates for the stored episode identities; per-unit serialized event logs were not present. Its persistent upgrade wrapper is the new semantic implementation. Pilot outcome envelopes are conditional assumptions on unknown divergent outcomes, not newly observed task outcomes.

# Reproduction commands (CPU only)

Repository: /mnt/cpfs/zbl-cpfs-new/share/leon/codex-archives/Explore-claude-local-worktrees/r16p19-step8-s1b-20260907
Historical source: b7f7c8862acef48438318c49658f0af97927e05e
B0 pre-analysis commit:8871bf9792cf029c765becfb61aad16a6b0253c6
B2 pre-heldout seal commit:a4bb9fb
Python:/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python

Every workload command used PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Scripts set CPU environment internally as well.

Executed in order after B0 hashing:
```
python experiments/r16p19_phase7/run_b1.py
python experiments/r16p19_phase7/run_nonformal.py B2
# B2_CALIBRATION + B2_SELECTION_SEAL hashed, chmod read-only, committed here
python experiments/r16p19_phase7/run_nonformal.py remaining
python -m pytest -q -p no:cacheprovider experiments/r16p19_phase7/test_phase7.py
python experiments/r16p19_phase7/b5_available_model.py
python experiments/r16p19_phase7/prepare_handoff.py
```

Do not rerun into frozen output artifacts. B0 and B2 files are immutable; reproduce in a new explicitly authorized remote workspace and preserve original seals. No new repository or worktree was created on the Mac. Local /tmp/r16p19-step8-publication holds only small publication artifacts, no .git.

B_DECISION is the computation stop. Subsequent actions only verify existing artifacts and publish to GitHub/Feishu. SHA256SUMS lists every deliverable except itself (a manifest cannot hash itself without recursion).

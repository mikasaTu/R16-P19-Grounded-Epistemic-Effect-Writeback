# Phase-2 PAI registry records

These are the exact payload wrappers and declarative templates used for the
single-GPU Phase-2 template calibration and qualification workloads. They were
validated and submitted by the external `pai-job-registry` controller from the
verified `rtx4090-general-1gpu` carrier (`quotaq1herttfeuy`).

The controller's fail-closed profile required, for each stage:

- exact source commit and tree;
- exact wrapper and repository launcher SHA-256;
- exactly one worker and one RTX 4090 rendering GPU;
- UID/GID 2254:2254 and the three pinned CPFS mounts;
- no injected secrets and W&B disabled;
- application-level fsynced JSONL resume units;
- no AIMaster, automatic fault tolerance, or platform restarts;
- the exact output and control-plane paths;
- no second idle GPU.

The qualification profile additionally pins the selected executor/template
manifest hashes, init 60--79, 40 expected rollouts, all failure videos enabled,
the three frozen gate thresholds, and zero formal-init access.

Job history and full GetJob readbacks are under
`artifacts/phase2_pai/control_plane/`. The first two calibration jobs are
retained as honest implementation failures; the third demonstrates completed
cell resume. The qualification job completed successfully but its scientific
gate failed.

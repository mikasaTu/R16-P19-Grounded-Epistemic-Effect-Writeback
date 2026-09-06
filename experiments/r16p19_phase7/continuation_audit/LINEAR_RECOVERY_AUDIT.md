# Exact-lineage linear artifact recovery audit

Read-only searches on dev14; no training, inference, rollout, writes by reviewer, or PAI/GPU use.

## Campaign storage

Root searched: /mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/

Six attempt directories were inspected. Only r16p19-phase5-idle-fixed-20260818-0855 contained artifacts; the other five contained zero files/NPZ. The active artifact run has1121 files and the frozen split counts natural210, calibration30, qualification30, pilot15, formal240. No additional same-lineage nonformal episode was found.

The full successful run has exactly one checkpoint-named file: results/verifier_checkpoint.npz. The earlier reviewer statement of zero checkpoint basename matches applied only to the rollouts subtree, excluding the sibling results directory; it is explicitly corrected here. No additional linear/score/prediction/logit-named file was found.

Unique checkpoint:6644 bytes, SHA256 edba3ff23ae71c5768760af161b601b3d1a9f65d37d1ddeb8304b2510e938005. NPZ fields model_type,mean,std,threshold,calibration_scale,calibration_bias,w1,b1,w2,b2. model_type=small_mlp; no linear w/b.

The same bytes were verified in the raw campaign, current step8 remote archive, and /mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P19-Grounded-Epistemic-Effect-Writeback-phase5.

## Saved scores and metrics

verifier_training_metrics.jsonl contains only two aggregate rows (linear and small_mlp), no parameters or per-sample score vectors. The checkpoint metrics copies have SHA256495589fdf47f3909debfdd7d8a8aa7586759c85e17d12fc88b5c5118fae22463.

learned_verifier_formal_results.jsonl has1680 rows of selected-model scores, SHA2567cd40fbd6ff84e6db49e5527b90b67e26f55dcbeca8e0b3013ffa28fca30d999. Code phase5_formal_runner.py:181-190 computes these from the selected checkpoint. They are not linear scores and cannot substitute for it.

## Source and Git history

phase5_verifier_model.py:145-153 fits both models in memory;:184-190 serializes only the selected model;:196-205 selects prediction code by saved model_type.

Reviewer inspected refs/reflogs with git rev-list --objects --all --reflog in canonical source, phase5 clone and step8 archive. Relevant matches were11 paths in canonical source and17 each in the latter two; the Phase5 result matches were the same7 known verifier-related files, not an additional model.

Reviewer also inspected19 dangling blobs returned by git fsck --full --no-reflogs --unreachable; they were historical Phase1–4/Feishu documents and did not supply the missing Phase5 candidate. Phase3 verifier or actor/state-BC weights are wrong lineage and were excluded.

## Conclusion and limits

No recoverable exact original linear weights or original linear per-sample scores were found in the bounded campaign/source/history scope. This is not a claim that no copy exists anywhere. A user-provided external exact-lineage copy could change the result. Reconstruction by replaying1200 optimizer iterations would be training, so it was not performed under the original no-retraining instruction.

# Protocol deviations and withdrawal

The claims that step9 was completed in 720cb38/585814e are withdrawn. The original C0 is unchanged; its late seal and missing selection/significance specifications cannot be repaired retrospectively. All new computations are transparent retrospective diagnostics. No READY_FOR_FORMAL certification can be asserted.

- **C0_ORDER**: run_step9.py fits per-effect maps before writing C0_SAMPLE_BUDGET; SHA256SUMS generated only after script completed; original C0 lacks pooled rule and significance criterion. Repair: not recoverable retrospectively; keep original C0 bytes and publish deviation record

- **C0_RATE**: rates = positive frames / total frames was divided into minimum positives to label estimated_sets. Repair: use per-task positive frames and positive episodes / 15 task episodes

- **C1_TEMPERATURE**: only fit_platt implemented. Repair: fit both positive-slope maps; preserve AUC and actual ceiling search

- **C1_CEILING_NULL**: B1 has operating_points, original lookup expects fpr_caps or top-level threshold_vector; old/new thresholds are null/empty. Repair: actual formal diagnostic-only rerun in isolated process

- **C2_FALSE_PASS**: status COMPLETE checks non-null LCB rather than each LCB >=0.90. Repair: explicit threshold comparisons and heldout/selection caveat

- **C2_PER_EFFECT**: old per-effect comparison uses len(y) instead of number of positives, and max positive score rather than highest feasible threshold. Repair: sort positive scores and solve actual Wilson rule

- **C3_EMPTY_RECEIPT**: all receipt_fpr null; no receipt AUC, no actual B4 numeric comparison. Repair: full receipt ROC and effect-level decisions at each threshold

- **C4_SUBSTITUTION**: 585814e copies phase6 S1 C variant 102/18 without applying step9 mappings or independent ledger semantics. Repair: withdraw old claim and execute actual event replay

- **SEAL_UNVERIFIED**: formal_access_count hard-coded zero; no actual paths and read hook in old script, formal diagnostic precedes C2 in same process. Repair: separate audited nonformal process with rejecting open hook; new diagnostic execution seal cannot restore preregistration

- **HASH_SELF**: sha256sum output redirect included SHA256SUMS itself after it existed, invalid self entry. Repair: exclude SHA256SUMS itself; document standard self exclusion

- **NO_TESTS**: published commits contain no tests or test outputs. Repair: behavior/property tests and release validation

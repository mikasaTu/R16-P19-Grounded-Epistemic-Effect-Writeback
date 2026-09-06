# Independent read-only review and acceptance

Reviewer: luna_worker /review_ceiling; no files modified, no GPU/PAI/rollouts.

Accepted on the current frozen data:
- Threshold candidates cover every decision attainable by score >= threshold, including ties and all-reject endpoint.
- Exact per-task decomposition and global FP budget combination match the unfactored objective, including min-TPR, FP, sum-TPR and lexicographic tie breaks.
- Independent DP on actual receipts matched both caps: 840 receipts, 612 negative, allowed FP 12/30, achieved FP10 and minimum effect TPR1.0. All five thresholds matched.
- Statistical identity: 120 units, seven conditions per unit; three disjoint tasks with280 receipts each; positive effect counts80/76/80/74/72.
- S1 1680-row reproduction mismatch0 and C concordant units102 match prior evidence.
- B3/B5/B6 uncomputable metrics remain null, not fabricated zeros. F is diagnostic-only and not used for selection.

Primary acceptance evidence: `TEST_REPORT.txt` records7 passed; `test_phase7.py` contains independently recomputable random global-bruteforce comparison across25 datasets and4 budgets, Wilson score-test inversion, minimum-sample certification, inclusive threshold ties, prohibited-path guard, tied AUC and missing-vector semantics. Reviewer additionally reported larger in-memory checks; those unpersisted checks are supplementary, not the reproducible test result.

Two nonblocking robustness observations remain scoped to generalization beyond the frozen task: the solver relies on effect traversal matching sorted global effect keys and consistency checks assume120 units. Reviewer independently verified both conditions on this dataset. They do not change any reported value, and no post-B0 rule changes were made.

Lineage reviewer independently inspected source, raw nonformal metadata and checkpoint structure. Primary live B2 counts matched371/14/14/381/15, and source selection criterion/checkpoint hash matched. See B1_TRAINING_SPLIT_AUDIT.json.

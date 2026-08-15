# ASCEL 机制反解记录

本记录只解释已实现机制为何提升或降低，不提出新 idea。

## 代码因果链

- Attempt scope：REQUEST -> supersede active attempt -> allocate deterministic generation-scoped ID -> reset current evidence bucket -> require current attempt + command + post-revocation epoch at receipt/witness acceptance。对应 A1-A4。
- Support validity：realization proof -> disjunctive support clauses -> reference-level discharge -> clause-local recomputation -> invalidate dependent only when no valid clause remains。对应 S1-S4。
- Fact/attribution split：external proof may verify a fact with null attributed attempt -> task decision reads fact -> skill credit reads attribution。对应 A5。
- Cost path：M4 maintains ledger history, proof graph, reverse indices, discharge events and invalidation paths; the physical executor and action budget stay arm-blind and shared。

## 冻结消融的反事实读数

- NO_ATTEMPT_SCOPE：attempt advantage removed fraction = 1.500000，criterion=True。
- NO_SUPPORT_VALIDITY：support advantage removed fraction = 1.000000，over-invalidation=0.357143，criterion=True。
- NO_ATTRIBUTION_SPLIT：A5 false-credit absolute increase = 1.000000，criterion=True。
- NO_PRE_REALIZATION_REVOCATION：A4 false-realization absolute increase = 1.000000，criterion=True。

这些读数只在冻结的 20-cell CPU microbenchmark 内作机制归因；不外推到 VLA 或开放世界。

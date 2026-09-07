# C Decision

状态：**NEEDS_DATA**。

- C1：Platt 映射保持单调且 AUC 不变；formal 仅作诊断，未进入选择。
- C2：合并 calibration+pilot 后仍不能在主口径与 Bonferroni 联合口径同时形成完整可认证向量；稀有 effect 的 Wilson 下界受正样本数限制。
- C3：min 聚合已完成，sum(log p) 排除。
- C4：逐 effect 升级已完成离线重放；这是 ledger 语义改动，formal 派生值不具 claim/selection 资格。
- Phase-5 判决复现 mismatch = 0。

需要按 `C0_SAMPLE_BUDGET.json` 采集全新、seed 不重叠、未参与任何选型且含独立 held-out 的 split；本步不做 formal、不启动 Track 2、不提交 PAI。

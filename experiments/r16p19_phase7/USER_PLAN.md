# 任务：R16-P19 Phase-6 / step8 — S1b 天花板探测与重标定（零 rollout）

## 背景（只读事实，不要重新推导）
- 仓库 mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback，S1 已完成，终态 FAIL_G1
- S1 已确认：Phase-5 判决可精确复现（1680 行 mismatch 0）；per-effect 重标定把 oracle 一致率从 0.7333 提到 0.9571；C 的 102/120 个一致 unit 已贡献 oracle 全量增益的 87%
- S1 的两个已知缺陷，本步必须修正，不要沿用：
(a) G1 用点估计卡 0.90。C 最低的 tomato 是 64/76=0.8421，Wilson 95% CI [0.744, 0.907]，包含 0.90，该 FAIL 不成立
(b) 三个 effect 的阈值由 9/9/10 个 calibration 正样本决定，选择规则 max(threshold | 经验TPR>=0.90) 在 n=9 时等价于“取 9 个样本里的最小分”
- 变体 D 已判定为废臂：加权算术平均无法表达 AND；calibration 上强制零假阳导致它在碰 formal 前就是退化规则（min group TPR 0.0，receipt TPR 0.143）；且它的 per-effect TPR 与 A/B/C 不是同一个量。本步不得沿用该实现

## 硬约束
1. 禁止 GPU、PAI、仿真 rollout。全部 CPU。不启动 S2
2. experiments/r16p19_phase5/ 与 r16p19_phase6/ 只读；新产物写入 experiments/r16p19_phase7/
3. 选择规则必须在 B0 冻结并哈希封存，之后不得修改。任何后续观测都不得回头改选择规则
4. 变体 F（天花板）使用 formal 标签，仅作上界诊断，永远不得用于任何 claim、不得作为任何选择规则的输入、不得写入 seal
5. 每个产物记 SHA256 到 SHA256SUMS
6. 复算与 S1 已有结论不一致时停止报告，不要“修正”

## 执行顺序
### B0 预注册（必须最先完成并封存）
写 experiments/r16p19_phase7/B0_PREREGISTRATION.json，含：
- 阈值选择规则：对每个 effect，在合并的非 formal split 上，选使 Wilson 95% 下界 TPR >= 0.90 的最高阈值；不存在则记为 UNCERTIFIABLE 并记录该 effect 可达的最高 LCB
- G1' 门（四条）：
1. 所选变体每个 effect 的 formal TPR Wilson 95% 下界 >= 0.90
2. 该变体 formal 假升级率 Wilson 95% 上界 <= 0.05，且显著低于 B3 的 50%
3. 选择过程未读取任何 formal 路径（seal 证明）
4. Phase-5 判决重现 mismatch = 0
- 明确写出：n=72~80 时认证 LCB>=0.90 需要经验 TPR >= 0.973
哈希封存后才能进入 B1。

### B1 天花板探测（变体 F，决定分支）
从 Phase-5 artifacts 判定哪些 split 参与过模型拟合（training_samples 来源），列出结论。然后：
- 用 formal 自身标签，对每个 effect 逐点搜索阈值，在 receipt 级 FPR <= 0.02 的约束下最大化 min per-effect TPR
- 报告：该约束下可达的 min per-effect TPR、对应每 effect 的 TPR/阈值、以及放宽到 FPR <= 0.05 时的同一组数字
输出 B1_CEILING.json，文件内显式标注 claim_eligible: false。
分支：
- 天花板 min per-effect TPR >= 0.973 → 继续 B2
- 天花板 < 0.973 → 停在这里，写 B1_DECISION.md，结论为“检测器容量不足，重标定路径无法通过 G1'”，列出该结论所依赖的具体数字，等待人工确认。不要继续 B2~B6

### B2 合并 split 重估阈值
- 合并所有未参与模型拟合的非 formal split（至少 calibration + pilot；qualification 保留为 B3 的 held-out，不得进入估计）
- 先报告每个 effect 合并后的正样本数。任一 effect 正样本 < 35 → 标记为 UNCERTIFIABLE 并在报告中显式列出，不要静默通过
- 按 B0 冻结的 LCB 规则解阈值，写 B2_CALIBRATION.json 并生成 B2_SELECTION_SEAL.json（记录读取路径清单与 formal 访问数 0）

### B3 held-out 检验（碰 formal 之前）
在 qualification split 上评估 B2 阈值向量：每 effect TPR/FPR 及 Wilson 区间、receipt 级假升级数。输出 B3_HELDOUT.json。
这一步的作用是把“估计噪声”和“迁移失败”分开：若 qualification 上通过而 formal 上不通过，结论是迁移；若 qualification 上就不通过，结论是估计仍不足。在报告中明确写出落在哪一侧。

### B4 重做软聚合（变体 D2）
- 聚合函数改为 min(scores) 与 sum(log p) 两种，各自扫 receipt 阈值
- 不得在 calibration 上强制零假阳；报告完整 TPR/FPR 曲线与 AUC
- per-effect TPR 必须按 effect 级判决定义，与 A/B/C 同量纲；若某聚合方式不产生 effect 级判决，明确记录并排除出 G1' 候选，不要用 receipt 接受率冒充
输出 B4_SOFT.json。

### B5 重选 verifier（变体 E）
不重训。用 Phase-5 已有的 linear 与 small_mlp 两个候选，按“min per-effect TPR @ 固定 FPR”重新选型，与原按 loss/accuracy 选型对比。
报告两者在同一准则下的排序，以及换选型后 B2 阈值规则的结果变化。输出 B5_MODEL_SELECTION.json。

### B6 工作点前沿
在 qualification split 上，把 per-effect TPR 目标从 0.80 扫到 0.99，对每个目标记录（假升级率，oracle 判决一致率）。选出一个满足 G1' 的工作点，写入 B6_OPERATING_POINT_SEAL.json 并哈希封存。
formal 上的完整前沿曲线只能作为描述性图输出，标注 selection_eligible: false，不得用于选点。

### B7 重放与新分歧集
用封存的工作点在 formal 上评估，重做判决重放。报告：
- 一致 / 分歧 unit 数与分歧 unit ID
- 增益区间：下界按对抗假设（分歧 unit 每个 faulted cell 最坏），上界按继承 oracle；另外单独报告中性假设值并显式标注为假设而非点估计
- S2 重执行集合只取本工作点的分歧 unit，不做跨变体并集
输出 B7_REPLAY.json 与 B7_S2_UNITS.txt。

## G1' 判定
写 B_DECISION.md 与 B_DECISION.json，逐条给出四个门的判定值与所依据的数字。

## 终止条件
写完 B_DECISION.md 即停。不启动 S2、不提交 PAI、不写 Phase-6/7 最终报告。
交回 B_DECISION.md、B1_CEILING.json、B3_HELDOUT.json 和 S2 候选集，等待人工确认。

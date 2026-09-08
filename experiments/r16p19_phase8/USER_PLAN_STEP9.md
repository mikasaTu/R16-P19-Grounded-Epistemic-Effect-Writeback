# 任务：R16-P19 Phase-8 / step9 — S1c 每 effect 概率重校准与认证口径确定（零 rollout）

## 背景（只读事实，不要重新推导）

- step8 结论 UNCERTIFIABLE_NO_OPERATING_POINT。B1 天花板：formal 上存在阈值向量
  达到 min per-effect TPR=1.0，假升级 10/612=1.63%，两个 FPR 约束下相同
- 该向量为 0.9319 / 0.0204 / 0.0487 / 0.9906 / 0.0585，跨约 48 倍
- B2 阻塞原因是校准侧正样本不足：tomato/book/close 各 14/14/15，
  完美检测下 Wilson LCB 上限仅 0.7847/0.7847/0.7961
- linear checkpoint 已确认永久缺失，B5 关闭，本步不再尝试恢复
- qualification 参与过原始选型且与 pilot 共享 15 个环境 seed，不是干净 held-out
- 计数单位 frame-effect 存在 episode 内时序相关

## 本步目标

判定：per-effect 单调重校准能否把阈值选择从"5 个稀疏估计"降为"1 个稠密估计"，
从而在不新增数据的前提下产生可认证向量。全部离线。

## 硬约束

1. 禁止 GPU、PAI、rollout、S2、采集新数据。全部 CPU
2. phase5/6/7 全部只读。新产物写入 experiments/r16p19_phase8/
3. 禁止重训 verifier：不得修改 w1/b1/w2/b2。只允许在标量输出上拟合
   per-effect 单调映射（temperature 或 Platt），并明确记录这不是重训
4. 任何使用 formal 标签的计算标 claim_eligible=false、selection_eligible=false，
   不得写入任何 selection seal
5. C0 封存前不得开始 C1
6. 每个产物记 SHA256；缺失值记 null，不得填 0

## 执行顺序

### C0 预注册与功效计算（最先，封存后不可改）

写 C0_PREREGISTRATION.json，定死并给出依据：

- 认证方法：per-effect TPR 的二项/Wilson 下界，或等价的 split-conformal 阈值规则
- 多重性口径：逐 effect 95% 与五 effect 联合 95%（Bonferroni α=0.01）两套都要
  给出数值要求，并标注本轮主口径为哪一套
- 独立性单位：frame-effect 与 episode 聚类两种,给出各自的区间宽度差异
- 产出 C0_SAMPLE_BUDGET.json：在每套口径 × 每种独立性单位下，
  每个稀有 effect 达到可认证所需的最少正样本数与对应的非 formal 集数估计
  （用 calibration+pilot 观测到的每集正样本率外推，并标注为外推）
  封存哈希后进入 C1。

### C1 per-effect 重校准与阈值收敛检验（决定性）

- 在 calibration+pilot 上，对每个 effect 分别拟合 temperature 与 Platt 映射，
  使用该 effect 的全部 frame-effect 行（正负样本都用）
- 显式验证并记录：映射是单调的，每 effect AUC 在重校准前后不变
- 用重校准后的分数重跑 B1 式天花板搜索，报告新的五个 formal 最优阈值
  （诊断用，claim_eligible=false）
- 报告收敛度量：重校准前后五个最优阈值的极差与变异系数
  输出 C1_RECALIBRATION.json。
  若极差没有显著收窄，明确写出"尺度失配假设未被支持"，并在报告中列出该结论
  所依据的具体数字。不要因为后续步骤需要就淡化这一条。

### C2 单一全局阈值选择

- 在 calibration+pilot 上，用重校准后的分数、按 C0 冻结的规则选一个全局阈值，
  合并全部 effect 的正样本估计
- 报告：合并正样本总数、所选阈值、在两套多重性口径下每个 effect 的
  校准侧 LCB、以及是否形成完整可认证向量
- 同时给出"每 effect 各自阈值"的对照结果，说明两者的样本效率差异
- 生成 C2_SELECTION_SEAL.json，记录读取路径清单与 formal 访问数
  输出 C2_GLOBAL_THRESHOLD.json。

### C3 min 聚合复测

在重校准分数下重跑 min 聚合，与 step8 的 B4 结果对比。
只报告能给出 effect 级判决的聚合方式；sum(log p) 仍排除出候选。
输出 C3_MIN_AGG.json。

### C4 per-effect 升级变体（机制改动，诊断优先）

把 receipt 级 AND 升级改为逐 effect 独立升级：某个 effect 通过 witness 就升级
该 effect，不因同 receipt 其他 effect 漏检而否决。

- 在已存事件流上重放 ledger，报告一致 unit 数、分歧 unit 数
- 双向指标必须同时报告：找回的增益区间（对抗下界 / oracle 上界 / 中性假设，
  中性值显式标注为假设）与假升级计数及其 Wilson 上界
- 明确记录这是 ledger 语义改动，不是分析口径改动
  输出 C4_PER_EFFECT_UPGRADE.json。

### Gate C（写入 C_DECISION.md 与 C_DECISION.json）

逐条给出判定值与依据：

1. C1 阈值极差是否显著收窄（尺度失配假设是否被支持）
2. C2 是否在不新增数据的前提下产生完整可认证向量（分两套多重性口径各判一次）
3. C4 是否在不抬高假升级上界的前提下提高一致 unit 数
4. Phase-5 判决复现 mismatch 是否为 0
   分支：

- C2 在主口径下产生完整向量 → 状态 READY_FOR_FORMAL，列出待评估的封存工作点，
  停下等人工确认后再做 formal 单次评估
- 未产生 → 状态 NEEDS_DATA，输出 C0_SAMPLE_BUDGET.json 中对应的采集集数，
  并给出该采集应满足的 split 设计要求（与现有 calibration/pilot/qualification/
  formal 的 seed 全部不重叠、不参与任何选型、其中一部分预留为全新 held-out）

## 终止条件

写完 C_DECISION.md 即停。不做 formal 评估、不启动 Track 2 采集、不提交 PAI、
不写 Phase-8 最终报告。交回 C_DECISION.md、C1_RECALIBRATION.json、
C0_SAMPLE_BUDGET.json，等待人工确认。

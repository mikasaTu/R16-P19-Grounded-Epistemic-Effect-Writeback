# 任务：R16-P19 Phase-9 / step10 — D 负类保形 witness 阈值，一次走到 formal（零 rollout）

## 背景（只读事实，不要重新推导）

- phase8 结论 NEEDS_DATA，且 step9 无有效预注册（C0 在拟合后才封存），其全部内容只能当回顾性诊断
- 已证伪：per-effect 单调重校准无法让单一全局阈值成立。Platt @0.6477 下稀有 effect 实际命中 4/14、7/14、1/15；temperature 更差（0/14、0/14、1/15）
- 已证伪：合并正样本是假通过模式，795 个 pooled 正样本里 752 个（94.6%）来自两个常见 effect
- 已证伪：C4 逐 effect 独立升级更差（一致 18/60 vs 19/60，假升级 2/510 vs 0/510）
- B1 天花板（仅诊断）：五个 per-effect 阈值在分数空间跨 48 倍，在 FPR 空间全部落在 0–2.9%，该处 min per-effect TPR = 1.0，receipt 假升级 10/612
- 估计集样本量：正 371/14/14/381/15，负 521/878/596/437/803
- Spearman(base_rate, 阈值) 在 raw 和 platt 下都是 0.90

## 本步目标

把 witness gate 的认证对象从 TPR 换成 per-effect FPR，用负类 split-conformal 选阈值，封存后在 formal 上做一次评估并完成 ledger 重放。全部离线。

## 硬约束

1. 禁止 GPU、PAI、rollout、采集新数据、重训 verifier（不得改 w1/b1/w2/b2）
2. phase5/6/7/8 全部只读。新产物写入 experiments/r16p19_phase9/
3. D0 必须在**独立 commit** 中先落地，记录该 commit hash；本轮任何分析脚本都不得早于该 commit 存在。事后补封存一律判为违规并如实记录
4. D3 选点过程不得读取任何正样本标签。用 read hook 强制拦截，并提供一个断言 hook 会拒绝正样本路径的测试
5. formal 只允许在工作点 seal 完成**独立 commit** 之后读取一次；seal commit hash 写入 D4 输出
6. 每个门判定写成具名函数，且必须配一个断言它在已知失败输入上返回 False 的单元测试。禁止用“字段非空”代替“字段满足阈值”
7. 测试与结果同一个 commit 提交。SHA256SUMS 排除自身
8. 缺失值记 null，不得填 0；任何用 formal 标签的量标 claim_eligible=false

## 执行顺序

### D0 预注册（独立 commit，先于一切分析）

写 D0_PREREGISTRATION.json：

- 认证对象：per-effect FPR ≤ α，方法为负类 split-conformal，阈值取第 ⌈(n_neg+1)(1-α)⌉ 阶统计量；同时给出 FPR 的精确二项上界
- α 网格：0.005 / 0.01 / 0.02 / 0.05
- 多重性：逐 effect 95% 与五 effect 联合 95%（Bonferroni α=0.01）各算一套，主口径写明
- 独立性单位：receipt 级与 episode 聚类各算一套；主口径为 episode 聚类（证据：B1 的 10 个假升级 receipt 仅来自 5 个 unit，误差是 unit 级的）
- 工作点选择规则：在满足 FPR 认证的 α 中，选使 estimation 上 receipt 级假升级上界最低者；平局按 α 小者。规则冻结后不得改
- Gate D 四条判定的具名函数签名
  封存并单独 commit，记录 hash。

### D1 转移性检查（早停门）

对每个 effect，比较 estimation / qualification / formal 三个 split 上**负类**分数的分位数曲线（至少 0.90/0.95/0.98/0.99/0.995 分位）。报告分位数差异、KS 统计量、以及按 estimation 分位数施加到 formal 时实际 FPR 与名义 α 的偏差。
输出 D1_EXCHANGEABILITY.json。
分支：若任一稀有 effect 的名义-实际 FPR 偏差超过 2 倍，写 D1_DECISION.md 判为 NOT_EXCHANGEABLE 并停止，不继续 D2–D6。

### D2 base-rate 因果检验

对负样本重采样，把五个 effect 的正样本比率拉平到同一水平（多个水平各做一次），重拟合 Platt，重算 formal 天花板阈值向量，报告阈值极差与 CV 随 base rate 均衡程度的变化。
输出 D2_BASERATE.json。
明确写出结论方向：极差随均衡收窄 → base-rate 机制被确认；不收窄 → 机制诊断不成立，须在报告中如实写出并标注这会影响 D3 的动机（但不改变 D3 的执行）。
本步使用 formal 标签，claim_eligible=false。

### D3 负类保形阈值与工作点封存

- 只用 estimation 负样本，按 D0 冻结规则对每个 α 解出 per-effect 阈值向量
- 对每个 α 报告：每 effect 的 FPR 精确二项上界（两套多重性口径）、阈值、所用阶统计量的秩
- 在 estimation 上评估该向量的 receipt 级假升级数与 episode 聚类区间
- 按冻结规则选出唯一工作点，写 D3_OPERATING_POINT_SEAL.json
- 提供 read hook 审计记录，证明选点过程零正样本标签访问
  **单独 commit 该 seal，记录 hash，然后才允许进入 D4。**

### D4 formal 单次评估

读取 formal 一次。报告封存工作点下：

- 每 effect 的 TPR、FPR 及区间（receipt 级与 episode 聚类各一套）
- receipt 级假升级数、率、Wilson 上界与 episode 聚类上界
- 与 oracle 判决的 receipt 级一致率
- 与 B1 天花板（TPR 1.0 / 假升级 10/612）的差距
  输出 D4_FORMAL.json，写入 D3 seal 的 commit hash。
  不得在看到 D4 结果后回头改任何规则或重跑 D3。

### D5 ledger 重放与增益

在 D4 工作点上重放 ledger（沿用 receipt-AND 语义，C4 已出局）：

- 一致 unit 数、分歧 unit 数与 ID 清单
- 增益区间：对抗下界 / 继承 oracle 上界 / 中性假设（显式标注为假设）
- 与 phase6 S1 变体 C 的 102/120、197/227 做同口径对比
  输出 D5_REPLAY.json 与 D5_S2_UNITS.txt（只取本工作点的分歧 unit）。

### D6 假升级失效模式刻画

对 D4 中产生假升级的 unit：

- 列出 unit ID、task、condition 分布
- 检验是否集中在 A5_EXTERNAL_REALIZATION / C0_CLEAN 等特定条件
- 检验这些是否为时序/归属边界情形（效果即将实现，或由外部实现），即 realized 与 observed/external 的类型混淆
- 报告集中度：产生假升级的 unit 数 / 总 unit 数
  输出 D6_FAILURE_MODES.json 与一段文字描述。
  不要为了配合叙事而夸大集中性；若分布是弥散的，如实写出。

## Gate D（写入 D_DECISION.md 与 D_DECISION.json）

四条具名函数判定，逐条给出值与依据：

1. D1 可交换性通过
2. D4 每 effect FPR 上界 ≤ 所选 α（主口径与联合口径各判一次）
3. D4 receipt 级假升级率的 episode 聚类上界 ≤ 0.05，且显著低于 B3 的 50%
4. D5 ledger 一致率 ≥ 0.85
   四条全过 → READY_FOR_S2，交出 D5_S2_UNITS.txt。
   未全过 → 逐条写明失败在哪一条、差多少，并明确写出“认证路线是否应终止”的判断依据（不替人做决定）。

## 终止条件

写完 D_DECISION.md 即停。不启动 S2 执行、不提交 PAI、不采集数据、不写 Phase-9 最终报告。交回 D_DECISION.md、D1_EXCHANGEABILITY.json、D4_FORMAL.json、D6_FAILURE_MODES.json，等待人工确认。

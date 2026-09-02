# step7 实验报告：S1 离线重标定与判决重放

> 本文是 Phase-6 的 S1 阶段报告，不是 Phase-6 最终报告。S2 未启动。

## 结论

G1 结果为 `FAIL_G1`。本轮已完成 S1.0–S1.4 的全部离线实验，没有 GPU/PAI 作业、没有 simulator rollout，也没有因为中间 gate 失败而跳过后续矩阵。

Phase-5 learned 侧归零确实有很大一部分来自“单一高阈值 + AND 聚合”的放大作用，但不能只归因于聚合规则。per-effect 重标定明显恢复了判决能力，却在 formal split 上仍未达到 G1 的可靠性要求。

## S1.0 底物

- 冻结 checkpoint、metrics、逐-effect 原始连续 score 均存在。
- Learned / oracle / support 行数为 1,680 / 4,200 / 960。
- 冻结 calibration/formal episode 为 30 / 240；既有 init-index split 足以做无泄漏重标定。
- Phase-5 目录和 frozen protocol 未修改。

## S1.1 重现

- 比对行数：1,680。
- 唯一 receipt：840。
- score、threshold、最终判决不一致：0。
- 最大 score 绝对误差：0。

因此后续归因建立在对 Phase-5 learned 判决的精确复现上。

## S1.2 Calibration-only 校准

五个 effect 的 AUC 为 0.9856–0.9993，说明冻结 verifier 有较强排序能力，但不同 effect 的 score 尺度严重不同。按 calibration split 反解的 TPR=0.90 阈值分别为：

| Effect | AUC | 原阈值 TPR | 新阈值 | calibration TPR/FPR |
| --- | ---: | ---: | ---: | ---: |
| alphabet soup 入篮 | 0.9922 | 0.7890 | 0.8282 | 0.9030 / 0.0403 |
| tomato sauce 入篮 | 0.9893 | 0.0000 | 0.1663 | 1.0000 / 0.0367 |
| book 入 caddy | 0.9924 | 0.0000 | 0.1326 | 1.0000 / 0.0259 |
| mug 入 microwave | 0.9993 | 0.9042 | 0.9529 | 0.9000 / 0.0000 |
| microwave 关闭 | 0.9856 | 0.0000 | 0.1024 | 0.9000 / 0.0308 |

所有规则选择只读取 calibration split；formal receipt 访问数为 0。选择规则在 formal 评估前写入并哈希封存于 `S1_CALIBRATION_SEAL.json`。

## S1.3 归因结果

| 变体 | Oracle 一致率 | Min per-effect TPR | Receipt FPR | False-upgrade |
| --- | ---: | ---: | ---: | ---: |
| A oracle × AND | 1.0000 | 1.0000 | 0.0000 | 0 |
| B global 0.9395 × AND | 0.7333 | 0.0000 | 0.0000 | 0 |
| C per-effect threshold × AND | 0.9571 | 0.8421 | 0.0131 | 8 |
| D weighted soft aggregation | 0.7833 | 0.0000 | 0.0065 | 4 |

C 将 oracle 一致率从 0.7333 提升到 0.9571，并把三个原本为 0 的 calibration TPR 拉起，因此“score 尺度错配被 AND 放大”是确认的主要机制。但 C 的 formal 最低 TPR 只有 0.8421，且产生 8 个 false-upgrade；D 的最低 TPR 仍为 0，并产生 4 个 false-upgrade。说明 calibration 到 formal 的检测/校准迁移仍不足，不能把 Phase-5 失败完全归因于聚合。

## S1.4 判决重放

- C：120 个 unit 中 102 个判决序列完全一致、18 个分歧；faulted-success 增益区间 `[0.1236, 0.3153]`。
- D：120 个 unit 中 29 个一致、91 个分歧；增益区间 `[-0.6875, 0.3153]`。
- C/D 分歧集合并集：93 个 unit，已写入 `S1_DISAGREEMENT_UNITS.txt`。

区间下界将每个分歧 unit 中所有 faulted cell 置为对增益最不利的 Core 失败 / baseline 成功；上界令分歧 unit 继承 oracle 结局。没有报告点估计。

## G1 判定

- C/D min per-effect TPR ≥ 0.90：失败。
- 同一变体 false-upgrade=0：失败。
- Calibration 选择未接触 formal：通过。
- Phase-5 判决重现 mismatch=0：通过。

最终：`FAIL_G1`。按计划在此停止，等待人工决定是否执行 S2 的 93-unit 重执行集合。

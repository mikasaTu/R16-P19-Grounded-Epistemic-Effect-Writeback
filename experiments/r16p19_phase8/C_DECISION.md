# C Decision — step9 审计交接

**NEEDS_DATA；补算属于事后诊断，不能声称原计划已全部合规完成。**

此前两次“已完成”声明撤回。C0 未在拟合前完整封存，不能追溯补救；原 C0 哈希保持不变。以下为补齐后的计算证据，不是 Phase-8 最终报告。

| C1 映射 | formal 诊断阈值极差 | CV |
|---|---:|---:|
| raw | 0.9702354446 | 1.0990518919 |
| temperature | 0.9976348452 | 1.1447047510 |
| platt | 0.9998241462 | 1.1744823373 |

两种映射均未收窄：尺度失配假设未被支持。formal ceiling 仅作诊断，绝不用于选点。每 effect AUC 不变不意味着 receipt min AUC 不变。

| C2 映射/口径 | global threshold | pooled positives | 每 effect LCB（固定顺序） | 完整向量 |
|---|---:|---:|---|---|
| platt/individual95 | 0.6477173870 | 795 | 0.902419, 0.117214, 0.267992, 0.966073, 0.011867 | 否 |
| platt/joint95_bonferroni | 0.5932677565 | 795 | 0.904001, 0.089347, 0.216479, 0.961481, 0.007876 | 否 |
| temperature/individual95 | 0.7400212612 | 795 | 0.931100, 0.000000, 0.000000, 0.973320, 0.011867 | 否 |
| temperature/joint95_bonferroni | 0.6971601416 | 795 | 0.934591, 0.000000, 0.000000, 0.969396, 0.007876 | 否 |

effect 顺序：alphabet_soup、tomato、book、mug、close。每 effect 独立阈值对照、TP 数、区间宽度和聚类 bootstrap 均在 C2_GLOBAL_THRESHOLD.json。读路径审计实际拒绝 formal，seal 不代表补救了原 C0。

| C3 split | raw min AUC | temperature | Platt |
|---|---:|---:|---:|
| estimation | 0.98310711 | 0.98626303 | 0.98684520 |
| qualification | 0.98370595 | 0.98479538 | 0.98219022 |

C3 包含完整曲线；资格集曾参与选型并与 pilot 重叠，不能宣称新 held-out 泛化。sum(log p) 排除。

C4：实际逐 effect ledger 状态、相同分母的假升级与区间、unit ID、增益假设见 C4_PER_EFFECT_UPGRADE.json 和 C4_MECHANISM.md。旧 S1 的 102/18 不是本轮结果，已撤回。

Phase-5 精确复算：1680 行，mismatch=0；verifier 权重文件与原提交字节一致。

新数据预算（按 task 独立全新认证分配，不是总采集启动命令）：

| effect | 逐 effect 95% 独立正样本最低数 | task episodes 外推 | 联合 95% 独立正样本最低数 | task episodes 外推 |
|---|---:|---:|---:|---:|
| task0:effect1:in tomato_sauce_1 basket_1_contain_region | 35 | 38 | 60 | 65 |
| task5:effect0:in black_book_1 desk_caddy_1_back_contain_region | 35 | 38 | 60 | 65 |
| task9:effect1:close microwave_1 | 35 | 35 | 60 | 60 |

仅为全成功且保持观测产出率的外推；不能保证实际产出。总认证分配外推为 111 集或 190 集（各 task 分别分配），另加独立拟合/选点数据；不可把已参与拟合的数据算作新的独立认证集。新 seed 与 calibration/pilot/qualification/formal 全部不重叠，认证 held-out 不参与任何选型。

独立性口径：frame Wilson 为描述性计数；episode-all-positive-detected 是不同 estimand，另给 episode bootstrap 的 frame-ratio 区间，不把时间相关的帧当作独立集。原 C0 使用 central two-sided 95%/99% z，保留数值并纠正 one-sided 文字。

按计划停在 C_DECISION：不做新 formal 评估，不启动采集、PAI、GPU、S2。C0 时序违规仍未被、也无法被追溯消除。

C4 数值判定：
- receipt AND: 一致 19/60，分歧 41；假升级 0/510，Wilson 95% 上界 0.00747596。
- per-effect independent: 一致 18/60，分歧 42；假升级 2/510，Wilson 95% 上界 0.01418437。
C4 Gate = False。相同分母的假升级上界上升且一致 unit 未增加。事件由冻结代码/seed 确定性重建，并非存在逐 unit 序列化日志；这是可复算的回放证据，不能伪称原始日志。

C4 pilot 任务成功率包络（相对 M0_TYPED_MATCHED；非 formal，不能把 effect TPR 当作 task_success）：
- legacy_receipt_and: 对抗下界 -0.466667；继承 oracle 上界 0.333333；中性假设 0.133333（分歧 cell 增益为 0，非点估计）。分母 90 个 faulted task cells。
- per_effect_independent: 对抗下界 -0.466667；继承 oracle 上界 0.333333；中性假设 0.133333（分歧 cell 增益为 0，非点估计）。分母 90 个 faulted task cells。
calibration 与非 pilot qualification 缺少已存 task outcomes，其 task-success 增益保持 null；不以检测率代填。

# 任务：R16-P19 Phase-11 / step12 — F 迁移性诊断与真机桥接（零 rollout，CPU）

## 背景（只读事实，不要重新推导）
- 目标已明确为真机部署，仿真只作为真机的准备。仿真侧的“分布无关 FPR 认证”路线本轮起停止，不再追求 certified operating point
- phase5 verifier 的特征构造见 r16p19/phase5_verifier_data.py::_feature_rows：视觉部分仅为 32x32 图的逐通道 mean/std（各 6 维），另有 temporal 差分 6 维、proprio、log1p(contact_count)、归一化 time、task/effect onehot
- 待检验假设：AUC 0.986~0.999 主要来自 time + proprio 捷径，而非效果识别
- 已知支持该假设的现象：单一阈值失效但每 effect AUC 极高；all-frame 与 injection-frame 的 KS 达 0.697/0.765/0.777/0.223/0.483；B1 天花板下 10 个假升级全部落在 A5_EXTERNAL_REALIZATION 与 C0_CLEAN 两个条件、仅来自 5 个 unit
- phase9 全部 formal 派生量 protocol_valid=false

## 本步目标
判定现有 verifier 是否为时间/位姿捷径；产出真机可迁移的特征与检测器规格；盘点真机接管数据能支撑的效果谓词与 provenance 标签规模。

## 硬约束
1. 禁止 GPU、PAI、仿真 rollout、采集新数据、S2
2. phase5~9 全部只读，原始 verifier checkpoint 字节不得改动。新产物写入 experiments/r16p19_phase11/
3. 本轮允许训练新的诊断模型，但必须是新文件、新目录，不得覆盖或替换 phase5 的 checkpoint；每个新模型记录特征集合、超参、随机种子、SHA256
4. 所有 AUC/指标按 effect 分别报，并给出 episode 聚类的置信区间；禁止只报 pooled 指标
5. 不得使用 formal split 做任何模型选择；formal 仅在 F1 的最终对照中只读一次，加载后立即落盘缓存，此后只读缓存
6. 门判定为具名函数，各配一个断言其在已知失败输入上返回 False 的测试
7. 测试与结果同 commit；SHA256SUMS 排除自身；缺失值记 null 不得填 0

## 执行顺序
### F1 捷径消融（本阶段阀门）
用 phase5 冻结的训练/校准 split，按同一模型结构与超参，训练以下特征子集的诊断模型，各报每 effect AUC、TPR@FPR=0.05、以及 episode 聚类区间：
- A 全特征（复现基线，须与 phase5 记录的 macro AUC 0.9917 一致，不一致则停止报告）
- B 去掉全部视觉（base_stats / wrist_stats / temporal）
- C 去掉 time
- D 去掉 contact_count
- E 只保留 time + task_onehot + effect_onehot
- F 只保留视觉 + effect_onehot（去掉 time、proprio、contact）
输出 F1_ABLATION.json 与一张对照表。
判定写入 F1_DECISION.md：
- 若 B 的 macro AUC 与 A 之差 < 0.02，或 E 单独就达到 macro AUC > 0.90 → 判定 SHORTCUT_CONFIRMED
- 若 F 单独就接近 A → 判定视觉确有效果信息
- 两者都不成立 → 写明实际情况，不要强行归类

### F2 传感器可迁移性审计
逐维列出 phase5 特征向量，每一维标注三选一：
- onboard_observable：真机上可直接获得（说明用什么传感器）
- privileged：依赖仿真器内部状态（如 contact_count）
- shortcut：在真机上不成立或会导致捷径（如绝对归一化 time）
对 privileged 项给出真机替代方案候选（例如力/力矩信号替代 contact_count），并注明该替代是否已在既有真机平台上可用。
输出 F2_SENSOR_PARITY.json 与 F2_SENSOR_PARITY.md。
不要臆测硬件配置；未知的写 null 并列为待确认项。

### F3 真机接管数据盘点
先检查是否有 QPILOTS 仓库（arrange_3_flowers 接管数据）的读取权限。
- 有权限：盘点 episode 数、接管发生率、每条 episode 的策略段/接管段划分是否可判、可导出的效果谓词候选（例如每支花是否入瓶）、以及可由接管段自动导出的 provenance 标签（realized-by-policy vs realized-by-external）数量
- 无权限：不要猜测内容。改为输出一份数据需求规格：为支撑一个 ledger 实验，需要哪些字段、哪些时间戳、哪些标注粒度、每个谓词需要多少正负样本
输出 F3_REAL_ROBOT_INVENTORY.json（或 F3_DATA_REQUIREMENTS.md）。

### F4 可迁移检测器规格与仿真侧上界
基于 F1/F2 的结论，定义一个“真机可迁移特征集”：剔除 privileged 与 shortcut 维度，视觉改为带空间结构的表示（在冻结的 32x32 图上可用的最强表示，或明确说明现有落盘数据的分辨率不足以支撑，并给出所需分辨率）。
在该特征集下训练诊断模型，报告每 effect AUC 与 TPR@FPR=0.05，与 F1-A 的全特征结果并列。
明确回答：在只用真机可观测信息的前提下，现有仿真数据能支撑的检测上界是多少。
输出 F4_TRANSFERABLE_SPEC.json 与 F4_TRANSFERABLE_SPEC.md。
若结论是现有落盘数据分辨率不足，如实写出，并给出重采集所需的最小字段与分辨率。

### F5 真机最小实验设计
基于 F2/F3/F4，写真机侧最小可行实验的设计草案，必须包含：
- 采用哪些效果谓词，各自如何在真机上判定
- 接管段如何映射到 realized / external / imagined 三类
- 不依赖 oracle、不依赖故障注入、不依赖轨迹分叉的对照设计
- 每个谓词所需的标注 episode 数估计，以及估计所依据的方差假设
- 明确标注哪些仿真结论可以迁移、哪些不能
输出 F5_REAL_ROBOT_PROTOCOL.md。这是设计草案，不得声称任何真机结果。

## Gate F（写入 F_DECISION.md 与 F_DECISION.json）
四条具名函数判定：
1. F1-A 成功复现 phase5 基线（macro AUC 一致）
2. F1 捷径判定已明确给出（SHORTCUT_CONFIRMED / NOT_CONFIRMED / 其他，附数字）
3. F2 每一维特征都已分类，无遗漏
4. F4 给出了真机可观测特征下的检测上界数字，或明确说明数据不足及所需条件
逐条给出值与依据。不要对“idea 是否该继续”下结论，那由人工判断。

## 终止条件
写完 F_DECISION.md 即停。不执行任何真机实验、不采集数据、不提交 PAI、不写 Phase-11 最终报告。交回 F_DECISION.md、F1_ABLATION.json、F2_SENSOR_PARITY.md、F4_TRANSFERABLE_SPEC.md、F5_REAL_ROBOT_PROTOCOL.md，等待人工确认。

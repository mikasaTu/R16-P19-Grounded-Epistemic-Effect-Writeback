# R16-P19 Phase-6 / step7 — S1 离线重标定与判决重放（零 rollout）

## 背景（只读事实）

- Phase-5 终态为 `BLOCKED_BY_VERIFIER`。
- Oracle ASCEL Core faulted success 相对最强基线提升 `+0.3153`，clustered 95% CI `[0.3042, 0.3250]`。
- Support graph 分支提升 `+0.25`，precision/recall 为 `1.0/1.0`。
- 冻结 π0.5 qualification task 0/5/9 为 `1.0/1.0/0.9`。
- Phase-5 learned verifier 使用 `np.all(scores >= threshold)`，threshold=`0.9395`，min per-effect TPR=`0.0000`，learned 增益精确为 0。
- 本步不质疑 Phase-5 底物，不重跑任何 rollout。

## 目标

判定 learned 侧归零是“聚合规则 + 未校准阈值”造成，还是 verifier 检测能力本身不足；只允许离线计算。

## 硬约束

1. 禁止 GPU、PAI 和仿真 rollout；全部在 CPU 完成。
2. `experiments/r16p19_phase5/` 全部只读；新产物只写入 `experiments/r16p19_phase6/`。
3. 选择只能使用 calibration split；formal receipts 不得参与选择。若 split 不足则停止。
4. 不修改 frozen protocol / preregistration。
5. 每个产物记录 SHA256 到 `SHA256SUMS`。
6. 复算与原判决不一致则立即停止。

## 执行顺序

### S1.0 底物盘点

盘点 checkpoint/metrics、逐 receipt/effect 原始 score、1680 learned / 4200 oracle / 960 support schemas，以及 calibration/formal split 定义，输出 `S1_INVENTORY.md`。

### S1.1 重现性校验

以冻结 checkpoint 和冻结特征复算 score，在 threshold=0.9395 + `np.all` 下逐行重放，输出 `S1_REPRO.json`。不一致大于 0 即停止。

### S1.2 每 effect 校准曲线

只用 calibration split，输出每个 effect 的 TPR/FPR 曲线、AUC、TPR=0.90 阈值、对应 FPR 和现行阈值位置，写入 `S1_CALIBRATION.json` 与逐 effect 曲线图。

### S1.3 归因 2×2

- A：oracle label × AND。
- B：learned score × AND @0.9395。
- C：learned score × per-effect 校准阈值。
- D：learned score × 加权软聚合 + 单一 receipt 阈值。

报告 oracle 一致率、min per-effect TPR、receipt FPR 和 false-upgrade，输出 `S1_ATTRIBUTION.json` 与对比表。

### S1.4 判决重放与增益区间

判决序列与 oracle 完全一致的 unit 继承 oracle 结局；任一分歧 unit 标为 `UNKNOWN`。输出 `S1_REPLAY.json`、分歧 unit 清单和增益区间，禁止点估计。

## G1

四条全过才 PASS：

1. C 或 D 的 min per-effect TPR ≥ 0.90。
2. 同一变体 false-upgrade=0。
3. 选择过程未接触 formal receipts。
4. S1 重放 mismatch=0。

## 终止条件

写完 `S1_DECISION.md` 即停止；不启动 S2、不提交 PAI、不写 Phase-6 最终报告，等待人工确认。

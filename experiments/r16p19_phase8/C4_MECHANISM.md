# C4 逐 effect ledger upgrade 机制审计

本审计在 CPU 上重放冻结的非 formal 输入：calibration、qualification 和与
qualification 配对的 pilot NOOP，共 60 个 unit、700 个 effect-cell，其中
600 个是 fault condition，negative effect-cell 分母为 510。没有读取 formal
episode、formal learned matrix 或 formal ceiling；`claim_eligible` 与
`selection_eligible` 均为 false。

阈值 `0.6477173811021721` 和五个跨任务的 per-effect Platt 映射是固定的
retrospective diagnostic snapshot，来源 commit 为
`585814e6645a1d42b4ea5742d302bea041b3b49c`，映射快照和阈值快照的 SHA256
分别记录在 `C4_AUDITED.json`。本脚本不读取正在替换的 C1/C2 文件，也没有
重训 verifier；因此这些数值不能被当成新的选择 seal。

每个 unit-condition 的 receipt 只有该 task 所需的一个或两个 effect；五个
映射是整个数据集跨 task 的总数，不是每个 receipt 的 effect 数。每个 required
 effect 都用相同的 frozen `M3_ASCEL_CORE` ledger 消费对应的事件流，并保留
实际的 `fact_state`、proof id、proof validity、attempt attribution 和
witness decision；`C4_AUDITED.json` 的 `per_effect_ledger_records` 保存每个
effect cell 的 before/after upgrade state 与 decision。pilot 与已存
`oracle_pilot_results.jsonl` 的 105 行交叉
核验为 mismatch=0。

phase5/6/7 中没有可读取的逐 unit 序列化事件日志，因此事件流由冻结的
`phase5_arm_kernel.py:event_sequence` 和 Phase-5 ledger 确定性 materialize，
并由 pilot oracle 交叉核验；缺失日志没有被冒充为原始记录。

## 语义差异和持久化 ledger 实现

C4 在 witness ledger 外增加了一个持久化的 per-effect upgrade record。两个
变体先消费完全相同的事件流；witness 的事实、proof、撤销状态、attribution
和 decision 必须逐 effect 相同。差异只发生在写入 upgrade record 的 gate：

- 旧语义 `receipt_and`：只有当该 receipt 的所有 required effect detector
  都通过且所有对应 witness 都验证时，才把每个 effect 的 upgrade record 写为
  `REALIZED / ADVANCE_TO_NEXT_SUBTASK`；否则每个 effect 都写为
  `UNKNOWN / REOBSERVE`，并保留 `receipt_and_veto`。
- C4 `per_effect_independent`：每个 effect 只由自己的 detector 和自己的
  witness 决定；通过的 effect 独立写入 `REALIZED / ADVANCE_TO_NEXT_SUBTASK`，
  漏检 effect 写入 `UNKNOWN / REOBSERVE`，并保留具体拒绝原因。

这是 ledger 的语义边界改变，不是把分析数组换名。最小属性测试对同一事件流
使用 `[true, false]` detector 和两个已验证 witness，结果为：

```text
receipt-AND       effect_a = UNKNOWN / REOBSERVE
per-effect        effect_a = REALIZED / ADVANCE_TO_NEXT_SUBTASK
effect_b         两者均 UNKNOWN / REOBSERVE
```

测试还逐 condition 对实际 core ledger 的最终状态、decision、verified 和
attempt credit 与 `evaluate_arm` 参考实现核对，全部通过。

## 结果

| 变体 | effect TP / positive | effect TPR | false upgrades | negative denominator | Wilson 95% upper | concordant units | divergent units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| receipt-AND | 42 / 190 | 0.2211 | 0 | 510 | 0.007476 | 19 / 60 | 41 / 60 |
| per-effect independent | 108 / 190 | 0.5684 | 2 | 510 | 0.014184 | 18 / 60 | 42 / 60 |

逐 effect 语义把 effect TPR 从 0.2211 提升到 0.5684，但同一 negative
分母下产生 2 个 false upgrades，Wilson 上界从 0.007476 升到 0.014184；
完整 effect-vector 与 label oracle 的一致 unit 从 19 降到 18。因此 C4
的“提高一致 unit 数且不抬高假升级上界”条件不成立。

直接的 fault effect-cell 差为 `(54 - 21) / 600 = 0.0550`，但这只是 effect
级诊断，不能替代 task-success。按相同 fault-cell 分母 600 的条件性区间，
independent 相对 receipt-AND 的对抗下界为 `-0.7600`，继承 oracle 上界为
`0.123333`，中性值为 `0.0000`。后两者都是假设值，不是点估计。

## 任务级 pilot 增益包络

为避免把 effect TP 改善冒充 task-success，另外使用已有非 formal pilot 的
冻结 M0/M3 `task_success` 结果。pilot 有 15 个 unit；每个 variant 只有在
七个 condition 的完整升级 effect-vector（每个 receipt 一个或两个 effect）
都逐一等于 legacy oracle receipt-AND vector 时才算 concordant。两 variant
都得到 6/15 concordant、9/15 divergent。fault 条件共有 `15 * 6 = 90`
个 task cells；concordant cells 的 M3-minus-M0 观测和为 12，全部 15 个
unit 继承 oracle 的总和为 30。

因此两 variant 的相同 task-success 包络都是：

- 对抗下界：`(12 - 9*6) / 90 = -0.466667`，把 divergent faulted cell
  的增益全部置为最坏的 `-1`；
- 继承 oracle 上界：`30 / 90 = 0.333333`，条件是假设全部 divergent
  unit 继承存储的 pilot M3/M0 结果；
- 中性假设：`12 / 90 = 0.133333`，条件是假设 divergent faulted cell
  的 gain 为零。

中性值明确是假设，不是点估计；上、下界也没有把 partial effect prediction
当成 task outcome。calibration/qualification 没有对应的冻结 task-level
M0/M3 结果，所以它们的 task-success gain 保持 `null`。

## 事件机制核验

| Condition | Final fact state | Verified | Attempt credit | Final decision |
| --- | --- | ---: | ---: | --- |
| C0_CLEAN | REALIZED | true | true | ADVANCE_TO_NEXT_SUBTASK |
| A1_NOOP_RETRY_STALE | UNKNOWN | false | false | REOBSERVE |
| A2_CROSS_ATTEMPT_MIX | OBSERVED | false | false | REOBSERVE |
| A3_CONTRADICTION_LATE_WITNESS | REVOKED | false | false | ROLLBACK_OR_REPLAN |
| A4_POST_REALIZATION_REVERSAL | REVOKED | false | false | ROLLBACK_OR_REPLAN |
| A5_EXTERNAL_REALIZATION | REALIZED | true | false | ADVANCE_TO_NEXT_SUBTASK |
| V1_SINGLE_VIEW_FALSE_POSITIVE | OBSERVED | false | false | REOBSERVE |

示例中，当一个 task 的第一个 effect detector 通过、第二个未通过而两个
witness 都是 `REALIZED/VALID` 时，旧 receipt gate 把第一个 effect 留在
`UNKNOWN/REOBSERVE`，新语义只升级第一个 effect。这确认提升来自 detector
AND 的跨 effect veto 被移除；attempt scope、revocation 和 external
attribution 没有改变。两个 false upgrade 则来自 independent gate 接受了
错误的 effect detector 通过，不能被包装成 ledger 真实度提升。

## 边界

C0 没有在本审计之前由本脚本独立封存；`C4_AUDITED.json` 记录了该
retrospective breach，但不能修复它。C4 没有启动 formal、rollout、PAI、
S2 或新数据采集。

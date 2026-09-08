# C4 逐 effect ledger upgrade 机制审计

本审计只重放冻结的非 formal 数据与 Phase-5 事件实现。输入 split 是
calibration、qualification 和 pilot；共 60 个 unit、700 个 effect-cell，其中
600 个属于 fault condition，negative effect-cell 分母为 510。没有读取
formal episode、formal learned matrix 或 formal ceiling，claim_eligible 与
selection_eligible 均为 false。

阈值固定为 C2 已写入的 Platt global threshold
0.6477173811021721，五个 per-effect 映射来自 C1。verifier 没有重训。
每个 cell 都在同一个 M3_ASCEL_CORE ledger 上重放
r16p19/phase5_arm_kernel.py 的 event_sequence，并记录实际 event stream hash、
最终 fact_state、effect_fact_verified、attempt_attributed_success、
最终 Decision 和 decision sequence。pilot 与已有非 formal
oracle_pilot_results.jsonl 的 105 行交叉核验全部一致（mismatch=0）。
phase5/6/7 没有可读取的逐 unit 序列化事件日志，因此 event stream 是从
冻结 broker/arm 代码确定性 materialize 的；这点在 C4_AUDITED.json 的
event_stream_materialization 中明确记录，没有把缺失日志冒充成原始记录。

## 语义差异

旧语义先把同一个 receipt 的五个 detector 输出做 AND；只要一个 effect
漏检，整个 receipt 不升级，因此所有 effect 都得到 false 的升级结果。
新语义保留每个 effect 自己的 detector/witness 判定：该 effect 通过且其
ledger witness 已验证时，只升级该 effect。一个最小属性例子是：

~~~text
detector_pass = [true, false], ledger_verified = true
legacy receipt-AND      -> [false, false]
per-effect independent  -> [true, false]
~~~

这不是分析口径的重命名，而是 receipt 到 effect 的 ledger upgrade 语义改变。
两个变体使用完全相同的 threshold、事件流、ledger arm、effect labels 和
分母。

## 结果

| 变体 | effect TP / positive | effect TPR | false upgrades | negative denominator | Wilson 95% upper | concordant units | divergent units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| receipt-AND | 42 / 190 | 0.2211 | 0 | 510 | 0.007476 | 19 / 60 | 41 / 60 |
| per-effect independent | 108 / 190 | 0.5684 | 2 | 510 | 0.014184 | 18 / 60 | 42 / 60 |

逐 effect 语义使 effect TPR 从 0.2211 提升到 0.5684，但同一 negative
分母下产生 2 个 false upgrades，Wilson 上界从 0.007476 升到
0.014184；unit 完整向量与 effect oracle 的一致数从 19 降为 18。
因此 C4 的条件“提高一致 unit 数且不抬高假升级上界”不成立。

这里的 oracle 是每个 condition 的 frozen predicate label（在 Phase-5
同一 injection frame），逐 effect 比较 predicted_effect == oracle_effect。
它不是把 Phase-5 的 receipt-level task success 或 formal S1 计数继承成
C4 结果。任何 unit 级增益都必须在这个逐 effect oracle 下解释。

fault effect-cell 的直接描述性差为 (54 - 21) / 600 = 0.0550，但不把它
当作认证点估计。按预先记录的条件性区间规则，per-effect 变体相对旧语义的
fault-cell 增益为：

- 对抗下界：-0.7600，把 divergent unit 的每个 fault cell 赋予最坏
  delta -1；
- 继承 oracle 上界：0.123333，条件是假设 divergent unit 的结果继承
  其存储的逐 effect predicate label；
- 中性假设：0.0000，条件是假设 divergent unit 的增益率等于 concordant
  unit 的观测增益率。

上界和中性值都是描述性假设值，不是点估计，也没有 formal 资格。三个值
共享同一个 faulted effect-cell 分母 600。

## 事件机制核验

真实 core ledger 的最终输出为：

| Condition | Final fact state | Verified | Attempt credit | Final decision |
| --- | --- | ---: | ---: | --- |
| C0_CLEAN | REALIZED | true | true | ADVANCE_TO_NEXT_SUBTASK |
| A1_NOOP_RETRY_STALE | UNKNOWN | false | false | REOBSERVE |
| A2_CROSS_ATTEMPT_MIX | OBSERVED | false | false | REOBSERVE |
| A3_CONTRADICTION_LATE_WITNESS | REVOKED | false | false | ROLLBACK_OR_REPLAN |
| A4_POST_REALIZATION_REVERSAL | REVOKED | false | false | ROLLBACK_OR_REPLAN |
| A5_EXTERNAL_REALIZATION | REALIZED | true | false | ADVANCE_TO_NEXT_SUBTASK |
| V1_SINGLE_VIEW_FALSE_POSITIVE | OBSERVED | false | false | REOBSERVE |

这说明新语义的增益来源是 detector 的部分 effect 被保留下来，而不是放宽
attempt scope、revocation 或 external attribution。A5 的 REALIZED 与
attempt credit=false 仍保持 truth/credit 分离。

## 边界

C0 预注册未在本审计之前由本脚本独立封存；C4_AUDITED.json 记录了该
retrospective breach，但不能修复它。C4 也没有启动 formal、rollout、PAI、
S2 或新数据采集。

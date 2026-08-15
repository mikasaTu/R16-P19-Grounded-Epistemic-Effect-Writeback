# R16-P19 Step5 / Phase-4 实验报告

## 结论

最终状态：`BLOCKED_BY_IMPLEMENTATION`。这是 CPU MuJoCo 微基准结论，不是 VLA、LIBERO、RMBench、N3 或论文就绪证据。

组件状态：attempt=`ATTEMPT_SCOPE_PASS`，support=`SUPPORT_PROOF_PASS`，truth-attribution=`TRUTH_ATTRIBUTION_PASS`，clean=`CLEAN_FAIL`。

## 完整执行与平台有效性

- 分支：`phase4-attempt-scoped-causal-ledger`
- rollout 生成 HEAD/tree：`1e30ff4055ab2681abbc376cfeaf9272fb22f442` / `ab97e31e6567444d1d9bab59e7896a12941a74a3`
- 最终分析 HEAD/tree：`3574bb0d397e16578ceeccc9a4ada0b846212f39` / `940117128ad3a3b4ddc161401dde38f0345591e2`
- 冻结 B6 SHA256：`4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5`
- Trace gate：TRACE_GATE_PASS，10,000 schedules
- Executor gate：pass=True，conditional=1.000000，full-chain=1.000000，backend errors=0
- Shared-prefix gate：pass=True，1,000 paired units，child failures=0
- Pilot：pass=True，1,000 rollouts；Formal：5,000 rollouts；Ablation：4,000 rollouts
- GPU/PAI：未使用；renderer：未初始化。

## Formal 主结果

| Arm | Clean | A1-A4 | S1-S4 | A5 truth | A5 false credit |
|---|---:|---:|---:|---:|---:|
| M0_TYPED_MATCHED | 1.000000 | 0.250000 | 0.500000 | 1.000000 | 1.000000 |
| M1_B6_ORIGINAL | 1.000000 | 0.500000 | 0.500000 | 0.000000 | 0.000000 |
| M2_ATTEMPT_ONLY | 1.000000 | 1.000000 | 0.500000 | 1.000000 | 0.000000 |
| M3_SUPPORT_ONLY | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| M4_ASCEL_FULL | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

- Attempt-family：M4 相对最强基线 `M1_B6_ORIGINAL` 的绝对差为 0.500000，paired cluster bootstrap 95% CI [0.500000, 0.500000]。
- Support-family：M4 相对最强基线 `M0_TYPED_MATCHED` 的绝对差为 0.500000，95% CI [0.450000, 0.548223]。
- Clean：成功率退化 0.000000，action-step overhead 0.000000，event-processing latency overhead 1.061801（M4 816233.520000 ns vs M0 395883.760000 ns）。

## 机制反解（基于代码与消融，不生成新 idea）

1. Attempt 改善来自 `AttemptScopedLedger.request()` 对旧 attempt 的 supersede、`_scope_match()` 的 attempt/command/epoch 三重绑定，以及新 request 清空当前证据池。它阻断了 A1/A2/A3 的跨尝试拼接；`negative_or_contradiction()` 在 VERIFIED 阶段撤销证据，又阻断 A4 的迟到 witness。
2. Support 改善来自 `SupportProofGraph` 的“clause 内合取、clauses 间析取”。父 proof 失效时只重算引用它的 clause；只有全部 clause 失效才递归撤销 dependent。`UNTIL_*` 在约定终点 discharge，因此 S2 不误杀；替代 clause 使 S3 保留；递归仅沿新失效 proof 传播，使 S4 保持 branch locality。
3. Truth/credit 改善来自 `external_realization()` 将 physical fact 置为 REALIZED，但在 attribution split 开启时保留 `attributed_attempt_id=null`。`decide()` 依据物理事实允许推进，而 capability credit 仍要求当前 attempt proof，所以 A5 能推进且不虚假记功。
4. `NO_ATTEMPT_SCOPE` 的 removed fraction 为 1.500000（超过 1 表示不只消除优势，还落到最强基线以下）；`NO_SUPPORT_VALIDITY` 移除了 support 优势的 1.000000；`NO_ATTRIBUTION_SPLIT` 使 A5 false credit 增加 1.000000；`NO_PRE_REALIZATION_REVOCATION` 使 A4 false realization 增加 1.000000。
5. 时延变化来自 M4 每个事件同时维护 append-only ledger、proof/clause 索引、discharge 和失效路径；它不改变物理动作数量。冻结的 10% overhead gate 未通过，因此该成本保留为 clean failure，不能用功能收益覆盖。

## 证据边界

本实验没有修改或重新解释 Phase-3：原状态仍是 `BLOCKED_BY_IMPLEMENTATION`，27/150 shared-prefix 不一致，且 B6 与 TYPED_MATCHED_RECOVERY 在 backend-valid units 上同为 0.913043、差值 0、95% CI [0,0]。Phase-4 使用新的 CPU 微基准和真 shared-prefix fork，只回答 ASCEL 三个窄机制在该合同内是否成立。

所有失败、逐 rollout 结果、paired audits、10,000-replicate bootstrap、McNemar/Holm、消融和 SHA256 均保存在本目录。

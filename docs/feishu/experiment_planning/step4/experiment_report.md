# 实验报告

## R16-P19 Step4 / Phase-3 Effect-Boundary Replay Causal Validation

## 1. 结论摘要

本阶段完成了计划要求的全部下游实验，没有因资格 gate 或 formal replay
gate 失败而提前停止：snapshot bank、replay qualification、K 选择、formal
replay、12-cell smoke、900-cell 主矩阵、180-cell D1、paired audit、首分歧因果
重放、10,000 次 cluster bootstrap、McNemar/Holm、90-cell 机制消融和失败视频
策略均被执行。

机器可读终态是 **`BLOCKED_BY_IMPLEMENTATION`**。同时存在两个独立且更早的
replay blocker：qualification replay gate FAIL、formal replay-only gate FAIL。
因此不能声称 B6 已被行为级验证，也不能声称 benchmark 泛化或 VLA 改进。

诊断矩阵的主要结果是：

- B6 消除了 B2/B3 的 command/imagination-as-progress 早退错误；
- B6 在 C4 上显著优于单视角 POSTCHECK，但与最强
`TYPED_MATCHED_RECOVERY` 的成功率逐条件、逐链完全相同；
- B6 相对最强 strong baseline 的配对差为 0，cluster bootstrap 95% CI 为
`[0, 0]`，McNemar `p=1.0`；
- 关闭 invalidation 后 C3 成功率从 0.667 降到 0，证明失效恢复对 B6 本身
必要；但简单 typed baseline 的通用 contradiction rollback 同样达到 0.667；
- command-parent provenance 消融没有带来任何下降，不能支持其增量机制主张；
- 27/150 paired unit 在第一次 memory decision 前已经出现不同 simulator/event
prefix，违反信息公平 gate；
- 首分歧自然决策 causal win rate 为 0.6136，低于 0.70 gate。

## 2. 冻结版本与执行环境

| 项目 | 冻结值 |
|-|-|
| Phase-3 分支 | `phase3-effect-boundary-replay` |
| 初始仓库基线 | `981ad1a64936b4e970e1f934be2d354497b5fc8e` |
| preregistration commit | `a219eb7411d84846a43b3356b8134d9d9ceca40b` |
| prepare 实现 commit | `a04f3c3ac809b37ed4b70dce5a6bbec3c08a9088` |
| prepare 证据冻结 commit | `857be7c9bee9bc344f9336dd2e6c57ed48a78df1` |
| formal source commit | `6089341084c9e39bb76b065a3c51fa3aa53ced25` |
| formal source tree | `5c8eb5e72c57d2620c0dad395e7d5c631691474c` |
| B6 受保护文件 SHA256 | `4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5` |
| selected manifest SHA256 | `53c6a9fa448267b3b77a9fa4458552e9383fda8b1a63fcf127acf80fe4904677` |
| formal contract SHA256 | `a78728e6bfe0bc33ae016a974c2bafa956e8eb1c3e744bc4616e276d3a4a438f` |
| official LIBERO commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| PAI formal JobId | `dlc1o37keilxr3sr` |
| PAI worker | 1 worker，2×A800 carrier，12 CPU，200 GiB |
| 科学进程 GPU | 仅 physical GPU0 可见；GPU1 不暴露 |
| W&B / 自动容错 | disabled / 0 次平台自动重启 |
| runtime identity | `2254:2254` |
| 启动测试 | 35/35 passed |

PAI prepare 尝试的 registry、日志与 CPFS 证据全部保留：v3
`dlc3r2jgqc8lu2p1` 因 controller 的
`env -i` 漏传 Phase-3 变量而失败；v4 `dlc1xieb62snpwdv` 因固定环境中的
LIBERO namespace/import 检查兼容性而失败；v5 `dlc6j1afexrn1am1` 成功。失败
尝试发生在 scientific application cells 之前，没有混入结果。v5 通过首个真实
cell 且终态成功后，v3/v4 两条已被替代的失败 PAI 服务记录依照两阶段删除协议于
12:26 UTC 清理并验证线上不存在；删除未触及本地 registry、ledger、日志或 CPFS
产物。没有创建 probe，未使用浏览器提交或监控。

## 3. 数据划分与 snapshot bank

- development：demo 0--19；
- calibration：demo 20--29；
- replay qualification：demo 30--39；
- formal controlled confirmation：demo 40--49；
- simulator init 0--19：保持未访问。

development/calibration/qualification 共抽取 320 个 effect segment。formal
访问只在 backend、链、预算、K 和源码哈希冻结后开始，并留下首个持久化账本
`formal_access_ledger/stove_moka/demo_40.complete.json`。formal demo bank 曾用于
先前 trace-level 工作，所以本阶段不是独立外部测试集。

backend 只允许从同 demonstration 的 effect-entry snapshot 复位，再通过正常
`env.step` 执行原 demonstration 的冻结 float32 action segment；没有直接设置
success 或目标 object state，也没有训练 actor/VLA/world model/effect verifier。

## 4. Qualification replay 与 K 选择

Qualification 共完成 400 次 exact replay。条件 effect 成功率为：

| 任务 | Effect | conditional success |
|-|-|-|
| `stove_moka` | `STOVE_TURNED_ON` | 0.56 |
| `stove_moka` | `MOKA_GRASPED` | 0.44 |
| `stove_moka` | `MOKA_ON_STOVE` | 0.88 |
| `stove_moka` | `MOKA_RELEASED_ON_STOVE` | 0.10 |
| `bowl_drawer` | `BOWL_GRASPED` | 0.90 |
| `bowl_drawer` | `BOWL_IN_BOTTOM_DRAWER` | 0.68 |
| `bowl_drawer` | `BOWL_RELEASED_IN_DRAWER` | 0.44 |
| `bowl_drawer` | `BOTTOM_DRAWER_CLOSED` | 0.00 |

六条候选链成功率依次为 0.30、0.44、0.10、0.68、0.32、0.00；没有链满足
冻结的 0.95 segment / 0.90 chain gate，eligible chain count=0，qualification
gate FAIL。按用户在 formal 访问前给出的 override，继续选择三条仅用于诊断的
链：

1. `S2_MOKA_GRASP_TO_ON_STOVE`；
2. `B1_BOWL_GRASP_TO_DRAWER`；
3. `B2_BOWL_IN_TO_RELEASED`。

Persistence calibration 每个 K 精确完成 150 行：

| K | chain success | mean action steps | backend failures |
|-|-|-|-|
| 2 | 0.2200 | 193.23 | 40 |
| 4 | 0.1267 | 202.81 | 40 |
| 8 | 0.0000 | 148.96 | 40 |

按预注册规则冻结选择 K=2。之后没有依据 formal 结果调整 K 或链。

## 5. Formal replay-only gate

三个链共享 `BOWL_IN_BOTTOM_DRAWER`，因此 formal gate 按 5 个唯一 effect ×
10 demo × 5 repetitions 完成 250 次 replay，而不是重复计算共享 segment。

| 任务 | Effect | conditional success |
|-|-|-|
| `stove_moka` | `MOKA_GRASPED` | 0.60 |
| `stove_moka` | `MOKA_ON_STOVE` | 0.66 |
| `bowl_drawer` | `BOWL_GRASPED` | 0.70 |
| `bowl_drawer` | `BOWL_IN_BOTTOM_DRAWER` | 0.58 |
| `bowl_drawer` | `BOWL_RELEASED_IN_DRAWER` | 0.58 |

链成功率为 0.50、0.48、0.44；有效 formal unit 数为 8、9、6。三个链均未达到
0.90 segment / 0.85 chain gate，第三条链也未达到至少 8 个有效 formal unit。
`formal_replay_gate.json` 因此为 `FORMAL_REPLAY_GATE_FAIL`，且明确记录后续结果
只能解释为 diagnostic。

## 6. 完整实验计数

| 实验 | 计划 | 实际 |
|-|-|-|
| 非 formal snapshot cells | 80 | 80 |
| qualification replay | ≥400 | 400 |
| K calibration | 450 | 450 |
| formal replay | unique-effect grid | 250 |
| non-formal smoke | 12 | 12 |
| formal main matrix | 900 | 900 |
| D1 delayed receipt | 180 | 180 |
| paired-unit audit | 150 | 150 |
| first-divergence interventions | data-dependent | 176（88 units） |
| mechanism ablations | 90 | 90 |
| cluster bootstrap | 10,000 | 10,000 |

900 行主矩阵严格包含 3 chains × 10 demos × 5 conditions × 6 arms。每个 arm
有 150 行，其中 35 行为同一批无效 snapshot unit 引发的
`REPLAY_BACKEND_FAILURE`。下表同时给出完整 raw grid 结果和排除预定义 backend /
broker / fault-injector error 后的 valid-primary 结果；后者只用于诊断，不能挽回
失败的 replay gate。

## 7. 主矩阵结果

### 7.1 完整 150-row/arm raw grid

| Arm | all chain success | clean success | faulted success | premature rate | mean steps |
|-|-|-|-|-|-|
| `B2_COMMAND_PROGRESS` | 0.1333 | 0.6333 | 0.0083 | 0.6333 | 95.91 |
| `B3_MONOLITHIC` | 0.1400 | 0.6667 | 0.0083 | 0.6267 | 97.10 |
| `POSTCHECK_RECOVERY` | 0.5533 | 0.7000 | 0.5167 | 0.1533 | 148.19 |
| `PERSISTENCE_RECOVERY` | 0.2333 | 0.5667 | 0.1500 | 0.0000 | 206.09 |
| `TYPED_MATCHED_RECOVERY` | 0.7000 | 0.7000 | 0.7000 | 0.0000 | 158.59 |
| `B6_FULL` | 0.7000 | 0.7000 | 0.7000 | 0.0000 | 164.44 |

B6 的 grounded advance precision=1.0、C3 contradiction recovery recall=1.0、
C4 false-positive advance rate=0、invalidated realization accuracy=1.0。它没有
premature advance，但有 10 个 timeout；`TYPED_MATCHED` 也有 10 个 timeout。B6
最大 resident slots=27，dangling parent=0，transition violation=0，truth/fault
leakage=0。

### 7.2 Backend-valid units（每 condition 23 个）

| Arm | C0 | C1 | C3 | C4 | C7 |
|-|-|-|-|-|-|
| `B2_COMMAND_PROGRESS` | 0.8261 | 0.0000 | 0.0000 | 0.0000 | 0.0435 |
| `B3_MONOLITHIC` | 0.8696 | 0.0000 | 0.0000 | 0.0000 | 0.0435 |
| `POSTCHECK_RECOVERY` | 0.9130 | 0.9130 | 0.8696 | 0.0000 | 0.9130 |
| `PERSISTENCE_RECOVERY` | 0.7391 | 0.2609 | 0.0000 | 0.2174 | 0.3043 |
| `TYPED_MATCHED_RECOVERY` | 0.9130 | 0.9130 | 0.8696 | 0.9565 | 0.9130 |
| `B6_FULL` | 0.9130 | 0.9130 | 0.8696 | 0.9565 | 0.9130 |

Valid primary endpoint 使用 C1/C3/C4/C7 的 92 个 paired units。B6 与
`TYPED_MATCHED` 均为 0.9130；POSTCHECK 为 0.6739；PERSISTENCE 为 0.1957。

### 7.3 失败分解

| Arm | replay backend | premature advance | timeout | success/150 |
|-|-|-|-|-|
| B2 | 35 | 95 | 0 | 20 |
| B3 | 35 | 94 | 0 | 21 |
| POSTCHECK | 35 | 23 | 9 | 83 |
| PERSISTENCE | 35 | 0 | 80 | 35 |
| `TYPED_MATCHED` | 35 | 0 | 10 | 105 |
| B6 | 35 | 0 | 10 | 105 |

没有 receipt-broker error、fault-injector error、action-budget exceed 或
unclassified failure。

## 8. 配对统计

最强 observed strong baseline 是 `TYPED_MATCHED_RECOVERY`。

| 比较 | paired n | B6-only wins | baseline-only wins | absolute diff | Holm-adjusted p |
|-|-|-|-|-|-|
| B6 vs POSTCHECK | 92 | 22 | 0 | 0.2391 | 9.54e-7 |
| B6 vs PERSISTENCE | 92 | 66 | 0 | 0.7174 | 8.13e-20 |
| B6 vs TYPED_MATCHED | 92 | 0 | 0 | 0.0000 | 1.0 |

以 source demo 为 cluster 的 10,000 次 bootstrap：

- B6 - POSTCHECK：0.2391，95% CI `[0.2174, 0.2500]`；
- B6 - PERSISTENCE：0.7174，95% CI `[0.6019, 0.8375]`；
- B6 - `TYPED_MATCHED`：0.0000，95% CI `[0.0000, 0.0000]`。

B6 相对最强 strong baseline 没有任何增量，失败
`margin>=0.10` 和 `bootstrap lower>0` 两个主要 gate。

完整 grid 上的绝对成功率 effect size（B6 - baseline）进一步定位为：

| Baseline | C0 | C1 | C3 | C4 | C7 |
|-|-|-|-|-|-|
| POSTCHECK | 0.0000 | 0.0000 | 0.0000 | 0.7333 | 0.0000 |
| PERSISTENCE | 0.1333 | 0.5000 | 0.6667 | 0.5667 | 0.4667 |
| `TYPED_MATCHED` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Baseline | S2 moka grasp→stove | B1 bowl grasp→drawer | B2 bowl in→released |
|-|-|-|-|
| POSTCHECK | 0.1600 | 0.1600 | 0.1200 |
| PERSISTENCE | 0.4200 | 0.6600 | 0.3200 |
| `TYPED_MATCHED` | 0.0000 | 0.0000 | 0.0000 |

这些 per-condition/per-chain effect size 包含配对的 replay-backend failure，主要
用于定位差异；primary inference 仍使用预定义的 92 个 backend-valid faulted units。

## 9. D1 delayed receipt

排除 replay-backend error 后每 arm 有 23 个 D1 units：

| Arm | chain success | steps | retries | reobserves |
|-|-|-|-|-|
| B2 | 0.8696 | 137.13 | 0.000 | 0.000 |
| B3 | 0.8696 | 137.48 | 0.000 | 0.000 |
| POSTCHECK | 0.9130 | 182.78 | 0.217 | 2.609 |
| PERSISTENCE | 0.7826 | 205.09 | 0.435 | 4.130 |
| `TYPED_MATCHED` | 0.9130 | 181.35 | 0.174 | 2.739 |
| B6 | 0.9130 | 184.13 | 0.261 | 2.478 |

D1 没有给 B6 带来成功率优势，只显示小幅成本/校准差异。

## 10. 首分歧因果重放

150 个 paired units 中，115 个观察到 decision divergence；其中只有 88 个通过
byte-identical prefix gate，产生 176 个 forced-decision replays。B6 natural
decision 在 54/88 个 unit 中并列为最优，`decision_causal_win_rate=0.6136`，低于
0.70 gate。

| 强制 decision | rows | immediate effect | eventual chain | irreversible failure |
|-|-|-|-|-|
| ADVANCE | 88 | 0.2159 | 0.5682 | 0.4091 |
| REOBSERVE | 60 | 0.8833 | 0.9667 | 0.0000 |
| RETRY | 23 | 0.7826 | 0.6957 | 0.0000 |
| ROLLBACK | 5 | 1.0000 | 1.0000 | 0.0000 |

这些 decision 子集不是随机同分布组，只能解释局部 first-divergence 反事实，不能
直接把表中差异当作总体策略效果。

## 11. 机制消融与代码反解

`B6_NO_PROVENANCE` 在 C4/C7 上与 B6 完全相同：全 60 行 success=0.7167，
排除 14 个 backend error 后 success=0.9348。C4 根本不发 valid witness，C7 的
后续成功 retry 发出的 witness 已有 command parent，所以当前 fault schedule
没有激活“unlinked witness 被拒绝”的关键路径；不能据此支持 provenance claim。

`B6_NO_INVALIDATION` 在 C3 上 30 行成功率为 0；排除 7 个 backend error 后仍为
0/23（21 premature advance、2 timeout）。B6 对应 valid C3 为 20/23=0.8696。
这证明 contradiction invalidation 对 B6 本身必要。但 `TYPED_MATCHED` 的简单通用
rollback 也是 20/23，所以没有证明 B6 lineage 表示的增量价值。

完整代码路径、提升/下降原因与不成立的机制路径见
`MECHANISM_REVERSE_ENGINEERING.md`。

## 12. 实现与公平性审计

paired audit 中 123/150 unit 通过、27/150 失败。失败全部发生在第一次 decision
index 0：各 arm 的 executed-action hashes 相同，但 simulator-state / event-prefix
hashes 已不同。分布为：

- `B2_BOWL_IN_TO_RELEASED`：14；
- `S2_MOKA_GRASP_TO_ON_STOVE`：11；
- `B1_BOWL_GRASP_TO_DRAWER`：2。

根因是当前矩阵为每个 arm 独立复位并执行接触敏感的 MuJoCo 轨迹，而不是真正
同步广播一次物理 prefix 的 shared broker。完整 flattened simulator state 和相同
action hash 仍不足以保证 controller/contact 轨迹 bit-identical。该问题在 formal
demo 已打开后不能通过事后改 audit 消除；需要新的冻结实验版本才能修复。

`final_decision.json` 的 frozen precedence 先检查 210 行
`REPLAY_BACKEND_FAILURE`（35×6 arms）和完整 grid，再检查 replay gate，因此机器
终态为 `BLOCKED_BY_IMPLEMENTATION`。即使改变 precedence，qualification 和 formal
replay gate 也都失败，最多只能得到 `BLOCKED_BY_REPLAY_BACKEND`，仍不允许 B6
行为优势主张。

## 13. Gate 总表

| Gate | 结果 |
|-|-|
| qualification replay | FAIL |
| formal replay-only | FAIL |
| exact 900-row grid | PASS |
| B6 margin ≥0.10 vs best strong | FAIL（0.0） |
| cluster CI lower >0 | FAIL（0.0） |
| grounded advance precision ≥0.95 | PASS（1.0） |
| C3 contradiction recovery ≥0.80 | PASS（1.0） |
| C4 false-positive advance ≤0.05 | PASS（0.0） |
| clean success degradation ≤0.02 | PASS（0.0） |
| clean step overhead ≤15% | PASS（3.09%） |
| decision causal win ≥0.70 | FAIL（0.6136） |
| provenance/resident/pairing correctness | FAIL（pairing 27/150） |

## 14. 视频与 PAI 终态

视频 policy 共枚举 748 个唯一 rollout 请求：590 个 failure、115 个 first-decision
divergence、73 个可获得的 representative recovery；三个集合允许重叠。最终生成
496 个 MP4，252 项报告确定性的 invalid-snapshot render error。后者恰好来自 7 个
无效 chain-demo unit × 6 conditions × 6 arms，并非额外的错误类型。

30 个 condition-arm recovery strata 中，14 个有至少 5 个成功样本，均保存 5 个；
D1 的 B6 与 POSTCHECK 只能分别提供 2 个和 1 个；另 14 个 strata 没有成功恢复可供
渲染。因此 73 是 formal 数据中所有可获得的代表性恢复，而不是任意截断。

496 个成功渲染中有 4 个 outcome mismatch：视频重放把冻结主表中的 failure 重放为
success，分别是 C0/B3/B2-chain/demo41、D1/B3/B2-chain/demo41、
D1/B2/B2-chain/demo46、C0/B2/S2-chain/demo47。这与 paired audit 的接触敏感
非 bit-identical 现象一致。视频只作诊断展示，不重写 `formal_results.jsonl`、统计量
或最终 gate。应用生成的 `SHA256SUMS` 已逐项校验通过。

PAI formal 作业 `dlc1o37keilxr3sr` 的终态为 `Succeeded`：2026-08-14
06:54:51 UTC 开始运行，12:01:18 UTC 结束，Duration=18,485 秒；单个 workload
Pod 成功，AIMaster/平台自动重启关闭，持久化标记 owner 为 `2254:2254`。控制面
曾出现一次短暂 GetJob 网络返回码 2，随后同一 JobId 持续运行并成功，不是作业重启
或实验失败。

## 15. 最终科学判断

本阶段不能验证原始 B6 incremental-value idea。更准确的证据边界是：

1. replay backend 本身未达到 competent executor gate；
2. 独立 arm 重放没有满足第一次 decision 前的严格信息公平；
3. 在可诊断的有效 units 上，typed evidence 和 contradiction recovery 很重要；
4. B6 相对弱臂和 POSTCHECK 有提升，但相对最强 `TYPED_MATCHED` 成功率完全打平；
5. invalidation 是必要机制，command-parent provenance 的增量未被当前干预激活；
6. 不启动 learned verifier、Mem-0、ACT、Pi0.5 或新的 executor。

因此不作 VLA improvement、LIBERO generalization、N3、paper acceptance 或
paper-level mechanism claim。

## 16. 主要产物

- 原始计划：`ORIGINAL_PHASE3_PLAN.txt`；
- preregistration 与冻结 contracts：本目录 YAML/JSON；
- snapshot/action manifests 与 SHA256：`snapshot_bank_*`、`selected_chain_manifest.json`；
- qualification/formal raw JSONL：`replay_qualification_results.jsonl`、
`formal_replay_results.jsonl`、`formal_results.jsonl`；
- D1、paired audit、causal replay、ablations：对应 JSONL；
- 统计：`behavior_summary.json`、`cluster_bootstrap.json`、`paired_tests.json`；
- 决策与失败分解：`final_decision.json`、`FINAL_DECISION.md`、
`failure_decomposition.jsonl`、`failure_cases.md`；
- 机制反解：`MECHANISM_REVERSE_ENGINEERING.md`；
- learned verifier readiness：`LEARNED_EFFECT_VERIFIER_READINESS.md`（结论为
NOT READY，未启动任何模型）；
- PAI 执行边界：`PAI_EXECUTION_AUDIT.md`；
- 测试、行数与结构校验：`VALIDATION.md`；
- PAI workload 原始证据：`pai/workload/`；
- PAI 控制面合同、终态与删除证据：`pai/control-plane/`；
- 完整字节清单：`SHA256SUMS`。

## 17. GitHub 发布记录

- main 与 phase3-effect-boundary-replay 最终提交：[5592c3e0686847aea6c55e887b964078fa5bedb0](https://github.com/mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback/commit/5592c3e0686847aea6c55e887b964078fa5bedb0)
- 完整 Step4 目录：[experiments/r16p19_libero_phase3](https://github.com/mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback/tree/main/experiments/r16p19_libero_phase3)
- 远端使用 SSH 地址 git@github.com:mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback.git；发布时使用非交互 BatchMode，没有执行 SSH 登录。

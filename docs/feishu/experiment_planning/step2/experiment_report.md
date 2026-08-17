# R16-P19 Phase-1B（Step2）Actor Upgrade 与闭环行为验证实验报告

实验日期：2026-08-13
Benchmark：LIBERO-10（2 个冻结任务）
唯一终态：`BLOCKED_BY_ACTOR_V2`

## 1. 结论摘要

本轮按预注册顺序完成了 actor–memory 解耦、小型共享 State-ACT 训练与
qualification，以及 primary 失败后唯一允许的 Per-Effect State-ACT fallback
训练与 qualification。两个 actor family 都未达到冻结门槛：

| Actor | Qualification rollout | Full-task success | Minimum per-effect success | 门槛 | 结论 |
|-|-|-|-|-|-|
| Primary：共享 Effect-Conditioned State-ACT | 40 | 18/40 = 0.45 | 0.40 | 0.80 | FAIL |
| Fallback：8 个 Per-Effect State-ACT | 40 | 24/40 = 0.60 | 0.50 | 0.80 | FAIL |

因此：

- 没有 actor 被冻结；
- formal init 0–19 没有执行任何 actor rollout；
- 2 tasks × 20 init × 5 conditions × 4 memory arms 的 800 次闭环矩阵没有被授权；
- 10,000 次 paired bootstrap 没有运行；
- 不能给出 `PASS_PHASE1_BEHAVIOR`，也不能据此给出
`REJECT_CORE_MECHANISM`；
- 按冻结决策规则，唯一合法结论是 `BLOCKED_BY_ACTOR_V2`。

这意味着当前证据仍被低层 actor competence 限制，而不是 R16-P19 核心
memory mechanism 已被闭环行为实验否证。

## 2. 科学问题与冻结边界

Phase-1B 只检验：在不向 actor 泄露 memory state 的条件下，小型 actor 是否
足以通过 clean competence gate，并由此解锁原样的 R16-P19 闭环行为矩阵。

本轮没有修改：

- `r16p19/memory.py` 及七类 epistemic state 语义；
- effect ontology、receipt/provenance 规则；
- C0–C7 fault 定义；
- B2/B3/B5/B6 memory 定义；
- demo 0–29 / 30–39 / 40–49 数据拆分；
- formal init 0–19；
- actor gate、behavior gate 或 paired-bootstrap 门槛。

关键受保护文件在 PAI runtime 中再次核验，SHA256 保持为：

| 受保护对象 | SHA256 |
|-|-|
| `r16p19/memory.py` | `4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5` |
| effect ontology | `1a8ee265bc23d714b38603617bb3d7cf426a50981d90f343745b281277dd6160` |
| fault matrix | `6ba25e31add729a5e623b3c18014190aa947d85586034dc4d38a2dc843e44797` |
| split manifest | `f60640dc65c7a403c560ff9cf6a1e9eef0e1b09240ba60ba77f8bb6cbdf3d343` |

预注册提交为 `fd0b3ef`，预注册文件 SHA256 为
`8226c2ae023c821b3bc9c59e0750f340655b09bb9e2d94eb5825b2885a88113c`。
官方 LIBERO 源码固定在
`8f1084e3132a39270c3a13ebe37270a43ece2a01`。

## 3. LIBERO 任务、数据与 init 审计

选择了两个具有连续多 effect、抓取、放置和机构操作的 LIBERO-10 任务：

1. `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
（简称 `stove_moka`）；
2. `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
（简称 `bowl_drawer`）。

两项任务各有 100 个官方 initial states。冻结用途为：

- actor development / qualification：init 20–39；
- formal actor 与 memory evaluation：init 0–19；
- init 40–99 不使用。

数据按 source episode 拆分，train、calibration、trace-test 两两互斥，没有跨
split 复制 episode。训练集 effect 样本/有效 padded action chunk 数如下：

| Task | Effect | Train samples/chunks |
|-|-|-|
| stove_moka | STOVE_TURNED_ON | 2,714 |
| stove_moka | MOKA_GRASPED | 3,742 |
| stove_moka | MOKA_ON_STOVE | 1,643 |
| stove_moka | MOKA_RELEASED_ON_STOVE | 206 |
| bowl_drawer | BOWL_GRASPED | 2,934 |
| bowl_drawer | BOWL_IN_BOTTOM_DRAWER | 1,569 |
| bowl_drawer | BOWL_RELEASED_IN_DRAWER | 587 |
| bowl_drawer | BOTTOM_DRAWER_CLOSED | 2,316 |

总计 train 15,711 个样本，calibration 5,404 个样本。state/action alignment
mismatch 为 0；缺失 effect segment 被排除而不伪造样本；短 chunk 仅在同一
effect segment 内用最后一个 action padding。详细 action 统计、gripper 类别
平衡、transition boundary 和 chunk-length 分布见
`experiments/r16p19_libero_phase1b/actor_data_audit.json`。

数据集 SHA256：

- stove_moka：
`6b30906a52a5741e98ef447d27e7066d6c0be4a5f7acd7ecaf1cb7468aca4aa9`；
- bowl_drawer：
`703950f48a3c49dfde61be489ade91527f16e1449b4f29a85f2e51153cef3638`。

## 4. Actor–memory 解耦与实现验证

统一低层接口实现为：

`SkillActor.action_chunk(state_history, task_id, effect_id, execution_mode)`

`execution_mode` 仅允许 `EXECUTE` 或 `RETRY`。Actor 输入只包含最近 4
帧 padded simulator state、task、effect 和 execution mode；不读取 memory
summary、epistemic state、simulator effect truth 或 future state。Memory
manager 只通过 ADVANCE、RETRY、REOBSERVE、ROLLBACK/REPLAN、SAFE_STOP
等高层 decision 影响行为。

回归测试验证了：相同 state history、task、effect、actor seed 且高层 decision
相同时，B2/B3/B5/B6 的 actor input hash 和 action bytes 一致；只有首次
memory decision 分叉之后才允许行为分叉。

开发机验证：

- pytest：16 passed，0 failed，0 skipped；
- CPU smoke：`CPU_SMOKE_PASS`，明确 `cuda_used=false`；
- primary 参数量 3,186,951，单 batch loss 1.053544879，输出 shape
`[2, 8, 7]`；
- 开发机 bounded GPU simulator smoke 在创建环境前因本机暴露 0 个 EGL
device 而停止，未启动训练、未产生科学 rollout。随后 PAI 固定环境完成了
80 个真实 LIBERO qualification rollout，因此该本地图形限制不构成
`BLOCKED_BY_IMPLEMENTATION`。

## 5. PAI 执行与可恢复性合同

正式训练和仿真均在 PAI DLC 完成。每个作业申请 2×A800，GPU 0 为唯一
active device，GPU 1 保留为空闲卡；GPU 1 的 dmon 记录中 SM、memory 和
framebuffer 使用均为 0。AIMaster/平台自动容错关闭，平台 restart 为 0；
应用层 auto-resume 和完整 checkpoint 合同
（model、optimizer、scheduler、RNG、global step）启用。所有 2,500-step
候选 checkpoint 均保留，没有只保存最佳 checkpoint。

| Stage | PAI JobId | 状态 | Duration | W&B |
|-|-|-|-|-|
| train-primary | `dlcvfiubutw83kz2` | Succeeded | 307 s | [j0a3zaly](https://wandb.ai/chen_jian-cj-workspace/r16p19-libero-phase1b/runs/j0a3zaly) |
| qualify-primary | `dlc1i6mya2tawqhu` | Succeeded | 1,605 s | [wwmm1fl7](https://wandb.ai/chen_jian-cj-workspace/r16p19-libero-phase1b/runs/wwmm1fl7) |
| train-fallback 首次尝试 | `dlc9iq5myqo4j49t` | Failed（已归档且不合格） | 470 s | [jh2v9bhf](https://wandb.ai/chen_jian-cj-workspace/r16p19-libero-phase1b/runs/jh2v9bhf) |
| train-fallback 修正后全量重跑 | `dlc1el69zetszgvb` | Succeeded | 760 s | [8kdfiu08](https://wandb.ai/chen_jian-cj-workspace/r16p19-libero-phase1b/runs/8kdfiu08) |
| qualify-fallback | `dlc1lszxmyuu1xkh` | Succeeded | 1,573 s | [ytm171kv](https://wandb.ai/chen_jian-cj-workspace/r16p19-libero-phase1b/runs/ytm171kv) |

W&B entity 固定为 `chen_jian-cj-workspace`，project 为
`r16p19-libero-phase1b`。

## 6. Primary actor

Primary 是一个跨两个 task、八个 effect 共享的 Effect-Conditioned State-ACT：
history 4、action horizon 8、executed prefix 4、hidden 256、4 层
Transformer、8 heads、dropout 0.1，共 3,186,951 参数。

训练使用 15,711 个 train 样本和 5,404 个 calibration 样本；运行从 step 0
开始，在 step 15,000 触发预注册 early stopping。完整 checkpoint 保留在
2,500、5,000、7,500、10,000、12,500、15,000。只按 calibration macro
loss 选择 step 10,000：

- calibration selection metric：1.1120355211712118；
- checkpoint SHA256：
`22f6079221f8e6bf3c8858ce41406848f0e26e595a0ae2f2c0ae36dd7d232c15`；
- normalization SHA256：
`9725c1da980f0416ca3f421f2e4bb2202ca248e608e0658854ecd874c4eac0c8`；
- qualification 结果没有参与 checkpoint selection。

Primary 在 init 20–39 的 qualification：

| Task | Full-task | 各 effect success |
|-|-|-|
| stove_moka | 8/20 = 0.40 | STOVE 1.00；MOKA_GRASPED 0.40；MOKA_ON_STOVE 0.40；MOKA_RELEASED 0.40 |
| bowl_drawer | 10/20 = 0.50 | BOWL_GRASPED 0.85；BOWL_IN 0.80；BOWL_RELEASED 0.80；BOTTOM_DRAWER_CLOSED 0.50 |
| 合计 | 18/40 = 0.45 | minimum = 0.40 |

Primary 有 22/40 repeated effect-chunk-limit failure，mean action steps 为
227.275；22 个失败视频全部保留。由于 minimum 0.40 < 0.80，primary gate
失败，按预注册规则激活唯一 fallback。

## 7. Fallback 训练、实现修正与 checkpoint

Fallback 为 8 个互相独立的 Per-Effect State-ACT。每个 actor 使用相同输入/
输出合同，hidden 128、2 层 Transformer、4 heads、410,503 参数；总参数量
3,284,024。它仍然不读取任何 memory 输入。

首次 fallback 训练在完成 5/8 个 actor 后，于
`BOWL_IN_BOTTOM_DRAWER` 的 normalization 阶段失败：该 effect 的 12,552
个 gripper target 全为 positive、negative 为 0，旧构造器错误地要求两类都
存在。这是一个可定位的实现边界条件，不是 actor 科学结果。

修正严格限制为：当任一 gripper class 缺失时令 positive weight = 1.0，即
标准 unweighted BCE；两类都存在时仍使用冻结的 inverse-frequency
weighting。没有合成样本，也没有修改 architecture、超参数、数据/split、
memory 语义或门槛。失败 run 的五个 partial actor 被完整归档且永不参与
selection。修正后在隔离的 `v2` 输出/checkpoint namespace 中将全部 8 个
actor 从 step 0 重跑；相关回归测试包含在 16 个通过测试中。

所有 fallback 都只用对应 effect 的 calibration loss 选择 step 5,000：

| Effect | Final complete step | Retained complete steps | Selection metric | Selected checkpoint SHA256 |
|-|-|-|-|-|
| STOVE_TURNED_ON | 9,500 | 2,500 / 5,000 / 7,500 / 9,500 | 0.534435251967 | `f5a486c6621f6e9bd13da7e0576f5c6a63838ff07bf466a7b5f5a44685569599` |
| MOKA_GRASPED | 9,000 | 2,500 / 5,000 / 7,500 / 9,000 | 1.041858033880 | `c4626eab2e68f10f800fd3b109a4e2a1c1a55f1cc8aa94e1d5187e295792618a` |
| MOKA_ON_STOVE | 9,000 | 2,500 / 5,000 / 7,500 / 9,000 | 0.271497289564 | `d968f318b93220f7290bb075ebb966355d58f36826284c28a135493beb0a7100` |
| MOKA_RELEASED_ON_STOVE | 9,500 | 2,500 / 5,000 / 7,500 / 9,500 | 3.932773947716 | `3dddeae3fd326f5c176891b60d51fd59f1963d96b5413296f9e856b5179e223d` |
| BOWL_GRASPED | 9,000 | 2,500 / 5,000 / 7,500 / 9,000 | 1.008216155612 | `1a698af86f2cc49b1a42d28e823bcdadf9dc754a517962bfa34e05637d6b6852` |
| BOWL_IN_BOTTOM_DRAWER | 10,000 | 2,500 / 5,000 / 7,500 / 10,000 | 0.316649632180 | `90405192f24361fe64a6a2aaaff58804685fcb00b666207f48aff36718e37b53` |
| BOWL_RELEASED_IN_DRAWER | 9,000 | 2,500 / 5,000 / 7,500 / 9,000 | 1.212063908577 | `b108b13bf8b50c707f7dec61561ce1a306710833a5905c52d863831f4869fbbc` |
| BOTTOM_DRAWER_CLOSED | 10,000 | 2,500 / 5,000 / 7,500 / 10,000 | 1.102613344874 | `1d5cc5df9ec53c5464036b6ae303d89abdd041d0603d0990bd27213b2f541907` |

八个 selected checkpoint 的组合身份 SHA256 为
`41f86b8b14a3f578ac33727814f7aafbad62c7797ee20240c21ce99ef4c63d36`，
组合 normalization SHA256 为
`7934b0fc17c7f793795fad2551e3fa2e5a5b80137803c11be22ec3209be1b0f0`。

## 8. Fallback qualification 与失败分解

Fallback 在同一冻结 init 20–39 上执行一次 40-rollout qualification：

| Task | Full-task | Mean action steps | 各 effect success |
|-|-|-|-|
| stove_moka | 10/20 = 0.50 | 226.05 | STOVE 0.95；MOKA_GRASPED 0.50；MOKA_ON_STOVE 0.50；MOKA_RELEASED 0.50 |
| bowl_drawer | 14/20 = 0.70 | 195.65 | BOWL_GRASPED 0.75；BOWL_IN 0.70；BOWL_RELEASED 0.70；BOTTOM_DRAWER_CLOSED 0.70 |
| 合计 | 24/40 = 0.60 | 210.85 | minimum = 0.50 |

失败分解：

- stove_moka：9 次首先卡在 `MOKA_GRASPED`，1 次卡在
`STOVE_TURNED_ON`；
- bowl_drawer：5 次首先卡在 `BOWL_GRASPED`，1 次卡在
`BOWL_IN_BOTTOM_DRAWER`；
- 合计 16/40 为 repeated effect-chunk-limit，16 个失败视频全部保留。

Primary 的对应 failure 分布为：`MOKA_GRASPED` 12 次、
`BOTTOM_DRAWER_CLOSED` 6 次、`BOWL_GRASPED` 3 次、
`BOWL_IN_BOTTOM_DRAWER` 1 次。

这些 clean qualification 没有 memory arm 和 fault injection，因此这里的
16 个 fallback failure 被归为 actor skill / timeout-or-repeated-loop；
memory decision、effect verifier 和 fault injector failure 均不能在此阶段
估计。日志能定位第一个未达到的物理 effect，但不能仅凭
`EFFECT_CHUNK_LIMIT` 进一步区分抓取几何、gripper timing、endpoint
control 或它们的组合。

## 9. Fail-closed 门禁与未运行阶段

Fallback minimum per-effect success 为 0.50，仍低于 0.80。门禁随即写入：

- `selected_actor_manifest.json`：`NO_ACTOR_PASSED_QUALIFICATION`；
- `formal_actor_gate.json`：`NOT_RUN`，formal init seen = []；
- `closed_loop_results.jsonl`：stage status `NOT_RUN`，observed rollouts = 0；
- `paired_bootstrap.json`：`NOT_RUN`，observed repetitions = 0；
- `mechanism_mediation.json`：`NOT_ESTIMABLE`；
- `final_status.json` / `FINAL_DECISION.md`：
`BLOCKED_BY_ACTOR_V2`。

这不是缺少算力或实验意外中断，而是预注册 actor competence gate 的正常
否决结果。继续查看 formal init、运行 800 次矩阵或根据 qualification 结果
调参都会破坏冻结协议，因此均未执行。

## 10. 与 Phase-1 actor-free trace 证据的关系

此前 actor-free trace gate 仍有效：

- B3 false completion = 0.50；
- B6 false completion = 0；
- B6 contradiction recovery recall = 1.00。

这些结果支持 R16-P19 在冻结 trace 上的机制级写回与矛盾恢复逻辑，但不能
升级为闭环行为 PASS。本轮也没有产生足以否定核心机制的 800-rollout 证据。
因此科学解释必须保持为：

> mechanism-level trace evidence remains positive; behavior-level validation
> remains blocked by actor competence.

## 11. 可复现性、代码与证据索引

GitHub 仓库：
`git@github.com:mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback.git`
（main 分支，使用 Git SSH transport 发布；未进行交互式 SSH 登录）。

关键提交：

| Commit | 内容 |
|-|-|
| `fd0b3ef` | 冻结 Phase-1B preregistration |
| `b38460a` | 数据与 init 审计 |
| `bf06ef5` | actor-memory 解耦、pipeline 与 regression tests |
| `b524327` | primary training 原始证据与 checkpoints |
| `35f314d` | primary qualification、视频与本地验证证据 |
| `393ee7e` | single-class gripper normalization 边界修正 |
| `49e2bfe` | 修正后 fallback 全量重跑的隔离 namespace |
| `b218d52` | 失败 run 全量归档与 PAI control patch |
| `b62f20b` | 最终 gate 结果、全部 corrected fallback 证据、报告与 checksums |

主要路径：

- 最终机器可读交付：`experiments/r16p19_libero_phase1b/`；
- 原始 training/qualification 与视频：`artifacts/phase1b/experiment/`；
- 所有 checkpoint：`artifacts/phase1b/checkpoints/`；
- PAI control-plane/runtime：`artifacts/phase1b/pai/`；
- 首次失败 fallback 完整证据：
`artifacts/phase1b/failed/dlc9iq5myqo4j49t/`；
- pytest、CPU smoke、开发机 GPU smoke：
`artifacts/phase1b/local-validation/`；
- PAI 模板与 launcher 的可审计 patch：
`experiments/r16p19_libero_phase1b/pai/controller_patches/`。

`experiments/r16p19_libero_phase1b/SHA256SUMS` 覆盖最终交付目录中的所有
结果文件；`artifacts/phase1b/SHA256SUMS` 覆盖 466 个 raw evidence 文件。
原始 PAI readback 中的凭证字段已做 redaction；W&B 只保存 run receipt/URL，
不提交私密 credential 或本地 cache。

## 12. 最终决策

`FINAL_STATUS = BLOCKED_BY_ACTOR_V2`

Primary 与唯一预注册 fallback 均未通过 clean qualification 的
minimum-per-effect ≥ 0.80 门槛。实验按规则停止在 actor gate；formal
competence、800 次 memory-conditioned 闭环矩阵和 paired bootstrap 均未
运行。若后续继续，必须新建独立 preregistration 和新的 actor 研究阶段，
不能在本轮 qualification/formal split 上继续调参，也不能把本轮结论改写为
R16-P19 核心机制失败。

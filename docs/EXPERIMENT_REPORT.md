# R16-P19 Phase-1 LIBERO 实验报告

- 实验日期：2026-08-13
- 科学终态：`BLOCKED_BY_ACTOR`
- PAI 作业终态：`Succeeded`

## 一句话结论

完整 R16-P19 写回（B6）在 actor-free trace gate 上通过全部正确性门槛，
并把 B3 的 50% false completion 降为 0，同时达到 100% contradiction
recovery；但 tiny state-BC actor 与预注册 fallback 都没有通过每个 effect
至少 80% 的 clean competence gate，因此 800 次记忆条件闭环比较按规则未
运行。本轮支持“显式区分认知状态有助于避免假完成和处理反证”这一机制层
判断，但尚不能证明它提高端到端 LIBERO 任务成功率。

## 1. 结论与决策

- **Actor-free 机制门槛：PASS。** 1,120 个
  task-demo-condition-arm case 完成，B6 的 decision accuracy、realized
  precision/recall、contradiction detection/recovery recall 和 recovery
  routing accuracy 均为 1.000。
- **Learned actor competence：FAIL。** 最差单 effect 成功率 0.45，低于
  预注册阈值 0.80；full-task clean success 为 0.60（24/40）。
- **Fallback competence：FAIL。** nearest-demo phase script 的 full-task
  success 与所有 effect success 均为 0。
- **800 次闭环与 paired bootstrap：NOT_RUN。** 这是预注册的 fail-closed
  行为，而非漏跑；actor 不合格时禁止解释 memory-conditioned task success。
- **最终状态：BLOCKED_BY_ACTOR。** 不是 `PASS_PHASE1`，也不是
  `REJECT_CORE_MECHANISM`。下一步需要先取得合格的共享 actor，再保持 memory
  实现与评估清单不变完成闭环门槛。

## 2. 被验证的 idea

核心 idea 是不能把“命令已发出、模型想象成功、传感器观察、独立验证、物理
实现”混成一个 progress bit。每个 effect 必须显式处于 `REQUESTED`、
`IMAGINED`、`OBSERVED`、`VERIFIED`、`REALIZED`、`STALLED` 或
`INVALIDATED_REALIZATION`；物理证据必须带 provenance receipt；`REALIZED`
之后出现反证时必须失效、阻断依赖进度并生成非空恢复路径。

本轮只验证 actor-free epistemic trace semantics，以及一个不足 10M 参数的
privileged-state tiny BC actor 所提供的因果 sanity gate。它不是 Pi0.5、
大型 VLA、DINO-WM、Mem-0 或 official ACT 的效果实验，也不能支持“VLA 性能
提升”主张。

## 3. LIBERO 任务选择

1. **turn on the stove and put the moka pot on it**：同时包含可逆开关、
   抓取、放置与释放，适合 no-op、delay、realization 后 reversal 和 observed
   contradiction。
2. **put the black bowl in the bottom drawer and close it**：包含
   containment、release 与依赖前序效果的抽屉闭合，能检验错误提前推进与依赖
   effect 失效后的恢复。

两项均来自官方 LIBERO-10，每项 50 条 demonstration。固定划分为 demo 0–29
train、30–39 calibration、40–49 trace-test，并固定 20 个 simulator
evaluation init。所有同源变体留在同一 split。

## 4. 实验设计

### Effect 与证据

每个任务定义四个顺序 effect。物理 evidence receipt 包含 `episode_id`、
`event_index`、`timestamp`、sensor/view、frame digest、`effect_id` 和 evidence
type。同一物理 frame 即使换 evidence ID，也不能充当独立验证。resident
memory 固定 32 slots，provenance ledger append-only，并对 dangling parent
fail closed。

### Fault conditions

- C0 clean；C1 command no-op；C2 delayed effect；C3 realized 后物理 reversal。
- C4 单相机 false positive；C5 同一 frame 换 evidence ID；C6 40 个无关事件
  施加 32-slot memory pressure；C7 imagined success 后 observed failure。

### Memory arms

- B1 sliding recent history；B2 command-as-progress；B3 monolithic writeback。
- B4 typed states、无 contradiction recovery；B5 typed states 加 verification、
  无 recovery。
- B6 full R16-P19；B7 oracle effect ledger upper bound。

B1–B6 在全部 160 个 task-demo-condition 组合上收到 byte-identical event
streams；核验结果为 160/160 每组只有一个 stream hash。B7 只作为 oracle
upper bound。

## 5. Actor-free trace gate

| 指标 | B3 | B4 | B5 | B6 | B7 oracle |
|---|---:|---:|---:|---:|---:|
| decision accuracy | 0.500 | 0.875 | 0.625 | **1.000** | 1.000 |
| false completion | 0.500 | 0.000 | 0.000 | **0.000** | 0.000 |
| contradiction detection recall | 0.000 | 1.000 | 1.000 | **1.000** | 1.000 |
| contradiction recovery recall | 0.000 | 0.000 | 0.000 | **1.000** | 1.000 |
| recovery routing accuracy | 0.000 | 0.667 | 0.000 | **1.000** | 1.000 |

其他 B6 结果：

- premature advance 0；realized precision/recall 均为 1.000；
- evidence alias acceptance 0，且 20 次 alias challenge 全部被拒绝；
- duplicate event idempotency 1.000；
- dangling parent 0；transition violation 0；最大 resident slots 恰为 32；
- p50/p95 latency 为 5.575/79.388 微秒；
- clean decision accuracy 1.000、unnecessary recovery 0，且决策不是退化为总是
  `REOBSERVE` 或 `SAFE_STOP`。

结果表明 typed state 本身不足以完成目标；verification 与显式
invalidation/recovery 路径需要同时存在。

## 6. Tiny state-BC actor

actor 是 4 层、hidden size 256、action chunk 8 的 retrieval-augmented MLP，
共 239,160 参数；使用 15,350 个训练样本，在 PAI 上训练 3,000 optimizer
steps。loss 从 step 1 的 1.13038 降至 step 3,000 的 0.04931。step 1,000、
2,000、3,000 均有包含 model、optimizer、scheduler、RNG 和 global step 的
`COMPLETE` checkpoint。

### Learned actor clean competence（40 rollouts）

- `stove_moka`：STOVE_TURNED_ON 1.00；MOKA_GRASPED 0.80；MOKA_ON_STOVE
  0.75；MOKA_RELEASED_ON_STOVE 0.75；full task 15/20。
- `bowl_drawer`：BOWL_GRASPED 0.90；BOWL_IN_BOTTOM_DRAWER 0.65；
  BOWL_RELEASED_IN_DRAWER 0.60；BOTTOM_DRAWER_CLOSED 0.45；full task 9/20。
- 总体 full task 24/40 = 0.60；minimum per-effect 0.45，低于 0.80 门槛。

### 预注册 fallback（40 rollouts）

nearest-demo phase script 在两个任务的 40 个 clean init 上均未触达首 effect，
full task 0/40，minimum per-effect 0。因此不能用 fallback 解锁 closed-loop
comparison。

## 7. 能说与不能说什么

已得到支持：

- 在冻结的 LIBERO trace fault matrix 上，分开 command、imagination、
  observation、verification、realization 与 invalidation，可以消除 B3 的假
  完成，并在反证到来时可靠生成恢复路由。
- provenance receipt 与 frame-digest alias 防护有效；32-slot memory pressure
  下没有悬空依赖或非法状态转换。

尚未得到支持：

- 不能声称 B6 提高端到端任务成功率、减少动作重试或改善 clean-case success；
- 不能计算 B6 相对 B3 的 faulted task-success uplift、95% paired bootstrap CI
  或 clean degradation；
- 不能外推到 Pi0.5、大型 VLA、真实机器人或更多 LIBERO suite。

## 8. PAI 与可复现证据

- 正式成功作业：`dlc6sr1fu466f1g9`；run ID
  `r16p19-libero-phase1-20260813-013200`；duration 764 秒；platform restart 0；
  launcher attempt 1；AIMaster disabled。
- 资源：2×NVIDIA A800-SXM4-80GB 合约；GPU0 执行，GPU1 合约保留。GPU1
  dmon 全程 SM 0%、memory 0%、framebuffer 0 MB。
- 实验源码：`ae362efeba68643ab4dd2a99cfd295c72a9cbdcc`；LIBERO：
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- demo SHA-256：stove
  `6b30906a52a5741e98ef447d27e7066d6c0be4a5f7acd7ecaf1cb7468aca4aa9`；
  drawer
  `703950f48a3c49dfde61be489ade91527f16e1449b4f29a85f2e51153cef3638`。
- 正式输出中 `SHA256SUMS` 的 24 个文件全部校验通过；`RUN_COMPLETE.json`
  为 true。
- W&B：[63azf17y](https://wandb.ai/chen_jian-cj-workspace/r16p19-libero-phase1/runs/63azf17y)。

第一次正式提交 `dlc1rycl56e4nvac` 在 85 秒时因 clean worker 首次 import
LIBERO 触发交互式数据目录询问而 EOF 失败，尚未产生 optimizer step。修复为
源码内固定并显式指定 `LIBERO_CONFIG_PATH` 后，以新 commit/run ID 重跑成功。
失败记录与 CPFS 日志均保留用于审计。

## 9. 下一步

1. 保持 effect ontology、C0–C7、B1–B7、split、20 个 eval init、阈值和
   统计规则冻结。
2. 只替换 actor，优先使用 clean per-effect success ≥0.80 的共享 small
   ACT/BC policy；训练和推理仍在 PAI，开发机仅做 CPU/GPU smoke。
3. competence 通过后补跑 2 tasks × 20 init × 5 conditions × 4 arms = 800 次
   闭环，并执行预注册的 10,000 次 paired bootstrap。
4. 只有 behavior gates 全部通过才标记 `PASS_PHASE1`；若合格 actor 下仍失败，
   才进入 `REJECT_CORE_MECHANISM`。

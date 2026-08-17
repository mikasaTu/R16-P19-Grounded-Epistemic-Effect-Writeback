# 实验报告

# R16-P19 Phase-1：LIBERO Actor-Decoupled 初步验证

**实验日期：**2026-08-13　**科学终态：**BLOCKED_BY_ACTOR　**PAI 作业终态：**Succeeded

**一句话结论：**完整 R16-P19 写回（B6）在 actor-free trace gate 上通过全部正确性门槛，并把 B3 的 50% false completion 降为 0，同时达到 100% contradiction recovery；但 tiny state-BC actor 与预注册 fallback 都没有通过每个 effect 至少 80% 的 clean competence gate，因此 800 次记忆条件闭环比较按规则未运行。本轮结果支持“显式区分认知状态有助于避免假完成和处理反证”这一机制层判断，但尚不能证明它提高了端到端 LIBERO 任务成功率。

## 1. 结论与决策

- **Actor-free 机制门槛：PASS。**1120 个 task-demo-condition-arm case 全部完成，B6 的 decision accuracy、realized precision/recall、contradiction detection/recovery recall 和 recovery routing accuracy 均为 1.000。
- **Actor competence：FAIL。**learned actor 的最差单 effect 成功率为 0.45，低于预注册阈值 0.80；full-task clean success 为 0.60（24/40）。
- **Fallback competence：FAIL。**nearest-demo phase script 的 full-task success 与所有 effect success 均为 0。
- **800 次闭环与 paired bootstrap：NOT_RUN。**这是预注册的 fail-closed 行为，不是缺失实验；actor 不合格时禁止解释 memory-conditioned task success。
- **最终状态：BLOCKED_BY_ACTOR。**不是 PASS_PHASE1，也不是 REJECT_CORE_MECHANISM。下一步必须先获得合格的共享 actor，再保持 memory 实现和评估清单不变完成闭环门槛。

## 2. 被验证的 idea

原 idea 的核心是：不能把“命令已发出、模型想象成功、传感器观察、独立验证、物理实现”混成一个 progress bit。每个 effect 必须显式处于 REQUESTED、IMAGINED、OBSERVED、VERIFIED、REALIZED、STALLED 或 INVALIDATED_REALIZATION；物理证据必须带 provenance receipt；REALIZED 之后出现反证时必须失效、阻断依赖进度并生成非空恢复路径。

本轮只验证两个边界清楚的部分：actor-free epistemic trace semantics，以及一个不足 10M 参数的 privileged-state tiny BC actor 所提供的因果 sanity gate。它不是 Pi0.5、大型 VLA、DINO-WM、Mem-0 或 official ACT 的效果实验，也不能支持“VLA 性能提升”主张。原始 idea 见[R16-P19：Grounded Epistemic Effect Writeback](https://icnbwz7kd1ui.feishu.cn/wiki/AfN7wFfFWi7dBOkBBtucoroanff)。

## 3. 为什么改用这两个 LIBERO 任务

1. **turn on the stove and put the moka pot on it**：同时包含可逆的开关、抓取、放置与释放，适合制造 no-op、delay、realization 后 reversal 与 observed contradiction。
2. **put the black bowl in the bottom drawer and close it**：包含 containment、release 与依赖前序效果的抽屉闭合，能检验错误提前推进与依赖 effect 失效后的恢复。

两项均来自官方 LIBERO-10，每项 50 条 demonstration。固定划分为 demo 0–29 train、30–39 calibration、40–49 trace-test，并固定 20 个 simulator evaluation init。所有同源变体留在同一 split。

## 4. 实验设计

### 4.1 Effect 与证据

每个任务定义 4 个顺序 effect。物理证据 receipt 包含 episode_id、event_index、timestamp、sensor/view、frame digest、effect_id 和 evidence type。同一物理 frame 即使换 evidence ID，也不能充当独立验证。resident memory 固定为 32 slots，provenance ledger append-only，并对 dangling parent fail closed。

### 4.2 Fault conditions

- C0 clean；C1 command no-op；C2 delayed effect；C3 realized 后物理 reversal。
- C4 单相机 false positive；C5 同一 frame 换 evidence ID；C6 40 个无关事件施加 32-slot memory pressure；C7 imagined success 后 observed failure。

### 4.3 Memory arms

- B1 sliding recent history；B2 command-as-progress；B3 monolithic writeback。
- B4 typed states、无 contradiction recovery；B5 typed states 加 verification、无 recovery。
- B6 full R16-P19；B7 oracle effect ledger upper bound。

B1–B6 在全部 160 个 task-demo-condition 组合上收到 byte-identical event streams；核验结果为 160/160 每组只有一个 stream hash。B7 只作为 oracle upper bound。

## 5. Actor-free trace gate 结果

- **B6：**decision accuracy 1.000；false completion 0；premature advance 0；realized precision 1.000；realized recall 1.000。
- **B6 contradiction：**detection recall 1.000；recovery recall 1.000；recovery routing accuracy 1.000。
- **Provenance：**evidence alias acceptance 0，且 alias attack 确实被触发并拒绝 20 次；duplicate event idempotency 1.000。
- **结构约束：**dangling parent 0；transition violation 0；最大 resident slots 恰为 32；B6 p50/p95 latency 为 5.575/79.388 微秒。
- **非退化：**B6 clean decision accuracy 1.000、unnecessary recovery 0，且候选决策并非总是 REOBSERVE 或 SAFE_STOP。

**对照解释：**B3 的 false completion 为 0.50、contradiction recovery recall 为 0；B4/B5 虽把 false completion 降为 0，但 recovery recall 仍为 0。只有 B6 与 oracle B7 同时达到 0 false completion 和 1.000 recovery recall。结果说明“typed state”本身不够，verification 与显式 invalidation/recovery 路径需要同时存在。

## 6. Tiny state-BC actor 结果

actor 是 4 层、hidden size 256、action chunk 8 的 retrieval-augmented MLP，共 239,160 参数；使用 15,350 个训练样本，在 PAI 上训练 3000 optimizer steps。loss 从 step 1 的 1.13038 降至 step 3000 的 0.04931。step 1000、2000、3000 均有包含 model、optimizer、scheduler、RNG 与 global step 的 COMPLETE checkpoint。

### 6.1 Learned actor clean competence（40 rollouts）

- **stove_moka：**STOVE_TURNED_ON 1.00；MOKA_GRASPED 0.80；MOKA_ON_STOVE 0.75；MOKA_RELEASED_ON_STOVE 0.75；full task 15/20。
- **bowl_drawer：**BOWL_GRASPED 0.90；BOWL_IN_BOTTOM_DRAWER 0.65；BOWL_RELEASED_IN_DRAWER 0.60；BOTTOM_DRAWER_CLOSED 0.45；full task 9/20。
- **总体：**full task 24/40 = 0.60；minimum per-effect 0.45，低于 0.80 门槛。

### 6.2 预注册 fallback（40 rollouts）

nearest-demo phase script 在两个任务的 40 个 clean init 上均未触达首 effect，full task 0/40，minimum per-effect 0。因此不能用 fallback 解锁 closed-loop comparison。

## 7. 本轮能说与不能说什么

### 已得到支持

- 在冻结的 LIBERO trace fault matrix 上，把 command、imagination、observation、verification、realization 与 invalidation 分开，可以消除 B3 的假完成，并在反证到来时可靠地产生恢复路由。
- provenance receipt 与 frame-digest alias 防护有效；32-slot memory pressure 下没有悬空依赖或非法状态转换。

### 尚未得到支持

- 不能声称 B6 提高了端到端任务成功率、降低了动作重试或改善了 clean-case success，因为 800 次闭环未运行。
- 不能计算 B6 相对 B3 的 faulted task-success uplift、95% paired bootstrap CI 或 clean degradation。
- 不能外推到 Pi0.5、大型 VLA、真实机器人或更多 LIBERO suite。

## 8. PAI 执行与可复现证据

- **正式成功作业：**dlc6sr1fu466f1g9；run-id r16p19-libero-phase1-20260813-013200；PAI 控制面 Succeeded；duration 764 秒；platform restart 0；launcher attempt 1；AIMaster disabled。
- **资源：**2×NVIDIA A800-SXM4-80GB 合约；GPU0 执行，GPU1 合约保留。GPU1 全程 dmon 显示 SM 0%、memory 0%、framebuffer 0 MB。
- **身份与存储：**runtime UID:GID 2254:2254；全部 artifact/checkpoint 位于 /mnt/cpfs/zbl-cpfs-new。
- **实验源码：**ae362efeba68643ab4dd2a99cfd295c72a9cbdcc；**LIBERO：**8f1084e3132a39270c3a13ebe37270a43ece2a01。
- **demo SHA256：**stove 6b30906a52a5741e98ef447d27e7066d6c0be4a5f7acd7ecaf1cb7468aca4aa9；drawer 703950f48a3c49dfde61be489ade91527f16e1449b4f29a85f2e51153cef3638。
- **artifact：**/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19_libero_phase1/tiny_state_bc_v1/r16p19-libero-phase1-20260813-013200/experiments/r16p19_libero_phase1。
- **checkpoint：**/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16p19_libero_phase1/tiny_state_bc_v1；保留 1000/2000/3000 三个完整点，符合“所有正 10k milestone 加最新 3 个完整非 milestone”策略。
- **完整性：**SHA256SUMS 中 24 个交付文件全部校验 OK；RUN_COMPLETE.json 为 true。
- **W&B：**[r16p19-libero-phase1-20260813-013200](https://wandb.ai/chen_jian-cj-workspace/r16p19-libero-phase1/runs/63azf17y)，entity 固定为 chen_jian-cj-workspace。

**透明记录：**第一次正式提交 dlc1rycl56e4nvac 在 85 秒时因 clean worker 首次 import LIBERO 触发交互式数据目录询问而 EOF 失败，尚未产生 optimizer step。修复为源码内固定、显式 LIBERO_CONFIG_PATH 后，以新 commit 和新 run-id 重跑成功。该失败 PAI 记录未删除，因为删除服务端记录需要单独的精确授权；CPFS 日志保留用于审计。

## 9. 下一步建议

1. **保持机制测试冻结。**不修改 effect ontology、C0–C7、B1–B7、split、20 个 eval init、阈值与统计规则。
2. **只替换 actor。**优先使用能在这两个任务上达到每个 effect clean success ≥0.80 的共享 small ACT/BC policy；训练与推理仍在 PAI，开发机只做 CPU 或 GPU smoke。
3. **competence 通过后补跑 800 次闭环。**即 2 tasks × 20 init × 5 conditions × 4 arms，并执行预注册的 10,000 次 paired bootstrap。
4. **判定规则不变。**若 B6 在 false completion、contradiction recovery、faulted task success、CI 与 clean degradation 上全部通过，才可标记 PASS_PHASE1；若合格 actor 下仍失败，才进入 REJECT_CORE_MECHANISM。
5. **再决定是否接 Pi0.5。**当前不建议直接进入大型 VLA 实验，以免 actor 能力与 memory 机制再次混淆。

## 10. 交付物索引

核心文件包括 preregistration.yaml、benchmark_manifest.json、split_manifest.json、fault_matrix.json、effect_ontology.json、actor_free_metrics.json、actor_competence.json、fallback_actor_competence.json、metrics.json、trace_events.jsonl、memory_outputs.jsonl、failure_cases.md、readiness_report.md、SHA256SUMS 与 RUN_COMPLETE.json。

**最终建议：**保留 actor-free 机制实现与冻结协议；把本轮视为“机制层 positive、行为层 blocked”的初步验证。下一轮唯一优先事项是获得合格共享 actor，而不是调整 B6 或降低门槛。

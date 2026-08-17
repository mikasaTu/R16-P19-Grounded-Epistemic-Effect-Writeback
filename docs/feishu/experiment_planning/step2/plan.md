# step2

# Agent Prompt：R16-P19 Phase-1B Actor Upgrade 与闭环行为验证

你需要继续完成仓库：

`mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback`

当前科学状态为：

`BLOCKED_BY_ACTOR`

已有结论：

- actor-free trace gate 已通过；
- B6 在冻结 trace matrix 上将 B3 的 false completion 从 0.50 降到 0；
- B6 contradiction recovery recall 为 1.00；
- 当前 tiny state-BC actor 的 clean full-task success 为 24/40=0.60；
- 当前 minimum per-effect success 为 0.45，低于预注册门槛 0.80；
- nearest-demo fallback 为 0/40；
- 800 次 memory-conditioned closed-loop matrix 因 actor gate 失败而未运行。

本任务不是重新设计 R16-P19，也不是启动 π0.5。

本任务的唯一目标是：

1. 获得一个合格、共享、与 memory state 解耦的小型 actor；
2. actor 通过冻结 competence gate 后，原样运行预注册的 800 次闭环矩阵；
3. 对 R16-P19 给出 behavior-level PASS、REJECT 或 BLOCKED 结论。

## 一、最高优先级约束

不得修改以下内容的科学语义：

- `r16p19/memory.py`
- effect ontology
- REQUESTED / IMAGINED / OBSERVED / VERIFIED / REALIZED / STALLED / INVALIDATED_REALIZATION
- evidence receipt 和 provenance 规则
- C0–C7 fault definitions
- B2/B3/B5/B6 memory definitions
- demo 0–29 / 30–39 / 40–49 split
- formal init 0–19
- 原有行为指标和 pass/fail thresholds
- 10,000 次 paired bootstrap 规则

不得：

- 把 trace PASS 写成闭环性能 PASS；
- 使用 π0.5、DINO-WM 或大型 VLA；
- 在 actor 不合格时运行 800 次矩阵；
- 在看到 formal init 0–19 结果后修改 actor；
- 只保存最佳 checkpoint 或最佳 seed；
- 构建 formal activation、加密发布、inode 审计或大型 mutation infrastructure；
- 修改门槛来获得正结果。

使用普通 Git commit、JSONL、SHA256 和 Markdown 即可。

## 二、首先修复 Actor–Memory 实验混杂

当前 actor 训练只见过 `ideal_memory_summary(phase)`，但闭环评测会向 actor 输入不同 arm 的真实 memory summary。

这会使不同 memory arm 同时改变：

1. 高层决策；
2. actor 的低层输入分布。

必须将两者解耦。

新增统一接口：

```Plain Text
SkillActor.action_chunk(
    state_history,
    task_id,
    effect_id,
    execution_mode
)
```

其中：

- `execution_mode` 只允许 `EXECUTE` 或 `RETRY`；
- actor 不得直接读取七类 epistemic state；
- actor 不得读取完整 memory summary；
- actor 不得读取 simulator effect truth；
- actor 不得读取未来状态。

Memory manager 只通过以下决策影响行为：

```Plain Text
ADVANCE_TO_NEXT_SUBTASK
RETRY_CURRENT_EFFECT
REOBSERVE
ROLLBACK_OR_REPLAN
SAFE_STOP
```

映射规则：

- `ADVANCE`：切换到下一个 effect skill；
- `RETRY`：从当前物理状态重新执行当前 effect skill；
- `REOBSERVE`：执行固定、arm-independent 的短 no-op observation window；
- `ROLLBACK_OR_REPLAN`：重置 actor history，并从当前状态重新执行当前 effect skill；
- `SAFE_STOP`：停止 episode。

在 memory arms 首次决策分叉前，actor 输入和动作必须完全一致。

增加 regression test：

- 同一 state history、task、effect、actor seed 下；
- B2/B3/B5/B6 在相同 high-level decision 时必须输出相同 action bytes；
- 只有 memory decision 不同后，行为才允许分叉。

## 三、冻结 Phase-1B Preregistration

在训练任何新 actor 前创建：

```Plain Text
experiments/r16p19_libero_phase1b/
  preregistration.yaml
  actor_data_audit.json
  actor_candidates.yaml
  init_split_manifest.json
  frozen_behavior_gates.json
  README.md
```

记录：

- 当前仓库 HEAD/tree；
- 原正式实验 commit；
- LIBERO commit；
- 两个 dataset SHA256；
- actor architecture；
- actor selection procedure；
- actor qualification init；
- formal init；
- training/calibration/test demos；
- competence gate；
- closed-loop matrix；
- final statuses。

先提交 preregistration commit，之后才能训练。

## 四、Actor 数据审计

对两个任务分别统计：

```Plain Text
stove_moka:
- STOVE_TURNED_ON
- MOKA_GRASPED
- MOKA_ON_STOVE
- MOKA_RELEASED_ON_STOVE

bowl_drawer:
- BOWL_GRASPED
- BOWL_IN_BOTTOM_DRAWER
- BOWL_RELEASED_IN_DRAWER
- BOTTOM_DRAWER_CLOSED
```

输出：

- 每个 effect 的训练帧数；
- 每个 effect 的有效 action chunks；
- state/action alignment；
- transition boundary；
- action dimension mean/std；
- gripper class balance；
- 各 effect chunk-length 分布；
- train/calibration/trace-test 是否互斥；
- 当前 actor 的失败视频与失败 effect 分布。

禁止通过复制同一个 episode 到不同 split 增加样本。

## 五、Actor Qualification Init

先读取官方 `.init` 文件并统计可用 initial states。

优先规则：

```Plain Text
actor development / qualification: init 20–39
formal memory evaluation: init 0–19
```

若可用 init 不足 40：

1. 不得默默复用 formal init 调参；
2. 报告实际可用数量；
3. 在 preregistration 中冻结一个唯一 actor config；
4. formal init 0–19 只能在 actor 完全冻结后运行一次；
5. 失败后不得继续在同一批 init 上改 actor。

## 六、Primary Actor：Effect-Conditioned State-ACT

实现一个小型低维 ACT：

输入：

- 最近 4 帧 padded simulator state；
- task embedding；
- effect embedding；
- execution mode。

不输入：

- memory summary；
- epistemic state；
- oracle effect truth；
- future state；
- camera privileged labels。

默认配置冻结为：

```Plain Text
history length: 4
predicted action horizon: 8
executed prefix: 4
hidden dim: 256
transformer layers: 4
attention heads: 8
dropout: 0.1
parameter count: <= 10M
optimizer: AdamW
learning rate: 3e-4
batch size: 256
gradient clipping: 1.0
```

训练要求：

- 使用 demo 0–29；
- demo 30–39 只用于 calibration、early stopping 和 checkpoint selection；
- demo 40–49 不参与 actor 训练或选择；
- 对 8 个 effect 做 balanced sampling；
- 连续 6D 动作用 Smooth L1；
- gripper dimension 使用单独加权 loss；
- 保存全部训练曲线和所有候选 checkpoint；
- 不能只保留最佳结果。

推理使用 receding horizon，每次执行前 4 步后重新预测。

## 七、唯一允许的 Fallback

如果共享 Effect-Conditioned State-ACT 在 actor qualification init 上未达到门槛，可启动一个预注册 fallback：

```Plain Text
Per-Effect State-ACT
```

要求：

- 每个 task/effect 一个小型 actor；
- 输入和输出合同与主 actor 相同；
- 总参数量和训练预算完整报告；
- 仍然不读取 memory summary；
- 不允许第三种 actor 或无边界超参数搜索。

如果 primary 和 fallback 都失败：

```Plain Text
FINAL_STATUS = BLOCKED_BY_ACTOR_V2
```

停止任务，不运行闭环 memory matrix。

## 八、Actor Gate

先在 actor qualification init 上运行 clean rollouts。

报告：

- 每个 effect success；
- 每个 task full-task success；
- action steps；
- repeated loop；
- endpoint/transition failure；
- gripper failure；
- 失败视频。

Actor qualification 要求：

```Plain Text
minimum per-effect success >= 0.80
```

同时报告但不用于放宽门槛：

```Plain Text
per-task full-task success
repeated-action-loop rate
mean action steps
```

actor checkpoint 完全冻结后，再在 formal init 0–19 上执行一次正式 competence gate。

如果 formal minimum per-effect success < 0.80：

```Plain Text
FINAL_STATUS = BLOCKED_BY_ACTOR_V2
```

不得运行 800 次闭环比较。

## 九、运行冻结的 800 次闭环矩阵

Actor 正式通过后，运行：

```Plain Text
2 tasks
× 20 formal init
× 5 conditions
× 4 memory arms
= 800 rollouts
```

Conditions：

```Plain Text
C0 clean
C1 command no-op
C2 delayed realization
C3 post-realization reversal
C7 imagined success followed by observed failure
```

Memory arms：

```Plain Text
B2 command-as-progress
B3 monolithic writeback
B5 typed + verification without contradiction recovery
B6 full R16-P19
```

所有 arm 必须绑定：

- 同一个 actor checkpoint；
- 同一个 actor source commit；
- 同一个 normalization；
- 同一个 init state；
- 同一个 fault target effect；
- 同一个 fault schedule；
- 同一个 actor seed；
- 同一个 max steps；
- 同一个 reobserve duration；
- 同一个 recovery mapping。

每个 paired unit 保存：

- initial-state hash；
- actor-input hash；
- first decision divergence step；
- first action divergence step；
- memory state；
- chosen decision；
- physical effect truth；
- action chunks；
- task success；
- failure type；
- retry count；
- action steps；
- video path。

## 十、行为指标与冻结门槛

沿用已有实现中的指标：

- full task success；
- clean task success；
- faulted task success；
- C1/C3/C7 target-fault success；
- false completion；
- premature transition；
- repeated-action loop；
- contradiction recovery；
- recovery routing；
- retry count；
- action steps；
- safe stop；
- resident slots；
- dangling parent count。

执行原有 10,000 次 paired bootstrap。

不得修改以下原门槛：

```Plain Text
B6 false completion relative reduction vs B3 >= 0.50
B6 contradiction recovery >= 0.80
B6 target-fault success > B3
B6 target-fault success > B5
95% paired bootstrap CI lower bound for B6-B3 > 0
clean success degradation vs B3 <= 0.03
B6 statistically distinguishable from B2
```

额外报告但不得替代原门槛：

- B6 相比 max(B3, B5) 的绝对成功率差；
- number needed to recover；
- 每个 condition 的 effect size；
- task-specific confidence interval。

## 十一、机制解释要求

闭环结束后，必须区分：

```Plain Text
memory decision failure
actor skill failure
effect verifier failure
fault injector failure
timeout / repeated loop
```

对于 B6 的每次成功恢复，记录：

```Plain Text
contradiction
→ INVALIDATED_REALIZATION / STALLED
→ recovery decision
→ actor retry
→ effect re-realized
→ task completion
```

对于每次失败，判断：

- memory 是否作出正确 decision；
- actor 是否未能执行正确 decision；
- actor 成功但 effect verifier 是否误判；
- actor 和 memory 是否同时失败。

不允许只报告最终 success rate。

## 十二、最终决策

最终只能使用以下四个状态之一：

```Plain Text
PASS_PHASE1_BEHAVIOR
REJECT_CORE_MECHANISM
BLOCKED_BY_ACTOR_V2
BLOCKED_BY_IMPLEMENTATION
```

判定规则：

### PASS_PHASE1_BEHAVIOR

- actor formal gate 通过；
- 全部 correctness gates 通过；
- 全部 frozen behavior gates 通过。

### REJECT_CORE_MECHANISM

- actor formal gate 通过；
- 正式 800 rollouts 完成；
- B6 未通过冻结 behavior gates。

不得通过继续调 memory threshold 挽救。

### BLOCKED_BY_ACTOR_V2

- primary 和 fallback 均未通过 actor gate；
- 或 formal init 0–19 上 actor minimum per-effect < 0.80。

### BLOCKED_BY_IMPLEMENTATION

- 数据、环境或实现问题使正式实验无法完成；
- 必须指出精确阻塞点，不能写成算法失败。

## 十三、交付物

生成：

```Plain Text
experiments/r16p19_libero_phase1b/
  preregistration.yaml
  actor_data_audit.json
  actor_candidates.yaml
  init_split_manifest.json
  actor_training_metrics.jsonl
  actor_qualification_results.jsonl
  selected_actor_manifest.json
  formal_actor_gate.json
  closed_loop_results.jsonl
  paired_bootstrap.json
  behavior_summary.json
  mechanism_mediation.json
  failure_cases.md
  FINAL_DECISION.md
  SHA256SUMS
```

另外保存：

- 所有 checkpoint；
- actor qualification 视频；
- 800 次闭环中的全部失败视频；
- 每种 condition 至少 5 个代表性成功/失败视频；
- PAI JobId；
- W&B run；
- exact source commit；
- exact actor checkpoint SHA256。

## 十四、执行顺序

严格按以下顺序执行：

1. 读取当前仓库并复核现状；
2. 冻结 preregistration；
3. 完成数据和 init 审计；
4. 修改 actor-memory 接口并运行 regression；
5. 训练 primary actor；
6. qualification gate；
7. 只有 primary 失败时才训练 fallback；
8. 冻结唯一合格 actor；
9. formal init 0–19 actor gate；
10. 只有正式 gate 通过才运行 800 次矩阵；
11. paired bootstrap；
12. failure decomposition；
13. 输出唯一 final status。

下一次进度报告优先汇报：

- 当前 repo HEAD；
- preregistration commit；
- 可用 init 数量与 qualification/formal split；
- 每个 effect 的训练样本数；
- actor-memory 是否已完全解耦；
- primary actor 训练和 qualification 结果；
- minimum per-effect success；
- 是否获得运行 800 rollouts 的授权。

不要启动 π0.5、Mem-0、视觉 ACT 或新的 benchmark。

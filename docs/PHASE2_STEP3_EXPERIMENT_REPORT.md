# 实验报告

## R16-P19 step3：Phase-2 Competent-Executor Causal Behavior Validation

## 结论先行

本阶段最终状态为 **`BLOCKED_BY_EXECUTOR_V3`**。

冻结后的 `RetargetedGeometricSkillExecutor` 在 LIBERO 的 2 个任务、init
60--79、共 40 条 clean qualification rollouts 上得到：

| 指标 | 预注册门槛 | 结果 | 是否通过 |
|---|---:|---:|---|
| 最低 per-effect success | >= 0.90 | 0.70 | 否 |
| stove_moka full-task success | >= 0.80 | 0.80（16/20） | 是 |
| bowl_drawer full-task success | >= 0.80 | 0.70（14/20） | 否 |
| repeated-effect-loop rate | <= 0.10 | 0.25（10/40） | 否 |
| 总 full-task success | 仅报告 | 0.75（30/40） | — |

PAI 作业本身正常完成，因此不是基础设施阻塞。按照预注册 stop rule，formal
init 0--19、800-cell memory matrix、first-divergence replay、10,000 次 paired
bootstrap 和 McNemar 检验均保持 **NOT_RUN**。本结果不能判断 B6 是否优于
B3/B5，也不能据此拒绝核心 memory mechanism。

下一阶段 readiness 仅记录为 **`LEARNED_EFFECT_VERIFIER_BLOCKED`**，没有自动
启动下一阶段。

## 1. 研究问题与边界

目标是先用一个确定性、非神经、与 memory/fault 解耦的低层执行器通过行为能力
门槛，再检验原始 R16-P19 memory mechanism 在 faulted LIBERO 行为上是否对
B3/B5 有因果提升。执行器输入只允许当前/近期低维几何、机器人状态、task/effect
ID、EXECUTE/RETRY 和 retry index；禁止 memory summary、epistemic state、fault
identity、reward、task success、future state、init index 等输入。

本阶段没有修改 `r16p19/memory.py`、effect ontology、receipt/provenance、fault
定义、memory arms、formal seeds 或 bootstrap gate，也没有训练第三个神经 actor。

数据边界严格保持为：demo 0--29 提取模板，demo 30--39 校准模板，init 40--59
仅用于开发，init 60--79 用于冻结后的 qualification，formal init 0--19 只有在
qualification 通过后才能访问。本次 qualification 失败，所以 formal access 为 0。

## 2. 实现与冻结身份

实现了 `RetargetedGeometricSkillExecutor`：每个 effect 从成功演示段提取最多 3
条 effect-relative 轨迹模板；运行时按当前几何选择模板并重定向到当前 object /
fixture frame；每次输出 8×7 action chunk，只执行前 4 步后重新观测和规划；控制
由 Cartesian pose feedback 加演示动作 feed-forward 组成；retry 使用预注册的
固定 1 cm offsets 和冻结模板顺序。

确定性与泄漏回归包括：相同输入产生逐 byte 相同 action、seeded rollout byte
identity、执行器不 import memory、不接收 fault identity、不在执行器内部调用
effect truth。完整测试为 **23 passed**。

冻结配置：position gain 6.0、orientation gain 1.5、position tolerance 0.015 m、
action horizon 8、executed prefix 4、每 effect 最多 4 次 attempt、每 attempt 40
chunks、每 episode 最多 700 action steps。`monotonic_progress` 和
`retry_reapproach` 因开发集负向结果而冻结为 false。

关键身份：

- qualification source commit：`8963f8cb3b10201095a47c48cec13ce11b0832f0`
- executor implementation commit：`136e8923829c0436ca27755078a609f91bcf75a5`
- selected executor manifest SHA-256：`384957cae10f96b6a53645a555e53d38876573a06a452fc93af0d95b4f254b6b`
- selected template manifest SHA-256：`6a123a334bb901da880baf3b14e72576015bc013ed1f3224e9dd2c6d3c49d431`
- official LIBERO commit：`8f1084e3132a39270c3a13ebe37270a43ece2a01`

## 3. PAI 校准与可恢复执行

由于开发机 EGL 不支持 MuJoCo 要求的 `PLATFORM_DEVICE`，正式 simulator 工作在
PAI 的已验证单卡 RTX 4090 rendering carrier 上运行。预注册明确禁止为该非神经
工作负载保留第二张空闲 GPU，因此没有申请 2×A800。所有任务均为 1 worker、1
GPU、14 CPU、120 GiB、UID/GID 2254:2254、W&B disabled、无 secret 注入、无
AIMaster、PAI 自动容错关闭、平台重启上限和实际重启均为 0。

模板校准共经历三次可审计任务：

1. `dlcv23myy6u8w5a7`：LIBERO 首次非交互启动因缺少
   `LIBERO_CONFIG_PATH` 在首个 cell 前失败；已保留原始失败日志。
2. `dlc151qfbenna348`：完成并 fsync 全部 237 个实际可用 template×demo cells，
   最终汇总时因相对路径未先 resolve 而失败；无平台重启。
3. `dlcq2f5sauntc59x`：修复路径规范化后复用原工作目录，在 34 秒内跳过 237 个
   已完成 cells，只完成汇总并成功退出，验证了 application-level cell resume。

校准从 24 个 extracted templates 中保留 14 个。值得注意的是，demo 30--39 上
MOKA_GRASPED 的最佳单模板 success 只有 0.40，BOWL_GRASPED 最佳为 0.889；选择
规则可以保留“当前最佳”，但这并不等价于已经满足 0.90 qualification 门槛。

## 4. 开发集正向与负向机理反解

以下均使用 init 40--59 的 paired 40 rollouts，未使用 qualification/formal 结果
调参。

| Arm / 单因素变化 | Full task | 最低 effect | Loop | Mean steps |
|---|---:|---:|---:|---:|
| A0 world-frame open loop | 0.000 | 0.000 | 1.000 | 375.025 |
| A1 + effect-local retargeting | 0.000 | 0.000 | 1.000 | 365.750 |
| A2 + Cartesian closed loop / feed-forward | 0.300 | 0.200 | 0.700 | 337.950 |
| A3 + 4-step receding-horizon prefix | 0.725 | 0.650 | 0.275 | 192.100 |
| A3 但关闭 demonstrated feed-forward | 0.300 | 0.150 | 0.700 | 257.400 |
| A3 但启用 monotonic cursor | 0.475 | 0.400 | 0.525 | 173.800 |

反解如下：

- Local frame 本身没有产生 competence。它只让轨迹对 frame 变化协变，却不会
  纠正 contact 后的累计误差或 stale action，所以 A0→A1 的 full success 仍为 0。
- Closed-loop feedback 让 action 随当前几何误差变化，是首次得到非零成功的
  组件。对照关闭 feed-forward 的实验进一步表明，仅 pose feedback 能到达 free
  space，却难以稳定驱动 stove/drawer 的约束接触；演示动作提供持续的方向和
  类 force delta，使 full success 从 0.300 升至 0.725。
- A2 与 A3 都只有 1 次 attempt，retry offset 实际未激活，因此 A2→A3 的差异可
  归于 executed prefix 8→4：stale-command window 缩短，接触后更快重规划，full
  +0.425、最低 effect +0.450、loop -0.425、平均 steps -145.85。
- Monotonic cursor 防止 waypoint 后退，但在前一 effect 结束姿态不利时，会锁在
  feed-forward 已经推成不可达的 post-contact waypoint，因而 full 从 0.725 降到
  0.475。该机制被明确排除出冻结配置。
- tolerance 从 0.010 放宽到 0.015 时，预注册的 bottleneck 指标从 0.35 升到
  0.40，但总体 full 从 0.525 降到 0.450，bowl closing 从 0.70 降到 0.40。
  代码在位置误差进入 tolerance 后会前跳两个 waypoint，因此“覆盖提高、精确
  contact 降低”与代码一致；具体 contact mediator 仍标记为 hypothesis，而非新
  idea 或已证实因果链。

## 5. Qualification 结果

PAI Job `dlceyy7m2jhmxc4o` 正常 `Succeeded`，耗时 231 秒，单 Pod UID
`a849b067-d9c3-45e7-8f95-bd75e9154027`，完成 frozen 2×20 grid。逐 effect：

| Task | Effect | Success |
|---|---|---:|
| stove_moka | STOVE_TURNED_ON | 0.85 |
| stove_moka | MOKA_GRASPED | 0.80 |
| stove_moka | MOKA_ON_STOVE | 0.80 |
| stove_moka | MOKA_RELEASED_ON_STOVE | 0.80 |
| bowl_drawer | BOWL_GRASPED | 0.80 |
| bowl_drawer | BOWL_IN_BOTTOM_DRAWER | 0.75 |
| bowl_drawer | BOWL_RELEASED_IN_DRAWER | 0.75 |
| bowl_drawer | BOTTOM_DRAWER_CLOSED | 0.70 |

10 个失败的首个失败 effect 为：STOVE_TURNED_ON 3、MOKA_GRASPED 1、
BOWL_GRASPED 4、BOWL_IN_BOTTOM_DRAWER 1、BOTTOM_DRAWER_CLOSED 1。全部失败
都耗尽 3 次 retry 并进入 repeated loop；10 个失败视频全部保留。

## 6. Qualification 失败机理

第一层是 **contact-mode aliasing**。三个 stove failure 最终位置误差平均约
0.0121 m，几何上接近模板 waypoint，但 orientation error 约 0.155 rad，且 switch
仍未触发。视频显示手臂在相同 knob 接触姿态附近重复。单纯的 pose proximity
无法表达“是否从有效一侧接触、是否产生有效受约束运动”。

第二层是 **retry projection skips reacquisition**。当前 retry 先整体平移局部轨迹，
再从已经失败的当前姿态选择最近 waypoint；冻结配置没有强制 pre-grasp
re-approach。由于两个 grasp effect 校准后都只保留一个模板，四次 attempt 的
template rank 都是 0。失败 trace 最终停在 moka 47/48 或 bowl 18--27/28 的中后段，
而不是可靠地回到 open-gripper pre-grasp。视频也呈现持续 miss 状态。这是由代码、
trace 和视频共同支持的定位，但没有把 qualification 数据用于事后改代码。

第三层是 **有限模板的 contact coverage**。校准阶段已经显示 MOKA_GRASPED 的
最佳模板远低于 qualification 门槛；固定小 offset 能提高部分覆盖，却不能保证
重新建立 contact。init 66 把 bowl 搬到 drawer 附近后未进入内部谓词，init 71
完成 grasp/place/release 后仍在 drawer closing contact 上耗尽预算。

上述问题发生在 memory 激活之前，所以错误分解为 executor failure after a clean
effect request，而不是 memory decision error、receipt error 或 fault injector error。

## 7. Stop rule 与未运行项

qualification 三个 gate 中有三个未全部满足，协议要求立即停止并使用
`BLOCKED_BY_EXECUTOR_V3`。因此：

- formal executor gate：0 rollouts，formal init access 为空；
- closed-loop matrix：0/800；
- first-divergence replay：0；
- paired bootstrap：0/10,000；
- representative recovery videos by fault condition：由于 fault matrix 未激活而不存在；
- 没有 post-qualification tuning，也没有第二 executor family。

这保持了结论的可证伪性：不能把 executor 能力不足误写为 memory mechanism 的
支持或反证。

## 8. 复现、代码与证据

代码、原始 JSONL、10 个 MP4、冻结 manifests、PAI templates/launchers、三次校准
任务和 qualification 的 GetJob/运行合同均已上传到 GitHub `main`：

`https://github.com/mikasaTu/R16-P19-Grounded-Epistemic-Effect-Writeback`

Phase-2 结果与证据发布提交为
`b60ec1654088234446f9bd2f32a87be1f16026b8`，对应 tree 为
`bc3d73c209a817b16c10de5085dfcc0789726273`。

核心路径：

- `experiments/r16p19_libero_phase2/executor_qualification_results.jsonl`
- `experiments/r16p19_libero_phase2/executor_qualification_summary.json`
- `experiments/r16p19_libero_phase2/mechanism_mediation.json`
- `experiments/r16p19_libero_phase2/failure_cases.md`
- `experiments/r16p19_libero_phase2/FINAL_DECISION.md`
- `artifacts/phase2_pai/`
- `pai/registry/phase2/`

完整性使用 `experiments/r16p19_libero_phase2/SHA256SUMS` 校验。测试命令为：

```bash
PYTHONPATH=. /mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python -m pytest -q
python scripts/summarize_phase2_blocked.py
(cd experiments/r16p19_libero_phase2 && sha256sum -c SHA256SUMS)
```

最终 checkout 的独立验证结果：

| 检查 | 结果 |
|---|---|
| 单元/回归测试 | `23 passed in 14.29s` |
| `pipeline.py static-check` | 通过；LIBERO commit、六项输入 SHA-256、运行身份与依赖均可回读 |
| `pipeline.py retention-test` | `CHECKPOINT_RETENTION_OK`，不完整 checkpoint 保留且 RNG resume exact |
| 阻塞产物 finalizer | `PHASE2_BLOCKED_ARTIFACTS_COMPLETE`；formal=0、matrix=0 |
| JSON / JSONL 解析 | 全部通过 |
| qualification failure MP4 | 10/10 使用 bundled ffmpeg 完整解码通过 |
| 完整性清单 | Step3 56 项、总 artifact bundle 683 项均通过 `sha256sum -c` |

开发机只进行了上述静态、CPU 测试和 GPU 环境 smoke；没有在开发机执行大规模训练
或正式 simulator sweep。

最终科学结论仍是：**当前 phase 被低层 executor competence 阻塞；B6 的行为级因果
效应保持未识别。**

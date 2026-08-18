# R16-P19 Phase-5 Bounded ASCEL Embodied Bridge 实验报告

最终状态：`BLOCKED_BY_VERIFIER`。所有预注册矩阵均已执行；未通过的 gate 不被改写，后续矩阵按用户明确要求仅作为 diagnostic continuation。

## 核心结果

- bounded ledger：100,000 events，reference mismatch=0，audit break=0，event P99=0.0528 ms，hot memory=0.0101 MB，systems pass=True。
- 冻结策略 qualification：pass=True；逐任务结果为 `{"0": {"backend_errors": 0, "episodes": 10, "loop_rate": 0.0, "minimum_effect_success": 1.0, "pass": true, "success_rate": 1.0}, "5": {"backend_errors": 0, "episodes": 10, "loop_rate": 0.0, "minimum_effect_success": 1.0, "pass": true, "success_rate": 1.0}, "9": {"backend_errors": 0, "episodes": 10, "loop_rate": 0.0, "minimum_effect_success": 0.9, "pass": true, "success_rate": 0.9}}`。
- shared prefix：1000 units，全部字段精确一致=True。
- oracle Core：最强 baseline=M0_TYPED_MATCHED；faulted success 风险差=0.3153，95% CI=[0.30416666666666664, 0.325]，pass=True。
- learned verifier：selected=small_mlp，qualified=False；formal 风险差=0.0000，95% CI=[0.0, 0.0]，pass=False。
- support proof：Full-Core 风险差=0.2500，95% CI=[0.25, 0.25]，cascade precision=1.0000，recall=1.0000，pass=True。

## 机制反解（不生成新 idea）

采用 code-first 的 first-divergence 方法：先固定同一物理/观察/动作前缀，再定位 arm 首次不同的 ledger 状态与 decision。提升只能归因到首次分叉前唯一不同的机制；降低同样按该路径追踪，不按最终分数倒推故事。

- `NO_ATTEMPT_SCOPE`：target-error 增量=1.0000，对应优势移除比例=1.0000，支持该机制归因=True。
- `NO_PRE_REALIZATION_REVOCATION`：target-error 增量=0.5000，对应优势移除比例=0.5000，支持该机制归因=True。
- `NO_SUPPORT_GRAPH`：target-error 增量=0.3333，对应优势移除比例=0.3333，支持该机制归因=True。
- `NO_TRUTH_CREDIT_SPLIT`：target-error 增量=0.0000，对应优势移除比例=0.0000，支持该机制归因=False。

learned 分支的 0 增益来自 verifier 门控而不是 ASCEL ledger 退化：选中的 small MLP threshold=0.9395，qualification min TPR=0.0000。`phase5_formal_runner.py` 对一条 receipt 使用 `np.all(scores >= threshold)`；任一 effect 漏检就整体不承认真实完成，因此 Core 与 baseline 同时不能推进，formal 风险差精确为 0。该结论与 oracle Core 的正向结果并存，最终按预注册优先级记为 `BLOCKED_BY_VERIFIER`。

`NO_TRUTH_CREDIT_SPLIT` 的零消融效应有指标边界：本轮 `target_error` 只统计 false advance，A5 的 task success 也没有因 `active_attempt_credit` 单独扣分；因此它只能说明选定 outcome 对 credit 字段不敏感，不能证明 truth 与 credit 在代码语义上等价。

代码路径上，attempt scope 在 `phase5_ledger_live.py` 的 active-attempt/command 检查处拒绝 stale 与 cross-attempt receipt；pre-realization revocation 通过 revocation epoch 使旧 witness 不再复活事实；truth-credit split 允许 external realization 更新物理事实但不把成功计给 active skill；support graph 只递归失效新近失去全部有效 clause 的 proof，并保留 alternative branch 或已 discharge 的结果。

## 证据边界

LIBERO 的策略轨迹是真实官方 simulator + frozen π0.5 推理；arm 级 4,200/1,680 rows 使用一次真实轨迹形成的 shared-prefix 事件/决策反事实，避免为每个 arm 独立重跑前缀。A1 具有真实 no-op 物理轨迹；A2/A3/V1 是绑定真实观察时点的 receipt fault。A4 与 A5 的因果归属主要由预注册事件 broker/归因干预实现，不等价于机器人外力硬件干预，因此即便数值为正，也不能外推为真实机器人证据。support 的 960 cells 来自重力、接触和约束启用的 MuJoCo 物理任务，不是独立 slider。

本阶段不改变 Phase-1 至 Phase-4 的任何结论，也不把 oracle-only、diagnostic continuation 或 learned-verifier 未过 gate 的结果称为 VLA 改进。

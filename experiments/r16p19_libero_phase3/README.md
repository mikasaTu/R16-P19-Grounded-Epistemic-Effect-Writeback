# R16-P19 Phase-3 preregistration

This directory freezes the Effect-Boundary Replay Causal Validation before any
Phase-3 access to formal demonstrations 40--49. The base repository state is
`981ad1a64936b4e970e1f934be2d354497b5fc8e`; the protected B6 implementation
hash is `4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5`.

The backend resets an exact same-demonstration effect-entry simulator snapshot
and executes that demonstration's frozen action segment with normal
`env.step`. It is intentionally not an actor or general-purpose executor.
Simulator truth is available only to the shared broker and evaluation labels,
never directly to a memory arm.

The user explicitly overrode the plan's early-stop clauses before formal
access. Every gate remains frozen and is reported, but the full matrix, D1,
first-divergence replay, statistics, and both B6 mechanism ablations will run
even if replay or primary gates fail. Such downstream results are diagnostic
and cannot turn a failed replay gate into a Phase-3 pass.

Splits are demo 0--19 development, 20--29 boundary/action calibration, 30--39
qualification, and 40--49 formal confirmation. Simulator init 0--19 remain
reserved and forbidden. The formal bank was used by earlier trace-level work,
so this phase is controlled confirmation rather than independent external
benchmark validation.

Feishu planning nodes:

- step4: https://icnbwz7kd1ui.feishu.cn/wiki/SqxXwnTIFiistlk1bOrcprPWnmb
- experiment report: https://icnbwz7kd1ui.feishu.cn/wiki/DvI3wxPQ7ixoSdkA9kncspgdnUc

No statement about B6 behavioral advantage is permitted until the complete
formal matrix and paired analysis are finished.

# Gate F 决策记录

状态：DIAGNOSTICS_COMPLETE_WITH_FORMAL_PROTOCOL_DEVIATION。

- gate_f1_baseline: True
- gate_f1_conclusion: True
- gate_f2_complete: True
- gate_f4_evidence: True

F1-A=0.991666989，精确复现；B−A=+0.003042821，E=0.986506104，F=0.784207743，判定SHORTCUT_CONFIRMED。F2覆盖35/35维。F4可观测输入诊断macro AUC=0.982006483，真实容量上界未识别，需求条件见F4。

formal存在提前访问违规，所有formal派生量标protocol_valid=false，仅描述，不可宣称合规final evaluation。QPILOTS可读但目标arrange_3_flowers数据未找到，实际数量均null；F3数据要求与F5设计已给出。

F1–F5计算/审计/设计产物均已交付；没有GPU、PAI、rollout、新采集或S2。是否继续idea由人工决定。本文件为阶段门记录，不是Phase-11最终报告。

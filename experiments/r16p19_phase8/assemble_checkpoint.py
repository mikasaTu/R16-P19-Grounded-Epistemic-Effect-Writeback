import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
J=lambda n:json.loads((O/n).read_text())
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
c1=J('C1_AUDITED.json');c2=J('C2_GLOBAL_THRESHOLD_AUDITED.json');c3=J('C3_AUDITED.json');c4=J('C4_AUDITED.json');b=J('C0_SAMPLE_BUDGET_AUDITED.json');repro=J('PHASE5_REPRO_AUDITED.json')
assert repro['mismatch_rows']==0
for src,dst in [('C1_AUDITED.json','C1_RECALIBRATION.json'),('C2_GLOBAL_THRESHOLD_AUDITED.json','C2_GLOBAL_THRESHOLD.json'),('C3_AUDITED.json','C3_MIN_AGG.json'),('C4_AUDITED.json','C4_PER_EFFECT_UPGRADE.json'),('C0_SAMPLE_BUDGET_AUDITED.json','C0_SAMPLE_BUDGET.json'),('C2_SELECTION_SEAL_AUDITED.json','C2_SELECTION_SEAL.json')]:
 (O/dst).write_bytes((O/src).read_bytes())
gates={'C1':{'passed':False,'reason':'尺度失配假设未被支持；两种映射阈值极差均变宽。原始预注册没有显著性检验定义，不能事后报告确认性显著性。','threshold_convergence':c1['threshold_convergence'],'claim_eligible':False,'selection_eligible':False},'C2':{'individual95':False,'joint95_bonferroni':False,'reason':'Both mappings fail at least three per-effect bounds; pooled positive count is not per-effect certification; independent calibration validity also absent.','methods':{m:{q:{'global_threshold':v['global_threshold'],'pooled_positive_count':v['pooled_positive_count'],'pooled_tpr_ci':v['pooled_tpr_ci'],'per_effect_lcb':{k:x['wilson'][0] for k,x in v['effects'].items()},'complete_frame_count_vector':v['complete_frame_count_vector'],'complete_certified_vector':v['complete_certified_vector']} for q,v in x.items()} for m,x in c2['methods'].items()}},'C4':{'passed':False,'source':'C4_AUDITED.json','reason':'See paired actual ledger replay; do not substitute legacy S1 counts.'},'Phase5_reproduction':{'passed':True,'rows':repro['compared_rows'],'mismatch':repro['mismatch_rows'],'purpose':repro['purpose'],'claim_eligible':False,'selection_eligible':False}}
paired=c4['pooled_metrics']; before=paired['legacy_receipt_and']; after=paired['per_effect_independent']
assert before['false_upgrade_denominator']==after['false_upgrade_denominator']
gates['C4']={'passed':after['concordant_units']>before['concordant_units'] and after['false_upgrade_wilson_95_upper']<=before['false_upgrade_wilson_95_upper'],'before':{k:before[k] for k in ['concordant_units','divergent_units','unit_count','false_upgrade_count','false_upgrade_denominator','false_upgrade_wilson_95_upper']},'after':{k:after[k] for k in ['concordant_units','divergent_units','unit_count','false_upgrade_count','false_upgrade_denominator','false_upgrade_wilson_95_upper']},'source':'C4_AUDITED.json','scope':c4['data_scope'],'claim_eligible':False,'selection_eligible':False}
d={'status':'NEEDS_DATA','execution_status':'RETROSPECTIVE_DIAGNOSTICS_COMPLETED_WITH_PROTOCOL_BREACH','fully_preregistered_plan_completed':False,'protocol_blocker':'Original C0 was not fully sealed before fitting; cannot repair retrospectively. Original C0 bytes retained.','gates':gates,'claim_eligible':False,'selection_eligible':False,'formal_selected_point_evaluated':False,'formal_usage':'C1 ceiling and frozen Phase5 reproduction only; no C4 formal or new workpoint evaluation','gpu_jobs_submitted':0,'pai_jobs_submitted':0,'rollouts_started':0,'S2_started':False,'sample_budget':'C0_SAMPLE_BUDGET.json','future_data_collection_started':False,'selected_workpoint':None}
d['C4_pilot_task_success_gain']=c4['pilot_task_success_gain']
write('C_DECISION.json',d)
lines=['# C Decision — step9 审计交接','', '**NEEDS_DATA；补算属于事后诊断，不能声称原计划已全部合规完成。**','', '此前两次“已完成”声明撤回。C0 未在拟合前完整封存，不能追溯补救；原 C0 哈希保持不变。以下为补齐后的计算证据，不是 Phase-8 最终报告。','','| C1 映射 | formal 诊断阈值极差 | CV |','|---|---:|---:|']
for m in ['raw','temperature','platt']:
 s=c1['threshold_convergence'][m]['spread'];lines.append(f"| {m} | {s['range']:.10f} | {s['cv']:.10f} |")
lines+=['','两种映射均未收窄：尺度失配假设未被支持。formal ceiling 仅作诊断，绝不用于选点。每 effect AUC 不变不意味着 receipt min AUC 不变。','','| C2 映射/口径 | global threshold | pooled positives | 每 effect LCB（固定顺序） | 完整向量 |','|---|---:|---:|---|---|']
for m,x in c2['methods'].items():
 for q,v in x.items():
  lcb=', '.join(f"{x['wilson'][0]:.6f}" for x in v['effects'].values());lines.append(f"| {m}/{q} | {v['global_threshold']:.10f} | {v['pooled_positive_count']} | {lcb} | 否 |")
lines+=['','effect 顺序：alphabet_soup、tomato、book、mug、close。每 effect 独立阈值对照、TP 数、区间宽度和聚类 bootstrap 均在 C2_GLOBAL_THRESHOLD.json。读路径审计实际拒绝 formal，seal 不代表补救了原 C0。','','| C3 split | raw min AUC | temperature | Platt |','|---|---:|---:|---:|']
for split,x in c3['splits'].items():lines.append(f"| {split} | {x['raw']['auc_rank']:.8f} | {x['temperature']['auc_rank']:.8f} | {x['platt']['auc_rank']:.8f} |")
lines+=['','C3 包含完整曲线；资格集曾参与选型并与 pilot 重叠，不能宣称新 held-out 泛化。sum(log p) 排除。','','C4：实际逐 effect ledger 状态、相同分母的假升级与区间、unit ID、增益假设见 C4_PER_EFFECT_UPGRADE.json 和 C4_MECHANISM.md。旧 S1 的 102/18 不是本轮结果，已撤回。','','Phase-5 精确复算：1680 行，mismatch=0；verifier 权重文件与原提交字节一致。','','新数据预算（按 task 独立全新认证分配，不是总采集启动命令）：','','| effect | 逐 effect 95% 独立正样本最低数 | task episodes 外推 | 联合 95% 独立正样本最低数 | task episodes 外推 |','|---|---:|---:|---:|---:|']
for key,v in b['effects'].items():
 if v['frame_positive_count']>35:continue
 a=v['requirements']['individual95']['episode_all_positive_frames_detected'];z=v['requirements']['joint95_bonferroni']['episode_all_positive_frames_detected'];lines.append(f"| {key} | {a['minimum_independent_positive_units_if_all_success']} | {a['total_task_episodes_extrapolated']} | {z['minimum_independent_positive_units_if_all_success']} | {z['total_task_episodes_extrapolated']} |")
lines+=['','仅为全成功且保持观测产出率的外推；不能保证实际产出。总认证分配外推为 111 集或 190 集（各 task 分别分配），另加独立拟合/选点数据；不可把已参与拟合的数据算作新的独立认证集。新 seed 与 calibration/pilot/qualification/formal 全部不重叠，认证 held-out 不参与任何选型。','','独立性口径：frame Wilson 为描述性计数；episode-all-positive-detected 是不同 estimand，另给 episode bootstrap 的 frame-ratio 区间，不把时间相关的帧当作独立集。原 C0 使用 central two-sided 95%/99% z，保留数值并纠正 one-sided 文字。','','按计划停在 C_DECISION：不做新 formal 评估，不启动采集、PAI、GPU、S2。C0 时序违规仍未被、也无法被追溯消除。']
lines+=['', 'C4 数值判定：']
for name,v in [('receipt AND',before),('per-effect independent',after)]:
 lines.append(f"- {name}: 一致 {v['concordant_units']}/{v['unit_count']}，分歧 {v['divergent_units']}；假升级 {v['false_upgrade_count']}/{v['false_upgrade_denominator']}，Wilson 95% 上界 {v['false_upgrade_wilson_95_upper']:.8f}。")
lines.append('C4 Gate = '+str(gates['C4']['passed'])+'。相同分母的假升级上界上升且一致 unit 未增加。事件由冻结代码/seed 确定性重建，并非存在逐 unit 序列化日志；这是可复算的回放证据，不能伪称原始日志。')
lines+=['', 'C4 pilot 任务成功率包络（相对 M0_TYPED_MATCHED；非 formal，不能把 effect TPR 当作 task_success）：']
for name,v in c4['pilot_task_success_gain']['variants'].items():
 g=v['task_success_gain_interval'];lines.append(f"- {name}: 对抗下界 {g['lower_adversarial']:.6f}；继承 oracle 上界 {g['upper_inherited_oracle']:.6f}；中性假设 {g['neutral_divergence_zero']:.6f}（分歧 cell 增益为 0，非点估计）。分母 {v['faulted_task_cell_denominator']} 个 faulted task cells。")
lines.append('calibration 与非 pilot qualification 缺少已存 task outcomes，其 task-success 增益保持 null；不以检测率代填。')
(O/'C_DECISION.md').write_text('\n'.join(lines)+'\n')

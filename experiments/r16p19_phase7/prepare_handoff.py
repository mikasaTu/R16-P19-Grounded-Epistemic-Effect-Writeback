#!/usr/bin/env python3
"""Assemble stage checkpoint and human handoff; never selects a working point."""
import json,hashlib,subprocess,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(n):return json.loads((P/n).read_text())
def write(n,v):(P/n).write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
b1=read('B1_CEILING.json');b2=read('B2_CALIBRATION.json');b3=read('B3_HELDOUT.json');b4=read('B4_SOFT.json');b5=read('B5_MODEL_SELECTION.json');b6=read('B6_FRONTIER.json');repro=read('B1_PHASE5_REPRO.json')
assert b1['branch']=='CONTINUE_B2' and not b2['complete_vector'] and b6['eligible_operating_points']==0
old=json.loads((R/'experiments/r16p19_phase6/S1_REPLAY.json').read_text())['variants']['C_PER_EFFECT_CALIBRATED_AND'];div=set(old['divergent_unit_ids'])
oracle=[json.loads(l) for l in (R/'experiments/r16p19_phase5/artifacts/results/oracle_formal_results.jsonl').read_text().splitlines() if l]
oracle=[r for r in oracle if r['condition']!='C0_CLEAN' and r['arm'] in ['M0_TYPED_MATCHED','M3_ASCEL_CORE']]
gain=lambda rows:sum((1 if r['arm']=='M3_ASCEL_CORE' else -1)*int(r['task_success']) for r in rows)
full=gain(oracle);known=gain([r for r in oracle if r['cluster_id'] not in div]);total=len(oracle)//2
assert (known-len(div)*6)/total==old['faulted_success_gain_interval']['lower'] and full/total==old['faulted_success_gain_interval']['upper'],'STOP S1 gain discrepancy'
write('MECHANISM_DIAGNOSTIC.json',{'claim_eligible':False,'new_idea_generated':False,'S1_recheck':{'oracle_gain_numerator':full,'concordant_gain_numerator':known,'faulted_cells':total,'concordant_share_of_oracle_gain':known/full,'concordant_units':120-len(div),'status':'MATCH'},'code_confirmed_mechanisms':['Global pooled threshold suppresses low-score rare effects; effect-specific thresholds repair scale mismatch without changing weights.','n=14/15 prevents Wilson LCB certification even at empirical TPR=1. This is an exact sample-size obstruction, not formal transfer failure.','Arithmetic weighted mean permits compensating a low effect by a high effect; min enforces componentwise conjunction at a common threshold.','sum(log p) is a receipt-only score; its AUC cannot substitute for effect TPR.','Only selected small_mlp weights were serialized, preventing no-retrain alternative ranking.'],'hypotheses_not_confirmed':['Sparse positive coverage can make tail thresholds unstable across splits; no new rollout or causal test performed.','High AUC can coexist with poor rare-effect operation at the original pooled threshold; model-capacity inadequacy is not established by these diagnostics.']})
lineage={
 'base_fit':{'split':'natural','seeds':[0,1,2,3,4],'episodes':150,'frame_effect_rows':13138,'positive_rows':2632,'natural_total_episodes':210,'unused_natural_seed_5_6_episodes':60,'source':'r16p19/phase5_verifier_model.py:145-153; r16p19/phase5_verifier_data.py:49-67'},
 'calibration':{'episodes':30,'rows':2711,'positive_rows':505,'historical_use':'fit calibrator scale/bias and choose threshold','source':'r16p19/phase5_verifier_model.py:156-160'},
 'pilot':{'episodes':15,'verifier_fit_use':False,'historical_use':'oracle pilot matrix','source':'r16p19/phase5_runner.py:207-208'},
 'qualification':{'episodes':30,'rows':2615,'positive_rows':501,'historical_use':'per-group qualification, robustness metrics and model selection','untouched_holdout':False,'source':'r16p19/phase5_verifier_model.py:160-182'},
 'pilot_qualification_environment_overlap':{'count':15,'initializations':[20,21,22,23,24],'policy_seed':0,'tasks':[0,5,9],'environment_seed_formula':'500000 + task_id*10000 + init_id*100 + policy_seed; variant excluded','source':'r16p19/phase5_rollout.py:37-45,89-90','implication':'Threshold-heldout only, not independent environment-heldout.'},
 'sample_dependence':'Rows are per-frame/per-effect expansions within episodes, not independent trials. Wilson intervals use the user-requested counting convention, not a guarantee of independence.',
 'candidate_preservation':b5['available_checkpoint'],
 'audit_provenance':{'independent_read_only_audit':True,'primary_crosscheck':'B2 live counts matched independently audited 371/14/14/381/15; checkpoint hash and original source criterion verified.'}
}
write('B1_TRAINING_SPLIT_AUDIT.json',lineage)
write('B7_REPLAY.json',{'status':'NOT_EXECUTED_NO_SEALED_OPERATING_POINT','selected_variant':None,'operating_point_seal':None,'concordant_units':None,'divergent_units':None,'divergent_unit_ids':None,'gain_interval':{'adversarial_lower':None,'inherited_oracle_upper':None,'neutral_assumption_value':None,'neutral_is_assumption_not_point_estimate':True},'S2_candidate_set_valid':False,'S2_candidate_count':None,'S2_started':False,'reason':'B2 and all B6 targets are uncertifiable. F formal ceiling thresholds and prior S1 disagreement sets cannot substitute for a sealed current working point.'})
# Empty transport file is not a valid selected set; paired status explicitly disallows its execution.
(P/'B7_S2_UNITS.txt').write_text('')
write('B7_S2_UNITS_STATUS.json',{'status':'NO_VALID_CANDIDATE_SET','file':'B7_S2_UNITS.txt','file_is_empty_placeholder':True,'candidate_count':None,'execution_allowed':False,'reason':'No sealed working point. Empty file does not mean zero disagreements.'})
softrows=[]
for split in ['estimation','qualification']:
 for m in ['min','sum_log_p']:
  v=b4[split][m];softrows.append(f'| {split} | {m} | {v["auc"]:.9f} | {len(v["points"])} | {v["produces_effect_decisions"]} |')
lines=['# step8 / S1b checkpoint（非 Phase-6/7 最终报告）','',
'当前状态：B1 上界诊断通过继续条件；B2 无法认证完整阈值向量。B3 做完可定义的部分检验，B4 两类曲线已完成，B5 因 linear 权重缺失无法完成双候选重选，B6 的 20 个目标均无法形成认证工作点；B7 依赖缺失，未重放、不生成有效 S2 候选。','',
'## B0 与边界','',
'B0 在任何 B1 数据分析前封存，commit 8871bf9792cf029c765becfb61aad16a6b0253c6，SHA256 7fd8f1cc3392aea5a375abe293843305178c789d5e38f243e34f9738f04f17ec。B2 在 qualification 计算前提交封存，commit a4bb9fb。选择规则始终未改；B2/B5/B6 选择过程有文件访问阻断，formal 读取数 0。F 从未进入任何选择 seal。','',
'全部 CPU，GPU/PAI/rollout/S2 均为 0。Phase-5/6 原产物只读。','',
'## B1 天花板（仅诊断，claim_eligible=false）','',
'| Receipt FPR 约束 | 可达 min effect TPR | 假升级 | 实际 receipt FPR |','| --- | ---: | ---: | ---: |']
for v in b1['operating_points']:lines.append(f'| {v["receipt_fpr_cap"]:.2f} | {v["min_per_effect_tpr"]:.6f} | {v["false_upgrade_count"]}/{v["negative_receipts"]} | {v["receipt_fpr"]:.6f} |')
lines+=['','840 个去重 receipt、120 个 unit；228 positive receipt、612 negative receipt。下面阈值使用了 formal 标签，仅供上界诊断，永远不能被选作工作点。','','| Effect | 上界诊断阈值 | TP/positive | TPR |','| --- | ---: | ---: | ---: |']
for k,v in b1['operating_points'][0]['effects'].items():lines.append(f'| {k} | {v["threshold"]:.12g} | {v["tp"]}/{v["positive_count"]} | {v["tpr"]:.6f} |')
lines+=['','不能据此下结论“检测器容量不足”。F 只说明这个冻结 formal 样本上的分数存在可分工作区间；它既不能证明泛化，也不能替代独立阈值认证。','',
'## B1 拟合来源与独立性审计','',
'natural 共 210 集，但 base-model fit 只读取 seed 0–4 的 150 集：13138 frame-effect rows。seed 5/6 的 60 集未用于拟合，但 natural split 整体是训练来源，未事后重定义成新的 B2 split。calibration 曾拟合 Platt-like scale/bias 和原阈值；本次依用户明确要求纳入 calibration+pilot。qualification 已用于原始模型选型，且其 clean 与 pilot noop 共享 15 个环境 seed，所以 B3 仅对本轮阈值估计 held-out，不是严格独立的全流程验证集。','',
'计数单位为 frame-effect，episode 内存在时序相关性；以下 Wilson 是计划指定的计数区间，不能把相关帧声称为独立样本。','',
'## B2 合并估计与 B3 held-out','',
'calibration+pilot 共45集、4030 frame-effect rows。','','| Effect | 正样本数 | 可达最高 Wilson LCB | 阈值 | 状态 |','| --- | ---: | ---: | ---: | --- |']
for k,v in b2['effects'].items():lines.append(f'| {k} | {v["positive_count"]} | {v["maximum_attainable_lcb"]:.6f} | {v["threshold"]} | {v["status"]} |')
lines+=['','三个稀有 effect 的 14/14/15 个正样本即使全对，LCB 也不足 0.80，更不可能达到 0.90；因此不能产生完整向量。这是认证数据不足，不是已证实的 formal 迁移失败。','','| 可估计 effect | qualification TP/positive | TPR | Wilson 95% |','| --- | ---: | ---: | --- |']
for k,v in b3['effects'].items():
 if v['status']=='EVALUATED':lines.append(f'| {k} | {v["tp"]}/{v["positive_count"]} | {v["tpr"]:.6f} | [{v["tpr_wilson95"][0]:.6f}, {v["tpr_wilson95"][1]:.6f}] |')
lines+=['','soup 的 held-out LCB=0.872347 也未达到 0.90；mug 的 LCB=0.911920 达到该条。三个缺阈值 effect 的 TPR/FPR 和完整 receipt 假升级数均记 null，不能写成 0。','',
'## B4 D2 软聚合','',
'完整曲线见 B4_SOFT.json，不强制零假阳。','','| Split | 聚合 | ROC AUC | 曲线点数 | 有 effect 级判决 |','| --- | --- | ---: | ---: | --- |',*softrows,'',
'min(scores) >= t 严格等价于每个 score >= t，可给出同量纲 effect 判决。sum(log p) 只给出 receipt 级分数，已排除出 G1\' 候选；其略高 AUC 不能被当作 per-effect TPR 提升或认证成功。','',
'## B5 原始选型与可复用产物','',
'实际原始选择规则为 max(qualified, macro_auroc, -ece, prefer linear)，并非 loss/accuracy。两个候选原始 qualified 都为 false；small_mlp 的 macro AUC 0.991667 高于 linear 0.982177，因此被保存。','',
'Phase-5 只序列化 selected small_mlp（6644 bytes；SHA256 edba3ff23ae71c5768760af161b601b3d1a9f65d37d1ddeb8304b2510e938005）。linear 只有历史聚合指标，无 w/b 或逐样本分数。因禁止重训，无法恢复其固定 FPR 排名或换模型后的 B2 结果。这部分明确为 BLOCKED_MISSING_LINEAR_CHECKPOINT，未伪称完成。']
if 'available_model_fixed_FPR_result' in b5:
 v=b5['available_model_fixed_FPR_result'];ev=v['evaluation'];lines+=['',f'可用 small_mlp 在 calibration+pilot、receipt FPR<=0.02 下的 min effect TPR={v["minimum_per_effect_TPR"]:.6f}，假升级 {ev["false_upgrade_count"]}/{ev["negative_receipts"]}。这是选型准则的描述值；B2 Wilson 认证结果仍不变。linear 同准则值为 null，二者无法排序。']
lines+=['','## B6 与 B7','',
'按冻结规则扫描 0.80–0.99 共20个目标，所有目标都缺三个稀有 effect 的认证阈值。qualification 前沿条目全部保留，假升级率与 oracle 一致率为 null；不能画出不存在的完整向量曲线。formal 前沿同样不可定义且 selection_eligible=false。未创建 B6_OPERATING_POINT_SEAL.json。','',
'B7 不具备封存工作点前提，未执行新重放。增益上下界、中性假设和分歧 unit 都为 null；B7_S2_UNITS.txt 是空占位文件，配套状态明确 candidate_count=null、execution_allowed=false，不能解释为“零分歧”或执行空集合。没有继承 S1 的18个/93个分歧集合，也没有使用 F 阈值。','',
'## 机制反解（从代码和复算证据，不生成新 idea）','',
f'S1 固定分数复算：1680 行 mismatch=0；B/C oracle 一致率精确复现 0.733333/0.957143，C 一致 unit=102/120；一致 unit 的 oracle 增益分子为 {known}/{full}={known/full:.6%}，对应原先约87%的结论。原始 S1 结论和文件未更改。','',
'已确认的实现机制：原始全局阈值把不同 effect 的分数尺度混合，低分的稀有 effect 被 AND 规则直接否决；per-effect 阈值不改模型权重即可降低这些假阴性。B1 中 tomato 的上界阈值约0.02039，与 S1 calibration 阈值0.16626差距大；这只是 formal 诊断，不可反向写入选择。少量正样本能决定经验最小分，却无法认证总体召回率，解释了“经验判决改善”和“认证仍失败”同时出现。','',
'已确认的统计区别：64/76 的 Wilson 区间 [0.743986,0.907305] 包含0.90，不能用点估计断言总体TPR<0.90；但同一区间下界也没有达到新G1\'的认证要求。n=72–80 所需最少成功数是 n−2，经验TPR范围0.972222–0.975；用户计划的0.973是近似值。B0在分析前已写明按精确Wilson公式认证，规则未事后改变。','',
'已确认的聚合机制：算术平均允许一个高分补偿另一个低分，而 AND 要求每一项都满足；min保留这个逻辑。sum(log p)使用联合低分惩罚，ROC略有变化，但不产生独立effect判决。未把微小AUC差异解释成已验证泛化收益。','',
'待证机制：稀有正样本覆盖不足可能导致阈值尾部不稳定；现有相关帧和重用qualification不足以做独立因果验证。没有新rollout、没有新idea。','',
'## 验证与交接','',
'独立求解器审查与7项CPU测试见 TEST_REPORT.txt 和 CODE_REVIEW.md；测试含全局暴力枚举对照、Wilson反演、阈值tie、访问阻断、AUC ties及缺失值不伪装成0。','',
'交接以 B_DECISION.md / B_DECISION.json 为准。停止在本轮 checkpoint，不启动S2、不提交PAI、不写Phase-6/7最终报告。']
(P/'STEP8_CHECKPOINT.md').write_text('\n'.join(lines)+'\n')
write('B_DECISION.json',{'status':'UNCERTIFIABLE_NO_OPERATING_POINT','not_a_phase6_or_phase7_final_report':True,'G1_prime_pass':False,'selected_variant':None,'gates':{'1_formal_effect_LCB':{'status':'UNCERTIFIABLE_NO_SELECTED_VARIANT','value':None,'evidence':'B2 positive counts 14/14/15; max LCB 0.784689/0.784689/0.796117; B6 all20 targets lack vector'},'2_formal_false_upgrade_UCB_and_significance':{'status':'NOT_EVALUATED_NO_SELECTED_VARIANT','UCB':None,'p_vs_B3_0_50':None,'F_diagnostic_is_not_gate_evidence':True},'3_selection_no_formal_paths':{'status':'PASS_FOR_EXECUTED_SELECTION','formal_access_count':0,'B2_seal':'B2_SELECTION_SEAL.json','B6_operating_point_seal':None,'evidence':['B2_SELECTION_SEAL.json','B2_B6_READ_AUDIT.json','B5_READ_AUDIT.json']},'4_phase5_reproduction':{'status':'PASS','compared_rows':repro['compared_rows'],'mismatch_rows':repro['mismatch_rows']}},'stage_status':{'B0':'SEALED','B1':'DIAGNOSTIC_CEILING_CONTINUE','B2':'UNCERTIFIABLE','B3':'PARTIAL_EFFECT_TESTS_NO_FULL_VECTOR','B4':'COMPLETED_CURVES_AND_AUC','B5':'PARTIAL_AVAILABLE_MODEL_BLOCKED_MISSING_LINEAR','B6':'20_TARGETS_NO_CERTIFIABLE_POINT','B7':'NOT_EXECUTED_MISSING_SEAL'},'S2_candidate_set_valid':False,'awaiting_human_confirmation':True,'GPU':0,'PAI':0,'rollouts':0,'S2':0})
(P/'B_DECISION.md').write_text('''# B_DECISION — step8 / S1b 人工交接

**UNCERTIFIABLE_NO_OPERATING_POINT；G1\' 未获认证。** 这不是 Phase-6/7 最终报告。

1. 每 effect formal TPR LCB≥0.90：无所选变体，不能判通过。B2三个稀有effect正样本14/14/15，可达最高LCB仅0.784689/0.784689/0.796117；B6全部20个目标无完整认证向量。B3可估计的soup LCB=0.872347也未过0.90，mug LCB=0.911920。
2. formal假升级UCB≤0.05且显著低于B3的50%：未评估，无所选变体；UCB与p值均为null。F的10/612只用于诊断，不作为门证据。
3. 选择过程formal访问：已执行的B2/B5/B6选择阶段为0，B2 seal及读路径审计已保存；没有伪造B6工作点seal。
4. Phase-5判决复现：1680行、mismatch=0，PASS。

B1 在FPR≤2%和≤5%下均可达min effect TPR=1.0，假升级10/612（1.633987%），因此没有走容量不足的早停分支，继续执行了后续可行步骤。

B4的min与sum(log p)完整曲线已保存；B5已算可用small_mlp的同准则值，但linear权重/逐样本分数未保存，禁止重训下无法完成双模型重选。qualification曾参与原始选型，且和pilot共享15个环境seed，不能声称全流程独立held-out。

B7因缺少封存工作点未执行；S2候选没有有效集合。空B7_S2_UNITS.txt仅为交付占位，必须结合B7_S2_UNITS_STATUS.json（execution_allowed=false、candidate_count=null）读取，不能解释为零分歧。

交回：B1_CEILING.json、B3_HELDOUT.json、B7_REPLAY.json及候选状态。详见STEP8_CHECKPOINT.md。等待人工确认是否另行修订数据/候选保存条件；本任务不擅自改B0、不重训、不新增rollout、不启动S2、不提交PAI。
''')
print('handoff prepared; S1 gain fraction',known,full,known/full)

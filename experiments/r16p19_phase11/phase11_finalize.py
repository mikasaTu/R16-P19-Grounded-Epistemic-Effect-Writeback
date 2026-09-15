import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run_phase11 import OUT,write,sha

def gate_f1_baseline(value,target=.9916669890228063):
 return isinstance(value,(int,float)) and math.isfinite(value) and abs(value-target)<1e-7

def gate_f1_conclusion(a,b,e,f,verdict):
 if not all(isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=1 for x in (a,b,e,f)):return False
 expected='SHORTCUT_CONFIRMED' if abs(b-a)<.02 or e>.90 else 'NOT_CONFIRMED'
 return verdict==expected

def gate_f2_complete(rows):
 return len(rows)==35 and sorted(r.get('index',-1) for r in rows)==list(range(35)) and all(r.get('classification') in ['onboard_observable','privileged','shortcut'] for r in rows)

def gate_f4_evidence(d):
 rows=d.get('per_effect',[])
 numeric=len(rows)==5 and all(isinstance(r.get('auroc'),(int,float)) and 0<=r['auroc']<=1 and isinstance(r.get('tpr_at_fpr_0.05'),(int,float)) and 0<=r['tpr_at_fpr_0.05']<=1 for r in rows)
 insufficient=d.get('absolute_ceiling_identified') is False and d.get('evidence_insufficient_for_true_ceiling') is True and d.get('required_resolution',[0,0])[0]>=224 and set(['rgb_native','camera_calibration','effect_onset','control_owner']).issubset(d.get('required_fields',[]))
 return numeric and insufficient

def main():
 f1=json.loads((OUT/'F1_ABLATION.json').read_text());fs=f1['sets'];formal=json.loads((OUT/'F1_FORMAL_COMPARISON.json').read_text());t=json.loads((OUT/'F4_TRANSFERABLE_METRICS.json').read_text())
 a,b,c,d,e,f=[fs[k]['macro_auc'] for k in ['A_full','B_no_visual','C_no_time','D_no_contact','E_shortcut_only','F_visual_only']]
 verdict='SHORTCUT_CONFIRMED' if abs(b-a)<.02 or e>.90 else 'NOT_CONFIRMED'
 f1.update(baseline_reference=.9916669890228063,baseline_match=gate_f1_baseline(a),shortcut_verdict=verdict,formal_comparison_file='F1_FORMAL_COMPARISON.json',formal_claim_eligible=False,formal_protocol_valid=False)
 write('F1_ABLATION.json',f1)
 lines=['# F1 捷径消融','',f'A={a:.9f}，精确复现 Phase-5 qualification macro AUC。B−A={b-a:+.9f}；E={e:.9f}；F−A={f-a:+.9f}。判定 **{verdict}**。','', '|变体|qualification macro AUC|相对A|formal macro AUC（无效协议，仅描述）|','|---|---:|---:|---:|']
 for k,v in fs.items():lines.append(f"|{k}|{v['macro_auc']:.6f}|{v['macro_auc']-a:+.6f}|{formal['models'][k]['macro_auc']:.6f}|")
 lines+=['','## 五个效果的 AUC / TPR@FPR=0.05','', '|变体|task/effect|AUC [episode 95% CI]|TPR@FPR≤0.05 [episode 95% CI]|','|---|---|---|---|']
 for k,v in fs.items():
  for r in v['per_effect']:
   ci=r['episode_cluster_ci95'];ac=ci['auroc'];tc=ci['tpr_at_fpr_0.05'];lines.append(f"|{k}|{r['task_id']}/{r['effect_index']}|{r['auroc']:.4f} [{ac[0]:.4f}, {ac[1]:.4f}]|{r['tpr_at_fpr_0.05']:.4f} [{tc[0]:.4f}, {tc[1]:.4f}]|")
 lines+=['','TPR@FPR≤0.05 是同一评价 split 的描述性 ROC 指标；没有用它选模型。calibration 负类确定的独立工作点 TPR/FPR 及区间另存 JSON。稀有 effect 只有9–10个qualification正行；bootstrap零宽区间不代表总体无不确定性。','qualification 曾参加原始选型，且与 pilot 共用seed；本轮无模型/超参选型，但不是新独立测试。','E 在task9/effect0上的TPR仅约0.3945，说明高macro AUC并不保证低FPR工作点效用。']
 (OUT/'F1_DECISION.md').write_text('\n'.join(lines)+'\n')
 (OUT/'F1_COMPARISON_TABLE.md').write_text('\n'.join(lines[4:12])+'\n')
 dims=[]
 for i in range(35):
  if i<18:
   section=['base','wrist','base_temporal_delta'][i//6];sub=i%6;name=f'{section}_{"mean" if sub<3 else "std"}_{"RGB"[sub%3]}';cl='onboard_observable';sensor='RGB camera (matching base/wrist view), hardware and synchronization unverified';replacement=None
  elif i<26:name=f'proprio_{i-18}';cl='onboard_observable';sensor='observation/state channel; encoder/FK semantic mapping requires verification';replacement=None
  elif i==26:name='log1p(contact_count)';cl='privileged';sensor='simulator trace contact_count';replacement='force/torque or tactile contact estimator; not semantically equivalent without validation'
  elif i==27:name='linspace_episode_time';cl='shortcut';sensor='future-dependent complete episode length normalization; no deployable equivalent';replacement='exclude from effect detector'
  elif i<31:name=f'task_onehot_{[0,5,9][i-28]}';cl='onboard_observable';sensor='query/task metadata, not physical evidence';replacement=None
  else:name=f'effect_onehot_{i-31}';cl='onboard_observable';sensor='query predicate metadata, not ground-truth label';replacement=None
  dims.append({'index':i,'name':name,'classification':cl,'sensor_or_source':sensor,'real_platform_available':None,'replacement':replacement,'replacement_available':None})
 write('F2_SENSOR_PARITY.json',{'dimensions':dims,'dimension_count':35,'coverage_complete':gate_f2_complete(dims),'real_platform_hardware_verified':False,'unknowns':['camera model/FOV/extrinsics/synchronization','proprio channel meaning/units/FK mapping','force torque/tactile availability']})
 md=['# F2 逐维传感器审计','','下表分类描述原则上的真机可观测性；当前没有目标真机硬件清单，所以真实平台可用性一律为 null，未把仿真记录当作硬件证据。','', '|维度|名称|分类|来源/真机替代|','|---:|---|---|---|']
 for r in dims:md.append(f"|{r['index']}|{r['name']}|{r['classification']}|{r['sensor_or_source']}; {r['replacement'] or ''}|")
 md+=['','contact_count 的力/触觉替代是否已可用：null。proprio 仍可能携带位姿/进度捷径；可观测不等于可迁移。task/effect one-hot 是查询条件，不是效果发生证据。','原始 `_feature_rows` 18:26 的状态通道语义与真实机器人8维状态的对应关系待确认；不臆造关节数量或硬件。']
 (OUT/'F2_SENSOR_PARITY.md').write_text('\n'.join(md)+'\n')
 t.update(absolute_ceiling_identified=False,evidence_insufficient_for_true_ceiling=True,empirical_attainable_macro_auc=t['macro_auc'],formal_cache_comparison=formal['models']['T_spatial'],required_resolution=[224,224],required_resolution_status='proposed engineering acquisition floor, not empirically established universal minimum',required_fields=['rgb_native','camera_calibration','effect_onset','control_owner','proprio_timestamp','action_timestamp','query_predicate','episode_id'],capacity_statement='0.982006483 qualification macro AUC under the tested spatial model; this is attained diagnostic performance, not proof of strongest possible representation or a real-robot upper bound',sim_split='qualification',selection_eligible=False)
 write('F4_TRANSFERABLE_SPEC.json',t)
 md=['# F4 真机可观测检测器规格与仿真侧可达水平','',f"测试的可迁移特征：双相机4×4空间块RGB均值/标准差192维 + proprio8维 + task/effect查询7维，共207维；剔除contact_count与归一化time。模型保持32隐藏单元MLP、1600步、种子29。qualification macro AUC={t['macro_auc']:.9f}；A全特征={a:.9f}。",'','|task/effect|A AUC|空间模型AUC [episode CI]|A TPR@5%FPR|空间模型TPR@5%FPR [episode CI]|','|---|---:|---|---:|---|']
 for ra,rt in zip(fs['A_full']['per_effect'],t['per_effect']):
  ci=rt['episode_cluster_ci95'];md.append(f"|{rt['task_id']}/{rt['effect_index']}|{ra['auroc']:.4f}|{rt['auroc']:.4f} [{ci['auroc'][0]:.4f},{ci['auroc'][1]:.4f}]|{ra['tpr_at_fpr_0.05']:.4f}|{rt['tpr_at_fpr_0.05']:.4f} [{ci['tpr_at_fpr_0.05'][0]:.4f},{ci['tpr_at_fpr_0.05'][1]:.4f}]|")
 md+=['','**上界问题的边界：**现有数据证实了以上可达数值，但不能识别真正的检测容量上界，也不能证明4×4池化是32×32上的最强表示。低分辨率是否是瓶颈，本轮没有多分辨率对照，不能强行断言已证实。该问题的严格答案是上界未识别；不把一项新模型的分数冒充数学上界。','若要建立面向真机效果谓词的上界，需要原始双相机RGB、相机内外参和时间同步、效果起始/稳定区间/遮挡标注、控制权来源及动作时间戳。建议保留原始分辨率且至少224×224作为工程起点，使花茎/瓶口关键几何特征可分辨；224并非已验证最小值，须以目标物体像素覆盖与人工可辨性验证，不在本轮采集。','本轮空间表示与C_no_time相比还同时去contact并改变视觉维数，差值不能单独归因于空间结构。proprio与task identity仍可能承载任务特定相关性，真机效果识别未验证。','formal空间模型macro AUC='+str(formal['models']['T_spatial']['macro_auc'])+'，使用已缓存分数，仅描述，claim_eligible=false、protocol_valid=false。']
 (OUT/'F4_TRANSFERABLE_SPEC.md').write_text('\n'.join(md)+'\n')
 mechanism=f'''# 代码机理反解（不生成新idea）

`r16p19/phase5_verifier_data.py:16-41` 把每幅图像32×32×3压成通道均值/标准差，像素重排不改变这些特征；temporal也是统计量的差。该路径无法直接保留“物体在容器内”等空间关系。与此同时time=linspace(0,1,chunks)包含完整episode长度的后验进度，本体状态与task/effect one-hot形成任务特定的进度查询。

A→B去全部视觉后macro AUC从{a:.9f}到{b:.9f}（{b-a:+.9f}）；说明原高排序性能无需这些视觉统计。A→C去time下降{c-a:+.9f}，time对现有泛化分布有贡献。E只保留time与查询ID即达{e:.9f}；在task9/effect0的低FPR TPR却约0.3945，不能用平均AUC掩盖失效。F仅视觉统计降至{f:.9f}，不接近A，不能据此说“所有视觉都无信息”，因为这里只消融了退化全局统计。

去contact仅变为{d:.9f}；并不支持“主要由接触真值泄漏导致”这一更强说法。B仍含proprio与contact，不能把B结果单独归因于proprio。没有单独移除proprio的预设变体，故time/姿态贡献不可完全分解。权重L2、正类加权和全批量MLP保持一致，差异来自输入子集及其维数改变下重新拟合，而非固定模型的局部干预。

F4空间特征达到{t['macro_auc']:.9f}，但同时更改视觉表示、删除time/contact；它是可迁移输入规格的诊断，不是空间结构的单变量因果实验。所有训练使用natural seed0–4，模型字节另存；原phase5权重未改。
'''
 (OUT/'MECHANISM_REVERSE_ENGINEERING.md').write_text(mechanism)
 gates={'gate_f1_baseline':gate_f1_baseline(a),'gate_f1_conclusion':gate_f1_conclusion(a,b,e,f,verdict),'gate_f2_complete':gate_f2_complete(dims),'gate_f4_evidence':gate_f4_evidence(t)}
 dec={'status':'DIAGNOSTICS_COMPLETE_WITH_FORMAL_PROTOCOL_DEVIATION','shortcut_verdict':verdict,'gates':gates,'formal_protocol_valid':False,'claim_eligible':False,'formal_deviation':'schema probe of one formal NPZ and preview of phase9 formal cache before freeze; cannot claim read-once protocol','baseline_macro_auc':a,'B_minus_A':b-a,'E_macro_auc':e,'F_macro_auc':f,'F4_macro_auc':t['macro_auc'],'true_capacity_ceiling_identified':False,'target_real_robot_data_available':False,'S2_started':False,'PAI_submitted':False,'real_robot_results':None,'idea_continue_decision':'reserved for human'}
 write('F_DECISION.json',dec)
 (OUT/'F_DECISION.md').write_text('# Gate F 决策记录\n\n状态：'+dec['status']+'。\n\n'+ '\n'.join(f'- {k}: {v}' for k,v in gates.items())+f'\n\nF1-A={a:.9f}，精确复现；B−A={b-a:+.9f}，E={e:.9f}，F={f:.9f}，判定{verdict}。F2覆盖35/35维。F4可观测输入诊断macro AUC={t["macro_auc"]:.9f}，真实容量上界未识别，需求条件见F4。\n\nformal存在提前访问违规，所有formal派生量标protocol_valid=false，仅描述，不可宣称合规final evaluation。QPILOTS可读但目标arrange_3_flowers数据未找到，实际数量均null；F3数据要求与F5设计已给出。\n\nF1–F5计算/审计/设计产物均已交付；没有GPU、PAI、rollout、新采集或S2。是否继续idea由人工决定。本文件为阶段门记录，不是Phase-11最终报告。\n')
 write('PROTOCOL_DEVIATIONS.json',{'formal_early_schema_probe':True,'phase9_mixed_cache_preview':True,'formal_protocol_valid':False,'excluded_failed_development_runs':['natural seed0-6 contamination','grouped effect index across tasks','max threshold yielded zero TPR','missing calibrator import'],'final_corrections':{'training_rows':13138,'group_count':5,'baseline_match':gate_f1_baseline(a),'threshold_rule':'tie-safe ROC threshold at FPR<=.05'},'F0_nonbinding_extra_protocol_deviation':'F0 mentioned calibration+pilot; exact Phase5 calibration uses calibration only. User plan exact Phase5 split takes precedence; no retroactive preregistration claim.'})
if __name__=='__main__':main()

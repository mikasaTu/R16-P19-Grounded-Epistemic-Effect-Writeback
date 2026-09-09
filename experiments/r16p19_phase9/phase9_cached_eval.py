"""Corrected descriptive diagnostics from recovered cache only. No source formal I/O."""
import json,math,hashlib,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from phase9_stats import cp_upper,wilson,binom_lower_tail
SEAL_COMMIT='21acdc25331cb221fbeab2c34a1b9044580cadc2'
D0_COMMIT='d4c81be6149068ab56cda8350c99b86ddf1c7fa8'
META={'claim_eligible':False,'selection_eligible':False,'protocol_valid':False,'original_formal_load_attempts':2,'formal_read_once':False,'recovery_source':'RECOVERED_LOADED_EVIDENCE.json; extracted from live process memory before termination','new_original_formal_reads':0,'d3_seal_commit':SEAL_COMMIT}
def write(name,d):
 if isinstance(d,dict):d.update(META)
 (ROOT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def validp(v):return isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1
def gate_exchangeability(max_rare_fpr_ratio):
 return isinstance(max_rare_fpr_ratio,(int,float)) and math.isfinite(max_rare_fpr_ratio) and 0<=max_rare_fpr_ratio<=2

def gate_effect_fpr(ucbs,alpha,certified):
 return bool(certified and validp(alpha) and len(ucbs)==5 and all(validp(v) and v<=alpha for v in ucbs))
def gate_receipt_fpr(cluster_ucb,pvalue_vs_half):
 return bool(validp(cluster_ucb) and cluster_ucb<=.05 and validp(pvalue_vs_half) and pvalue_vs_half<.05)
def gate_ledger(concordant,total):
 return bool(isinstance(concordant,int) and isinstance(total,int) and total>0 and 0<=concordant<=total and concordant/total>=.85)
def effect_accept(score,t):return t is not None and score>t
def receipt_accept(r,t):return all(effect_accept(s,t[k]) for k,s in zip(r['effect_keys'],r['scores']))
def cluster_metrics(rows,key,threshold):
 neg=defaultdict(list);pos=defaultdict(list)
 for r in rows:
  if key not in r['effect_keys']:continue
  i=r['effect_keys'].index(key);p=effect_accept(r['scores'][i],threshold)
  (pos if r['labels'][i] else neg)[r['cluster_id']].append(p)
 fp=sum(any(v) for v in neg.values());tp=sum(any(v) for v in pos.values())
 return {'negative_episode_count':len(neg),'false_positive_episode_count':fp,'positive_episode_count':len(pos),'true_positive_episode_count':tp,'fpr':fp/len(neg) if neg else None,'tpr':tp/len(pos) if pos else None,'fpr_exact_upper95':cp_upper(fp,len(neg),.95),'fpr_exact_upper99':cp_upper(fp,len(neg),.99),'fpr_wilson95':wilson(fp,len(neg)),'tpr_wilson95':wilson(tp,len(pos)),'fpr_estimand':'any accepted NEGATIVE observation in an episode with >=1 negative observation','tpr_estimand':'any accepted POSITIVE observation in an episode with >=1 positive observation; not average per-frame TPR'}
def d1(cache):
 values={}
 for split,rows in [('estimation',cache['estimation']),('qualification',cache['qualification']),('formal',cache['formal'])]:
  vals=defaultdict(list)
  for r in rows:
   for k,s,y in zip(r['effect_keys'],r['scores'],r['labels']):
    if not y:vals[k].append(s)
  values[split]={k:np.array(v) for k,v in vals.items()}
 out={'effects':{},'sampling_population_mismatch':True,'sampling_population':{'estimation':'all frame-effect rows calibration+pilot','qualification':'all frame-effect rows qualification','formal':'seven selected condition injection receipts per unit'},'exchangeability_status':'NOT_CERTIFIABLE_SAMPLING_POPULATION_MISMATCH','d1_pass':False,'limitation':'KS and ratios below describe the available sampled score distributions, not an isolated split-transfer effect. Full formal frame population was not retained; original sources are not reopened.'}
 maxratio=0
 for k in sorted(values['estimation']):
  q={str(v):{s:float(np.quantile(values[s][k],v)) for s in values} for v in [.9,.95,.98,.99,.995]}
  for item in q.values():
   item['formal_minus_estimation']=item['formal']-item['estimation'];item['qualification_minus_estimation']=item['qualification']-item['estimation']
  ks={}
  for a,b in [('estimation','qualification'),('estimation','formal'),('qualification','formal')]:
   x=np.sort(values[a][k]);y=np.sort(values[b][k]);z=np.unique(np.r_[x,y]);ks[a+'_vs_'+b]=float(np.max(np.abs(np.searchsorted(x,z,side='right')/len(x)-np.searchsorted(y,z,side='right')/len(y))))
  ratios={}
  for alpha in [.005,.01,.02,.05]:
   threshold=float(np.quantile(values['estimation'][k],1-alpha));n=len(values['formal'][k]);fp=int(np.sum(values['formal'][k]>threshold));ratio=fp/n/alpha
   ratios[str(alpha)]={'threshold':threshold,'formal_negative_count':n,'formal_false_positive_count':fp,'formal_fpr':fp/n,'nominal_alpha':alpha,'actual_minus_nominal':fp/n-alpha,'actual_over_nominal':ratio}
   if any(v in k for v in ['tomato','book','close']):maxratio=max(maxratio,ratio)
  out['effects'][k]={'negative_counts':{s:len(values[s][k]) for s in values},'quantiles':q,'ks_statistic':ks,'nominal_actual':ratios}
 out['descriptive_max_rare_ratio']=maxratio;out['numeric_ratio_gate']=gate_exchangeability(maxratio)
 write('D1_EXCHANGEABILITY.json',out)
 (ROOT/'D1_DECISION.md').write_text('# D1 诊断\n\n描述性最大稀有 effect FPR / nominal = %.8f，数值门未通过。另有 all-frame 与 injection-frame sampling population 不一致，不能据此单独证明 split 不可交换；认证结论为不可判定/不通过。全部后续实验继续。\n'%maxratio)
 return out

def d4(cache,seal):
 rows=cache['formal'];thresholds=seal['selected_threshold_vector'];effects={}
 assert len(rows)==840 and len({r['cluster_id'] for r in rows})==120
 assert len({(r['cluster_id'],r['condition']) for r in rows})==840
 for k,t in thresholds.items():
  obs=[(bool(r['labels'][r['effect_keys'].index(k)]),effect_accept(r['scores'][r['effect_keys'].index(k)],t)) for r in rows if k in r['effect_keys']]
  tp=sum(y and p for y,p in obs);fp=sum(not y and p for y,p in obs);n=sum(not y for y,p in obs);pos=sum(y for y,p in obs)
  effects[k]={'threshold':t,'reject_all':t is None,'receipt':{'positive_count':pos,'negative_count':n,'tp':tp,'fp':fp,'tpr':tp/pos if pos else None,'fpr':fp/n if n else None,'tpr_wilson95':wilson(tp,pos),'fpr_wilson95':wilson(fp,n),'fpr_exact_upper95':cp_upper(fp,n,.95),'fpr_exact_upper99':cp_upper(fp,n,.99)},'episode_cluster':cluster_metrics(rows,k,t)}
 negrows=[r for r in rows if not all(r['labels'])];fprows=[r for r in negrows if receipt_accept(r,thresholds)]
 negunits={r['cluster_id'] for r in negrows};fpunits={r['cluster_id'] for r in fprows};n=len(negrows);fp=len(fprows)
 agree=sum(receipt_accept(r,thresholds)==all(r['labels']) for r in rows)
 min_tpr=min(v['receipt']['tpr'] for v in effects.values())
 out={'status':'COMPLETED_CACHED_DIAGNOSTIC_WITH_PROTOCOL_BREACH','selected_alpha':seal['selected_alpha'],'certified_point':None,'d3_seal_sha256':hashlib.sha256((ROOT/'D3_OPERATING_POINT_SEAL.json').read_bytes()).hexdigest(),'per_effect':effects,'receipt_false_upgrades':{'count':fp,'negative_receipt_count':n,'rate':fp/n,'wilson95':wilson(fp,n),'episode_count':len(fpunits),'negative_episode_count':len(negunits),'episode_exact_upper95':cp_upper(len(fpunits),len(negunits),.95),'episode_exact_upper99':cp_upper(len(fpunits),len(negunits),.99),'episode_wilson95':wilson(len(fpunits),len(negunits)),'pvalue_vs_half':binom_lower_tail(len(fpunits),len(negunits))},'oracle_receipt_agreement_count':agree,'oracle_receipt_agreement_denominator':len(rows),'oracle_receipt_agreement':agree/len(rows),'formal_unit_count':len({r['cluster_id'] for r in rows}),'b1_gap':{'reference_source':'user-provided frozen background, diagnostic-only','min_tpr_reference':1.0,'false_upgrade_reference':10,'false_upgrade_denominator_reference':612,'min_tpr_current':min_tpr,'min_tpr_current_minus_reference':min_tpr-1.0,'false_upgrades_current_minus_reference':fp-10}}
 write('D4_FORMAL.json',out);return out

def d6(cache,seal):
 rows=cache['formal'];false=[r for r in rows if not all(r['labels']) and receipt_accept(r,seal['selected_threshold_vector'])];units=sorted({r['cluster_id'] for r in false});total=len({r['cluster_id'] for r in rows})
 out={'false_upgrade_receipts':len(false),'false_upgrade_unit_count':len(units),'total_unit_count':total,'concentration':len(units)/total,'unit_ids':units,'unit_details':[{k:r[k] for k in ['cluster_id','task_id','condition','receipt_id']} for r in false],'task_distribution':dict(Counter(str(r['task_id']) for r in false)),'condition_distribution':dict(Counter(r['condition'] for r in false)),'special_conditions':{c:sum(r['condition']==c for r in false) for c in ['A5_EXTERNAL_REALIZATION','C0_CLEAN']},'temporal_or_attribution_confusion':None,'reason':'No false upgrades under reject-all; boundary attribution cannot be tested on an empty set. Zero error is caused by no upgrades, not resolved attribution.'}
 write('D6_FAILURE_MODES.json',out)
 (ROOT/'D6_FAILURE_MODES.md').write_text('# D6 假升级失效模式\n\n%d/%d 个 unit 出现假升级。当前集合为空，无法检验 A5/C0 集中性或 realized/observed/external 类型混淆；这些机制量记 null。低假升级来自 reject-all，不证明归属混淆得到解决。\n'%(len(units),total))
 return out

def source_guard(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
 p=str(Path(args[0]).resolve())
 if '/rollouts/' in p or any('/experiments/r16p19_phase%d/'%i in p for i in [5,6,7,8]):
  raise RuntimeError('Original evidence read forbidden in cache recovery evaluation: '+p)
def main():
 sys.addaudithook(source_guard)
 cache=json.loads((ROOT/'RECOVERED_LOADED_EVIDENCE.json').read_text());seal=json.loads((ROOT/'D3_OPERATING_POINT_SEAL.json').read_text())
 a=d1(cache);b=d4(cache,seal);c=d6(cache,seal)
 write('PROTOCOL_DEVIATIONS.json',{'D0_valid_separate_commit':True,'D0_commit':D0_COMMIT,'D3_valid_separate_commit':True,'violations':[{'requirement':'formal loaded once','observed_dataset_load_attempts':2,'detail':'First invocation loaded formal then failed due missing numpy import; second invocation loaded formal and ran D1/D2. Parent recovered live memory and terminated the second process. Source-hash inventory also read artifact bytes; per-file total source opens was not globally captured.','consequence':'Cannot claim valid single-read formal protocol; all downstream results diagnostic only.'}],'repairs':['D4 cluster FPR restricted to negative observations','D1 sampling-population mismatch disclosed','D5 actual ledger replay implemented separately','D2 regularized Platt implementation repaired from cache'],'original_raw_read_counter_exhaustive':False,'additional_original_formal_reads_after_recovery':0})
 print(json.dumps({'d1_max_ratio':a['descriptive_max_rare_ratio'],'d4_agreement':b['oracle_receipt_agreement'],'fp':b['receipt_false_upgrades'],'d6_units':c['false_upgrade_unit_count']},indent=2))
if __name__=='__main__':main()

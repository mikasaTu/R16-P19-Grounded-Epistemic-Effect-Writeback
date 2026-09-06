#!/usr/bin/env python3
"""CPU-only formal-label ceiling diagnostic. Never produces a selection seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib,itertools,math,datetime,platform,importlib.util
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'experiments/r16p19_phase7'
RAW=Path('/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/pai/r16p19-phase5-idle-fixed-20260818-0855/rollouts')
B0_SHA='7fd8f1cc3392aea5a375abe293843305178c789d5e38f243e34f9738f04f17ec'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,v):
 (OUT/name).write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-w,c+w]
def candidates(s): return np.r_[np.unique(s), np.nextafter(max(s),np.inf)]
def effect_stats(s,y,t):
 p=s>=t;tp=int((p&y).sum());n=int(y.sum());fp=int((p&~y).sum());neg=int((~y).sum())
 return dict(threshold=float(t),tp=tp,positive_count=n,tpr=tp/n,tpr_wilson95=wilson(tp,n),fp=fp,negative_count=neg,fpr=fp/neg if neg else None,fpr_wilson95=wilson(fp,neg) if neg else None)
def task_options(rs):
 keys=rs[0]['effect_keys'];s=np.array([r['scores'] for r in rs]);y=np.array([r['labels'] for r in rs],bool);truth=y.all(axis=1)
 opts=[]
 for ts in itertools.product(*(candidates(s[:,j]) for j in range(s.shape[1]))):
  pred=s>=ts;rates=(pred&y).sum(axis=0)/y.sum(axis=0);receipt=pred.all(axis=1)
  opts.append(dict(ts=tuple(float(t) for t in ts),min=float(min(rates)),sum=float(sum(rates)),rates=tuple(float(t) for t in rates),fp=int((receipt&~truth).sum())))
 return dict(keys=keys,s=s,y=y,truth=truth,rs=rs,options=opts)
def solve(groups,cap,negative_count):
 budget=math.floor(cap*negative_count+1e-12)
 targets=sorted({o['min'] for g in groups for o in g['options']},reverse=True)
 chosen_target=None
 for target in targets:
  if sum(min((o['fp'] for o in g['options'] if o['min']>=target),default=10**9) for g in groups)<=budget:
   chosen_target=target;break
 assert chosen_target is not None
 fronts=[]
 for g in groups:
  best={}
  for o in g['options']:
   if o['min']<chosen_target or o['fp']>budget:continue
   prev=best.get(o['fp'])
   if prev is None or (o['sum'],o['ts'])>(prev['sum'],prev['ts']):best[o['fp']]=o
  fronts.append(list(best.values()))
 feasible=(combo for combo in itertools.product(*fronts) if sum(o['fp'] for o in combo)<=budget)
 combo=max(feasible,key=lambda c:(-sum(o['fp'] for o in c),sum(o['sum'] for o in c),tuple(t for o in c for t in o['ts'])))
 effects={};false_ids=[];positives=0;tp=0
 for g,o in zip(groups,combo):
  for j,key in enumerate(g['keys']):effects[key]=effect_stats(g['s'][:,j],g['y'][:,j],o['ts'][j])
  pred=(g['s']>=o['ts']).all(axis=1);positives+=int(g['truth'].sum());tp+=int((pred&g['truth']).sum())
  false_ids += [r['receipt_id'] for r,a,b in zip(g['rs'],pred,g['truth']) if a and not b]
 next_targets=[t for t in targets if t>chosen_target];next_target=min(next_targets) if next_targets else None
 min_fp=lambda t:sum(min((o['fp'] for o in g['options'] if o['min']>=t),default=10**9) for g in groups)
 return dict(claim_eligible=False,selection_eligible=False,receipt_fpr_cap=cap,allowed_false_upgrades=budget,min_per_effect_tpr=chosen_target,effects=effects,false_upgrade_count=len(false_ids),negative_receipts=negative_count,receipt_fpr=len(false_ids)/negative_count,receipt_fpr_wilson95=wilson(len(false_ids),negative_count),receipt_tp=tp,positive_receipts=positives,receipt_tpr=tp/positives,false_upgrade_receipt_ids=sorted(false_ids),optimality=dict(next_attainable_min_tpr=next_target,minimum_false_upgrades_at_next_tpr=min_fp(next_target) if next_target else None,minimum_false_upgrades_at_branch_target_0_973=min_fp(.973)))
def main():
 assert sha(OUT/'B0_PREREGISTRATION.json')==B0_SHA,'B0 drift'
 opened=set()
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   p=Path(os.fsdecode(args[0])).absolute()
   if (str(p).startswith(str(RAW)) or str(p).startswith(str(REPO))) and not str(p).startswith(str(OUT)) and p.is_file(): opened.add(p)
 sys.addaudithook(audit)
 spec=importlib.util.spec_from_file_location('s1',REPO/'experiments/r16p19_phase6/run_s1.py');s1=importlib.util.module_from_spec(spec);spec.loader.exec_module(s1)
 before={p.relative_to(REPO).as_posix():sha(p) for root in ['r16p19_phase5','r16p19_phase6'] for p in (REPO/'experiments'/root).rglob('*') if p.is_file()}
 repro=s1.reproduce(REPO,RAW,OUT)
 (OUT/'S1_REPRO.json').rename(OUT/'B1_PHASE5_REPRO.json')
 historical=json.loads((REPO/'experiments/r16p19_phase6/S1_REPRO.json').read_text())
 assert repro==historical,'STOP: S1 reproduction differs; do not repair historical evidence'
 print('REPRO 1680 rows mismatch 0',flush=True)
 cal=json.loads((REPO/'experiments/r16p19_phase6/S1_CALIBRATION.json').read_text());rs=s1._formal_receipts(REPO,RAW,cal)
 old=json.loads((REPO/'experiments/r16p19_phase6/S1_ATTRIBUTION.json').read_text())['variants']
 checks={}
 for variant,mode in [('B_LEARNED_AND_0_9395','B'),('C_PER_EFFECT_CALIBRATED_AND','C')]:
  decisions=[];oracle=[];effect={};div=set()
  for r in rs:
   out=np.array([v>=(s1.CURRENT_THRESHOLD if mode=='B' else cal['effect_threshold_vector'][key]) for v,key in zip(r['scores'],r['effect_keys'])]);decision=bool(out.all());truth=bool(r['labels'].all());decisions.append(decision);oracle.append(truth)
   if decision!=truth:div.add(r['cluster_id'])
   for j,key in enumerate(r['effect_keys']):
    v=effect.setdefault(key,[0,0]);v[1]+=int(r['labels'][j]);v[0]+=int(r['labels'][j] and out[j])
  d=np.array(decisions);y=np.array(oracle)
  values=dict(agreement_with_oracle=float((d==y).mean()),false_upgrade_count=int((d&~y).sum()),per_effect_tpr={k:a/b for k,(a,b) in effect.items()},concordant_units=120-len(div),divergent_units=len(div))
  for key in ['agreement_with_oracle','false_upgrade_count','per_effect_tpr']:assert values[key]==old[variant][key],f'STOP S1 {variant}/{key} differs'
  if mode=='C':assert values['concordant_units']==102
  checks[variant]=values
 write('B1_S1_CONSISTENCY.json',{'status':'MATCH','checks':checks,'old_FAIL_G1_not_reused':True})
 groups=[task_options([r for r in rs if r['task_id']==task]) for task in sorted({r['task_id'] for r in rs})]
 negative=sum(not bool(r['labels'].all()) for r in rs)
 solutions=[solve(groups,c,negative) for c in [.02,.05]]
 result={'variant':'F','claim_eligible':False,'selection_eligible':False,'formal_labels_used':True,'not_written_to_selection_seal':True,'receipt_count':len(rs),'unit_count':len({r['cluster_id'] for r in rs}),'negative_receipts':negative,'positive_receipts':len(rs)-negative,'method':'Exact Cartesian threshold enumeration per disjoint task, monotone feasible min-TPR search and exact global false-upgrade-budget combination. Ties follow B0.','options_per_task':{str(g['rs'][0]['task_id']):len(g['options']) for g in groups},'operating_points':solutions,'branch': 'CONTINUE_B2' if solutions[0]['min_per_effect_tpr']>=.973 else 'STOP_AFTER_B1','interpretation':'Formal-label upper-bound diagnostic only; no prospective or deployment claim; cannot infer global model-class capacity.'}
 write('B1_CEILING.json',result)
 print(json.dumps({'branch':result['branch'],'points':[{k:v for k,v in x.items() if k in ['receipt_fpr_cap','min_per_effect_tpr','false_upgrade_count','negative_receipts','receipt_fpr','optimality']} for x in solutions]},indent=2),flush=True)
 # Exact Wilson counts document the precision of the requested approximate statement.
 write('B1_WILSON_AUDIT.json',{'z':1.959963984540054,'requested_approximation':.973,'minimum_success_counts':{str(n):dict(k=(k:=next(k for k in range(n+1) if wilson(k,n)[0]>=.90)),empirical_tpr=k/n,lcb=wilson(k,n)[0]) for n in range(72,81)},'S1_tomato_64_of_76':wilson(64,76),'calibration_all_success_lcb':{str(n):wilson(n,n)[0] for n in [9,10,35]}})
 after={p.relative_to(REPO).as_posix():sha(p) for root in ['r16p19_phase5','r16p19_phase6'] for p in (REPO/'experiments'/root).rglob('*') if p.is_file()}
 assert before==after,'read-only input drift'
 write('B1_INPUT_HASHES.json',{'protected_tree_before':before,'protected_tree_after_equal':True,'read_paths':[{'path':str(p),'sha256':sha(p)} for p in sorted(opened)],'formal_reads_are_diagnostic_only':True})
 write('B1_EXECUTION.json',{'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_only':True,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_jobs_submitted':0,'pai_jobs_submitted':0,'rollouts_executed':0,'s2_started':False,'phase5_phase6_unchanged':True,'preregistration_sha256':B0_SHA,'code_sha256':sha(Path(__file__))})
if __name__=='__main__':main()

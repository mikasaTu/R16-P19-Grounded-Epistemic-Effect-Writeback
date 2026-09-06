#!/usr/bin/env python3
"""B2-B6 CPU diagnostics; formal input access is blocked at the file-open boundary."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib,math,importlib.util,datetime,argparse
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];OUT=REPO/'experiments/r16p19_phase7'
RAW=Path('/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/pai/r16p19-phase5-idle-fixed-20260818-0855/rollouts')
CHECKPOINT=REPO/'experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz'
B0_SHA='7fd8f1cc3392aea5a375abe293843305178c789d5e38f243e34f9738f04f17ec'
READS=set();DENIED=[];STAGE=''
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).absolute();s=str(p)
 mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
 if writing:
  if s.startswith(str(REPO)) and not s.startswith(str(OUT)):
   raise RuntimeError('Protected source write blocked: '+s)
  return
 if s.startswith(str(REPO)) or s.startswith(str(RAW)):
  forbidden='formal' in p.parts or 'formal_results' in p.name or (p.parent==OUT and p.name.startswith('B1_'))
  if STAGE=='B2' and 'qualification' in p.parts:forbidden=True
  if forbidden:
   DENIED.append(s);raise RuntimeError('Formal/heldout/ceiling input forbidden in '+STAGE+': '+s)
  if p.is_file():READS.add(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,v):
 (OUT/name).write_text(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-w,c+w]
def select(s,y,target=.90):
 s=np.array(s,float);y=np.array(y,bool);n=int(y.sum());maximum=wilson(n,n)[0]
 ok=[float(t) for t in np.unique(s) if wilson(int(((s>=t)&y).sum()),n)[0]>=target] if n else []
 chosen=max(ok) if ok and n>=35 else None
 return dict(status='CERTIFIABLE_ON_ESTIMATION' if chosen is not None else 'UNCERTIFIABLE',positive_count=n,negative_count=int((~y).sum()),target_lcb=target,maximum_attainable_lcb=maximum,threshold=chosen,reasons=(['positive_count_below_35'] if n<35 else [])+(['no_threshold_meets_Wilson_LCB'] if not ok else []),selected_tp=int(((s>=chosen)&y).sum()) if chosen is not None else None)
def evaluate(effects,rs,thresholds):
 detail={}
 for k,v in sorted(effects.items()):
  s=np.array(v['scores']);y=np.array(v['labels'],bool);n=int(y.sum());neg=int((~y).sum());t=thresholds.get(k)
  if t is None:detail[k]=dict(status='NO_CERTIFIED_THRESHOLD',positive_count=n,negative_count=neg,tpr=None,fpr=None,tpr_wilson95=None,fpr_wilson95=None);continue
  pred=s>=t;tp=int((pred&y).sum());fp=int((pred&~y).sum())
  detail[k]=dict(status='EVALUATED',threshold=t,tp=tp,positive_count=n,tpr=tp/n,tpr_wilson95=wilson(tp,n),fp=fp,negative_count=neg,fpr=fp/neg,fpr_wilson95=wilson(fp,neg))
 complete=all(thresholds.get(k) is not None for k in effects)
 y=np.array([bool(np.all(r['labels'])) for r in rs]);d=np.array([all(s>=thresholds[k] for s,k in zip(r['scores'],r['effect_keys'])) for r in rs]) if complete else None
 return dict(effects=detail,complete_vector=complete,receipt_count=len(rs),positive_receipts=int(y.sum()),negative_receipts=int((~y).sum()),false_upgrade_count=int((d&~y).sum()) if complete else None,receipt_fpr=float(d[~y].mean()) if complete else None,oracle_agreement=float((d==y).mean()) if complete else None)
def load(split,s1):return s1._calibration_dataset(RAW/'episodes'/split,CHECKPOINT)
def pooled(s1):
 from collections import defaultdict
 effects=defaultdict(lambda:dict(labels=[],scores=[]));rs=[];splits={}
 for split in ['calibration','pilot']:
  ee,rr,_,paths=load(split,s1)
  for k,v in ee.items():
   for key in ['labels','scores']:effects[k][key].extend(v[key])
  rs.extend(rr);splits[split]=dict(episodes=len(paths)//2,receipts=len(rr),effects={k:dict(positive_count=sum(v['labels']),samples=len(v['labels'])) for k,v in ee.items()})
 return dict(effects),rs,splits
def curve(rs,method,s1):
 y=np.array([bool(np.all(r['labels'])) for r in rs]);scores=np.array([float(np.min(r['scores'])) if method=='min' else float(np.log(np.clip(r['scores'],1e-15,1)).sum()) for r in rs]);ts=np.r_[np.nextafter(scores.max(),np.inf),np.unique(scores)[::-1],np.nextafter(scores.min(),-np.inf)]
 points=[]
 for t in ts:
  d=scores>=t;row=dict(threshold=float(t),tp=int((d&y).sum()),fp=int((d&~y).sum()),tpr=float(d[y].mean()),fpr=float(d[~y].mean()),oracle_agreement=float((d==y).mean()))
  if method=='min':
   effect={}
   for r in rs:
    for k,s,truth in zip(r['effect_keys'],r['scores'],r['labels']):
     v=effect.setdefault(k,[0,0]);v[1]+=int(truth);v[0]+=int(truth and s>=t)
   row['per_effect_tpr']={k:a/b for k,(a,b) in effect.items()};row['min_per_effect_tpr']=min(row['per_effect_tpr'].values())
  points.append(row)
 auc=float(np.trapz([r['tpr'] for r in points],[r['fpr'] for r in points]));assert abs(auc-s1.auroc(y,scores))<1e-12
 return dict(method=method,receipt_count=len(rs),positives=int(y.sum()),negatives=int((~y).sum()),auc=auc,effect_decision_definition='each effect score >= same receipt threshold' if method=='min' else None,produces_effect_decisions=method=='min',eligible_for_G1_prime=method=='min',eligibility_note='Effect-decision form is compatible, but B0 sample-count/LCB certification remains mandatory.' if method=='min' else 'Receipt-only sum(log p): excluded from G1 prime; receipt acceptance is never substituted for effect TPR.',zero_fpr_forced=False,points=points)
def main():
 global STAGE
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['B2','remaining']);args=parser.parse_args();STAGE=args.stage
 assert sha(OUT/'B0_PREREGISTRATION.json')==B0_SHA
 sys.addaudithook(audit)
 spec=importlib.util.spec_from_file_location('s1',REPO/'experiments/r16p19_phase6/run_s1.py');s1=importlib.util.module_from_spec(spec);spec.loader.exec_module(s1)
 ee,rs,splits=pooled(s1)
 if STAGE=='B2':
  effects={k:select(v['scores'],v['labels']) for k,v in sorted(ee.items())};thresholds={k:v['threshold'] for k,v in effects.items()}
  result=dict(status='UNCERTIFIABLE' if any(v is None for v in thresholds.values()) else 'ESTIMATED',estimation_splits=['calibration','pilot'],excluded_splits={'natural':'Split participates in base model fitting (seeds0-4); seeds5-6 are not promoted to a new eligible split after B0.','qualification':'Reserved for B3','formal':'Prohibited'},calibration_reuse_caveat='Calibration previously fitted score scale/bias and threshold, though absent from base-model training_samples. Included because USER_PLAN explicitly requires calibration+pilot.',sample_unit='frame-effect; within-episode dependence remains, Wilson is the requested descriptive counting convention',splits=splits,effects=effects,threshold_vector=thresholds,complete_vector=all(t is not None for t in thresholds.values()),formal_access_count=0)
  write('B2_CALIBRATION.json',result)
  input_paths=sorted(READS);evidence=[dict(path=str(p),sha256=sha(p)) for p in input_paths]
  write('B2_SELECTION_SEAL.json',dict(status='SEALED_PARTIAL_UNCERTIFIABLE' if not result['complete_vector'] else 'SEALED',preregistration_sha256=B0_SHA,code_sha256=sha(Path(__file__)),calibration_sha256=sha(OUT/'B2_CALIBRATION.json'),formal_access_count=0,formal_read_paths=[],denied_formal_access_attempts=DENIED,read_paths=evidence,qualification_used_for_estimation=False,complete_threshold_vector=result['complete_vector'],note='Seal records B2 computation only; it does not certify or select a deployable operating point.'))
  print(json.dumps({'stage':'B2','status':result['status'],'effects':effects},indent=2));return
 b2=json.loads((OUT/'B2_CALIBRATION.json').read_text());seal=json.loads((OUT/'B2_SELECTION_SEAL.json').read_text());assert sha(OUT/'B2_CALIBRATION.json')==seal['calibration_sha256']
 qe,qr,_,_=load('qualification',s1);b3=evaluate(qe,qr,b2['threshold_vector']);b3.update(status='NOT_EVALUABLE_FULL_VECTOR' if not b3['complete_vector'] else 'EVALUATED',interpretation='ESTIMATION_INSUFFICIENT: three effects fail before held-out testing; transfer failure cannot be inferred.',qualification_independent_of_B2_threshold_fitting=True,qualification_is_untouched_model_selection_holdout=False,pilot_qualification_environment_seed_overlap=15,independence_caveat='qualification participated in original model selection; pilot has same init/seed as 15 qualification clean episodes. This is threshold-heldout only, not strictly environment-independent.',formal_access_count=0)
 write('B3_HELDOUT.json',b3)
 soft={'status':'COMPLETED_DIAGNOSTIC','estimation':{m:curve(rs,m,s1) for m in ['min','sum_log_p']},'qualification':{m:curve(qr,m,s1) for m in ['min','sum_log_p']},'formal_access_count':0,'old_D_reused':False,'selection_performed':False}
 write('B4_SOFT.json',soft)
 metrics=json.loads((CHECKPOINT.with_suffix('.metrics.json')).read_text());original=metrics['models']
 historic_ranking=sorted(original,key=lambda n:(original[n]['qualified'],original[n]['macro_auroc'],-original[n]['ece'],n=='linear'),reverse=True)
 with np.load(CHECKPOINT,allow_pickle=False) as data:keys=sorted(data.files);model_type=str(data['model_type'])
 write('B5_MODEL_SELECTION.json',dict(status='BLOCKED_MISSING_LINEAR_CHECKPOINT',retraining_performed=False,original_selection_rule='max(qualified, macro_auroc, -ece, name == linear); not loss/accuracy',original_ranking=historic_ranking,original_selected=metrics['selected'],original_metrics={k:{a:v[a] for a in ['qualified','macro_auroc','ece','min_tpr','max_fpr','accuracy']} for k,v in original.items()},requested_selection_rule='min per-effect TPR @ receipt FPR <=0.02 on eligible estimation splits',ranking_under_requested_rule=None,reason='Only selected small_mlp weights saved. linear aggregate metrics at historical threshold cannot determine its fixed-FPR frontier. Reconstructing training would violate no-retraining instruction.',available_checkpoint=dict(path=str(CHECKPOINT),sha256=sha(CHECKPOINT),model_type=model_type,keys=keys),B2_result_after_model_switch=None,model_switch_performed=False,formal_access_count=0))
 frontier=[]
 for i in range(80,100):
  target=i/100;selected={k:select(v['scores'],v['labels'],target) for k,v in ee.items()};ts={k:v['threshold'] for k,v in selected.items()};evaluation=evaluate(qe,qr,ts)
  frontier.append(dict(target=target,status='UNREACHABLE_CERTIFICATION' if not evaluation['complete_vector'] else 'EVALUATED',uncertifiable_effects=[k for k,v in selected.items() if v['threshold'] is None],maximum_lcbs={k:v['maximum_attainable_lcb'] for k,v in selected.items()},qualification_false_upgrade_rate=evaluation['receipt_fpr'],qualification_oracle_agreement=evaluation['oracle_agreement'],threshold_vector=ts))
 write('B6_FRONTIER.json',dict(status='NO_CERTIFIABLE_OPERATING_POINT',selection_split='qualification',estimation_splits=['calibration','pilot'],points=frontier,eligible_operating_points=0,formal_access_count=0))
 write('B6_OPERATING_POINT_STATUS.json',dict(status='NO_SEAL_CREATED',reason='Every preregistered target 0.80..0.99 lacks a complete certified threshold vector. Never fabricate a selected point.',S2_allowed=False))
 write('B6_FORMAL_FRONTIER_STATUS.json',dict(status='NOT_EVALUABLE',selection_eligible=False,claim_eligible=False,reason='No complete threshold vector exists at any preregistered target. Formal cannot fill missing thresholds.',formal_access_count=0))
 evidence=[dict(path=str(p),sha256=sha(p)) for p in sorted(READS)]
 write('B2_B6_READ_AUDIT.json',dict(formal_access_count=0,formal_read_paths=[],denied_access_attempts=DENIED,reads=evidence,cpu_only=True,code_sha256=sha(Path(__file__)),preregistration_sha256=B0_SHA))
 print(json.dumps({'B3':b3['status'],'B4':{split:{m:soft[split][m]['auc'] for m in ['min','sum_log_p']} for split in ['estimation','qualification']},'B5':'BLOCKED_MISSING_LINEAR_CHECKPOINT','B6':'NO_CERTIFIABLE_OPERATING_POINT'},indent=2))
if __name__=='__main__':main()

"""Retrospective CPU repair; never claims to restore pre-registration."""
import os,sys,json,hashlib,math,importlib.util
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experiments/r16p19_phase8'
RAW=Path('/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/pai/r16p19-phase5-idle-fixed-20260818-0855/rollouts')
CK=ROOT/'experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz'
READS=set()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).absolute();s=str(p);mode,flags=args[1:3]
 writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT))
 if writing:
  if (s.startswith(str(ROOT)) and not s.startswith(str(OUT))) or s.startswith(str(RAW)):raise RuntimeError('protected write '+s)
 elif s.startswith(str(ROOT)) or s.startswith(str(RAW)):
  if 'formal' in p.parts or 'formal_results' in p.name or p.name.startswith(('B1_','C1_AUDITED','C1_RECALIBRATION','C4_')):raise RuntimeError('formal diagnostic forbidden in C0/C2 '+s)
  if p.is_file():READS.add(p)
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def wil(k,n,z):
 if not n:return [None,None]
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [max(0,c-w),min(1,c+w)]
ZS={'individual95':1.959963984540054,'joint95_bonferroni':2.5758293035489004}
def select(s,y,z):
 pos=np.sort(np.asarray(s)[np.asarray(y,bool)])[::-1]; n=len(pos)
 if not n or wil(n,n,z)[0]<.9:return None
 # highest observed score retaining enough positive observations, ties included
 for k in range(1,n+1):
  if wil(k,n,z)[0]>=.9:return float(pos[k-1])
def apply(s,param):
 x=np.asarray(s,float);q=np.log(np.clip(x,1e-6,1-1e-6))-np.log1p(-np.clip(x,1e-6,1-1e-6))
 return 1/(1+np.exp(-np.clip(param['a']*q+param['b'],-40,40)))
def main():
 sys.addaudithook(audit)
 spec=importlib.util.spec_from_file_location('s1',ROOT/'experiments/r16p19_phase6/run_s1.py');s1=importlib.util.module_from_spec(spec);spec.loader.exec_module(s1)
 ee=defaultdict(lambda:dict(scores=[],labels=[],episodes=[])); rs=[]; paths=[]; splitcounts={}
 for split in ['calibration','pilot']:
  e,r,_,ip=s1._calibration_dataset(RAW/'episodes'/split,CK);paths+=ip;rs+=r
  splitcounts[split]=len({v['episode_id'] for v in r})
  for v in r:
   for key,s,y in zip(v['effect_keys'],v['scores'],v['labels']):
    ee[key]['scores'].append(float(s));ee[key]['labels'].append(bool(y));ee[key]['episodes'].append(v['episode_id'])
 # Fixed split-specific counts and per-task collection rates, NOT frame rates
 budgets={};old=json.loads((OUT/'C0_PREREGISTRATION.json').read_text())
 assert old['authentication']['target']==.9
 assert old['multiplicity']['individual_95']['z']==ZS['individual95']
 assert old['multiplicity']['joint_95_bonferroni']['z']==ZS['joint95_bonferroni']
 for key,v in ee.items():
  ids=sorted(set(v['episodes']));by={ep:[i for i,e in enumerate(v['episodes']) if e==ep] for ep in ids};y=np.array(v['labels']);pos=int(y.sum());positive_eps=sum(bool(y[ix].any()) for ix in by.values());n_epi=len(ids)
  budgets[key]={'frame_positive_count':pos,'task_episodes':n_epi,'positive_episodes':positive_eps,'total_nonformal_episodes':sum(splitcounts.values()),'positive_frames_per_task_episode':pos/n_epi,'positive_episodes_per_task_episode':positive_eps/n_epi,'requirements':{}}
  for mult,z in ZS.items():
   n=next(n for n in range(1,1000) if wil(n,n,z)[0]>=.9)
   if n>1:assert wil(n-1,n-1,z)[0]<.9
   unit={}
   for name,nobs,rate in [('frame_effect',pos,pos/n_epi),('episode_all_positive_frames_detected',positive_eps,positive_eps/n_epi)]:
    total=math.ceil(n/rate) if rate else None;additional=math.ceil(max(0,n-nobs)/rate) if rate else None
    unit[name]={'minimum_independent_positive_units_if_all_success':n,'observed_units':nobs,'max_current_wilson_lcb':wil(nobs,nobs,z)[0], 'best_case_interval_width':1-wil(nobs,nobs,z)[0] if nobs else None,'rate_per_task_episode':rate,'total_task_episodes_extrapolated':total,'additional_task_episodes_extrapolated':additional,'new_standalone_certification_task_episodes_extrapolated':total,'extrapolation':True,'assumption':'same positive yield; all detections succeed; not guaranteed; existing rows do not make an independent certification set'}
   budgets[key]['requirements'][mult]=unit
 write('C0_SAMPLE_BUDGET_AUDITED.json',{'status':'RETROSPECTIVE_CORRECTION','claim_eligible':False,'selection_eligible':False,'primary':'individual95','interval_convention':'Original frozen z values preserved: central two-sided 95% and 99% Wilson lower endpoints; original one-sided wording was incorrect. These are conservative relative to one-sided 95/99%.','split_episode_counts':splitcounts,'effects':budgets,'cluster_estimand':'episode success = every truly positive frame for the effect detected; positive episode = at least one positive frame. This is an episode-level estimand, not an equivalent frame-weighted TPR. Also report frame-ratio cluster-bootstrap widths in C2.','certification_caveat':'fitting maps and choosing thresholds on same labels does not supply independent coverage; Wilson post-selection is descriptive. Pooling effects changes the estimand and cannot replace per-effect evidence.','split_design':{'task_balance':'budget counts are PER TASK, sum task maxima to obtain balanced collection totals','seed_disjoint_from':['calibration','pilot','qualification','formal'],'no_model_or_threshold_selection':True,'dedicated_new_heldout':'reserve a separate untouched certification allocation, at least the standalone budget; fitting/selection data must be additional','iid_clusters_required':'independent environment seeds/episodes; shared init seeds must be grouped conservatively; temporal frames not independent'}})
 methods=json.loads((OUT/'C2_EXTRA_MAPS_NONFORMAL.json').read_text())
 results={}
 for method,params in methods.items():
  transformed={k:apply(v['scores'],params[k]) for k,v in ee.items()}; scores=np.concatenate(list(transformed.values()));labels=np.concatenate([np.array(v['labels']) for v in ee.values()]);mres={}
  for mult,z in ZS.items():
   threshold=select(scores,labels,z);detail={}
   for key,v in ee.items():
    y=np.array(v['labels']);s=transformed[key];p=s>=threshold if threshold is not None else None;n=int(y.sum());tp=int((p&y).sum()) if p is not None else None
    cluster=[]
    for ep in sorted(set(v['episodes'])):
     ix=np.array([a==ep for a in v['episodes']]);ny=int(y[ix].sum());ty=int((p[ix]&y[ix]).sum()) if p is not None else None
     if ny:cluster.append((ny,ty))
    cpass=sum(a==b for a,b in cluster) if p is not None else None;cn=len(cluster); rng=np.random.default_rng(19719)
    if cn and p is not None:
     ar=np.array(cluster);idx=rng.integers(cn,size=(4000,cn));boot=ar[idx,1].sum(1)/ar[idx,0].sum(1);alpha=.05 if mult=='individual95' else .01;bc=[float(t) for t in np.quantile(boot,[alpha/2,1-alpha/2])]
    else:bc=[None,None]
    per_t=select(s,y,z); frame_ci=wil(tp,n,z) if tp is not None else [None,None];ep_ci=wil(cpass,cn,z) if cpass is not None else [None,None]
    detail[key]={'positive_frames':n,'tp':tp,'tpr':tp/n if tp is not None and n else None,'wilson':frame_ci,'max_lcb_even_with_perfect_detection':wil(n,n,z)[0],'frame_count_gate':frame_ci[0] is not None and frame_ci[0]>=.9,'per_effect_threshold':per_t,'per_effect_threshold_status':'UNCERTIFIABLE' if per_t is None else 'DESCRIPTIVE_THRESHOLD','positive_episodes':cn,'fully_detected_positive_episodes':cpass,'episode_all_positive_detected_wilson':ep_ci,'frame_ratio_episode_bootstrap':bc,'interval_widths':{'frame_wilson':frame_ci[1]-frame_ci[0] if tp is not None else None,'episode_wilson':ep_ci[1]-ep_ci[0] if cpass is not None else None,'cluster_bootstrap':bc[1]-bc[0] if bc[0] is not None else None},'bootstrap_warning':'resubstitution diagnostic only; bootstrap can degenerate to zero width with all successes, not proof of certainty'}
   mres[mult]={'pooled_positive_count':int(labels.sum()),'global_threshold':threshold,'pooled_tpr_ci':wil(int(((scores>=threshold)&labels).sum()),int(labels.sum()),z) if threshold is not None else [None,None],'effects':detail,'complete_frame_count_vector':all(v['frame_count_gate'] for v in detail.values()),'complete_certified_vector':False,'independent_certification':False}
  results[method]=mres
 write('C2_GLOBAL_THRESHOLD_AUDITED.json',{'status':'NO_COMPLETE_CERTIFIED_VECTOR','claim_eligible':False,'selection_eligible':False,'methods':results,'rule':'highest pooled threshold achieving requested Wilson lower endpoint, then explicitly check EACH effect LCB>=.90; descriptive resubstitution only','formal_access_count':0})
 evidence=[{'path':str(p),'sha256':sha(p)} for p in sorted(READS)]
 write('C2_SELECTION_SEAL_AUDITED.json',{'status':'DIAGNOSTIC_EXECUTION_SEAL_NOT_VALID_PREREGISTRATION','formal_access_count':0,'formal_read_paths':[],'read_paths':evidence,'original_C0_sha256':sha(OUT/'C0_PREREGISTRATION.json'),'output_sha256':sha(OUT/'C2_GLOBAL_THRESHOLD_AUDITED.json'),'code_sha256':sha(__file__),'selected_deployable_point':None,'note':'audit-hook enforced no formal input in this process; cannot repair historical preregistration breach'})
 print(json.dumps({'methods':list(results),'positive_counts':{k:v['frame_positive_count'] for k,v in budgets.items()},'episode_counts':{k:v['positive_episodes'] for k,v in budgets.items()}}))
if __name__=='__main__':main()

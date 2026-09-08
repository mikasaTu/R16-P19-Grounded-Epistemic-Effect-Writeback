import os,json,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';sys.dont_write_bytecode=True
import numpy as np
import audit_calibration as a
from pathlib import Path
O=Path(__file__).resolve().parent
maps=json.loads((O/'C2_EXTRA_MAPS_NONFORMAL.json').read_text());c3=json.loads((O/'C3_AUDITED.json').read_text())
def cmp(s,y):
 p=s[y][:,None];n=s[~y][None,:];return (p>n).astype(float)+.5*(p==n)
result={}
for split in ['estimation','qualification']:
 if split=='estimation':_,rs,_=a.collect_estimation()
 else:_,rs,_,_=a.collect_split('qualification')
 y=np.array([all(r['labels']) for r in rs],bool);raw=np.array([min(r['scores']) for r in rs]);old=cmp(raw,y)
 out={}
 for family in ['temperature','platt']:
  transformed=[np.array([a.latent_from_raw(np.array([s]),maps[family][k])[0] for s,k in zip(r['scores'],r['effect_keys'])]) for r in rs]
  scores=np.array([min(x) for x in transformed]);new=cmp(scores,y);delta=new-old
  changes=sum(np.argmin(r['scores'])!=np.argmin(s) for r,s in zip(rs,transformed))
  d=float(delta.mean());target=c3['splits'][split][family]['auc_rank']-c3['splits'][split]['raw']['auc_rank'];assert abs(d-target)<1e-12
  out[family]={'positive_negative_pairs':int(delta.size),'improved_pair_count':int((delta>0).sum()),'worsened_pair_count':int((delta<0).sum()),'unchanged_pair_count':int((delta==0).sum()),'improvement_credit':float(delta[delta>0].sum()),'worsening_credit':float(-delta[delta<0].sum()),'delta_auc_exact':d,'C3_auc_delta':target,'bottleneck_effect_changed_receipts':int(changes),'total_receipts':len(rs),'interpretation':'exact pair-order accounting, including half-credit ties; per-effect monotone maps change cross-effect bottlenecks and receipt ordering, not effect AUC'}
 result[split]=out
(O/'MECHANISM_PAIR_AUDIT.json').write_text(json.dumps({'claim_eligible':False,'selection_eligible':False,'formal_access_count':0,'splits':result},indent=2)+'\n');print(json.dumps(result,indent=2))

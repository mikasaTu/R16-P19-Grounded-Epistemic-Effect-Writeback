from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json,hashlib,sys,time
from pathlib import Path
import numpy as np
def rankdata(x):
 order=np.argsort(x,kind="stable");v=x[order];starts=np.r_[0,1+np.flatnonzero(v[1:]!=v[:-1])];ends=np.r_[starts[1:],len(x)];r=np.repeat((starts+ends+1)/2,ends-starts);out=np.empty(len(x));out[order]=r;return out
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from r16p19.phase5_verifier_data import _feature_rows
from r16p19.phase5_verifier_model import Standardizer,MLPVerifier,fit_calibrator,apply_calibrator
OUT=Path(__file__).resolve().parent
ROOT=Path('/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/pai/r16p19-phase5-idle-fixed-20260818-0855/rollouts')
KEYS=[(0,0),(0,1),(5,0),(9,0),(9,1)]
SETS={'A_full':list(range(35)),'B_no_visual':list(range(18,35)),'C_no_time':[i for i in range(35) if i!=27],'D_no_contact':[i for i in range(35) if i!=26],'E_shortcut_only':list(range(27,35)),'F_visual_only':list(range(18))+list(range(31,35))}
def write(name,d):
 def clean(x):
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [clean(v) for v in x]
  if isinstance(x,np.ndarray):return clean(x.tolist())
  if isinstance(x,(np.integer,)):return int(x)
  if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
  return x
 (OUT/name).write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spatial(d,task):
 n=len(d['proprio']); parts=[]
 for camera in ['base_rgb_32','wrist_rgb_32']:
  im=d[camera].astype(np.float32)/255.; bins=im.reshape(n,4,8,4,8,3)
  parts.extend([bins.mean((2,4)).reshape(n,-1),bins.std((2,4)).reshape(n,-1)])
 taskoh=np.zeros((n,3),np.float32);taskoh[:,{0:0,5:1,9:2}[task]]=1
 shared=np.concatenate(parts+[d['proprio'].astype(np.float32),taskoh],1); result=[]
 for e in range(d['predicate_values'].shape[1]):
  one=np.zeros((n,4),np.float32);one[:,e]=1;result.append(np.concatenate([shared,one],1))
 return np.concatenate(result)
def load(split):
 x=[];y=[];g=[];ep=[];sx=[];manifest=[]
 for p in sorted((ROOT/'episodes'/split).glob('*.npz')):
  meta=json.loads(p.with_suffix('.json').read_text())
  if split=='natural' and meta['policy_seed'] not in range(5):continue
  task=int(meta['task_id'])
  with np.load(p,allow_pickle=False) as d:
   a,b,e=_feature_rows(d,task);c=spatial(d,task)
  x.append(a);y.append(b);sx.append(c);g.extend([(task,int(i)) for i in e]);ep.extend([p.stem]*len(b));manifest.append({'npz':str(p),'sha256':sha(p),'metadata':str(p.with_suffix('.json'))})
 return {'x':np.concatenate(x),'y':np.concatenate(y).astype(int),'spatial':np.concatenate(sx),'group':np.array(g),'episode':np.array(ep),'manifest':manifest}
def auc(y,s):
 pos=y==1;n1=pos.sum();n0=len(y)-n1
 if not n1 or not n0:return None
 return float((rankdata(s)[pos].sum()-n1*(n1+1)/2)/(n1*n0))
def threshold_at_fpr(y,s,target=.05):
 neg=np.sort(s[y==0])[::-1]
 if not len(neg):return None
 allowed=int(np.floor(target*len(neg)+1e-12))
 # Strictly exceed next disallowed negative, then >= implements tie-safe FPR.
 return float(np.nextafter(neg[allowed],np.inf)) if allowed<len(neg) else float('-inf')
def point(y,s,t):
 return {'tpr':float((s[y==1]>=t).mean()) if np.any(y==1) else None,'fpr':float((s[y==0]>=t).mean()) if np.any(y==0) else None}
def basic(y,s):
 t=threshold_at_fpr(y,s);r=point(y,s,t) if t is not None else {'tpr':None,'fpr':None}
 return auc(y,s),r['tpr'],r['fpr'],t
def cluster_ci(y,s,episode,threshold,nboot=2000):
 # Episode resampling with all frame rows retained; weighted rank calculation avoids copying repeated episodes.
 names,inv=np.unique(episode,return_inverse=True);n=len(names);rng=np.random.default_rng(20260915)
 order=np.argsort(s,kind='stable'); scores=s[order];yy=y[order];ii=inv[order];start=np.r_[0,1+np.flatnonzero(scores[1:]!=scores[:-1])]
 vals=[]
 for _ in range(nboot):
  counts=np.bincount(rng.integers(n,size=n),minlength=n);w=counts[ii];pw=w*(yy==1);nw=w*(yy==0);P=pw.sum();N=nw.sum()
  if P==0 or N==0:continue
  pbin=np.add.reduceat(pw,start);nbin=np.add.reduceat(nw,start)
  av=float(np.sum(pbin*(np.cumsum(nbin)-nbin/2))/(P*N))
  allowed=.05*N;before=np.cumsum(nbin[::-1])-nbin[::-1];accept=(before+nbin[::-1])<=allowed+1e-12
  # Tied score groups are accepted atomically in descending order.
  tp=float(pbin[::-1][accept].sum()/P)
  pred=scores>=threshold
  vals.append([av,tp,float(pw[pred].sum()/P),float(nw[pred].sum()/N)])
 if not vals:return {'auroc':None,'tpr_at_fpr_0.05':None,'calibration_point_tpr':None,'calibration_point_fpr':None,'valid_replicates':0}
 q=np.quantile(vals,[.025,.975],axis=0).T
 return dict(zip(['auroc','tpr_at_fpr_0.05','calibration_point_tpr','calibration_point_fpr'],q.tolist()),valid_replicates=len(vals))
def evaluate(data,scores,cal,calscores,ci=True):
 rows=[]
 for task,effect in KEYS:
  m=(data['group']==[task,effect]).all(1);mc=(cal['group']==[task,effect]).all(1);y=data['y'][m];s=scores[m];t=threshold_at_fpr(cal['y'][mc],calscores[mc]);a,tp,fp,roc_t=basic(y,s)
  row={'task_id':task,'effect_index':effect,'positive_rows':int((y==1).sum()),'negative_rows':int((y==0).sum()),'episode_count':len(np.unique(data['episode'][m])),'auroc':a,'tpr_at_fpr_0.05':tp,'achieved_roc_fpr':fp,'calibration_threshold':t,'calibration_point':point(y,s,t),'roc_threshold_selection_eligible':False}
  if ci:row['episode_cluster_ci95']=cluster_ci(y,s,data['episode'][m],t)
  rows.append(row)
 return {'macro_auc':float(np.mean([r['auroc'] for r in rows if r['auroc'] is not None])),'per_effect':rows,'interval_method':'episode bootstrap 2000 resamples seed 20260915; rows within episode jointly weighted','roc_metrics_descriptive':True}
def fit(name,tr,cal,q,cols,spatial_mode=False):
 field='spatial' if spatial_mode else 'x';cols=list(range(tr[field].shape[1])) if cols is None else cols
 st=Standardizer().fit(tr[field][:,cols]);model=MLPVerifier().fit(st.transform(tr[field][:,cols]),tr['y']);raw=model.predict(st.transform(cal[field][:,cols]));scale,bias=fit_calibrator(cal['y'],raw)
 cs=apply_calibrator(raw,scale,bias);qs=apply_calibrator(model.predict(st.transform(q[field][:,cols])),scale,bias)
 path=OUT/'models'/f'{name}.npz';path.parent.mkdir(exist_ok=True)
 np.savez_compressed(path,feature_columns=cols,mean=st.mean,std=st.std,w1=model.w1,b1=model.b1,w2=model.w2,b2=model.b2,calibration_scale=scale,calibration_bias=bias)
 np.savez_compressed(OUT/f'{name}_predictions.npz',cal_scores=cs,qualification_scores=qs)
 result=evaluate(q,qs,cal,cs);result.update(feature_columns=cols,training_rows=len(tr['y']),calibration_rows=len(cal['y']),qualification_rows=len(q['y']),model_path=str(path.relative_to(OUT)),model_sha256=sha(path),seed=29,hyperparameters={'hidden':32,'steps':1600,'learning_rate':.01,'l2':1e-4,'class_weight':'negative_count/positive_count','calibrator_steps':1200,'calibrator_learning_rate':.03},feature_field=field)
 print(name,result['macro_auc'],flush=True);return result

def main():
 tr=load('natural');cal=load('calibration');q=load('qualification');assert len(tr['y'])==13138;assert len(cal['y'])==2711;assert len(q['y'])==2615
 write('NONFORMAL_INPUT_MANIFEST.json',{'natural':tr['manifest'],'calibration':cal['manifest'],'qualification':q['manifest']})
 out={'evaluation_split':'qualification','formal_used_for_selection':False,'sets':{}}
 for name,cols in SETS.items():out['sets'][name]=fit(name,tr,cal,q,cols)
 write('F1_ABLATION.json',out)
 trans=fit('T_spatial',tr,cal,q,None,True);trans.update(feature_definition='4x4 spatial RGB means/stds (192), proprio (8), task (3), effect (4); excludes time/contact',raw_resolution=[32,32],upper_bound_interpretation='best tested CPU diagnostic attainable performance; not a mathematical upper bound or real-robot guarantee',claim_eligible=False)
 write('F4_TRANSFERABLE_METRICS.json',trans)
 write('NONFORMAL_EXECUTION.json',{'complete':True,'formal_read_count':0,'cpu_only':True,'training_rows':len(tr['y']),'phase5_baseline':.9916669890228063,'observed_baseline':out['sets']['A_full']['macro_auc']})
if __name__=='__main__':main()

import io,json,hashlib,sys
from pathlib import Path
import numpy as np
from run_phase11 import OUT,ROOT,SETS,KEYS,sha,write,_feature_rows,spatial,evaluate,load,apply_calibrator

def main():
 assert (OUT/'NONFORMAL_EXECUTION.json').exists()
 models={p.name:sha(p) for p in sorted((OUT/'models').glob('*.npz'))}
 freeze={'models':models,'selection_inputs':['natural policy_seed 0-4','calibration'],'qualification_role':'fixed ablation comparison; no architecture/hyperparameter selection','formal_selection_inputs':0,'source_formal_probe_before_freeze':True,'protocol_valid':False,'deviation':'one formal NPZ was inspected for shape before cache; phase9 mixed cache formal preview was also read. No selection used those values. This cannot be described as compliant single-access formal protocol.'}
 write('F1_MODEL_FREEZE.json',freeze)
 cache=OUT/'F1_FORMAL_CACHE.npz'
 if cache.exists():raise RuntimeError('Refusing to reread formal source: cache already exists')
 xs=[];ys=[];gs=[];eps=[];sp=[];manifest=[]
 for p in sorted((ROOT/'episodes'/'formal').glob('*.npz')):
  raw=p.read_bytes();meta=json.loads(p.with_suffix('.json').read_text());task=int(meta['task_id'])
  with np.load(io.BytesIO(raw),allow_pickle=False) as d:x,y,e=_feature_rows(d,task);s=spatial(d,task)
  xs.append(x);ys.append(y);sp.append(s);gs.extend([(task,int(i)) for i in e]);eps.extend([p.stem]*len(y));manifest.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'read_count_this_loader':1})
 np.savez_compressed(cache,x=np.concatenate(xs),y=np.concatenate(ys).astype(int),group=np.asarray(gs),episode=np.asarray(eps),spatial=np.concatenate(sp))
 del xs,ys,gs,eps,sp
 write('F1_FORMAL_READ_AUDIT.json',{'source_files':manifest,'cache_sha256':sha(cache),'cache_path':cache.name,'logical_source_loads_this_loader':1,'prior_probe_violation':True,'protocol_valid':False,'selection_eligible':False,'claim_eligible':False})
 with np.load(cache,allow_pickle=False) as z:data={k:z[k] for k in z.files}
 cal=load('calibration');result={}
 for name in list(SETS)+['T_spatial']:
  with np.load(OUT/'models'/f'{name}.npz') as m:
   field='spatial' if name=='T_spatial' else 'x';cols=m['feature_columns'];x=(data[field][:,cols]-m['mean'])/m['std'];p=1/(1+np.exp(-np.clip(np.maximum(x@m['w1']+m['b1'],0)@m['w2']+m['b2'],-30,30)));scores=apply_calibrator(p,float(m['calibration_scale']),float(m['calibration_bias']))
  with np.load(OUT/f'{name}_predictions.npz') as z:cs=z['cal_scores']
  result[name]=evaluate(data,scores,cal,cs)
  result[name].update(claim_eligible=False,selection_eligible=False,protocol_valid=False)
  print(name,result[name]['macro_auc'],flush=True)
 write('F1_FORMAL_COMPARISON.json',{'claim_eligible':False,'selection_eligible':False,'protocol_valid':False,'reason':'early formal schema and mixed-cache preview access disclosed; final metrics descriptive only','models':result,'source':'F1_FORMAL_CACHE.npz'})
 for name,h in models.items():assert sha(OUT/'models'/name)==h
if __name__=='__main__':main()

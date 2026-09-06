"""Complete the available-model portion of B5 without formal inputs or retraining."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import importlib.util,json,math,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nf',P/'run_nonformal.py');nf=importlib.util.module_from_spec(spec);spec.loader.exec_module(nf)
nf.STAGE='B5';sys.addaudithook(nf.audit)
spec=importlib.util.spec_from_file_location('s1',nf.REPO/'experiments/r16p19_phase6/run_s1.py');s1=importlib.util.module_from_spec(spec);spec.loader.exec_module(s1)
e,rs,splits=nf.pooled(s1)
pos={k:np.sort(np.asarray(v['scores'])[np.asarray(v['labels'],bool)])[::-1] for k,v in e.items()}
targets=sorted({i/len(s) for s in pos.values() for i in range(1,len(s)+1)},reverse=True)
y=np.array([bool(np.all(r['labels'])) for r in rs]);negative=int((~y).sum());cap=.02;budget=math.floor(cap*negative+1e-12)
for target in targets:
 thresholds={k:float(s[max(i for i in range(len(s)) if (i+1)/len(s)>=target and all((j+1)/len(s)<target for j in range(i)))]) for k,s in pos.items()}
 # Above index is the smallest k-1 with k/n>=target; exact float comparisons match discrete TPR objective.
 d=np.array([all(s>=thresholds[k] for s,k in zip(r['scores'],r['effect_keys'])) for r in rs]);fp=int((d&~y).sum())
 if fp<=budget:break
metrics=nf.evaluate(e,rs,thresholds)
p=P/'B5_MODEL_SELECTION.json';result=json.loads(p.read_text())
result['available_model_fixed_FPR_result']={'model':'small_mlp','estimation_splits':['calibration','pilot'],'receipt_FPR_cap':cap,'allowed_false_upgrades':budget,'minimum_per_effect_TPR':min(v['tpr'] for v in metrics['effects'].values()),'thresholds_for_model_ranking_only':thresholds,'evaluation':metrics,'method':'Exact monotone TPR feasibility: for each possible TPR target use the highest threshold covering required positives; this minimizes receipt false upgrades at that target.','not_B2_certified_thresholds':True,'B2_Wilson_result_unchanged':True}
result['same_criterion_ranking']={'small_mlp':result['available_model_fixed_FPR_result']['minimum_per_effect_TPR'],'linear':None,'ranking_determinable':False}
nf.write('B5_MODEL_SELECTION.json',result)
nf.write('B5_READ_AUDIT.json',{'formal_access_count':0,'denied_access_attempts':nf.DENIED,'read_paths':[{'path':str(p),'sha256':nf.sha(p)} for p in sorted(nf.READS)],'code_sha256':nf.sha(Path(__file__)),'no_retraining':True})
print(json.dumps(result['same_criterion_ranking']));print('false_upgrades',fp,'negative',negative)

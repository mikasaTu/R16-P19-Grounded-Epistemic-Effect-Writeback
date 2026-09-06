"""Independent numerical and access-boundary checks; no rollout imports or execution."""
import importlib.util,itertools,math,os
from pathlib import Path
import numpy as np
import pytest
P=Path(__file__).resolve().parent

def module(name):
 spec=importlib.util.spec_from_file_location(name,P/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
b1=module('run_b1');nf=module('run_nonformal')

def test_wilson_against_scipy_inversion():
 # Wilson interval must invert the score test at both endpoints.
 for n in [9,14,15,35,72,76,80,612]:
  for k in [0,1,n//2,n-1,n]:
   lo,hi=nf.wilson(k,n)
   for q in [lo,hi]:
    if 1e-12<q<1-1e-12:
     assert abs((k-n*q)**2/(n*q*(1-q))-1.959963984540054**2)<1e-9

def test_small_samples_cannot_be_certified_even_if_perfect():
 for n in [9,10,14,15,34]:
  r=nf.select([1.]*n,[True]*n,.9)
  assert r['threshold'] is None and r['status']=='UNCERTIFIABLE'
 assert nf.select([1.]*35,[True]*35,.9)['threshold']==1.

def test_highest_threshold_and_inclusive_ties():
 s=np.r_[np.repeat(.8,35),.9,.1];y=np.r_[np.ones(36,bool),False]
 result=nf.select(s,y,.9)
 assert result['threshold']==.8
 assert nf.wilson(36,36)[0]>=.9
 assert nf.wilson(1,36)[0]<.9

def test_exact_ceiling_matches_unfactored_global_bruteforce():
 rng=np.random.default_rng(708)
 for trial in range(25):
  groups=[];receipts=[]
  for task,effects in [(0,2),(5,1)]:
   s=rng.choice([.1,.4,.8],(5,effects));y=rng.integers(0,2,(5,effects)).astype(bool);y[0]=True;y[1]=False
   rs=[dict(task_id=task,effect_keys=[f't{task}e{j}' for j in range(effects)],scores=a,labels=b,receipt_id=f'{task}-{i}') for i,(a,b) in enumerate(zip(s,y))]
   receipts+=rs;groups.append(b1.task_options(rs))
  neg=sum(not r['labels'].all() for r in receipts)
  for cap in [0.,.2,.5,1.]:
   budget=math.floor(cap*neg+1e-12)
   combos=[c for c in itertools.product(*(g['options'] for g in groups)) if sum(o['fp'] for o in c)<=budget]
   key=lambda c:(min(o['min'] for o in c),-sum(o['fp'] for o in c),sum(o['sum'] for o in c),tuple(t for o in c for t in o['ts']))
   expected=max(combos,key=key);got=b1.solve(groups,cap,neg)
   assert got['min_per_effect_tpr']==key(expected)[0]
   assert got['false_upgrade_count']==-key(expected)[1]
   assert tuple(v['threshold'] for v in got['effects'].values())==key(expected)[3]

def test_formal_and_ceiling_access_are_blocked_by_guard():
 nf.STAGE='B2'
 for path in [nf.RAW/'episodes/formal/one.npz',nf.RAW/'episodes/qualification/one.npz',nf.OUT/'B1_CEILING.json',nf.REPO/'experiments/r16p19_phase5/artifacts/results/learned_verifier_formal_results.jsonl']:
  with pytest.raises(RuntimeError):nf.audit('open',(str(path),'r',0))

def test_sum_log_is_receipt_only_and_tied_auc_is_half():
 class A:
  @staticmethod
  def auroc(y,s):
   pos=s[y];neg=s[~y]
   return np.mean([float(a>b)+.5*float(a==b) for a in pos for b in neg])
 rs=[dict(scores=np.array([.5,.5]),labels=np.array(y),effect_keys=['a','b']) for y in [[True,True],[False,True],[True,False],[True,True]]]
 c=nf.curve(rs,'sum_log_p',A)
 assert c['auc']==.5 and not c['produces_effect_decisions'] and not c['eligible_for_G1_prime']
 assert all('per_effect_tpr' not in p for p in c['points'])

def test_partial_vector_does_not_invent_receipt_metrics():
 r=nf.evaluate({'a':dict(scores=[.8,.2],labels=[True,False])},[dict(scores=[.8],labels=[True],effect_keys=['a']),dict(scores=[.2],labels=[False],effect_keys=['a'])],{'a':None})
 assert not r['complete_vector'] and r['false_upgrade_count'] is None and r['oracle_agreement'] is None

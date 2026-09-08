import importlib.util,math
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('audit_c0_c2.py'); spec=importlib.util.spec_from_file_location('audit_c0',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_best_case_sample_boundaries():
 for z,n in [(m.ZS['individual95'],35),(m.ZS['joint95_bonferroni'],60)]:
  assert m.wil(n,n,z)[0]>=.9
  assert m.wil(n-1,n-1,z)[0]<.9
  assert abs(m.wil(14,14,z)[0]-14/(14+z*z))<1e-12
def test_sparse_effect_cannot_be_certified_by_perfect_scores():
 for z in m.ZS.values():
  assert m.select(np.ones(14),np.ones(14,bool),z) is None
def test_pooled_certification_does_not_imply_group_certification():
 z=m.ZS['individual95'];t=m.select(np.ones(795),np.ones(795,bool),z)
 assert t==1 and m.wil(795,795,z)[0]>.9
 assert m.wil(14,14,z)[0]<.9
def test_selector_is_highest_feasible_and_tie_safe():
 s=np.linspace(0,1,80);y=np.ones(80,bool);z=m.ZS['individual95'];t=m.select(s,y,z)
 assert m.wil(int((s>=t).sum()),80,z)[0]>=.9
 assert all(m.wil(int((s>=u).sum()),80,z)[0]<.9 for u in s if u>t)
 assert m.select(np.ones(35),np.ones(35,bool),z)==1
def test_missing_denominators_remain_null():
 assert m.wil(0,0,m.ZS['individual95'])==[None,None]
 assert m.select([],[],m.ZS['individual95']) is None
def test_false_upgrade_upper_bound_has_nonzero_uncertainty():
 assert m.wil(0,14,m.ZS['individual95'])[1]>.2
def test_budget_correct_episode_rate():
 assert math.ceil(35/(14/15))==38
 assert math.ceil(60/(14/15))==65
 assert math.ceil(35/(15/15))==35
def test_formal_inputs_are_rejected_before_open():
 import pytest
 with pytest.raises(RuntimeError,match='forbidden'):
  m.audit('open',(str(m.RAW/'episodes/formal/x.npz'),'r',0))
 with pytest.raises(RuntimeError,match='forbidden'):
  m.audit('open',(str(m.ROOT/'experiments/r16p19_phase7/B1_CEILING.json'),'r',0))
def test_prior_phase_writes_are_rejected():
 import pytest,os
 with pytest.raises(RuntimeError,match='protected write'):
  m.audit('open',(str(m.ROOT/'experiments/r16p19_phase7/anything'),'w',os.O_WRONLY))
def test_original_preregistration_bytes_preserved():
 assert m.sha(m.OUT/'C0_PREREGISTRATION.json')=='de415c807142cecfba8469b3f3a589bc704784a378d1b54251e03f50bdeb6c6a'

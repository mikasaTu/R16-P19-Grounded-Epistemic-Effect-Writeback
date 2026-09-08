import json
from pathlib import Path
import audit_c4 as m
O=Path(__file__).resolve().parent

def test_persistent_ledger_semantics_are_distinct():
 r=m.run_property_tests();assert r['passed']
 d=r['persistent_ledger_distinction'];a=d['legacy']['records'];b=d['independent']['records']
 assert d['same_event_stream']
 assert a['effect_a']['upgrade_fact_state']=='UNKNOWN'
 assert b['effect_a']['upgrade_fact_state']=='REALIZED'
 assert a['effect_a']['proof_ids']==b['effect_a']['proof_ids']
 assert b['effect_b']['upgrade_fact_state']=='UNKNOWN'

def test_same_denominator_and_actual_unit_counts():
 d=json.loads((O/'C4_AUDITED.json').read_text());p=d['pooled_metrics'];b=p['legacy_receipt_and'];a=p['per_effect_independent']
 assert b['false_upgrade_denominator']==a['false_upgrade_denominator']==510
 assert b['false_upgrade_count']==0 and a['false_upgrade_count']==2
 assert b['concordant_units']==19 and a['concordant_units']==18
 assert a['false_upgrade_wilson_95_upper']>b['false_upgrade_wilson_95_upper']
 assert d['stored_nonformal_oracle_replay']['mismatch_count']==0

def test_task_outcome_gain_is_separate_from_effect_gain():
 d=json.loads((O/'C4_AUDITED.json').read_text())['pilot_task_success_gain']
 assert d['scope']=='pilot_only_nonformal'
 assert d['nonpilot_task_success_gain'] is None
 for v in d['variants'].values():
  assert v['faulted_task_cell_denominator']==90
  g=v['task_success_gain_interval']
  assert abs(g['lower_adversarial']+0.4666666666666667)<1e-12
  assert abs(g['upper_inherited_oracle']-1/3)<1e-12

def test_c4_formal_path_is_blocked():
 import pytest
 with pytest.raises(RuntimeError,match='formal read forbidden'):
  m.access_audit('open',(str(m.RAW_ROOT/'episodes/formal/x.npz'),'r',0))

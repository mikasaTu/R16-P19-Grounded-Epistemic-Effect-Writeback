import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from phase9_ledger_cache import *

def events(condition,unit='test'):
 return {k:event_sequence(condition,unit+'|'+k,0) for k in ['a','b']}

def test_receipt_and_veto_preserves_witness_proofs():
 a=PersistentPerEffectUpgradeLedger(['a','b'],'test','receipt_and')
 r=a.process(events('C0_CLEAN'),np.array([True,False]))
 assert r['receipt_gate'] is False
 assert all(x['witness_effect_fact_verified'] for x in r['records'].values())
 assert all(x['upgrade_fact_state']=='UNKNOWN' and not x['upgrade_accepted'] for x in r['records'].values())
 assert all(x['proof_ids'] and x['proof_validity'] for x in r['records'].values())
 assert a.snapshot()['records']==r['records']

def test_all_pass_clean_advances_actual_ledger():
 a=PersistentPerEffectUpgradeLedger(['a','b'],'test','receipt_and')
 r=a.process(events('C0_CLEAN'),np.array([True,True]))
 assert r['receipt_gate'] is True
 assert all(x['upgrade_fact_state']=='REALIZED' for x in r['records'].values())

def test_detector_cannot_override_invalid_late_witness():
 a=PersistentPerEffectUpgradeLedger(['a','b'],'test','receipt_and')
 r=a.process(events('A3_CONTRADICTION_LATE_WITNESS'),np.array([True,True]))
 assert r['receipt_gate'] is False
 assert not any(x['witness_effect_fact_verified'] for x in r['records'].values())

def test_duplicate_condition_evidence_rejected():
 with pytest.raises(AssertionError):index_unique([{'u':1,'c':2},{'u':1,'c':2}],('u','c'))

def test_completed_real_ledger_result():
 d=json.loads((ROOT/'D5_REPLAY.json').read_text())
 assert d['unit_count']==120 and d['concordant_unit_count']==6 and d['divergent_unit_count']==114
 assert len(d['ledger_records'])==840 and d['effect_stream_count']==1400 and d['event_count']==8000
 assert d['reference_kernel_mismatch']==0
 assert d['oracle_increment_count']==683-456==227
 assert d['captured_oracle_increment_count']==4
 assert set((ROOT/'D5_S2_UNITS.txt').read_text().splitlines())==set(d['divergent_unit_ids'])
 assert d['gain_interval']['lower_adversarial']<=d['gain_interval']['neutral_assumption']<=d['gain_interval']['upper_inherit_oracle']
 assert d['gain_interval']['neutral_is_assumption'] is True
 for r in d['ledger_records']:
  for e in r['records'].values():assert e['event_sha256']==canonical_sha(e['events'])

import sys,subprocess,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from phase9_cached_eval import *

def test_gate_exchangeability_known_failures():
 assert gate_exchangeability(2.01) is False
 assert gate_exchangeability(None) is False
 assert gate_exchangeability(float('nan')) is False
 assert gate_exchangeability(2.0) is True

def test_gate_effect_fpr_known_failures():
 assert gate_effect_fpr([.006]*5,.005,True) is False
 assert gate_effect_fpr([0.0]*5,.005,False) is False
 assert gate_effect_fpr([],.005,True) is False
 assert gate_effect_fpr([None]*5,.005,True) is False
 assert gate_effect_fpr([-.01]*5,.005,True) is False
 assert gate_effect_fpr([.001]*5,.005,True) is True

def test_gate_receipt_fpr_known_failures():
 assert gate_receipt_fpr(.051,.001) is False
 assert gate_receipt_fpr(.01,.06) is False
 assert gate_receipt_fpr(None,.001) is False
 assert gate_receipt_fpr(.01,.001) is True

def test_gate_ledger_known_failures():
 assert gate_ledger(101,120) is False
 assert gate_ledger(0,0) is False
 assert gate_ledger(121,120) is False
 assert gate_ledger(102,120) is True

def test_cluster_negative_observation_not_or_truth():
 rows=[{'cluster_id':'one','effect_keys':['effect'],'labels':[True],'scores':[.9]}, {'cluster_id':'one','effect_keys':['effect'],'labels':[False],'scores':[.8]}]
 m=cluster_metrics(rows,'effect',.5)
 assert m['negative_episode_count']==1 and m['false_positive_episode_count']==1
 assert m['positive_episode_count']==1 and m['true_positive_episode_count']==1
 assert m['fpr']==1.0

def test_strict_threshold_and_missing_data():
 assert not effect_accept(.5,.5)
 assert not effect_accept(1.0,None)
 assert cp_upper(0,0) is None
 assert abs(cp_upper(0,15)-.18103627252208465)<1e-12

def test_real_open_rejected_by_selector_hook():
 code="import sys; from phase9_nonformal import selector_read_hook; sys.addaudithook(selector_read_hook); open('positive_labels.json')"
 p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,text=True,capture_output=True)
 assert p.returncode!=0 and 'RuntimeError' in p.stderr and 'positive' in p.stderr

def test_original_source_open_rejected():
 code="import sys; from phase9_cached_eval import source_guard; sys.addaudithook(source_guard); open('/fake/rollouts/episodes/formal/file.json')"
 p=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,text=True,capture_output=True)
 assert p.returncode!=0 and 'Original evidence read forbidden' in p.stderr

def test_recovered_cache_and_metrics_consistency():
 cache=json.loads((ROOT/'RECOVERED_LOADED_EVIDENCE.json').read_text())
 assert len(cache['formal'])==840 and len(cache['oracle_rows'])==4200
 assert len({(r['cluster_id'],r['condition']) for r in cache['formal']})==840
 assert sum(not all(r['labels']) for r in cache['formal'])==612
 result=json.loads((ROOT/'D4_FORMAL.json').read_text())
 assert result['oracle_receipt_agreement_count']==612
 assert result['receipt_false_upgrades']['negative_episode_count']==120
 for row in result['per_effect'].values():
  assert row['episode_cluster']['negative_episode_count']==40
  assert row['receipt']['tpr']==0.0
 assert result['formal_read_once'] is False
 assert result['protocol_valid'] is False

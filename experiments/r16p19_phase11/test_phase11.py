import unittest,json,math
from pathlib import Path
import numpy as np
from phase11_finalize import gate_f1_baseline,gate_f1_conclusion,gate_f2_complete,gate_f4_evidence
from run_phase11 import threshold_at_fpr,point,auc,cluster_ci,OUT,sha
class Gates(unittest.TestCase):
 def test_baseline_rejects_wrong_finite_value(self):
  self.assertFalse(gate_f1_baseline(.991));self.assertFalse(gate_f1_baseline(float('nan')));self.assertFalse(gate_f1_baseline(None));self.assertTrue(gate_f1_baseline(.9916669890228063))
 def test_shortcut_rejects_inconsistent_verdict(self):
  self.assertFalse(gate_f1_conclusion(.99,.5,.5,.5,'SHORTCUT_CONFIRMED'));self.assertFalse(gate_f1_conclusion(.99,.5,None,.5,'NOT_CONFIRMED'));self.assertTrue(gate_f1_conclusion(.99,.5,.5,.5,'NOT_CONFIRMED'))
 def test_coverage_rejects_missing_or_duplicate_dimension(self):
  rows=[{'index':i,'classification':'onboard_observable'} for i in range(35)];self.assertTrue(gate_f2_complete(rows));self.assertFalse(gate_f2_complete(rows[:-1]));rows[-1]['index']=0;self.assertFalse(gate_f2_complete(rows))
 def test_f4_rejects_empty_or_out_of_range_metrics(self):
  self.assertFalse(gate_f4_evidence({'macro_auc':.99}));d=json.loads((OUT/'F4_TRANSFERABLE_SPEC.json').read_text());self.assertTrue(gate_f4_evidence(d));d['per_effect'][0]['auroc']=1.2;self.assertFalse(gate_f4_evidence(d))
class Metrics(unittest.TestCase):
 def test_fpr_threshold_direction(self):
  y=np.array([0]*20+[1]*3);s=np.r_[np.arange(20)/20,[.94,.98,.99]];t=threshold_at_fpr(y,s,.05);p=point(y,s,t);self.assertLessEqual(p['fpr'],.05);self.assertEqual(p['tpr'],1.0);self.assertLess(t,1)
 def test_ties_do_not_silently_exceed_fpr(self):
  y=np.array([0]*20+[1]);s=np.r_[np.ones(20)*.5,.6];t=threshold_at_fpr(y,s,.05);self.assertEqual(point(y,s,t)['fpr'],0);self.assertEqual(point(y,s,t)['tpr'],1)
 def test_rank_ties_auc(self):
  self.assertEqual(auc(np.array([0,1]),np.array([.4,.4])),.5)
 def test_cluster_bootstrap_perfect_and_all_tied(self):
  y=np.array([0,1,0,1]);ep=np.array(['a','a','b','b']);s=np.array([0.,1.,0.,1.]);ci=cluster_ci(y,s,ep,.5,100);self.assertEqual(ci['auroc'],[1,1]);self.assertEqual(ci['tpr_at_fpr_0.05'],[1,1]);ci2=cluster_ci(y,np.ones(4),ep,1.1,100);self.assertEqual(ci2['auroc'],[.5,.5]);self.assertEqual(ci2['tpr_at_fpr_0.05'],[0,0])
class Artifacts(unittest.TestCase):
 def test_exact_splits_and_five_groups(self):
  d=json.loads((OUT/'F1_ABLATION.json').read_text());
  for v in d['sets'].values():
   self.assertEqual(v['training_rows'],13138);self.assertEqual(v['calibration_rows'],2711);self.assertEqual(v['qualification_rows'],2615);self.assertEqual(len(v['per_effect']),5);self.assertEqual(sha(OUT/v['model_path']),v['model_sha256'])
   for r in v['per_effect']:self.assertLessEqual(r['achieved_roc_fpr'],.05);self.assertEqual(r['episode_cluster_ci95']['valid_replicates'],2000)
 def test_full_checkpoint_weights_and_freeze(self):
  freeze=json.loads((OUT/'F1_MODEL_FREEZE.json').read_text())
  for name,h in freeze['models'].items():
   p=OUT/'models'/name;self.assertEqual(sha(p),h)
   with np.load(p) as z:
    for k in ['w1','b1','w2','b2','mean','std','calibration_scale','calibration_bias','feature_columns']:self.assertIn(k,z.files)
    self.assertEqual(z['w1'].shape[1],32)
 def test_formal_is_explicitly_invalid_and_not_selection(self):
  d=json.loads((OUT/'F1_FORMAL_COMPARISON.json').read_text());self.assertFalse(d['protocol_valid']);self.assertFalse(d['selection_eligible']);self.assertFalse(d['claim_eligible'])
if __name__=='__main__':unittest.main(verbosity=2)

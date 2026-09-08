from pathlib import Path
import json,hashlib,subprocess
O=Path(__file__).resolve().parent;R=O.parents[1]
J=lambda n:json.loads((O/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 checks={}
 checks['original_C0_unchanged']=sha(O/'C0_PREREGISTRATION.json')=='de415c807142cecfba8469b3f3a589bc704784a378d1b54251e03f50bdeb6c6a'
 d=J('C2_SELECTION_SEAL_AUDITED.json')
 checks['C2_formal_access_zero']=d['formal_access_count']==0 and not d['formal_read_paths']
 checks['C2_input_hashes_match']=all(sha(Path(x['path']))==x['sha256'] for x in d['read_paths'])
 checks['C2_output_hash_match']=sha(O/'C2_GLOBAL_THRESHOLD_AUDITED.json')==d['output_sha256']
 checks['C2_code_hash_match']=sha(O/'audit_c0_c2.py')==d['code_sha256']
 c1=J('C1_AUDITED.json');c2=J('C2_GLOBAL_THRESHOLD_AUDITED.json');c3=J('C3_AUDITED.json');c4=J('C4_AUDITED.json')
 checks['C1_ineligible']=c1['claim_eligible'] is False and c1['selection_eligible'] is False
 checks['C1_actual_ceiling_reproduced']=all(v[m]['min_per_effect_tpr']==1 and v[m]['false_upgrade_count']==10 for v in c1['operating_points'].values() for m in ['raw_recomputed','temperature','platt'])
 checks['C2_no_false_complete']=all(not v['complete_certified_vector'] and not v['complete_frame_count_vector'] for m in c2['methods'].values() for v in m.values())
 checks['C3_map_binding']=c3['maps_sha256']==sha(O/'C2_EXTRA_MAPS_NONFORMAL.json')
 checks['C4_ineligible']=c4['claim_eligible'] is False and c4['selection_eligible'] is False
 p=c4['pooled_metrics'];b=p['legacy_receipt_and'];a=p['per_effect_independent']
 checks['C4_same_denominator']=b['false_upgrade_denominator']==a['false_upgrade_denominator']>0
 checks['C4_count_integrity']=all(v['concordant_units']+v['divergent_units']==v['unit_count'] and len(v['divergent_unit_ids'])==v['divergent_units'] for v in p.values())
 checks['C4_persistent_records']=len(c4['per_effect_ledger_records'])==700
 checks['C4_properties_pass']=c4['property_tests']['passed'] is True
 checks['Phase5_exact_repro']=J('PHASE5_REPRO_AUDITED.json')['mismatch_rows']==0
 checks['Phase5_6_7_unchanged']=not subprocess.check_output(['git','diff','--name-only','585814e','--','experiments/r16p19_phase5','experiments/r16p19_phase6','experiments/r16p19_phase7'],cwd=R,text=True).strip()
 checks['decision_honest']=J('C_DECISION.json')['status']=='NEEDS_DATA' and J('C_DECISION.json')['fully_preregistered_plan_completed'] is False
 for canonical,audited in [('C0_SAMPLE_BUDGET.json','C0_SAMPLE_BUDGET_AUDITED.json'),('C1_RECALIBRATION.json','C1_AUDITED.json'),('C2_GLOBAL_THRESHOLD.json','C2_GLOBAL_THRESHOLD_AUDITED.json'),('C3_MIN_AGG.json','C3_AUDITED.json'),('C4_PER_EFFECT_UPGRADE.json','C4_AUDITED.json')]:
  checks['canonical_'+canonical]=sha(O/canonical)==sha(O/audited)
 result={'checks':checks,'all_pass':all(checks.values()),'scope':'release integrity and executed diagnostics; not restoration of valid preregistration'}
 (O/'RELEASE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));assert result['all_pass']
 # Manifest excludes itself; otherwise stable self-hash is impossible.
 files=sorted(p for p in O.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
 (O/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(R))+'\n' for p in files))
 subprocess.run(['sha256sum','--quiet','-c',str(O/'SHA256SUMS')],cwd=R,check=True)
 print('SHA256SUMS verified',len(files),'artifacts; self excluded')
if __name__=='__main__':main()

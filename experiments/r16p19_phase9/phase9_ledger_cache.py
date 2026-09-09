"""Actual receipt-AND ledger replay from recovered evidence. No raw source reads."""
from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
from typing import Any
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).parent
REPO=ROOT.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT))
from r16p19.phase5_arm_kernel import _make,event_sequence,evaluate_arm
from phase9_cached_eval import effect_accept,META,write,source_guard

def canonical_sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

# Class copied verbatim by AST from read-only phase8 audit_c4.py, without its module-wide hooks.
class PersistentPerEffectUpgradeLedger:
    """Persist detector-gated upgrade state around the frozen core ledger.

    The inner arm consumes the exact same Phase-5 events in both modes.  Its
    fact state, proof ids, validity, attribution, and decision are retained as
    witness provenance.  The outer persistent upgrade record is the C4
    semantic boundary: receipt-AND gates every effect together, whereas the
    independent mode gates each effect separately.
    """

    def __init__(self, effect_keys: list[str], unit_id: str, mode: str) -> None:
        if mode not in {"receipt_and", "per_effect_independent"}:
            raise ValueError(mode)
        self.effect_keys = list(effect_keys)
        self.unit_id = unit_id
        self.mode = mode
        self.arms = {
            key: _make("M3_ASCEL_CORE", f"{unit_id}|{key}")
            for key in self.effect_keys
        }
        self.records: dict[str, dict[str, Any]] = {}
        self.receipt_gate = False

    def process(
        self,
        events_by_effect: dict[str, list[Any]],
        detector_pass: np.ndarray,
    ) -> dict[str, Any]:
        detector_pass = np.asarray(detector_pass, dtype=bool)
        if len(detector_pass) != len(self.effect_keys):
            raise ValueError("detector/effect length mismatch")
        for key in self.effect_keys:
            if key not in events_by_effect:
                raise ValueError(f"missing event stream for {key}")
            for event in events_by_effect[key]:
                self.arms[key].process(event)
        witness_verified = np.asarray(
            [self.arms[key].effect_fact_verified("TASK_GOAL") for key in self.effect_keys],
            dtype=bool,
        )
        self.receipt_gate = bool(np.all(detector_pass) and np.all(witness_verified))
        records: dict[str, dict[str, Any]] = {}
        for index, key in enumerate(self.effect_keys):
            arm = self.arms[key]
            fact = arm.ledger.facts["TASK_GOAL"]
            witness_state = fact.fact_state.value
            witness_decision = arm.decide("TASK_GOAL").value
            proof_ids = list(fact.realization_proof_ids)
            proof_validity = {
                proof_id: arm.ledger.proofs[proof_id].validity_status.value
                for proof_id in proof_ids
            }
            if self.mode == "receipt_and":
                accepted = self.receipt_gate
                rejection_reason = "receipt_and_veto" if not accepted else None
            else:
                accepted = bool(detector_pass[index] and witness_verified[index])
                rejection_reason = (
                    "effect_detector_miss"
                    if not detector_pass[index]
                    else "ledger_witness_not_verified"
                ) if not accepted else None
            if bool(witness_verified[index]) and accepted:
                upgrade_state = "REALIZED"
                upgrade_decision = "ADVANCE_TO_NEXT_SUBTASK"
            elif bool(witness_verified[index]) and not accepted:
                # The witness and proof remain present above; only the C4
                # upgrade record is held back by the detector gate.
                upgrade_state = "UNKNOWN"
                upgrade_decision = "REOBSERVE"
            else:
                upgrade_state = witness_state
                upgrade_decision = witness_decision
            records[key] = {
                "detector_pass": bool(detector_pass[index]),
                "witness_fact_state": witness_state,
                "witness_effect_fact_verified": bool(witness_verified[index]),
                "witness_attempt_attributed_success": bool(
                    arm.attempt_attributed_success("TASK_GOAL")
                ),
                "witness_decision": witness_decision,
                "proof_ids": proof_ids,
                "proof_validity": proof_validity,
                "upgrade_fact_state": upgrade_state,
                "upgrade_decision": upgrade_decision,
                "upgrade_accepted": bool(accepted),
                "upgrade_rejection_reason": rejection_reason,
            }
        self.records = records
        return {
            "mode": self.mode,
            "receipt_gate": self.receipt_gate,
            "records": records,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "unit_id": self.unit_id,
            "receipt_gate": self.receipt_gate,
            "records": self.records,
        }

CONDITIONS={'C0_CLEAN','A1_NOOP_RETRY_STALE','A2_CROSS_ATTEMPT_MIX','A3_CONTRADICTION_LATE_WITNESS','A4_POST_REALIZATION_REVERSAL','A5_EXTERNAL_REALIZATION','V1_SINGLE_VIEW_FALSE_POSITIVE'}
def index_unique(rows,fields):
 out={}
 for r in rows:
  k=tuple(r[f] for f in fields)
  if k in out:raise AssertionError('Duplicate evidence key: '+str(k))
  out[k]=r
 return out

def replay_receipt(row,meta,thresholds):
 unit=f"learned-t{meta['task_id']}-i{meta['formal_init']}-s{meta['policy_seed']}-{row['condition']}-M3_ASCEL_CORE"
 events={k:event_sequence(row['condition'],unit+'|'+k,int(meta['policy_seed'])) for k in row['effect_keys']}
 detector=np.array([effect_accept(s,thresholds[k]) for k,s in zip(row['effect_keys'],row['scores'])])
 arm=PersistentPerEffectUpgradeLedger(row['effect_keys'],unit,'receipt_and')
 state=arm.process(events,detector)
 reference_mismatch=0
 for k in row['effect_keys']:
  reference=evaluate_arm('M3_ASCEL_CORE',row['condition'],unit+'|'+k,int(meta['policy_seed']))
  rec=state['records'][k]
  reference_mismatch+=int(rec['witness_effect_fact_verified']!=bool(reference['verified']) or rec['witness_attempt_attributed_success']!=bool(reference['credited']) or rec['witness_decision']!=reference['final_decision'])
  rec['stream_id']=unit+'|'+k;rec['events']=[e.to_dict() for e in events[k]];rec['event_sha256']=canonical_sha(rec['events'])
 # Oracle comparison retains the same receipt-AND definition as S1.
 oracle_gate=all(row['labels'])
 oracle_witness_gate=oracle_gate and all(v['witness_effect_fact_verified'] for v in state['records'].values())
 assert oracle_gate==oracle_witness_gate,'Oracle truth and frozen witness incompatible'
 return {'cluster_id':row['cluster_id'],'condition':row['condition'],'task_id':row['task_id'],'semantic_unit_id':unit,'receipt_gate':state['receipt_gate'],'oracle_receipt_gate':oracle_gate,'reference_kernel_mismatch':reference_mismatch,'records':state['records']}

def run():
 sys.addaudithook(source_guard)
 cache=json.loads((ROOT/'RECOVERED_LOADED_EVIDENCE.json').read_text());seal=json.loads((ROOT/'D3_OPERATING_POINT_SEAL.json').read_text())
 raw=index_unique(cache['formal'],('cluster_id','condition'))
 oracle=index_unique(cache['oracle_rows'],('cluster_id','condition','arm'))
 groups=defaultdict(set)
 for unit,condition in raw:groups[unit].add(condition)
 assert len(groups)==120 and all(v==CONDITIONS for v in groups.values())
 records=[]
 for k,row in sorted(raw.items()):records.append(replay_receipt(row,oracle[k+('M3_ASCEL_CORE',)],seal['selected_threshold_vector']))
 assert sum(r['reference_kernel_mismatch'] for r in records)==0
 divergent=sorted({r['cluster_id'] for r in records if r['receipt_gate']!=r['oracle_receipt_gate']});concordant=sorted(set(groups)-set(divergent))
 total_increment=0;captured_increment=0;baseline_total=0;core_total=0;cells=0;divergent_baseline_success=0
 for (unit,condition),r in raw.items():
  if condition=='C0_CLEAN':continue
  base=int(oracle[(unit,condition,'M0_TYPED_MATCHED')]['task_success']);core=int(oracle[(unit,condition,'M3_ASCEL_CORE')]['task_success'])
  baseline_total+=base;core_total+=core;cells+=1;total_increment+=core-base
  if unit in concordant:captured_increment+=core-base
  else:divergent_baseline_success+=base
 assert cells==720 and total_increment==227
 unknown_cells=len(divergent)*6
 out={'status':'COMPLETED_ACTUAL_RECEIPT_AND_LEDGER_REPLAY_DIAGNOSTIC','unit_count':len(groups),'concordant_unit_count':len(concordant),'divergent_unit_count':len(divergent),'concordant_unit_ids':concordant,'divergent_unit_ids':divergent,'ledger_concordance':len(concordant)/len(groups),'receipt_concordance':sum(r['receipt_gate']==r['oracle_receipt_gate'] for r in records)/len(records),'condition_count_per_unit':7,'effect_stream_count':sum(len(r['records']) for r in records),'event_count':sum(len(v['events']) for r in records for v in r['records'].values()),'reference_kernel_mismatch':0,'event_lineage':'Deterministic event_sequence from protected Phase5 kernel using original semantic unit identifiers and cached policy seeds; materialized frozen program replay, not newly collected/serialized physical rollout events.','receipt_and_semantics':'All detector decisions AND all ledger witnesses verified; retains proof validity, attribution and persistent upgrade record. C4 independent upgrade is not evaluated.','ledger_records':records,'outcome_denominator':cells,'oracle_baseline_success_count':baseline_total,'oracle_core_success_count':core_total,'oracle_increment_count':total_increment,'captured_oracle_increment_count':captured_increment,'captured_oracle_increment_fraction':captured_increment/total_increment,'gain_interval':{'lower_adversarial':(captured_increment-unknown_cells)/cells,'upper_inherit_oracle':total_increment/cells,'neutral_assumption':captured_increment/cells,'neutral_is_assumption':True,'lower_rule':'Each divergent unit faulted cell contributes -1; concordant cells inherit oracle difference.','upper_rule':'Assumes all divergent cells inherit oracle outcomes; not a model-free upper confidence bound.','neutral_rule':'Assumes zero incremental effect in every divergent cell; not a point estimate.','known_baseline_sharper_lower_descriptive':(captured_increment-divergent_baseline_success)/cells},'phase6_C_comparison':{'source':'User-provided read-only historical facts; C not rerun','C_concordant_units':102,'C_total_units':120,'current_minus_C_concordant_units':len(concordant)-102,'C_captured_oracle_increment':197,'C_total_oracle_increment':227,'current_captured_oracle_increment':captured_increment,'current_minus_C_captured_increment':captured_increment-197,'metric_note':'197/227 is captured incremental task successes, not faulted-cell concordance.'},'s2_reexecution_unit_ids':divergent,'s2_started':False,'cache_sha256':hashlib.sha256((ROOT/'RECOVERED_LOADED_EVIDENCE.json').read_bytes()).hexdigest()}
 write('D5_REPLAY.json',out)
 (ROOT/'D5_S2_UNITS.txt').write_text('\n'.join(divergent)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k in ['concordant_unit_count','divergent_unit_count','effect_stream_count','event_count','reference_kernel_mismatch','gain_interval','captured_oracle_increment_count']},indent=2))
if __name__=='__main__':run()

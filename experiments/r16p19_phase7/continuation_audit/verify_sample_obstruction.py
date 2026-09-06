#!/usr/bin/env python3
"""Count unused natural episodes and prove finite-sample obstruction. No inference."""
import json,math,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parents[1]
RAW=Path('/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/pai/r16p19-phase5-idle-fixed-20260818-0855/rollouts/episodes/natural')
counts=defaultdict(lambda:[0,0]);inputs=[];episodes=0
for p in sorted(RAW.glob('*.npz')):
 m=json.loads(p.with_suffix('.json').read_text())
 if int(m['policy_seed']) not in [5,6]:continue
 episodes+=1
 with np.load(p,allow_pickle=False) as d:
  for i,label in enumerate(d['predicate_labels'].tolist()):
   k=f'task{m["task_id"]}:effect{i}:{label}';counts[k][0]+=int(d['predicate_values'][:,i].sum());counts[k][1]+=len(d['predicate_values'])
 inputs += [{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in [p,p.with_suffix('.json')]]
b=json.loads((P/'B2_CALIBRATION.json').read_text());z=1.959963984540054
required=math.ceil(z*z*.9/(1-.9));assert required==35
items={}
for k,v in b['effects'].items():
 n=v['positive_count'];hyp=n+counts[k][0]
 items[k]={'frozen_B2_positive_count':n,'maximum_B2_LCB_even_all_correct':n/(n+z*z),'unused_natural_positive_count':counts[k][0],'hypothetical_combined_count_NOT_AUTHORIZED':hyp,'hypothetical_max_LCB':hyp/(hyp+z*z),'minimum_additional_positives_even_if_all_correct':max(0,required-hyp)}
result={'purpose':'read-only input-availability and mathematical-obstruction audit, not revised selection','B0_unchanged':True,'natural_added_to_B2':False,'unused_natural_episodes':episodes,'proof':'For k=n the Wilson lower bound is n/(n+z^2). This is the maximum for fixed n. LCB>=0.90 requires n>=ceil(9*z^2)=35 even with zero false negatives.','minimum_perfect_successes':required,'effects':items,'source_hashes':inputs,'formal_accesses':0,'new_model_inference':False,'new_training':False}
(P/'continuation_audit/SAMPLE_OBSTRUCTION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in items.items() if v['frozen_B2_positive_count']<35},indent=2))

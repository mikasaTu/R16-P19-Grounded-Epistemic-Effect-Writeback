import os,sys,importlib.util,json
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';sys.dont_write_bytecode=True
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'experiments/r16p19_phase8'
spec=importlib.util.spec_from_file_location('s1',R/'experiments/r16p19_phase6/run_s1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1:3];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT))
 if writing and str(p).startswith(str(R)) and not str(p).startswith(str(O)):raise RuntimeError('protected write '+str(p))
sys.addaudithook(audit)
r=m.reproduce(R,Path('/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/pai/r16p19-phase5-idle-fixed-20260818-0855/rollouts'),O)
r.update(claim_eligible=False,selection_eligible=False,purpose='frozen Phase-5 exact reproduction only; no step9 formal workpoint evaluation')
(O/'S1_REPRO.json').write_text(json.dumps(r,indent=2)+'\n')
(O/'PHASE5_REPRO_AUDITED.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

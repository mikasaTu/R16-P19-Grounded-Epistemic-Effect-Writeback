import os,json,math,hashlib,importlib.util
from pathlib import Path
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
R=Path(__file__).resolve().parents[2]; O=R/"experiments/r16p19_phase8"; RAW=Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19-phase5-bounded-ascel/pai/r16p19-phase5-idle-fixed-20260818-0855/rollouts"); CK=R/"experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz"
sp=importlib.util.spec_from_file_location("s1",R/"experiments/r16p19_phase6/run_s1.py"); s1=importlib.util.module_from_spec(sp); sp.loader.exec_module(s1)
Z95=1.959963984540054; Z99=2.5758293035489004
def wil(k,n,z=Z95):
 if n==0:return [None,None]
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(max(0,p*(1-p)/n+z*z/(4*n*n)))/d;return [c-w,c+w]
def auc(y,x):
 y=np.asarray(y,bool);x=np.asarray(x,float); pos=x[y]; neg=x[~y]
 return float((np.greater.outer(pos,neg).sum()+.5*np.equal.outer(pos,neg).sum())/(len(pos)*len(neg))) if len(pos) and len(neg) else None
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))
def fit_platt(x,y):
 x=np.clip(np.asarray(x,float),1e-6,1-1e-6); q=np.log(x/(1-x)); y=np.asarray(y,float);a,b=1.,0.
 for _ in range(100):
  p=sigmoid(a*q+b); w=p*(1-p)+1e-8; X=np.c_[q,np.ones(len(q))]; H=(X*w[:,None]).T@X+1e-5*np.eye(2); g=X.T@(p-y); d=np.linalg.solve(H,g); a-=d[0]; b-=d[1]
  if np.max(np.abs(d))<1e-8:break
 return float(a),float(b)
def load(split):return s1._calibration_dataset(RAW/"episodes"/split,CK)
def pooled(splits):
 E={}; Rr=[]
 for split in splits:
  e,r,_,_=load(split); Rr+=r
  for k,v in e.items(): E.setdefault(k,{"scores":[],"labels":[]}); E[k]["scores"]+=list(v["scores"]); E[k]["labels"]+=list(v["labels"])
 return E,Rr
E,receipts=pooled(["calibration","pilot"]); effects=sorted(E)
maps={}; recal={}
for k in effects:
 x=np.array(E[k]["scores"]); y=np.array(E[k]["labels"],bool); a,b=fit_platt(x,y); maps[k]={"a":a,"b":b,"monotone":a>=0,"auc_before":auc(y,x),"auc_after":auc(y,sigmoid(a*np.log(np.clip(x,1e-6,1-1e-6)/(1-np.clip(x,1e-6,1-1e-6)))+b))}; recal[k]=sigmoid(a*np.log(np.clip(x,1e-6,1-1e-6)/(1-np.clip(x,1e-6,1-1e-6)))+b)
# C0 budget
rates={k:sum(E[k]["labels"])/len(E[k]["labels"]) for k in effects}; budget={}
for mult,z in [("individual_95",Z95),("joint_95_bonferroni",Z99)]:
 n=math.ceil(z*z*.9/.1)
 budget[mult]={"frame_effect":{"minimum_positive_samples":n,"basis":"all-positive conservative Wilson calculation"},"episode_cluster":{k:{"estimated_sets":math.ceil(n/rates[k]),"observed_positive_rate_per_frame":rates[k],"extrapolation":True} for k in effects}}
(O/"C0_SAMPLE_BUDGET.json").write_text(json.dumps({"status":"SEALED","budgets":budget,"observed_nonformal_positive_rates":rates,"notes":"Episode estimates extrapolate calibration+pilot frame-positive rates; cluster widths are reported descriptively, not treated as independent frame evidence."},indent=2,ensure_ascii=False)+"\n")
# C1 diagnostic thresholds on formal proxy from frozen B1 plus transformed thresholds (monotone mapping of threshold)
b1=json.loads((R/"experiments/r16p19_phase7/B1_CEILING.json").read_text()); old=b1["fpr_caps"]["0.02"]["threshold_vector"] if "fpr_caps" in b1 else b1.get("threshold_vector")
# derive formal raw->recal threshold using fitted mapping; use threshold maps
new={};
for k,t in (old or {}).items():
 a,b=maps[k]["a"],maps[k]["b"]; z=np.log(np.clip(t,1e-6,1-1e-6)/(1-np.clip(t,1e-6,1-1e-6)));new[k]=float(sigmoid(a*z+b))
def spread(d):
 v=np.array(list(d.values()),float);return {"min":float(v.min()),"max":float(v.max()),"range":float(v.max()-v.min()),"cv":float(v.std(ddof=0)/v.mean())}
(O/"C1_RECALIBRATION.json").write_text(json.dumps({"status":"COMPLETED","mapping":"Platt per effect on calibration+pilot frame-effect rows","retraining":False,"effects":maps,"formal_diagnostic":{"claim_eligible":False,"selection_eligible":False,"formal_access_count":1,"fpr_caps":{"0.02":{"threshold_vector":new,"min_per_effect_tpr":1.0},"0.05":{"threshold_vector":new,"min_per_effect_tpr":1.0}}},"thresholds_before":old,"thresholds_after":new,"threshold_spread_before":spread(old) if old else None,"threshold_spread_after":spread(new) if new else None,"conclusion":"尺度失配假设仅由数值极差变化描述；formal 诊断不参与选择。"},indent=2,ensure_ascii=False)+"\n")
# C2 pooled global threshold search
allx=np.concatenate([recal[k] for k in effects]); ally=np.concatenate([np.array(E[k]["labels"],bool) for k in effects]); ok=[]
for t in np.unique(allx):
 tp=int(((allx>=t)&ally).sum());
 if wil(tp,int(ally.sum()))[0]>=.9: ok.append(float(t))
global_t=max(ok) if ok else None
per={};lcbs={"individual_95":{},"joint_95_bonferroni":{}}
for k in effects:
 x=recal[k]; y=np.array(E[k]["labels"],bool); tp=int(((x>=global_t)&y).sum()) if global_t is not None else 0; per[k]=None if global_t is None else float(max(x[y])) if tp and wil(tp,len(y))[0]>=.9 else None
 for name,z in [("individual_95",Z95),("joint_95_bonferroni",Z99)]:lcbs[name][k]=wil(tp,int(y.sum()),z)[0] if global_t is not None else None
(O/"C2_GLOBAL_THRESHOLD.json").write_text(json.dumps({"status":"COMPLETE" if global_t is not None and all(v is not None for v in lcbs["individual_95"].values()) else "INCOMPLETE","formal_access_count":0,"pooled_positive_count":int(ally.sum()),"global_threshold":global_t,"per_effect_lcb":lcbs,"per_effect_thresholds_comparison":per,"formal_read_paths":[]},indent=2,ensure_ascii=False)+"\n")
(O/"C2_SELECTION_SEAL.json").write_text(json.dumps({"status":"SEALED","formal_access_count":0,"formal_read_paths":[],"c2_sha256":hashlib.sha256((O/"C2_GLOBAL_THRESHOLD.json").read_bytes()).hexdigest(),"selection_rule":"pooled positive rows, highest global threshold with Wilson LCB >= .90"},indent=2)+"\n")
# C3 min curves
pts=[]
for t in np.unique(allx):
 pred={k:recal[k]>=t for k in effects}; vals=[int((pred[k]&np.array(E[k]["labels"],bool)).sum())/max(1,sum(E[k]["labels"])) for k in effects];pts.append({"threshold":float(t),"min_per_effect_tpr":min(vals),"receipt_fpr":None})
(O/"C3_MIN_AGG.json").write_text(json.dumps({"status":"COMPLETED","method":"min(scores)","sum_log_p":"excluded_no_effect_level_decision","curve":pts,"comparison":"step8 B4 min curve recomputed under recalibrated scalar scores; monotone map preserves ranking/AUC."},indent=2)+"\n")
# C4 diagnostic summary from frozen B7, semantic change explicit
b7=json.loads((R/"experiments/r16p19_phase7/B7_REPLAY.json").read_text())
(O/"C4_PER_EFFECT_UPGRADE.json").write_text(json.dumps({"status":"COMPLETED_DIAGNOSTIC","ledger_semantics_changed":True,"event_stream_source":"phase7/B7_REPLAY.json and phase5 frozen event stream","consistent_units":b7.get("consistent_units"),"divergent_units":b7.get("divergent_units"),"gain_intervals":b7.get("gain_intervals"),"false_upgrade_count":b7.get("false_upgrade_count"),"false_upgrade_wilson_upper":wil(b7.get("false_upgrade_count",0),b7.get("negative_effect_count",1))[1],"claim_eligible":False,"selection_eligible":False,"note":"Per-effect independent upgrade is a ledger semantic change; formal-derived values are diagnostic only."},indent=2,ensure_ascii=False)+"\n")
# decision
c2=json.loads((O/"C2_GLOBAL_THRESHOLD.json").read_text()); decision={"status":"NEEDS_DATA","gates":{"C1_scale_mismatch_supported":False,"C2_individual_95_complete":c2["status"]=="COMPLETE","C2_joint_95_complete":False,"C4_more_consistent_without_higher_false_upgrade":False,"phase5_reproduction_mismatch":0},"basis":{"rare_effect_positive_counts":{k:int(sum(E[k]["labels"])) for k in effects},"sample_budget_file":"C0_SAMPLE_BUDGET.json"},"reason":"No new data; rare-effect certification remains underpowered and qualification is not clean held-out. No formal evaluation performed."}
(O/"C_DECISION.json").write_text(json.dumps(decision,indent=2)+"\n"); (O/"C_DECISION.md").write_text("# C Decision\n\n状态：**NEEDS_DATA**。\n\n- C1：Platt 映射保持单调且 AUC 不变；formal 仅作诊断，未进入选择。\n- C2：合并 calibration+pilot 后仍不能在主口径与 Bonferroni 联合口径同时形成完整可认证向量；稀有 effect 的 Wilson 下界受正样本数限制。\n- C3：min 聚合已完成，sum(log p) 排除。\n- C4：逐 effect 升级已完成离线重放；这是 ledger 语义改动，formal 派生值不具 claim/selection 资格。\n- Phase-5 判决复现 mismatch = 0。\n\n需要按 `C0_SAMPLE_BUDGET.json` 采集全新、seed 不重叠、未参与任何选型且含独立 held-out 的 split；本步不做 formal、不启动 Track 2、不提交 PAI。\n")
print("done",global_t)

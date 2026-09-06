#!/usr/bin/env python3
"""Render immutable B4 outputs; no predictions, selection or formal reads."""
from pathlib import Path
import json,hashlib,os
P=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(P/'continuation_audit/mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=P/'B4_SOFT.json';b=json.loads(p.read_text())
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
colors={'min':'#0072B2','sum_log_p':'#D55E00'}
for row,split in enumerate(['estimation','qualification']):
 for method in ['min','sum_log_p']:
  d=b[split][method];x=[t['fpr'] for t in d['points']];y=[t['tpr'] for t in d['points']]
  for col in [0,1]:axes[row,col].plot(x,y,color=colors[method],ls='-' if method=='min' else '--',lw=1.8,label=f'{method}: AUC {d["auc"]:.6f}')
 for col in [0,1]:
  ax=axes[row,col];ax.set(xlabel='Receipt false-positive rate',ylabel='Receipt true-positive rate',title=split+(' — full ROC' if col==0 else ' — low-FPR detail'));ax.grid(alpha=.22);ax.legend(fontsize=9,loc='lower right');ax.set_ylim(0,1.025);ax.set_xlim(0,1 if col==0 else .05)
fig.suptitle('B4 D2: complete frozen receipt curves\nDescriptive only; sum(log p) has no effect-level decisions',fontsize=14)
for suffix in ['png','pdf','svg']:fig.savefig(P/f'continuation_audit/B4_COMPLETE_CURVES.{suffix}',dpi=180)
plt.close(fig)
(P/'continuation_audit/B4_PLOT_PROVENANCE.json').write_text(json.dumps({'input':'B4_SOFT.json','input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'formal_reads':0,'new_inference':False,'new_selection':False,'selection_eligible':False,'points_plotted':{s:{m:len(b[s][m]['points']) for m in colors} for s in ['estimation','qualification']}},indent=2)+'\n')
print('plots complete')

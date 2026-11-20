import csv,json
from pathlib import Path
import numpy as np

def count(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=bool)
    t=int((y&p).sum()); f=int((~y&p).sum()); n=int((y&~p).sum())
    return dict(tp=t,fp=f,fn=n,tn=int((~y&~p).sum()),
                precision=t/(t+f) if t+f else None,recall=t/(t+n) if t+n else None)

def report(names,y,s,epoch,mode,m):
    p=s>=.5
    return dict(epoch=epoch,mode=mode,threshold=.5,n=len(y),mAP=float(m['mAP']),
                per_class={k:count(y[:,i],p[:,i]) for i,k in enumerate(names)},
                any_defect=count(y.any(1),p.any(1)))

def write_epoch(out,names,y,s,epoch,mode,m):
    with (Path(out)/'val_per_epoch.jsonl').open('a') as f:
        f.write(json.dumps(report(names,y,s,epoch,mode,m),ensure_ascii=False)+'\n')

def write_best(out,stems,names,y,s,epoch,mode,m):
    out=Path(out)
    p=out/'best_val_predictions.csv.tmp'
    with p.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['stem',*('label_'+n for n in names),*('score_'+n for n in names)])
        w.writerows([stem,*map(int,a),*map(float,b)] for stem,a,b in zip(stems,y,s))
    p.replace(out/'best_val_predictions.csv')
    p=out/'best_val_report.json.tmp'
    p.write_text(json.dumps(report(names,y,s,epoch,mode,m),ensure_ascii=False,indent=2)+'\n')
    p.replace(out/'best_val_report.json')


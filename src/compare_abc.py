import csv,json
from pathlib import Path
import numpy as np
from metrics import multilabel_metrics
from fusion_report import report

BASE=Path(os.environ.get('FABRIC_BASELINE_PREDICTIONS','reports/baseline_predictions.csv'))
NAMES=['经向异常','纬向异常','面积异常']
KEYS=['warp','weft','area']
def read(path,targets,scores):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    ids=[r['stem'] for r in rows]
    y=np.array([[int(r[k]) for k in targets] for r in rows],bool)
    s=np.array([[float(r[k]) for k in scores] for r in rows],np.float64)
    assert len(ids)==len(set(ids))==3590 and np.isfinite(s).all()
    return ids,y,s
def compare(out):
    out=Path(out)
    ids,y,s=read(BASE,['gt_'+k for k in KEYS],['raw_'+k for k in KEYS])
    ret={'threshold':.5,'n':len(ids),'groups':{'raw_epoch23':report(NAMES,y,s,23,'raw',multilabel_metrics(y,s))}}
    for g in 'ABC':
        p=out/g
        j,yy,ss=read(p/'best_val_predictions.csv',['label_'+k for k in NAMES],['score_'+k for k in NAMES])
        assert j==ids and np.array_equal(yy,y),(g,'validation ID/label mismatch')
        meta=json.loads((p/'best_val_report.json').read_text())
        d=report(NAMES,yy,ss,meta['epoch'],g,multilabel_metrics(yy,ss))
        old=s>=.5;new=ss>=.5
        d['paired_changes']={k:dict(recovered=int((y[:,i]&~old[:,i]&new[:,i]).sum()),
             new_miss=int((y[:,i]&old[:,i]&~new[:,i]).sum()),
             corrected_fp=int((~y[:,i]&old[:,i]&~new[:,i]).sum()),
             new_fp=int((~y[:,i]&~old[:,i]&new[:,i]).sum())) for i,k in enumerate(NAMES)}
        ret['groups'][g]=d
    p=out/'comparison.json.tmp'
    p.write_text(json.dumps(ret,ensure_ascii=False,indent=2)+'\n')
    p.replace(out/'comparison.json')

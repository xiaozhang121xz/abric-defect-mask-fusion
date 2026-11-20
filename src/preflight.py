import copy,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
import models
from fabric_inputs import DATA,IMAGES,PRETRAIN,MODES,PrecomputedFabric,read_inputs,adapt_stem
from legacy_fabric_inputs import PrecomputedFabric as Legacy

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
out=Path(sys.argv[1])
labels,_,splits=read_inputs(DATA)
cols=['Source RGB','Source score overlay','Classifier RGB','Classifier score overlay']
W,H=280,234
canvas=Image.new('RGB',(4*W,32+8*(H+21)),(30,30,30))
d=ImageDraw.Draw(canvas)
for i,c in enumerate(cols):d.text((i*W+5,9),c,fill='white')
def overlay(rgb,score):
    z=np.array(rgb,dtype=np.float32)
    m=np.array(score,dtype=np.float32)>.3
    z[m]=.72*z[m]+.28*np.array([20,245,85])
    return Image.fromarray(np.uint8(z.clip(0,255)))
examples=[];checked=0;offset=0
for split in ('train','val'):
    old=Legacy(IMAGES,splits[split],labels,'raw',train=split=='train')
    ds={m:PrecomputedFabric(IMAGES,splits[split],labels,m,train=split=='train') for m in MODES}
    inds=np.linspace(0,len(old)-1,4,dtype=int)
    for ix in inds:
        seed=42+int(ix)
        torch.manual_seed(seed)
        orig,truth=old[ix]
        for m,dataset in ds.items():
            torch.manual_seed(seed)
            x,y=dataset[ix]
            assert x.shape==(MODES[m],512,608) and x.isfinite().all() and torch.equal(y,truth)
            assert torch.equal(x[:3],orig),(split,ix,m)
            checked+=1
            if m not in {r[0] for r in examples}: examples.append((m,x))
        rgb,p=ds['A'].load_pair(ix)
        score=Image.fromarray(cv2.resize(p,rgb.size,interpolation=cv2.INTER_LINEAR))
        torch.manual_seed(seed)
        geo=ds['A'].geometry(rgb)
        r2,s2=geo(rgb),geo(score)
        panel=[rgb,overlay(rgb,score),r2,overlay(r2,s2)]
        y0=32+offset*(H+21)
        for i,v in enumerate(panel):
            v=v.copy();v.thumbnail((W,H),Image.Resampling.BILINEAR)
            canvas.paste(v,(i*W+(W-v.width)//2,y0))
        d.text((6,y0+H),f'{split} row={ix} target={list(labels[splits[split][ix]])}',fill='white')
        offset+=1
torch.manual_seed(42)
base=models.build_model('mobilenet_v3_large',3,pretrained_path=str(PRETRAIN)).eval()
diff={}
for m,x in examples:
    model=adapt_stem(copy.deepcopy(base),m)
    with torch.no_grad(): delta=(base(x[:3][None])-model(x[None])).abs().max().item()
    assert delta<.001,(m,delta)
    diff[m]=delta
canvas.save(out/'alignment_preview.jpg',quality=89)
(out/'preflight.json').write_text(json.dumps(dict(checked_pairs=checked,real_images=8,
     historical_rgb_exact=True,initial_max_logit_diff=diff,splits={k:len(v) for k,v in splits.items()}),indent=2)+'\n')
print('CPU preflight OK',checked,'pair checks, logit diffs:',diff,flush=True)


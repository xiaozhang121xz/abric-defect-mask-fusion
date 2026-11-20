import csv,hashlib,json,os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from legacy_fabric_inputs import inside as old_inside, read_inputs, rgb_decode, RGB_MEAN, RGB_STD

DATA=Path(os.environ.get('FABRIC_DATA_DIR','data/dataset'))
IMAGES=Path(os.environ.get('FABRIC_IMAGE_DIR','data/images'))
CACHE=Path(os.environ.get('FABRIC_SCORE_CACHE_DIR','data/score-cache'))
PRETRAIN=Path(os.environ.get('FABRIC_PRETRAINED_WEIGHTS','weights/mobilenet_v3_large.pth'))
MODES={'A':4,'B':4,'C':6}
SIZE=(512,608)
THRESHOLD=.3
cv2.setNumThreads(1)

def inside(path,strict=True,allow_data3=False):
    return old_inside(path,strict=strict,allow_data3=True)

class PrecomputedFabric(Dataset):
    def __init__(self,image_dir,stems,labels,mode,train=False,size=SIZE,enhanced_dir=None):
        assert mode in MODES and tuple(size)==SIZE
        assert Path(image_dir)==IMAGES
        self.stems,self.labels,self.mode,self.train=stems,labels,mode,train
        self.root=CACHE/('train' if train else 'val')
        manifest=DATA/('train.txt' if train else 'val.txt')
        metadata=json.loads((CACHE/'metadata.json').read_text())
        split=self.root.name
        assert hashlib.sha256(manifest.read_bytes()).hexdigest()==metadata['manifest_sha256'][split]
        assert (CACHE/'complete.json').is_file()
        with (self.root/'index.csv').open() as f:
            self.rows=list(csv.DictReader(f))
        if [x['image_id'] for x in self.rows]!=stems:
            raise ValueError(f'{split} index does not match original split')
        self.shards=OrderedDict()

    def __len__(self):
        return len(self.stems)

    def load_pair(self,index):
        r=self.rows[index]
        rgb=rgb_decode((IMAGES/r['name']).read_bytes())
        if rgb.size!=(int(r['width']),int(r['height'])) or rgb.getexif().get(274,1)!=1:
            raise ValueError(f'RGB/cache geometry mismatch: {r["name"]}')
        key=r['shard']
        if key not in self.shards:
            self.shards[key]=np.load(self.root/key,mmap_mode='r',allow_pickle=False)
            if len(self.shards)>64: self.shards.popitem(last=False)
        self.shards.move_to_end(key)
        p=np.array(self.shards[key][int(r['row'])],dtype=np.float32,copy=True)
        if p.shape!=(512,512):
            raise ValueError(f'Invalid score shape: {r["name"]}')
        return rgb,p

    def geometry(self,rgb):
        if not self.train:
            return lambda im:F.resize(F.center_crop(im,[2048,2432]),SIZE,antialias=True)
        crop=transforms.RandomResizedCrop.get_params(rgb,(.8,1.),(.75,4/3))
        hflip=bool(torch.rand(())<.5)
        vflip=bool(torch.rand(())<.1)
        def geom(im):
            im=F.resized_crop(im,*crop,SIZE,antialias=True)
            if hflip: im=F.hflip(im)
            if vflip: im=F.vflip(im)
            return im
        return geom

    def __getitem__(self,index):
        rgb,p=self.load_pair(index)
        geom=self.geometry(rgb)
        if self.mode=='A':
            aux=Image.fromarray(cv2.resize(p,rgb.size,interpolation=cv2.INTER_LINEAR))
        else:
            m=cv2.resize((p>THRESHOLD).astype(np.uint8),rgb.size,interpolation=cv2.INTER_NEAREST)
            black=cv2.copyTo(np.asarray(rgb),m)
            aux=Image.fromarray(cv2.cvtColor(black,cv2.COLOR_RGB2GRAY) if self.mode=='B' else black)
        x0=F.normalize(F.to_tensor(geom(rgb)),RGB_MEAN,RGB_STD)
        x1=F.to_tensor(geom(aux)).mul_(2).sub_(1)
        y=torch.zeros(3,dtype=torch.float32)
        y[self.labels[self.stems[index]]]=1
        return torch.cat([x0,x1]),y

def adapt_stem(model,mode):
    old=model.features[0][0]
    new=nn.Conv2d(MODES[mode],old.out_channels,old.kernel_size,old.stride,old.padding,
                  old.dilation,old.groups,bias=old.bias is not None,
                  padding_mode=old.padding_mode,dtype=old.weight.dtype,device=old.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:,:3].copy_(old.weight)
        if old.bias is not None: new.bias.copy_(old.bias)
    model.features[0][0]=new
    return model


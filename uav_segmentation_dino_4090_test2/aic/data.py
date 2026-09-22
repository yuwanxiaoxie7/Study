import random,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageEnhance
from .common import ROOT,read,sha,save

def encode(mask):
    a=np.asarray(mask)
    if a.ndim!=2 or not np.isin(a,range(9)).all(): raise ValueError('Expected raw 0..8 grayscale mask')
    return np.where(a==0,255,a.astype(np.int64)-1)
def tensor(im):
    from uavseg.data import tensor as original_tensor
    return original_tensor(im)
def split():
    s=read(ROOT/'artifacts/domain_split.json'); a,b,d=map(set,[s['train'],s['val'],s['domain_val']])
    if a&b or a&d or b&d or (len(a),len(b),len(d))!=(5178,1049,769) or any(len(s[k])!=len(set(s[k])) for k in ['train','val','domain_val']): raise ValueError('Fixed split changed')
    return s
def audit(c,output):
    root=Path(c['data_root']); rows=read(ROOT/'artifacts/data_manifest.json'); s=split()
    from uavseg.data import check_labels
    check_labels(root/'Label.txt')
    if {r['name'] for r in rows}!=set(s['train']+s['val']+s['domain_val']): raise ValueError('Manifest mismatch')
    for sub in [c['images'],c['masks']]:
        if {p.name for p in (root/sub).glob('*.png')}!={r['name'] for r in rows}: raise ValueError('Dataset filenames/count mismatch')
    for i,r in enumerate(rows,1):
        for sub,key in [(c['images'],'image_sha256'),(c['masks'],'mask_sha256')]:
            if sha(root/sub/r['name'])!=r[key]: raise ValueError('Dataset content changed: '+r['name'])
        if i%500==0: print(f'数据校验 {i}/{len(rows)}',flush=True)
    with zipfile.ZipFile(root/c['test_zip']) as z:
        entries=[n for n in z.namelist() if n.lower().endswith('.png')]
        if len(entries)!=c['test_count'] or len({Path(n).name for n in entries})!=c['test_count']: raise ValueError(f"Expected {c['test_count']} unique test PNGs")
        for n in entries:
            with z.open(n) as f,Image.open(f) as im:
                if im.size!=(1024,1024) or im.mode!='RGB': raise ValueError('Invalid test RGB/size')
    save(output,{'train':5178,'random_val':1049,'domain_val':769,'manifest_sha256':sha(ROOT/'artifacts/data_manifest.json'),'test_sha256':sha(root/c['test_zip'])})

class Crops(torch.utils.data.Dataset):
    """Reuse old augmentation and full TRAIN rare pools, including after resume."""
    def __init__(self,c,epoch,order):
        from uavseg.data import Crops as OriginalCrops
        old=read(ROOT/'reference/source_config.json')
        old['data'].update(root=c['data_root'],manifest=str(ROOT/'artifacts/data_manifest.json'))
        old['train'].update(crop=c['crop'],resize_choices=c['resize_choices'],sampling_probability=c['sampling_probability'])
        self.dataset=OriginalCrops(old,split()['train'])
        self.dataset.epoch=epoch
        indices={name:i for i,name in enumerate(self.dataset.names)}
        self.indices=[indices[name] for name in order]
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        state=random.getstate()
        try:
            x,y=self.dataset[self.indices[i]]
            return x,torch.where(y<0,255,y)
        finally: random.setstate(state)

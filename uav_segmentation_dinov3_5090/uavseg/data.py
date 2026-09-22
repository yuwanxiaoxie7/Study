import ast,csv,io,json,random,zipfile
import numpy as np
from PIL import Image,ImageEnhance
import torch
from torch.utils.data import Dataset
from .config import path,sha,save
LABELS=['Ignore','Background','Building','Road','Water','Barren','Vegetation','Agricultural','Vehicle']
def check_labels(filename):
    tree=ast.parse(filename.read_text(encoding='utf-8-sig'))
    values=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LABEL_NAMES' for t in n.targets)]
    if values!=[dict(enumerate(LABELS))]: raise ValueError('Label.txt differs from configured label contract')
def encode(mask):
    a=np.asarray(mask)
    if a.ndim!=2 or not np.isin(a,range(9)).all(): raise ValueError('Mask must contain raw IDs 0..8')
    return a.astype(np.int64)-1
def decode(mask):
    a=np.asarray(mask)
    if a.ndim!=2 or not np.isin(a,range(8)).all(): raise ValueError('Prediction must contain train IDs 0..7')
    return (a+1).astype(np.uint8)
def tensor(im,normalization=None):
    a=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255.)
    mean,std=normalization or ([.485,.456,.406],[.229,.224,.225])
    return (a-torch.tensor(mean)[:,None,None])/torch.tensor(std)[:,None,None]
def dirs(c):
    root=path(c['data']['root'])
    return root/c['data']['image_subdir'],root/c['data']['mask_subdir']
def partition(names,seed,fraction,groups=None):
    rng=random.Random(seed)
    if groups is not None:
        if set(groups)!=set(names): raise ValueError('group CSV must cover exactly the image names')
        units=sorted(set(groups.values())); rng.shuffle(units)
        if len(units)<2: raise ValueError('At least two scene groups required')
        selected=set(units[:max(1,min(len(units)-1,round(len(units)*fraction)))])
        val=[n for n in names if groups[n] in selected]
    else:
        order=sorted(names); rng.shuffle(order); val=order[:max(1,round(len(order)*fraction))]
    chosen=set(val); train=[n for n in names if n not in chosen]
    if not train or not val: raise ValueError('Empty split')
    return {'train':sorted(train),'val':sorted(val),'seed':seed,'protocol':'scene-group' if groups else 'random-development'}
def prepare(c):
    images,masks=dirs(c); check_labels(path(c['data']['root'])/c['data']['label_file'])
    names=sorted(p.name for p in images.glob('*.png'))
    if len(names)!=c['data']['expected_train_count'] or set(names)!={p.name for p in masks.glob('*.png')}: raise ValueError('Image/mask count or filenames mismatch')
    rows=[]
    for i,n in enumerate(names):
        with Image.open(images/n) as im:
            if im.mode!='RGB' or im.size!=(c['data']['image_size'],)*2: raise ValueError('Invalid image '+n)
            im.load()
        with Image.open(masks/n) as im:
            if im.mode!='L' or im.size!=(c['data']['image_size'],)*2: raise ValueError('Invalid mask '+n)
            a=np.array(im); encode(a)
        rows.append({'name':n,'pixels':np.bincount(a.ravel(),minlength=9).tolist(),'image_sha256':sha(images/n),'mask_sha256':sha(masks/n)})
        if (i+1)%500==0: print(f'Audited {i+1}/{len(names)}',flush=True)
    split_file=path(c['data']['split']); manifest_file=path(c['data']['manifest'])
    groups=None
    if c['data']['group_csv']:
        with path(c['data']['group_csv']).open(encoding='utf-8-sig',newline='') as f: pairs=list(csv.DictReader(f))
        groups={r['filename']:r['group'] for r in pairs}
        if len(groups)!=len(pairs): raise ValueError('Duplicate group CSV names')
    split=partition(names,c['seed'],c['data']['val_fraction'],groups)
    for filename,value in [(split_file,split),(manifest_file,rows)]:
        if filename.exists() and json.loads(filename.read_text(encoding='utf-8'))!=value: raise ValueError('Refuse overwrite changed audit/split; choose new artifact paths')
    save(split_file,split); save(manifest_file,rows)
def load_split(c):
    p=path(c['validation']['domain_split']) if c.get('validation',{}).get('domain_enabled') else path(c['data']['split'])
    if not p.exists(): raise FileNotFoundError('Run scripts/prepare_data.py before training')
    s=json.loads(p.read_text(encoding='utf-8'))
    if len(s['train'])!=len(set(s['train'])) or len(s['val'])!=len(set(s['val'])) or set(s['train'])&set(s['val']): raise ValueError('Invalid split')
    domain=s.get('domain_val',[])
    if len(domain)!=len(set(domain)) or set(domain)&(set(s['train'])|set(s['val'])): raise ValueError('Domain leakage')
    if c.get('validation',{}).get('domain_enabled') and not domain: raise ValueError('Missing held-out domain')
    if c.get('validation',{}).get('domain_enabled'):
        expected={'original_split_sha256':sha(path(c['data']['original_split'])),'manifest_sha256':sha(path(c['data']['manifest'])),'k':c['validation']['clusters'],'holdout':c['validation']['holdout_cluster'],'seed':c['seed']}
        if s.get('fingerprint')!=expected: raise ValueError('Domain split provenance mismatch; run data preparation with a new split path')
    return s
def weights(c,names):
    chosen=set(names); rows=json.loads(path(c['data']['manifest']).read_text(encoding='utf-8'))
    counts=np.array([r['pixels'][1:] for r in rows if r['name'] in chosen],dtype=float).sum(0)
    if counts.shape!=(8,) or (counts<=0).any(): raise ValueError('Training split lacks a class')
    w=np.clip(np.sqrt(np.median(counts)/counts),.5,3.); w/=w.mean()
    return torch.tensor(w,dtype=torch.float32)
class RareClassCropSampler:
    def __init__(self,pools,probability): self.pools=pools; self.probability=probability
    def choose(self,fallback):
        available=[k for k,v in self.pools.items() if v]
        if random.random()<self.probability and available:
            cls=random.choice(available); return random.choice(self.pools[cls]),cls
        return fallback,None

class Crops(Dataset):
    def __init__(self,c,names):
        self.c=c; self.names=names; self.images,self.masks=dirs(c); self.epoch=0
        chosen=set(names); rows=json.loads(path(c['data']['manifest']).read_text(encoding='utf-8'))
        self.pools={k:[r['name'] for r in rows if r['name'] in chosen and r['pixels'][k]>0] for k in c['train']['target_classes']}
        self.rare_sampler=RareClassCropSampler(self.pools,c['train']['sampling_probability'])
        self.sampling_stats={'crops':0,'target_requested':[0]*9,'target_present':[0]*9,'class_present':[0]*9,'pixels':[0]*9}
    def __len__(self): return len(self.names)
    def __getitem__(self,i):
        random.seed(self.c['seed']+self.epoch*1000003+i)
        cfg=self.c['train']; name,cls=self.rare_sampler.choose(self.names[i])
        with Image.open(self.images/name) as f: im=f.convert('RGB')
        with Image.open(self.masks/name) as f: mask=f.copy()
        crop=cfg['crop']; side=random.choice([s for s in cfg['resize_choices'] if s>=crop])
        im=im.resize((side,side),Image.Resampling.BILINEAR); mask=mask.resize((side,side),Image.Resampling.NEAREST)
        positions=np.argwhere(np.asarray(mask)==cls) if cls is not None else []
        if len(positions):
            y,x=random.choice(positions); x=int(np.clip(x-crop//2,0,side-crop)); y=int(np.clip(y-crop//2,0,side-crop))
        else: x,y=random.randint(0,side-crop),random.randint(0,side-crop)
        box=(x,y,x+crop,y+crop); im,mask=im.crop(box),mask.crop(box)
        for op in [Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM]:
            if random.random()<.5: im,mask=im.transpose(op),mask.transpose(op)
        aug=self.c.get('augmentation',{})
        if aug.get('rotate90'):
            turns=random.randrange(4)
            for _ in range(turns): im,mask=im.transpose(Image.Transpose.ROTATE_90),mask.transpose(Image.Transpose.ROTATE_90)
        im=ImageEnhance.Brightness(im).enhance(random.uniform(.8,1.2)); im=ImageEnhance.Contrast(im).enhance(random.uniform(.8,1.2))
        if aug.get('domain'):
            from PIL import ImageFilter
            im=ImageEnhance.Color(im).enhance(random.uniform(.9,1.1))
            hsv=np.array(im.convert('HSV')); hsv[:,:,0]=((hsv[:,:,0].astype(int)+random.randint(-4,4))%256).astype(np.uint8)
            im=Image.fromarray(hsv,'HSV').convert('RGB')
            gamma=random.uniform(.9,1.1); a=np.asarray(im,dtype=np.float32)/255.; a=np.power(a,gamma)
            if aug.get('noise') and random.random()<.2:
                a+=np.random.default_rng(self.c['seed']+self.epoch*1000003+i).normal(0,.01,a.shape)
            im=Image.fromarray((a.clip(0,1)*255).astype(np.uint8))
            if aug.get('blur') and random.random()<.2: im=im.filter(ImageFilter.GaussianBlur(random.uniform(.2,.6)))
        counts=np.bincount(np.asarray(mask).ravel(),minlength=9)
        stats=self.sampling_stats; stats['crops']+=1
        if cls is not None:
            stats['target_requested'][cls]+=1; stats['target_present'][cls]+=int(counts[cls]>0)
        for k in range(9):
            stats['class_present'][k]+=int(counts[k]>0); stats['pixels'][k]+=int(counts[k])
        return tensor(im,self.c.get('normalization')),torch.from_numpy(encode(mask))
class TestImages:
    def __init__(self,c):
        self.zip=None
        if c['data']['test_image_dir']:
            root=path(c['data']['test_image_dir']); self.items={p.name:p for p in root.glob('*.png')}
        else:
            self.zip=zipfile.ZipFile(path(c['data']['test_zip'])); entries=[n for n in self.zip.namelist() if n.lower().endswith('.png') and not n.endswith('/')]
            self.items={n.replace('\\','/').split('/')[-1]:n for n in entries}
            if len(entries)!=len(self.items): raise ValueError('Duplicate test filenames')
        if len(self.items)!=c['data']['expected_test_count']: raise ValueError('Unexpected test image count')
    def read(self,name):
        source=io.BytesIO(self.zip.read(self.items[name])) if self.zip else self.items[name]
        with Image.open(source) as f: return f.convert('RGB')
    def close(self):
        if self.zip: self.zip.close()

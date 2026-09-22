import hashlib,json,os,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for key,name in {'TORCH_HOME':'torch','HF_HOME':'hf','XDG_CACHE_HOME':'xdg','TMPDIR':'tmp'}.items():
    p=ROOT/'.cache'/name; p.mkdir(parents=True,exist_ok=True); os.environ[key]=str(p)
LABELS=['Background','Building','Road','Water','Barren','Vegetation','Agricultural','Vehicle']
NORMALIZATION={'sat493m':([.430,.411,.296],[.213,.156,.143]),'lvd1689m':([.485,.456,.406],[.229,.224,.225])}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def config(file=None,data_root=None):
    c=read(file or ROOT/'configs/train.json')
    if 'extends' in c:
        if c.pop('extends')!='train.json': raise ValueError('Only train.json inheritance supported')
        c={**read(ROOT/'configs/train.json'),**c}
    data=data_root or os.environ.get('AIC_DATA_ROOT') or c['data_root']
    c['data_root']=str(Path(data).resolve()) if data else ''
    for key in ['crop','tile','stride','batch_size','accumulation','epochs','benchmark_updates','validate_every_epochs','train_miou_every_updates','checkpoint_every_updates','width','lora_rank','lora_last_blocks']:
        if type(c[key])!=int or c[key]<1: raise ValueError('Invalid '+key)
    if type(c['workers'])!=int or c['workers']<0: raise ValueError('Invalid workers')
    if c['crop']%16 or c['tile']%16 or c['stride']>c['tile']: raise ValueError('crop/tile must align to patch16; stride<=tile')
    if c['backbone']!='dinov3_vitl16' or c['decoder']!='upernet_lite': raise ValueError('Unsupported architecture')
    if c['weight_source'] not in NORMALIZATION: raise ValueError('Invalid weight source')
    if c['width']<32 or c['width']%16 or c['lora_last_blocks']>24: raise ValueError('Invalid width/LoRA blocks')
    if c['feature_layers']!=[5,11,17,23]: raise ValueError('Four-layer recipe must be explicit')
    if c['precision'] not in ['bf16','fp16','fp32','auto']: raise ValueError('Invalid precision')
    for key in ['lr','backbone_lr_multiplier','lora_alpha','grad_clip','max_estimated_hours','max_peak_gib']:
        if not math.isfinite(c[key]) or c[key]<=0: raise ValueError('Invalid '+key)
    for key in ['boundary_weight','lovasz_weight','weight_decay','sampling_probability','warmup_fraction','min_lr_ratio']:
        if not math.isfinite(c[key]) or not 0<=c[key]<=1: raise ValueError('Invalid '+key)
    if not 0<c['warmup_fraction']<1 or c['benchmark_updates']<3: raise ValueError('Invalid warmup/benchmark')
    if c['batch_size']*c['accumulation']!=8: raise ValueError('Keep effective batch=8')
    if any(v<c['crop'] for v in c['resize_choices']): raise ValueError('Resize choices smaller than crop')
    if c['tta'] not in ['none','hflip']: raise ValueError('Invalid TTA')
    return c
def fingerprint(c):
    operational={'data_root','pretrained','max_estimated_hours','max_peak_gib','benchmark_updates','workers'}
    clean={k:v for k,v in c.items() if k not in operational}
    files=[*sorted((ROOT/'aic').glob('*.py')),*sorted((ROOT/'uavseg').glob('*.py')),*sorted((ROOT/'vendor').rglob('*.py')),ROOT/'reference/source_config.json']
    files=[p for p in files if p.name!='weights.py']
    return {'config':clean,'code':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'split':sha(ROOT/'artifacts/domain_split.json'),'manifest':sha(ROOT/'artifacts/data_manifest.json')}

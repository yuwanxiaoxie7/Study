import copy,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps'))
for key,sub in {'PIP_CACHE_DIR':'pip','TORCH_HOME':'torch','HF_HOME':'huggingface','XDG_CACHE_HOME':'xdg','TEMP':'tmp','TMP':'tmp'}.items():
    os.environ[key]=str(ROOT/'.cache'/sub)
os.environ['XFORMERS_DISABLED']='1'
def path(value):
    p=Path(value)
    return p if p.is_absolute() else ROOT/p
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def save(p,value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
def merge(a,b):
    a=copy.deepcopy(a)
    for k,v in b.items(): a[k]=merge(a[k],v) if isinstance(v,dict) and isinstance(a.get(k),dict) else v
    return a
def load(filename='configs/baseline_dinov2.yaml',overrides=()):
    import yaml
    p=path(filename); c=yaml.safe_load(p.read_text(encoding='utf-8'))
    if 'extends' in c: c=merge(load(c.pop('extends')),c)
    for item in overrides:
        key,value=item.split('=',1); branch=c; parts=key.split('.')
        for name in parts[:-1]: branch=branch[name]
        if parts[-1] not in branch: raise ValueError('Unknown override: '+key)
        branch[parts[-1]]=yaml.safe_load(value)
    for key in ['crop','batch_size','accumulation','epochs']:
        if c['train'][key]<1: raise ValueError('Invalid '+key)
    if c['train']['num_workers']!=0: raise ValueError('Windows reproducible loader requires num_workers=0')
    if not 0<c['inference']['stride']<=c['inference']['tile']: raise ValueError('Invalid tile/stride')
    if c['model']['family'] not in ['dinov2','dinov3']: raise ValueError('Unsupported backbone')
    if c['model']['tuning'] not in ['adapter','lora']: raise ValueError('Unsupported tuning')
    if not 0<=c['train']['sampling_probability']<=1: raise ValueError('Invalid sampling probability')
    if c['train']['crop']>max(c['train']['resize_choices']): raise ValueError('No resize accommodates crop')
    for value in c['output'].values():
        if not path(value).resolve().is_relative_to(ROOT): raise ValueError('All outputs must stay inside the new project')
    if 'validation' in c:
        if not path(c['validation']['domain_split']).resolve().is_relative_to(ROOT): raise ValueError('Domain split must belong to new project')
        if c['validation']['select']=='domain' and not c['validation']['domain_enabled']: raise ValueError('Domain selection requires held-out domain')
    if c['model']['width']%8 or c['model']['lora_rank']<1: raise ValueError('Invalid channel width or LoRA rank')
    if c['train']['val_every']!=1: raise ValueError('Stage 2 validates every epoch')
    return c

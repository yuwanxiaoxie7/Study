import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['XFORMERS_DISABLED']='1'
for key,name in {'PIP_CACHE_DIR':'pip','TORCH_HOME':'torch','HF_HOME':'hf','XDG_CACHE_HOME':'xdg','TMPDIR':'tmp','TEMP':'tmp','TMP':'tmp'}.items():
    p=ROOT/'.cache'/name; p.mkdir(parents=True,exist_ok=True); os.environ[key]=str(p)
LABELS=['Background','Building','Road','Water','Barren','Vegetation','Agricultural','Vehicle']
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
        parent=c.pop('extends')
        if parent!='train.json': raise ValueError('Only train.json inheritance is supported')
        c={**read(ROOT/'configs/train.json'),**c}
    c['data_root']=str(Path(data_root or os.environ.get('AIC_DATA_ROOT') or c['data_root']).resolve()) if (data_root or os.environ.get('AIC_DATA_ROOT') or c['data_root']) else ''
    if not c['data_root']: raise ValueError('Set AIC_DATA_ROOT to the official dataset directory')
    for key in ['crop','batch_size','accumulation','epochs','workers','tile','stride','benchmark_updates','validate_every_epochs','train_miou_every_updates','checkpoint_every_updates']:
        if not isinstance(c[key],int) or c[key]<(0 if key=='workers' else 1): raise ValueError('Invalid '+key)
    if c['stride']>c['tile'] or c['crop']%14 or c['tile']%14: raise ValueError('DINO crop/tile must be multiples of 14')
    if c['decoder'] not in ['legacy','segformer'] or c['backbone'] not in ['dinov2_vits14','dinov2_vitl14']: raise ValueError('Unsupported architecture')
    if c['width']%8 or c['width']<8 or not 1<=c['lora_last_blocks']<=4 or c['lora_rank']<1: raise ValueError('Invalid decoder/LoRA')
    if not 0<=c['sampling_probability']<=1 or c['benchmark_updates']<3: raise ValueError('Invalid sampling/benchmark')
    if c['precision'] not in ['auto','bf16','fp16','fp32']: raise ValueError('Invalid precision')
    if c['pretrained'] != 'pretrained/'+c['backbone']+'_pretrain.pth': raise ValueError('Backbone and weights mismatch')
    if not c['resize_choices'] or any(not isinstance(v,int) or v<c['crop'] for v in c['resize_choices']): raise ValueError('Resize choices must fit crop')
    if c['lr']<=0 or c['max_estimated_hours']<=0: raise ValueError('Invalid learning rate/budget')
    return c
def fingerprint(c):
    # Dataset content is hashed separately. Absolute paths may change during cloud migration.
    clean={k:v for k,v in c.items() if k!='data_root'}
    files=[*sorted((ROOT/'aic').glob('*.py')),*sorted((ROOT/'uavseg').glob('*.py')),*sorted((ROOT/'vendor').rglob('*.py')),ROOT/'pipeline.py',ROOT/'requirements.txt',ROOT/'setup.sh',ROOT/'reference/source_config.json']
    return {'config':clean,'code':{str(p.relative_to(ROOT)):sha(p) for p in files},'split':sha(ROOT/'artifacts/domain_split.json'),'manifest':sha(ROOT/'artifacts/data_manifest.json')}

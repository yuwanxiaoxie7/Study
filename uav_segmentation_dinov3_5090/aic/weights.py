"""Authorized/local weights; resumable downloads without logging signed URLs."""
import hashlib,os,re,time,urllib.request,urllib.error,warnings
from pathlib import Path
from .common import ROOT,sha,read,save

def weight_path(c):
    chosen=os.environ.get('DINOV3_WEIGHTS') or c.get('pretrained')
    return Path(chosen).expanduser().resolve() if chosen and Path(chosen).is_absolute() else ROOT/(chosen or f'pretrained/dinov3_vitl16_{c["weight_source"]}.pth')

def load_checked(model,path):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('Expected official backbone state_dict, not training checkpoint')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        model.load_state_dict(state,strict=True)
    return state

def validate_weight(path,c):
    import torch
    from .model import build_backbone
    with torch.device('meta'): model=build_backbone(c['weight_source'])
    state=load_checked(model,path)
    for name,value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite pretrained tensor: '+name)

def download(url,target,attempts=5):
    if not url.startswith('https://'): raise ValueError('Use an official authorized HTTPS link')
    partial=target.with_suffix('.partial'); meta=partial.with_suffix('.partial.json')
    identity=hashlib.sha256(url.split('?',1)[0].encode()).hexdigest()
    info=read(meta) if meta.exists() else {}
    if info.get('resource')!=identity and partial.exists():
        raise RuntimeError('Partial belongs to another resource; move it aside explicitly')
    for attempt in range(attempts):
        offset=partial.stat().st_size if partial.exists() else 0
        headers={'User-Agent':'DINOv3-UAV/1.0','Accept-Encoding':'identity'}
        if offset:
            headers['Range']=f'bytes={offset}-'
            if info.get('etag'): headers['If-Range']=info['etag']
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as response:
                if 'text/' in response.headers.get('Content-Type',''): raise ValueError('Download returned a webpage')
                status=response.status
                if status==206:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=offset: raise ValueError('Invalid resume range')
                    total=int(match[3])
                    if info.get('etag') and response.headers.get('ETag')!=info['etag']: raise ValueError('Remote file changed during resume')
                elif status==200:
                    offset=0; total=int(response.headers.get('Content-Length',0))
                else: raise ValueError('Unexpected download response')
                info={'resource':identity,'etag':response.headers.get('ETag'),'total':total}; save(meta,info)
                started=time.monotonic(); done=offset; last=0
                with partial.open('ab' if offset else 'wb') as out:
                    while True:
                        data=response.read(1024*1024)
                        if not data: break
                        out.write(data); done+=len(data); now=time.monotonic()
                        if now-last>=3:
                            speed=(done-offset)/max(now-started,.001)
                            print(f'Weights {done/2**20:.1f} MiB / {total/2**20:.1f} MiB; {done/total:.1%}; {speed/2**20:.2f} MiB/s; ETA {(total-done)/max(speed,1)/60:.1f} min' if total else f'Weights {done/2**20:.1f} MiB',flush=True)
                            last=now
                if total and done!=total: raise OSError('Incomplete response')
                return partial
        except urllib.error.HTTPError as e:
            if e.code in (401,403,404): raise RuntimeError(f'Weight server HTTP {e.code}; check authorized URL/access. URL withheld.') from None
            if e.code==416 and info.get('total')==offset: return partial
        except (OSError,TimeoutError): pass
        if attempt+1<attempts:
            print('Download interrupted; retrying with retained partial.',flush=True); time.sleep(3)
    raise RuntimeError('Download failed after retries; partial retained, URL withheld')

def ensure_weights(c):
    target=weight_path(c); target.parent.mkdir(parents=True,exist_ok=True)
    receipt=ROOT/'artifacts'/f'weights_{c["weight_source"]}.json'
    expected=os.environ.get('DINOV3_SHA256')
    if not target.exists():
        url=os.environ.get('DINOV3_WEIGHT_URL')
        if not url: raise RuntimeError('Provide authorized DINOV3_WEIGHT_URL or local DINOV3_WEIGHTS. See README. No training started.')
        partial=download(url,target)
        digest=sha(partial)
        if expected and digest.lower()!=expected.lower(): raise ValueError('Weight SHA256 mismatch; partial retained')
        validate_weight(partial,c); partial.replace(target)
    digest=sha(target)
    if expected and digest.lower()!=expected.lower(): raise ValueError('Weight SHA256 mismatch')
    previous=read(receipt) if receipt.exists() else {}
    if previous.get('sha256')!=digest or previous.get('source')!=c['weight_source']:
        validate_weight(target,c)
    save(receipt,{'sha256':digest,'source':c['weight_source'],'bytes':target.stat().st_size,'strict_load':True,
                  'hash_note':'Local content digest; externally supplied SHA256 checked only when DINOV3_SHA256 set'})
    print(f'Weights verified/reused: {target.name}; SHA256 {digest}',flush=True)
    return digest

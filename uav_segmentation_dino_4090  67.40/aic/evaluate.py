import io,time,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import LABELS,save,sha
from .data import tensor,encode
from .precision import autocast

def confusion(pred,target):
    pred=np.asarray(pred).reshape(-1); target=np.asarray(target).reshape(-1)
    valid=target!=255
    return np.bincount(8*target[valid].astype(np.int64)+pred[valid],minlength=64).reshape(8,8)

def metrics(cm):
    tp=np.diag(cm); union=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,union,out=np.full(8,np.nan),where=union>0)
    return {'miou':float(np.nanmean(iou)) if np.any(union) else None,
            'iou':{n:float(v) if np.isfinite(v) else None for n,v in zip(LABELS,iou)},'confusion':cm.tolist()}

def positions(length,tile,stride):
    if length<tile: raise ValueError('Image smaller than tile')
    return sorted(set([*range(0,length-tile+1,stride),length-tile]))

@torch.inference_mode()
def predict(model,image,c):
    model.eval(); w,h=image.size; tile=c['tile']; device=next(model.parameters()).device
    scores=torch.zeros((8,h,w),device=device,dtype=torch.float32); weights=torch.zeros((h,w),device=device)
    hann=torch.hann_window(tile,periodic=False,device=device).clamp_min(.05)
    hann=hann[:,None]*hann[None,:]
    for y in positions(h,tile,c['stride']):
        for x in positions(w,tile,c['stride']):
            batch=tensor(image.crop((x,y,x+tile,y+tile))).unsqueeze(0).to(device)
            with autocast(c,device.type):
                p=model(batch)[0].float()
            scores[:,y:y+tile,x:x+tile]+=p*hann
            weights[y:y+tile,x:x+tile]+=hann
    if not torch.isfinite(scores).all() or not (weights>0).all(): raise RuntimeError('Invalid prediction/fusion')
    return (scores/weights).argmax(0).byte().cpu().numpy()

def evaluate(model,names,c,label,output=None):
    cm=np.zeros((8,8),np.int64); start=time.monotonic(); root=Path(c['data_root'])
    for i,name in enumerate(names,1):
        with Image.open(root/c['images']/name) as f: im=f.convert('RGB')
        with Image.open(root/c['masks']/name) as f: target=encode(f)
        cm+=confusion(predict(model,im,c),target)
        if i%50==0 or i==len(names):
            result=metrics(cm); elapsed=time.monotonic()-start
            if output:
                save(Path(output).parents[1]/'progress.json',{'stage':'validation','dataset':label,'images_done':i,'images_total':len(names),'cumulative_miou':result['miou'],'partial':i!=len(names),'remaining_minutes':(len(names)-i)*elapsed/i/60})
            print(f'{label} {i}/{len(names)} | 累计 mIoU {result["miou"]:.2%} | 剩余 {(len(names)-i)*elapsed/i/60:.1f} 分钟',flush=True)
    result=metrics(cm); result.update(images=len(names),seconds=time.monotonic()-start)
    if output: save(output,result)
    return result

def validate_zip(path,names):
    expected=set(names)
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(expected) or set(z.namelist())!=expected: raise ValueError('Submission filenames mismatch')
        if z.testzip() is not None: raise ValueError('ZIP checksum failed')
        for name in z.namelist():
            raw=z.read(name)
            # PNG IHDR bit depth 8, color type 0 = true grayscale; reject palette/RGB.
            if raw[:8]!=b'\x89PNG\r\n\x1a\n' or raw[24:26]!=bytes([8,0]): raise ValueError('Not 8-bit grayscale PNG')
            with Image.open(io.BytesIO(raw)) as im:
                a=np.asarray(im)
                if im.mode!='L' or im.size!=(1024,1024) or a.ndim!=2 or a.max()>8: raise ValueError('Invalid mask')
    return {'images':len(expected),'errors':[],'sha256':sha(path),'format':'8-bit grayscale PNG, 1024x1024, IDs 0..8'}

def submit(model,c,run,checkpoint):
    out=run/'submission'; out.mkdir(exist_ok=True); final=out/'submission.zip'; tmp=out/'submission.partial.zip'
    start=time.monotonic()
    with zipfile.ZipFile(Path(c['data_root'])/c['test_zip']) as source,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as dest:
        names=sorted(n for n in source.namelist() if n.lower().endswith('.png'))
        basenames=[Path(n).name for n in names]
        if len(names)!=500 or len(set(basenames))!=500: raise ValueError('Expected 500 test images')
        for i,(entry,name) in enumerate(zip(names,basenames),1):
            with source.open(entry) as f,Image.open(f) as im:
                if im.size!=(1024,1024): raise ValueError('Unexpected test dimensions')
                mask=predict(model,im.convert('RGB'),c)+np.uint8(1)
            stream=io.BytesIO(); Image.fromarray(mask).save(stream,format='PNG'); dest.writestr(name,stream.getvalue())
            if i%25==0 or i==len(names): print(f'测试集预测 {i}/{len(names)} | 剩余 {(len(names)-i)*(time.monotonic()-start)/i/60:.1f} 分钟',flush=True)
    report=validate_zip(tmp,basenames); report['checkpoint_sha256']=sha(checkpoint); report['single_model']=True
    tmp.replace(final); save(out/'validation.json',report)
    print(f'提交文件（检查通过）: {final.resolve()}',flush=True)
    return final

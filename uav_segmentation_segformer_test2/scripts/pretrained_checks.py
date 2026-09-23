"""Real official B4/B5 architecture and weights, bounded CPU smoke tests."""
import gc
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.common import ROOT, config, environment, save
from aic.weights import ensure_weights
from aic.model import Segmenter
from aic.losses import loss_fn


def main():
    torch.set_num_threads(4); records=[]
    for arch in ('b4','b5'):
        c=config(ROOT/f'configs/{arch}_offset.json'); receipt=ensure_weights(c)
        start=time.monotonic(); model=Segmenter(c).train()
        x=torch.randn(1,3,64,64); y=torch.randint(0,8,(1,64,64))
        loss=loss_fn(model(x),y,c); loss.backward()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        for name,p in model.named_parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise RuntimeError('Missing/nonfinite gradient: '+name)
        records.append({'backbone':arch,'head':'offset','official_weight_sha256':receipt['sha256'],
              'loading':model.load_report,'parameters':sum(p.numel() for p in model.parameters()),
              'input':[1,3,64,64],'device':'cpu','forward_backward_finite':True,'seconds':time.monotonic()-start,
              'note':'Synthetic 64x64 smoke only, no optimizer update and no competition training.'})
        print(records[-1],flush=True)
        del model,x,y,loss; gc.collect()
    save(ROOT/'checks/pretrained_models.json', {'environment':environment(),'models':records,'passed':True})


if __name__ == '__main__': main()

"""Real 300M DINOv3-L, random initialization: structure/gradient test, NOT pretrained validation."""
import sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root))
import torch
from aic.common import config
from aic.model import Segmenter
torch.set_num_threads(4); torch.manual_seed(2026)
started=time.monotonic(); c=config()
model=Segmenter(c,initialize=False).train()
x=torch.randn(1,3,48,64); y=torch.randint(0,8,(1,48,64)); y[:,:2]=255
losses,_=model(x,y); loss=sum(losses.values()); loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is None for n,p in model.backbone.named_parameters() if not p.requires_grad)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if p.requires_grad)
assert any(p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('qkv.b'))
with torch.no_grad(): assert model.eval()(x).shape==(1,8,48,64)
print(json.dumps({'real_backbone':'DINOv3 ViT-L/16 SAT structure','parameters':sum(p.numel() for p in model.parameters()),
 'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'loss':loss.item(),
 'seconds':time.monotonic()-started,'pretrained_weights':'NOT tested; random initialization','gpu':'NOT tested'}))

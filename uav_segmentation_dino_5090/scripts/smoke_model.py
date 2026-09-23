"""CPU-only structural smoke test; no downloads or training of the dataset."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from aic.common import ROOT,read
from aic.model import Segmenter
torch.set_num_threads(4)
c=read(ROOT/'configs/train.json')
model=Segmenter(c,initialize=False).train()
x=torch.randn(1,3,56,56)
y=torch.randint(0,8,(1,56,56)); y[:,:4]=255
losses,_=model(x,y)
loss=sum(losses.values()); loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
model.eval()
with torch.no_grad(): output=model(x)
assert output.shape==(1,8,56,56) and torch.isfinite(output).all()
print(json.dumps({'total_parameters':sum(p.numel() for p in model.parameters()),
    'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
    'loss':float(loss.detach()),'shape':list(output.shape),'device':'CPU',
    'pretrained_loaded':False,'note':'Random initialization; structural test, not accuracy measurement'}))

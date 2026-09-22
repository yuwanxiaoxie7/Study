"""Exercise real training controller using a tiny CPU model and synthetic data."""
import contextlib,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from aic import train as engine
from aic.common import ROOT,read

class Tiny(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__(); self.conv=torch.nn.Conv2d(3,8,1); self.drop=torch.nn.Dropout(.2)
    def cuda(self): return self
    def forward(self,x,y=None,return_train_prediction=False):
        scores=self.conv(self.drop(x))
        return {'ce':torch.nn.functional.cross_entropy(scores,y)}, scores.detach() if return_train_prediction else None

class Scaler:
    def __init__(self,**kwargs): pass
    def scale(self,loss): return loss
    def unscale_(self,opt): pass
    def step(self,opt): opt.step()
    def update(self): pass
    def get_scale(self): return 1.
    def is_enabled(self): return False
    def state_dict(self): return {}
    def load_state_dict(self,state): pass

class Writer:
    def __init__(self,*args,**kwargs): pass
    def add_scalar(self,*args): pass
    def close(self): pass

class ResumeTest(unittest.TestCase):
    def test_validation_interruption_and_best_selection(self):
        c=read(ROOT/'configs/train.json'); c.update(epochs=2,accumulation=2,validate_every_epochs=1,train_miou_every_updates=1,checkpoint_every_updates=1)
        split={'train':['a','b','c','d'],'val':['v'],'domain_val':['d']}
        def loader(config,epoch,names):
            for name in names:
                g=torch.Generator().manual_seed(ord(name)+epoch)
                yield torch.rand((1,3,4,4),generator=g),torch.zeros((1,4,4),dtype=torch.long)
        def evaluate(model,names,config,label,output):
            score=.6 if '01_' in str(output) else .5
            return {'miou':score,'iou':{'Background':score}}
        with tempfile.TemporaryDirectory() as directory,contextlib.ExitStack() as stack:
            for target,replacement in [
                ('aic.train.gpu',lambda:None),('aic.train.Segmenter',Tiny),('aic.train.loader',loader),
                ('aic.train.split',lambda:split),('aic.train.evaluate',evaluate),
                ('torch.Tensor.cuda',lambda self,**kw:self),('aic.train.autocast',lambda c:contextlib.nullcontext()),
                ('aic.train.make_scaler',lambda c:Scaler()),('torch.cuda.get_rng_state_all',lambda:[]),
                ('torch.cuda.set_rng_state_all',lambda states:None),('torch.cuda.max_memory_allocated',lambda:0)]:
                stack.enter_context(patch(target,replacement))
            stack.enter_context(patch.dict('sys.modules',{'torch.utils.tensorboard':types.SimpleNamespace(SummaryWriter=Writer)}))
            full=Path(directory)/'full'; resumed=Path(directory)/'resumed'; full.mkdir(); resumed.mkdir()
            engine.train(c,full)
            with patch('aic.train.evaluate',side_effect=RuntimeError('simulated interruption during validation')):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'): engine.train(c,resumed)
            state=torch.load(resumed/'last.pt',weights_only=False)
            self.assertEqual((state['epoch'],state['cursor'],state['updates']),(0,4,2))
            engine.train(c,resumed)
            a=torch.load(full/'last.pt',weights_only=False); b=torch.load(resumed/'last.pt',weights_only=False)
            self.assertEqual((a['epoch'],a['updates']),(2,4)); self.assertEqual(a['updates'],b['updates'])
            for name,tensor in a['model'].items(): self.assertTrue(torch.equal(tensor,b['model'][name]),name)
            best=torch.load(resumed/'best.pt',weights_only=False); self.assertEqual(best['epoch'],1)
            self.assertEqual(best['metrics']['domain_val']['miou'],.6)
            mid=Path(directory)/'mid'; mid.mkdir()
            def interrupted_loader(config,epoch,names):
                for i,batch in enumerate(loader(config,epoch,names)):
                    if i==2: raise RuntimeError('mid epoch interruption')
                    yield batch
            with patch('aic.train.loader',interrupted_loader):
                with self.assertRaisesRegex(RuntimeError,'mid epoch interruption'): engine.train(c,mid)
            state=torch.load(mid/'last.pt',weights_only=False)
            self.assertEqual(state['cursor'],2)
            engine.train(c,mid)
            state=torch.load(mid/'last.pt',weights_only=False)
            for name,tensor in a['model'].items(): self.assertTrue(torch.equal(tensor,state['model'][name]),name)

if __name__=='__main__': unittest.main()

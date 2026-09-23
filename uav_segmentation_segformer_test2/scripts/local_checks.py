"""Bounded CPU tests with synthetic data; never competition training."""
import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.common import ROOT, config, save
from aic.data import encode, TestImages
from aic.model import Segmenter, OffsetLearning, Head
from aic.losses import loss_fn
from aic.evaluate import confusion, metrics, positions, predict, submit, validate_zip
from aic.engine import train, checkpoint_load, seed_all


class SyntheticCrops(torch.utils.data.Dataset):
    def __init__(self, c, epoch, names): self.names, self.epoch = names, epoch
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        g = torch.Generator().manual_seed(int(self.names[i]) + 100 * self.epoch)
        return torch.randn(3, 16, 16, generator=g), torch.randint(0, 8, (16, 16), generator=g)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Dropout2d(.2))
        self.head = nn.Conv2d(16, 8, 1)
    def forward(self, x): return self.head(self.backbone(x))


class LocalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_configs(self):
        for path in (ROOT / 'configs').glob('*.json'):
            c = config(path)
            self.assertEqual(c['expected_test_count'], 1300)
            self.assertEqual(c['test_zip'], 'test_2.zip')

    def test_label_metrics_and_losses(self):
        self.assertEqual(encode(np.array([[0,1,8]], dtype=np.uint8)).tolist(), [[255,0,7]])
        with self.assertRaises(ValueError): encode(np.array([[9]], dtype=np.uint8))
        cm = confusion(np.array([0,0,7,3]), np.array([0,1,7,255]))
        self.assertAlmostEqual(metrics(cm)['miou'], .5)
        self.assertEqual(int(cm.sum()), 3)
        x = torch.randn(1,8,12,12, requires_grad=True)
        y = torch.randint(0,8,(1,12,12)); y[:,0,:] = 255
        c = config(); c.update(dice_weight=.3, lovasz_weight=.2)
        loss_fn(x,y,c).backward(); self.assertTrue(torch.isfinite(x.grad).all())
        x.grad = None; z = loss_fn(x, torch.full_like(y,255), c)
        self.assertEqual(float(z), 0.); z.backward(); self.assertEqual(float(x.grad.abs().sum()), 0.)

    def test_offset_equation_and_gradient(self):
        m = OffsetLearning(32)
        x = torch.randn(2,32,5,7, requires_grad=True)
        got = m(x)
        f = x.flatten(2).transpose(1,2); base = m.cls_repr.repeat(2,1,1)
        scores = torch.einsum('bnc,bkc->bkn', f, base)
        cr = base + (scores.softmax(2) @ f) @ m.cls_offset_proj.weight.T
        fr = f + (scores.softmax(1).transpose(1,2) @ base) @ m.feat_offset_proj.weight.T
        oracle = nn.functional.layer_norm(torch.einsum('bnc,bkc->bnk',fr,cr), (8,), m.mask_norm.weight,m.mask_norm.bias,m.mask_norm.eps)
        oracle = oracle.transpose(1,2).reshape_as(got)
        torch.testing.assert_close(got, oracle)
        got.square().mean().backward()
        for p in m.parameters():
            self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all())

    def test_encoder_matches_huggingface_and_checkpoint_gradients(self):
        c = config(); c['checkpoint_blocks'] = False
        model = Segmenter(c, initialize=False, tiny=True).eval()
        x = torch.randn(1,3,64,96)
        with torch.no_grad():
            expected = model.backbone(x, output_hidden_states=True).hidden_states
            for a,b in zip(model.features(x), expected): torch.testing.assert_close(a,b)
        checkpointed = copy.deepcopy(model); checkpointed.c = {**c, 'checkpoint_blocks':True}
        for m in (model,checkpointed):
            m.train(); torch.manual_seed(17); m(x).square().mean().backward()
        for p,q in zip(model.parameters(), checkpointed.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)
        model.eval()
        with torch.no_grad(): self.assertEqual(model(torch.randn(1,3,65,79)).shape, (1,8,65,79))

    def test_all_heads_backward(self):
        c = config()
        for name in ('mlp','fpn','ppm','msfe'):
            for offset in (False,True):
                head = Head({**c, 'head':name, 'offset':offset})
                features = [torch.randn(1,ch,h,h,requires_grad=True) for ch,h in ((64,16),(128,8),(320,4),(512,2))]
                y = head(features); self.assertEqual(y.shape,(1,8,16,16))
                y.square().mean().backward()
                for x in features: self.assertTrue(torch.isfinite(x.grad).all())

    def test_resume_exact_with_partial_accumulation(self):
        # Seven samples, batch=2, accumulation=2: final group has three samples.
        c = {**config(), 'batch_size':2, 'accumulation':2, 'workers':0, 'epochs':2,
             'validate_every':2, 'save_every':1, 'log_every':1, 'precision':'fp32'}
        s = {'train':list(map(str, range(7))), 'val':['val'], 'domain_val':['domain']}
        seed_all(77); initial = TinyNet()
        a, b = copy.deepcopy(initial), copy.deepcopy(initial)
        with tempfile.TemporaryDirectory() as tmp, patch('aic.engine.split',return_value=s), patch('aic.engine.Crops',SyntheticCrops), patch('aic.engine.evaluate', return_value={'miou':.25}), patch('aic.engine.class_weights',return_value=None):
            root = Path(tmp); full, resumed = root/'full', root/'resume'; full.mkdir(); resumed.mkdir()
            self.assertTrue(train(a,c,full,{'test':'synthetic'}))
            self.assertFalse(train(b,{**c,'stop_after_updates':1},resumed,{'test':'synthetic'}))
            self.assertTrue(train(b,c,resumed,{'test':'synthetic'},resume=True))
            for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,atol=0,rtol=0)
            final = checkpoint_load(resumed/'last.pt')
            self.assertEqual(final['state']['update'],4); self.assertEqual(final['state']['epoch'],2)
            with self.assertRaises(RuntimeError): train(b,c,resumed,{'test':'wrong'},resume=True)

    def test_sliding_and_submission_recovery(self):
        self.assertEqual(positions(103,32,20),[0,20,40,60,71])
        self.assertEqual(positions(7,32,20),[0])
        c = {**config(), 'image_size':40, 'tile':32, 'stride':17, 'expected_test_count':3, 'visualizations':0}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c['data_root']=tmp
            with zipfile.ZipFile(root/'test_2.zip','w') as z:
                for i in range(3):
                    out=io.BytesIO(); Image.fromarray(np.full((40,40,3),i*60,dtype=np.uint8)).save(out,format='PNG')
                    z.writestr(f'test_2/{i}.png',out.getvalue())
            m=TinyNet().eval(); img=Image.new('RGB',(40,40))
            self.assertEqual(predict(m,img,c).shape,(40,40))
            checkpoint=root/'fake.pt'; checkpoint.write_bytes(b'synthetic test identifier, not a model')
            result=submit(m,c,root/'run',checkpoint,{'synthetic':True})
            self.assertEqual(validate_zip(result,['0.png','1.png','2.png'],40)['images'],3)
            (root/'run/submission_test2/png/1.png').write_bytes(b'corrupted')
            submit(m,c,root/'run',checkpoint,{'synthetic':True})
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(RuntimeError): submit(m,c,root/'run',checkpoint,{'synthetic':True})
            with self.assertRaises(ValueError): TestImages({**c,'expected_test_count':1300})


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LocalChecks))
    save(ROOT/'checks/unit_tests.json', {'tests':result.testsRun, 'failures':len(result.failures),
         'errors':len(result.errors), 'passed':result.wasSuccessful(), 'data':'synthetic CPU only',
         'full_competition_training':False})
    sys.exit(0 if result.wasSuccessful() else 1)

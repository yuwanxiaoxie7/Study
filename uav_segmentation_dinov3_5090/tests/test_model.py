import copy,unittest,io,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
from PIL import Image
from aic.model import QVLoRA,CheckpointBlock,Decoder,feature_maps,build_backbone
from aic.common import ROOT,config,NORMALIZATION,fingerprint
from aic.data import tensor,split
from aic.evaluate import predict,validate_zip
from aic.train import lr_schedule,optimizer
from dinov3.models.vision_transformer import DinoVisionTransformer
from dinov3.layers.attention import LinearKMaskedBias

torch.set_num_threads(2)

class ModelTests(unittest.TestCase):
    def test_qv_mask_and_gradient(self):
        base=LinearKMaskedBias(32,96,bias=True)
        base.bias_mask.fill_(1); base.bias_mask[32:64]=0
        x=torch.randn(2,7,32); reference=base(x).detach(); adapted=QVLoRA(base,4,8)
        self.assertTrue(torch.equal(reference,adapted(x)))
        adapted(x).square().mean().backward()
        self.assertIsNone(base.weight.grad); self.assertGreater(adapted.b.grad.abs().sum().item(),0)
        with torch.no_grad(): adapted.b.normal_()
        self.assertTrue(torch.equal(reference[...,32:64],adapted(x)[...,32:64]))

    def test_official_features_checkpoint_and_rope(self):
        net=DinoVisionTransformer(img_size=64,patch_size=16,embed_dim=32,depth=4,num_heads=4,
                n_storage_tokens=4,mask_k_bias=True,pos_embed_rope_dtype='fp32',layerscale_init=1.)
        net.init_weights(); net.eval(); net.requires_grad_(False)
        x=torch.randn(1,3,43,61); padded=torch.nn.functional.pad(x,(0,3,0,5),mode='replicate')
        expected=net.get_intermediate_layers(padded,n=[0,1,2,3],reshape=True)
        actual=feature_maps(net,x,[0,1,2,3])
        for a,b in zip(actual,expected): self.assertTrue(torch.equal(a,b)); self.assertEqual(a.shape,(1,32,3,4))
        net.blocks[-1].attn.qkv=QVLoRA(net.blocks[-1].attn.qkv,4,8)
        net.blocks[-1]=CheckpointBlock(net.blocks[-1]); net.blocks[-1].enabled=True
        feature_maps(net,x,[0,1,2,3])[-1].square().mean().backward()
        grad=net.blocks[-1].block.attn.qkv.b.grad
        self.assertIsNotNone(grad); self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(grad.abs().sum().item(),0)

    def test_upernet_non_square_gradients(self):
        net=Decoder(32,32,True)
        features=[torch.randn(1,32,3,4,requires_grad=True) for _ in range(4)]
        image=torch.randn(1,3,43,61,requires_grad=True)
        output,edge=net(features,image,True)
        self.assertEqual(output.shape,(1,8,43,61)); self.assertEqual(edge.shape,(1,1,43,61))
        (output.square().mean()+edge.square().mean()).backward()
        for value in [image,*features]: self.assertTrue(torch.isfinite(value.grad).all()); self.assertGreater(value.grad.abs().sum(),0)

    def test_weight_variants_structure(self):
        with torch.device('meta'):
            sat=build_backbone('sat493m'); web=build_backbone('lvd1689m')
        self.assertEqual(sat.n_blocks,24); self.assertEqual(sat.embed_dim,1024); self.assertEqual(sat.patch_size,16)
        self.assertEqual(sat.n_storage_tokens,4)
        self.assertIn('local_cls_norm.weight',sat.state_dict())
        self.assertNotIn('local_cls_norm.weight',web.state_dict())

    def test_normalization(self):
        image=Image.new('RGB',(4,4),(128,64,32))
        for source,(mean,std) in NORMALIZATION.items():
            result=tensor(image,{'weight_source':source})
            expected=(torch.tensor([128,64,32])/255-torch.tensor(mean))/torch.tensor(std)
            self.assertTrue(torch.allclose(result[:,0,0],expected))

    def test_configs_schedule_and_split(self):
        for file in (ROOT/'configs').glob('*.json'): config(file)
        c=config(); self.assertEqual(len(split()['train']),5178)
        self.assertAlmostEqual(lr_schedule(c,0,100),.2)
        self.assertAlmostEqual(lr_schedule(c,5,100),1.)
        self.assertAlmostEqual(lr_schedule(c,99,100),.01)
        alternate={**c,'pretrained':'elsewhere.pth','max_estimated_hours':99}
        self.assertEqual(fingerprint(c),fingerprint(alternate))
        self.assertNotEqual(fingerprint(c),fingerprint({**c,'lr':.001}))

    def test_predict_and_zip(self):
        class Constant(torch.nn.Module):
            def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.zeros(()))
            def forward(self,x):
                result=x.new_zeros((len(x),8,*x.shape[-2:])); result[:,7]=1; return result
        c={**config(),'tile':32,'stride':16,'tta':'hflip'}
        result=predict(Constant(),Image.new('RGB',(93,77)),c)
        self.assertEqual(result.shape,(77,93)); self.assertTrue((result==7).all())
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'result.zip'
            for color,valid in [(1,True),(8,True),(0,False),(9,False)]:
                stream=io.BytesIO(); Image.new('L',(1024,1024),color).save(stream,format='PNG')
                with zipfile.ZipFile(file,'w') as z: z.writestr('001.png',stream.getvalue())
                if valid: validate_zip(file,['001.png'])
                else:
                    with self.assertRaises(ValueError): validate_zip(file,['001.png'])

    def test_optimizer_rates(self):
        net=torch.nn.Module(); net.backbone=QVLoRA(torch.nn.Linear(16,48)); net.decoder=Decoder(32,32)
        opt=optimizer(net,config()); groups={id(p):g for g in opt.param_groups for p in g['params']}
        self.assertAlmostEqual(groups[id(net.backbone.a)]['lr'],2e-5)
        self.assertAlmostEqual(groups[id(net.decoder.classifier.weight)]['lr'],2e-4)
        self.assertEqual(groups[id(net.decoder.classifier.bias)]['weight_decay'],0)

if __name__=='__main__': unittest.main()

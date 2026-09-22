import io,sys,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from aic.common import ROOT,read,config
from aic.data import encode,split,Crops
from aic.model import Decoder,LoRALinear,feature_maps
from aic.evaluate import predict,validate_zip,metrics,confusion
from uavseg.losses import segmentation_loss

class DinoTests(unittest.TestCase):
    def test_decoder_shapes_and_gradients(self):
        for kind in ['spatial_pyramid']:
            decoder=Decoder(32,16,kind)
            features=[torch.randn(1,32,3,4,requires_grad=True) for _ in range(4)]
            out=decoder(features,torch.randn(1,3,39,53))
            self.assertEqual(out.shape,(1,8,39,53))
            target=torch.randint(0,8,(1,39,53)); target[:,:3]=-1
            loss=segmentation_loss(out,target,torch.ones(8),{'kind':'ce','dice_weight':.3})
            loss.backward()
            for f in features: self.assertTrue(torch.isfinite(f.grad).all())

    def test_lora_matches_initial_linear(self):
        original=torch.nn.Linear(16,48); x=torch.randn(2,7,16)
        expected=original(x).detach(); model=LoRALinear(original,4,8)
        self.assertTrue(torch.equal(expected,model(x)))
        model(x).square().mean().backward()
        self.assertIsNone(model.base.weight.grad)
        self.assertGreater(model.b.grad.abs().sum().item(),0)

    def test_official_features_and_checkpoint_backward(self):
        # Exercise real official ViT with small dimensions, including register tokens.
        sys.path.insert(0,str(ROOT/'vendor/dinov2'))
        from dinov2.models.vision_transformer import DinoVisionTransformer
        for registers in [0,4]:
            net=DinoVisionTransformer(img_size=56,patch_size=14,embed_dim=32,depth=4,num_heads=4,block_chunks=0,num_register_tokens=registers)
            net.eval(); net.requires_grad_(False)
            x=torch.randn(1,3,43,55)
            padded=torch.nn.functional.pad(x,(0,1,0,13),mode='replicate')
            expected=net.get_intermediate_layers(padded,n=[0,1,2,3],reshape=True)
            actual=feature_maps(net,x,[0,1,2,3],False)
            for a,b in zip(actual,expected): self.assertTrue(torch.allclose(a,b,atol=1e-6))
            net.blocks[-1].attn.qkv=LoRALinear(net.blocks[-1].attn.qkv,4,8)
            actual=feature_maps(net,x,[0,1,2,3],True)
            actual[-1].square().mean().backward()
            self.assertIsNotNone(net.blocks[-1].attn.qkv.b.grad)
            self.assertTrue(torch.isfinite(net.blocks[-1].attn.qkv.b.grad).all())

    def test_ignore(self):
        scores=torch.randn(1,8,7,9,requires_grad=True)
        target=torch.full((1,7,9),-1)
        loss=segmentation_loss(scores,target,torch.ones(8),{'kind':'ce','dice_weight':.3})
        loss.backward(); self.assertEqual(loss.item(),0)
        self.assertEqual(scores.grad.abs().sum().item(),0)
        self.assertEqual(encode(np.arange(9,dtype=np.uint8)[None]).tolist(),[[255,0,1,2,3,4,5,6,7]])

    def test_split_config(self):
        self.assertEqual(len(split()['train']),5178)
        for file in (ROOT/'configs').glob('*.json'):
            c=config(file,str(ROOT)); self.assertEqual(c['batch_size']*c['accumulation'],8)

    def test_sliding(self):
        class Constant(torch.nn.Module):
            def __init__(self): super().__init__(); self.marker=torch.nn.Parameter(torch.zeros(1))
            def forward(self,x):
                out=x.new_zeros((len(x),8,*x.shape[-2:]));out[:,7]=2;return out
        output=predict(Constant(),Image.new('RGB',(93,77)),{'tile':28,'stride':14})
        self.assertTrue((output==7).all())
        self.assertEqual(output.shape,(77,93))

    def test_zip_rejects_wrong_format(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'submission.zip'
            for mode,size,color,valid in [('L',(1024,1024),8,True),('L',(1024,1024),9,False),('RGB',(1024,1024),0,False),('L',(512,512),1,False)]:
                data=io.BytesIO();Image.new(mode,size,color).save(data,format='PNG')
                with zipfile.ZipFile(file,'w') as z:z.writestr('001.png',data.getvalue())
                if valid: self.assertEqual(validate_zip(file,['001.png'])['errors'],[])
                else:
                    with self.assertRaises(ValueError):validate_zip(file,['001.png'])

    def test_optimizer_groups(self):
        from aic.train import optimizer
        model=torch.nn.Module(); model.backbone=LoRALinear(torch.nn.Linear(8,8),2,4);model.decoder=torch.nn.Linear(8,8)
        c=read(ROOT/'configs/train.json');opt=optimizer(model,c)
        rates={id(p):g['lr'] for g in opt.param_groups for p in g['params']}
        self.assertAlmostEqual(rates[id(model.backbone.a)],3e-5)
        self.assertAlmostEqual(rates[id(model.decoder.weight)],3e-4)
        self.assertNotIn(id(model.backbone.base.weight),rates)

if __name__=='__main__':unittest.main()

"""DINOv2 + LoRA + old spatial/detail adapters; selectable lightweight fusion.

SegFormer-inspired depth fusion is independently implemented, not the MiT model.
All four DINO feature maps have the same patch-grid resolution.
"""
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import ROOT, read
from uavseg.adapters import SpatialAdapter
from uavseg.losses import segmentation_loss

SPECS = {'dinov2_vits14': (384, 12, [2,5,8,11]),
         'dinov2_vitl14': (1024, 24, [5,11,17,23])}

class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.base = linear.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, linear.in_features))
        self.b = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b)*self.scale

def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(8,co),nn.GELU())

class Decoder(nn.Module):
    def __init__(self, dim, width, kind):
        super().__init__()
        self.kind = kind
        self.spatial = nn.ModuleDict({str(i):SpatialAdapter(dim) for i in [2,3]})
        self.project = nn.ModuleList([nn.Conv2d(dim,width,1) for _ in range(4)])
        if kind == 'segformer':
            self.fuse = nn.Sequential(nn.Conv2d(4*width,width,1,bias=False),nn.GroupNorm(8,width),nn.GELU())
        elif kind == 'legacy':
            self.refine = nn.ModuleList([block(width,width) for _ in range(4)])
        else: raise ValueError('Unsupported decoder')
        self.detail = nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(32,width,3,stride=2,padding=1),nn.GELU())
        self.classifier = nn.Conv2d(width,8,1)
    def forward(self, features, x):
        h,w = x.shape[-2:]
        features = list(features)
        for key, module in self.spatial.items(): features[int(key)] = module(features[int(key)])
        features = [proj(f) for proj,f in zip(self.project,features)]
        if self.kind == 'segformer':
            # Pointwise fusion at the native grid avoids expanding 4x channels
            # to quarter resolution. Interpolation cannot create new details.
            out = self.fuse(torch.cat(features,dim=1))
            out = F.interpolate(out,size=(max(1,h//4),max(1,w//4)),mode='bilinear',align_corners=False)
        else:
            out = None
            for i in range(3,-1,-1):
                f = F.interpolate(features[i],size=(max(1,h//(4*2**i)),max(1,w//(4*2**i))),mode='bilinear',align_corners=False)
                if out is not None: f = f+F.interpolate(out,size=f.shape[-2:],mode='bilinear',align_corners=False)
                out = self.refine[i](f)
        out = out+F.interpolate(self.detail(x),size=out.shape[-2:],mode='bilinear',align_corners=False)
        return F.interpolate(self.classifier(out),size=(h,w),mode='bilinear',align_corners=False)

def feature_maps(backbone, x, layers, use_checkpoint):
    h,w = x.shape[-2:]
    padded = F.pad(x,(0,(-w)%14,0,(-h)%14),mode='replicate')
    tokens = backbone.prepare_tokens_with_masks(padded)
    result = []
    for i, item in enumerate(backbone.blocks):
        trainable = any(p.requires_grad for p in item.parameters())
        if use_checkpoint and torch.is_grad_enabled() and trainable:
            tokens = checkpoint(item,tokens,use_reentrant=False)
        else: tokens = item(tokens)
        if i in layers:
            features = backbone.norm(tokens)[:,1+backbone.num_register_tokens:]
            result.append(features.transpose(1,2).reshape(x.shape[0],-1,padded.shape[-2]//14,padded.shape[-1]//14))
    if len(result)!=4: raise ValueError('Expected four DINO depth features')
    return result

class Segmenter(nn.Module):
    def __init__(self,c,initialize=True):
        super().__init__()
        self.c = c
        sys.path.insert(0,str(ROOT/'vendor/dinov2'))
        from dinov2.hub import backbones
        dim,depth,self.layers = SPECS[c['backbone']]
        self.backbone = getattr(backbones,c['backbone'])(pretrained=False)
        if initialize:
            state = torch.load(ROOT/c['pretrained'],map_location='cpu',weights_only=True)
            self.backbone.load_state_dict(state,strict=True)
            del state
            print('Official DINOv2 backbone strict load: 100% keys and shapes matched',flush=True)
        self.backbone.requires_grad_(False)
        for item in list(self.backbone.blocks)[-c['lora_last_blocks']:]:
            item.attn.qkv = LoRALinear(item.attn.qkv,c['lora_rank'],c['lora_alpha'])
        self.decoder = Decoder(dim,c['width'],c['decoder'])
        # Old project's inverse-square-root weights, TRAIN split only.
        import numpy as np
        chosen = set(read(ROOT/'artifacts/domain_split.json')['train'])
        counts = np.array([r['pixels'][1:] for r in read(ROOT/'artifacts/data_manifest.json') if r['name'] in chosen],dtype=float).sum(0)
        if (counts<=0).any(): raise ValueError('Training split lacks a class')
        weights = np.clip(np.sqrt(np.median(counts)/counts),.5,3.)
        self.register_buffer('class_weights',torch.tensor(weights/weights.mean(),dtype=torch.float32))
    def train(self,mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
    def forward(self,x,target=None,return_train_prediction=False):
        features = feature_maps(self.backbone,x,self.layers,self.c['activation_checkpoint'] and self.training)
        scores = self.decoder(features,x)
        if target is None: return scores
        mapped = torch.where(target==255,-1,target)
        with torch.autocast(device_type=x.device.type,enabled=False):
            loss = segmentation_loss(scores.float(),mapped,self.class_weights,{'kind':'ce','dice_weight':.3})
        return {'ce_dice':loss},scores.detach() if return_train_prediction else None

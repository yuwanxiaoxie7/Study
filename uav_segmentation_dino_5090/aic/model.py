"""DINOv2 with a coarse-to-fine spatial-prior decoder (independent implementation).
The backbone's four depth features share one patch grid, not four native scales.
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
from .objectives import boundary_loss, iou_surrogate

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
    def __init__(self, dim, width, kind='spatial_pyramid', boundary=False):
        super().__init__()
        if kind != 'spatial_pyramid': raise ValueError('Only spatial_pyramid is supported')
        self.spatial = nn.ModuleDict({str(i):SpatialAdapter(dim) for i in [2,3]})
        self.project = nn.ModuleList([nn.Conv2d(dim,width,1) for _ in range(4)])
        self.detail = nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(32,width,3,stride=2,padding=1),nn.GroupNorm(8,width),nn.GELU())
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,width),nn.GELU()) for _ in range(3)])
        self.refine = nn.ModuleList([block(width,width) for _ in range(4)])
        # Sigmoid gates start near zero; global semantic features dominate initially.
        self.prior_gate = nn.Parameter(torch.full((4,),-2.0))
        self.classifier = nn.Conv2d(width,8,1)
        self.boundary_head = nn.Conv2d(width,1,1) if boundary else None
    def forward(self, features, x, return_boundary=False):
        h,w = x.shape[-2:]
        features = list(features)
        for key, module in self.spatial.items(): features[int(key)] = module(features[int(key)])
        features = [proj(f) for proj,f in zip(self.project,features)]
        priors = [self.detail(x)]
        for down in self.down: priors.append(down(priors[-1]))
        out = None
        for i in range(3,-1,-1):
            f = F.interpolate(features[i],size=priors[i].shape[-2:],mode='bilinear',align_corners=False)
            f = f + self.prior_gate[i].sigmoid()*priors[i]
            if out is not None: f = f+F.interpolate(out,size=f.shape[-2:],mode='bilinear',align_corners=False)
            out = self.refine[i](f)
        scores = F.interpolate(self.classifier(out),size=(h,w),mode='bilinear',align_corners=False)
        if return_boundary:
            if self.boundary_head is None: raise RuntimeError('Boundary head is disabled')
            edges = F.interpolate(self.boundary_head(out),size=(h,w),mode='bilinear',align_corners=False)
            return scores, edges
        return scores

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
        self.decoder = Decoder(dim,c['width'],c['decoder'],boundary=c.get('boundary_weight',0)>0)
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
        use_boundary = target is not None and self.c.get('boundary_weight',0)>0
        decoded = self.decoder(features,x,return_boundary=use_boundary)
        scores, edges = decoded if use_boundary else (decoded,None)
        if target is None: return scores
        mapped = torch.where(target==255,-1,target)
        with torch.autocast(device_type=x.device.type,enabled=False):
            losses = {'ce_dice':segmentation_loss(scores.float(),mapped,self.class_weights,{'kind':'ce','dice_weight':.3})}
            if self.c.get('lovasz_weight',0)>0:
                losses['lovasz'] = self.c['lovasz_weight']*iou_surrogate(scores,mapped)
            if use_boundary:
                losses['boundary'] = self.c['boundary_weight']*boundary_loss(edges,mapped)
        return losses,scores.detach() if return_train_prediction else None

"""Official DINOv3-L/16 + Q/V LoRA + independent lightweight UPerNet-style head."""
import math
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import ROOT, read
from .objectives import boundary_loss, iou_surrogate
from uavseg.losses import segmentation_loss
sys.path.insert(0, str(ROOT/'vendor/dinov3'))

class QVLoRA(nn.Module):
    """Preserve official masked K bias; adapt only Q/V slices of fused projection."""
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        if self.out_features != 3*self.in_features: raise ValueError('Expected fused QKV')
        self.a = nn.Parameter(torch.empty(2, rank, self.in_features))
        self.b = nn.Parameter(torch.zeros(2, self.in_features, rank))
        for a in self.a: nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.scale = alpha/rank
    def forward(self, x):
        q = F.linear(F.linear(x, self.a[0]), self.b[0])*self.scale
        v = F.linear(F.linear(x, self.a[1]), self.b[1])*self.scale
        return self.base(x) + torch.cat((q, torch.zeros_like(q), v), dim=-1)

class CheckpointBlock(nn.Module):
    def __init__(self, block):
        super().__init__(); self.block = block; self.enabled = False
    def forward(self, x, rope=None):
        if self.enabled and torch.is_grad_enabled():
            return checkpoint(self.block, x, rope, use_reentrant=False)
        return self.block(x, rope)

def conv(ci, co, kernel=3, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, kernel, stride, kernel//2, bias=False), nn.GroupNorm(4, co), nn.GELU())

class Decoder(nn.Module):
    """PPM + FPN + all-level fusion; CNN priors supply actual spatial details.
    Independent UPerNet-inspired implementation, not a paper-exact reproduction.
    """
    def __init__(self, dim, width=128, boundary=False):
        super().__init__()
        self.detail = nn.Sequential(conv(3, 32, stride=2), conv(32, width, stride=2))
        self.down = nn.ModuleList([conv(width, width, stride=2) for _ in range(3)])
        self.project = nn.ModuleList([nn.Conv2d(dim, width, 1) for _ in range(4)])
        self.prior_gate = nn.Parameter(torch.full((4,), -2.))
        self.pool_sizes = (1, 2, 3, 6)
        self.pools = nn.ModuleList([conv(width, width//4, 1) for _ in self.pool_sizes])
        self.context = conv(width*2, width)
        self.smooth = nn.ModuleList([conv(width, width) for _ in range(3)])
        self.refine = conv(width*4, width)
        self.classifier = nn.Conv2d(width, 8, 1)
        self.boundary_head = nn.Conv2d(width, 1, 1) if boundary else None
    def forward(self, features, image, return_boundary=False):
        if len(features) != 4: raise ValueError('Expected four depth features')
        priors = [self.detail(image)]
        for down in self.down: priors.append(down(priors[-1]))
        levels = [F.interpolate(p(f), size=s.shape[-2:], mode='bilinear', align_corners=False)
                  + self.prior_gate[i].sigmoid()*s
                  for i, (p, f, s) in enumerate(zip(self.project, features, priors))]
        deep = levels[-1]
        pooled = [F.interpolate(p(F.adaptive_avg_pool2d(deep, k)), size=deep.shape[-2:], mode='bilinear', align_corners=False)
                  for k, p in zip(self.pool_sizes, self.pools)]
        levels[-1] = self.context(torch.cat([deep, *pooled], 1))
        for i in (2, 1, 0):
            levels[i] = self.smooth[i](levels[i] + F.interpolate(levels[i+1], size=levels[i].shape[-2:], mode='bilinear', align_corners=False))
        merged = self.refine(torch.cat([F.interpolate(f, size=levels[0].shape[-2:], mode='bilinear', align_corners=False) for f in levels], 1))
        scores = F.interpolate(self.classifier(merged), size=image.shape[-2:], mode='bilinear', align_corners=False)
        if return_boundary:
            if self.boundary_head is None: raise ValueError('Boundary head disabled')
            return scores, F.interpolate(self.boundary_head(merged), size=image.shape[-2:], mode='bilinear', align_corners=False)
        return scores

def feature_maps(backbone, image, layers):
    h, w = image.shape[-2:]
    image = F.pad(image, (0, (-w)%16, 0, (-h)%16), mode='replicate')
    return backbone.get_intermediate_layers(image, n=layers, reshape=True, norm=True)

def build_backbone(source):
    from dinov3.hub.backbones import dinov3_vitl16, Weights
    return dinov3_vitl16(pretrained=False, weights=Weights.SAT493M if source=='sat493m' else Weights.LVD1689M)

class Segmenter(nn.Module):
    def __init__(self, c, initialize=True):
        super().__init__(); self.c = c
        self.backbone = build_backbone(c['weight_source'])
        if initialize:
            from .weights import weight_path, load_checked
            load_checked(self.backbone, weight_path(c))
            print('Official DINOv3-L/16: strict 100% keys/shapes matched', flush=True)
        self.backbone.requires_grad_(False)
        for i in range(24-c['lora_last_blocks'], 24):
            block = self.backbone.blocks[i]
            block.attn.qkv = QVLoRA(block.attn.qkv, c['lora_rank'], c['lora_alpha'])
            self.backbone.blocks[i] = CheckpointBlock(block)
        self.decoder = Decoder(1024, c['width'], boundary=c['boundary_weight']>0)
        chosen = set(read(ROOT/'artifacts/domain_split.json')['train'])
        counts = np.array([r['pixels'][1:] for r in read(ROOT/'artifacts/data_manifest.json') if r['name'] in chosen], dtype=float).sum(0)
        if (counts <= 0).any(): raise ValueError('Training split lacks a class')
        weights = np.clip(np.sqrt(np.median(counts)/counts), .5, 3.)
        self.register_buffer('class_weights', torch.tensor(weights/weights.mean(), dtype=torch.float32))
    def train(self, mode=True):
        super().train(mode); self.backbone.eval()
        for block in self.backbone.blocks:
            if isinstance(block, CheckpointBlock): block.enabled = mode and self.c['activation_checkpoint']
        return self
    def forward(self, x, target=None, return_train_prediction=False):
        features = feature_maps(self.backbone, x, self.c['feature_layers'])
        edges_enabled = target is not None and self.c['boundary_weight']>0
        decoded = self.decoder(features, x, return_boundary=edges_enabled)
        scores, edges = decoded if edges_enabled else (decoded, None)
        if target is None: return scores
        mapped = torch.where(target==255, -1, target)
        with torch.autocast(device_type=x.device.type, enabled=False):
            losses = {'ce_dice': segmentation_loss(scores.float(), mapped, self.class_weights, {'kind':'ce', 'dice_weight':.3})}
            if self.c['lovasz_weight']>0: losses['lovasz'] = self.c['lovasz_weight']*iou_surrogate(scores, mapped)
            if edges_enabled: losses['boundary'] = self.c['boundary_weight']*boundary_loss(edges, mapped)
        return losses, scores.detach() if return_train_prediction else None

"""HF MiT encoder + MMSeg-style head. GN is an explicit small-batch adaptation."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import SegformerConfig, SegformerModel
from .weights import load_backbone, weight_path


def norm(channels, kind):
    return nn.GroupNorm(min(32, channels), channels) if kind == 'gn' else nn.BatchNorm2d(channels)


def conv(ci, co, kernel=1, kind='gn'):
    return nn.Sequential(nn.Conv2d(ci, co, kernel, padding=kernel // 2, bias=False), norm(co, kind), nn.ReLU())


def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class OffsetLearning(nn.Module):
    """OffSeg authors' dual offset equations, nn-only port; see THIRD_PARTY.md."""
    def __init__(self, channels, classes=8):
        super().__init__()
        self.cls_repr = nn.Parameter(torch.empty(1, classes, channels))
        self.cls_offset_proj = nn.Linear(channels, channels, bias=False)
        self.feat_offset_proj = nn.Linear(channels, channels, bias=False)
        self.mask_norm = nn.LayerNorm(classes)
        nn.init.trunc_normal_(self.cls_repr, std=.02)
        nn.init.trunc_normal_(self.cls_offset_proj.weight, std=.02)
        nn.init.trunc_normal_(self.feat_offset_proj.weight, std=.02)

    def forward(self, x):
        b, c, h, w = x.shape
        features = x.flatten(2).transpose(1, 2)
        classes = self.cls_repr.expand(b, -1, -1)
        # FP32 softmax/matmul for numerical stability; gradients still reach both projections.
        with torch.autocast(device_type=x.device.type, enabled=False):
            features, classes = features.float(), classes.float()
            coupled = (features @ classes.transpose(1, 2)).transpose(1, 2)
            classes = classes + self.cls_offset_proj(coupled.softmax(-1) @ features)
            original_classes = self.cls_repr.float().expand(b, -1, -1)
            features = features + self.feat_offset_proj(coupled.softmax(1).transpose(1, 2) @ original_classes)
            logits = self.mask_norm(features @ classes.transpose(1, 2))
        return logits.transpose(1, 2).reshape(b, -1, h, w)


class ContextPPM(nn.Module):
    """Ordinary PPM, explicitly NOT the paper's SFA-PPM."""
    def __init__(self, width, kind):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(width, width // 4, 1), nn.ReLU()) for s in (1, 2, 3, 6)])
        self.fuse = conv(width * 2, width, 3, kind)

    def forward(self, x):
        return self.fuse(torch.cat([x] + [resize(m(x), x.shape[-2:]) for m in self.branches], 1))


class AxisLocalAttention(nn.Module):
    """ELA-inspired axis pooling + depthwise 1D convolution; engineering adaptation."""
    def __init__(self, width):
        super().__init__()
        self.filter = nn.Sequential(nn.Conv1d(width, width, 7, padding=3, groups=width, bias=False), nn.GroupNorm(16, width))

    def forward(self, x):
        rows = self.filter(x.mean(3)).sigmoid().unsqueeze(3)
        cols = self.filter(x.mean(2)).sigmoid().unsqueeze(2)
        return x * rows * cols


class Head(nn.Module):
    def __init__(self, c, channels=(64, 128, 320, 512)):
        super().__init__()
        width, kind = c['head_width'], c['head_norm']
        self.kind, self.pool = c['head'], c['head_pool']
        self.project = nn.ModuleList([conv(ci, width, kind=kind) for ci in channels])
        self.fuse = conv(width * 4, width, kind=kind)
        if self.kind != 'mlp':
            self.refine = nn.ModuleList([conv(width, width, 3, kind) for _ in range(4)])
        self.context = ContextPPM(width, kind) if self.kind in ('ppm', 'msfe') else nn.Identity()
        self.local = nn.ModuleList([AxisLocalAttention(width) if self.kind == 'msfe' else nn.Identity() for _ in range(3)])
        self.dropout = nn.Dropout2d(.1)
        self.classifier = OffsetLearning(width) if c['offset'] else nn.Conv2d(width, 8, 1)

    def forward(self, features):
        fs = [p(x) for p, x in zip(self.project, features)]
        if self.kind != 'mlp':
            fs[-1] = self.context(fs[-1])
            for i in range(2, -1, -1):
                fs[i] = self.local[i](fs[i]) + resize(fs[i + 1], fs[i].shape[-2:])
            fs = [m(x) for m, x in zip(self.refine, fs)]
        out = self.fuse(torch.cat([resize(x, fs[0].shape[-2:]) for x in fs], 1))
        if self.pool == 2: out = F.max_pool2d(out, 2)
        return self.classifier(self.dropout(out))


class Segmenter(nn.Module):
    def __init__(self, c, initialize=True, tiny=False):
        super().__init__()
        self.c = c
        cfg = SegformerConfig(depths=[3, 8, 27, 3] if c['backbone'] == 'b4' else [3, 6, 40, 3],
                              hidden_sizes=[64, 128, 320, 512], num_attention_heads=[1, 2, 5, 8],
                              sr_ratios=[8, 4, 2, 1], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                              mlp_ratios=[4, 4, 4, 4], drop_path_rate=.1, layer_norm_eps=1e-6)
        if tiny: cfg.depths = [1, 1, 1, 1]
        self.backbone = SegformerModel(cfg)
        self.load_report = load_backbone(self.backbone, weight_path(c)) if initialize else {'initialized': False}
        self.head = Head(c)

    def features(self, x):
        # Mirror HF encoder.forward, with non-reentrant block checkpointing.
        b = x.shape[0]; outputs = []
        enc = self.backbone.encoder
        for embedding, blocks, layer_norm in zip(enc.patch_embeddings, enc.block, enc.layer_norm):
            x, h, w = embedding(x)
            for block in blocks:
                if self.c['checkpoint_blocks'] and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x, h, w, False, use_reentrant=False)[0]
                else:
                    x = block(x, h, w, False)[0]
            x = layer_norm(x).reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs

    def forward(self, x):
        h, w = x.shape[-2:]
        padded = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode='replicate')
        logits = self.head(self.features(padded))
        return resize(logits, padded.shape[-2:])[..., :h, :w]

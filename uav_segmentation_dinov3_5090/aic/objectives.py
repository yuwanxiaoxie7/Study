"""Training-only edge supervision and valid-pixel IoU surrogate.

Independent edge target construction, inspired by boundary-supervised segmentation;
not the PIDNet architecture or its exact target-generation procedure.
"""
import torch
from torch.nn import functional as F
from uavseg.losses import lovasz


def edge_targets(target):
    """Mark both sides of 4-neighbor class transitions; ignore unknown neighborhoods."""
    valid = (target >= 0) & (target != 255)
    edge = torch.zeros_like(valid)
    horizontal = valid[:, :, 1:] & valid[:, :, :-1] & (target[:, :, 1:] != target[:, :, :-1])
    vertical = valid[:, 1:, :] & valid[:, :-1, :] & (target[:, 1:, :] != target[:, :-1, :])
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal
    edge[:, 1:, :] |= vertical
    edge[:, :-1, :] |= vertical
    # Exclude the ignore pixel and its surrounding 3x3 neighborhood.
    # Image exterior is not treated as a labeled class or as an ignore pixel.
    safe = F.max_pool2d((~valid).float().unsqueeze(1),3,1,1).squeeze(1) == 0
    return edge.float(), safe


def boundary_loss(logits, target):
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError('Boundary logits must be Bx1xHxW matching target')
    truth, safe = edge_targets(target)
    errors = F.binary_cross_entropy_with_logits(logits[:,0].float(),truth,reduction='none')
    positive = safe & (truth > 0)
    negative = safe & (truth == 0)
    # Normalize each present group separately. Empty/all-edge crops remain valid.
    terms = [errors[mask].mean() for mask in (positive,negative) if mask.any()]
    return torch.stack(terms).mean() if terms else logits.float().sum()*0


def iou_surrogate(logits, target):
    valid = (target >= 0) & (target != 255)
    probabilities = logits.float().softmax(1).permute(0,2,3,1)
    return lovasz(probabilities[valid],target[valid])

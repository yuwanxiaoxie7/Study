import torch
from torch.nn import functional as F


def lovasz(prob, target):
    terms = []
    for cls in range(8):
        truth = (target == cls).float()
        if not truth.any(): continue
        errors, order = (truth - prob[:, cls]).abs().sort(descending=True)
        fg = truth[order]; total = fg.sum()
        grad = 1 - (total - fg.cumsum(0)) / (total + (1 - fg).cumsum(0))
        grad = torch.cat((grad[:1], grad[1:] - grad[:-1]))
        terms.append(torch.dot(errors, grad))
    return torch.stack(terms).mean() if terms else prob.sum() * 0


def loss_fn(logits, target, c, weights=None):
    logits = logits.float(); valid = target != 255
    if not valid.any(): return logits.sum() * 0
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
    if c['dice_weight'] or c['lovasz_weight']:
        prob = logits.softmax(1).permute(0, 2, 3, 1)[valid]
        truth = target[valid]
        if c['lovasz_weight']: ce = ce + c['lovasz_weight'] * lovasz(prob, truth)
        if c['dice_weight']:
            hot = F.one_hot(truth, 8).float(); present = hot.sum(0) > 0
            dice = 1 - ((2 * (prob * hot).sum(0) + 1) / ((prob + hot).sum(0) + 1))[present].mean()
            ce = ce + c['dice_weight'] * dice
    return ce

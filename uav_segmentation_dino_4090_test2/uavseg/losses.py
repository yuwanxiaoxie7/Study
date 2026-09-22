import torch
from torch.nn import functional as F
def lovasz(prob,target):
    terms=[]
    for cls in range(8):
        truth=(target==cls).float()
        if not truth.any(): continue
        errors=(truth-prob[:,cls]).abs(); errors,order=torch.sort(errors,descending=True); fg=truth[order]
        total=fg.sum(); intersection=total-fg.cumsum(0); union=total+(1-fg).cumsum(0)
        grad=1-intersection/union
        grad=torch.cat([grad[:1],grad[1:]-grad[:-1]])
        terms.append(torch.dot(errors,grad))
    return torch.stack(terms).mean() if terms else prob.sum()*0
def segmentation_loss(logits,target,weights,cfg,details=None):
    logits=logits.float(); valid=target>=0
    if not valid.any():
        if details is not None: details.update(primary=0.,dice=0.,hard_ratio=0.)
        return logits.sum()*0
    ce=F.cross_entropy(logits,target,weight=weights,ignore_index=-1,reduction='none')
    if cfg['kind']=='ce': primary=ce[valid].sum()/weights[target[valid]].sum()
    elif cfg['kind']=='ohem':
        # Rank unweighted errors so class weights do not determine difficulty.
        fraction=float(cfg.get('hard_fraction',.25)); mix=float(cfg.get('hard_mix',.5))
        if not 0<fraction<=1 or not 0<=mix<=1: raise ValueError('Invalid OHEM fraction/mix')
        with torch.no_grad():
            difficulty=F.cross_entropy(logits.detach(),target,ignore_index=-1,reduction='none')[valid]
            count=max(1,int(difficulty.numel()*fraction))
            selected=difficulty.topk(count,sorted=False).indices
        effective=weights[target[valid]]; values=ce[valid]
        whole=values.sum()/effective.sum().clamp_min(1e-12)
        hard=values[selected].sum()/effective[selected].sum().clamp_min(1e-12)
        primary=(1-mix)*whole+mix*hard
        if details is not None: details['hard_ratio']=count/difficulty.numel()
    elif cfg['kind']=='focal':
        p=logits.softmax(1).gather(1,target.clamp_min(0).unsqueeze(1)).squeeze(1)
        primary=(((1-p)**cfg['focal_gamma'])*ce)[valid].sum()/weights[target[valid]].sum()
    elif cfg['kind']=='lovasz': primary=lovasz(logits.softmax(1).permute(0,2,3,1)[valid],target[valid])
    elif cfg['kind']=='ce_lovasz': primary=ce[valid].sum()/weights[target[valid]].sum()+lovasz(logits.softmax(1).permute(0,2,3,1)[valid],target[valid])
    else: raise ValueError('Unknown loss kind')
    truth=F.one_hot(target.clamp_min(0),8).permute(0,3,1,2).float()*valid[:,None]
    prob=logits.softmax(1)*valid[:,None]; present=truth.sum((0,2,3))>0
    dice=1-((2*(truth*prob).sum((0,2,3))[present]+1)/((truth+prob).sum((0,2,3))[present]+1)).mean()
    if details is not None:
        details.update(primary=float(primary.detach()),dice=float(dice.detach()))
        details.setdefault('hard_ratio',1.)
    return primary+cfg['dice_weight']*dice

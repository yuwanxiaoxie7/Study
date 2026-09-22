import json,math,random,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,save,fingerprint
from .data import Crops,split
from .evaluate import confusion,metrics,evaluate
from .model import Segmenter
from .precision import autocast,scaler as make_scaler,dtype

def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)

def atomic_checkpoint(path,state):
    tmp=path.with_suffix('.partial.pt'); torch.save(state,tmp); tmp.replace(path)

def optimizer(model,c):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        factor=c['backbone_lr_multiplier'] if name.startswith('backbone.') else 1.
        decay=0. if p.ndim==1 or any(k in name for k in ['norm','absolute_pos_embed','relative_position_bias_table','query_embed','query_feat','level_embed']) else c['weight_decay']
        groups.setdefault((factor,decay),[]).append(p)
    return torch.optim.AdamW([dict(params=p,lr=c['lr']*factor,lr_factor=factor,weight_decay=decay) for (factor,decay),p in groups.items()],betas=(.9,.999),eps=1e-8)

def loader(c,epoch,order):
    generator=torch.Generator().manual_seed(c['seed']+epoch)
    return torch.utils.data.DataLoader(Crops(c,epoch,order),batch_size=c['batch_size'],num_workers=c['workers'],pin_memory=True,shuffle=False,generator=generator)

def gpu():
    if not torch.cuda.is_available(): raise RuntimeError('需要云端 NVIDIA CUDA GPU；本机不启动训练。')
    p=torch.cuda.get_device_properties(0)
    if p.total_memory<20*1024**3: raise RuntimeError(f'{p.name} 显存不足20GB；请在租用的4090上运行。')
    return {'name':p.name,'memory_gib':p.total_memory/1024**3,'torch':torch.__version__,'cuda':torch.version.cuda}

def benchmark(c,run):
    info=gpu(); seed(c['seed']); model=Segmenter(c).cuda(); opt=optimizer(model,c); scaler=make_scaler(c)
    torch.cuda.reset_peak_memory_stats()
    count=c['benchmark_updates']; names=split()['train'][:(count+2)*c['batch_size']*c['accumulation']]
    model.train(); opt.zero_grad(set_to_none=True); times=[]; successful=0; step_start=time.monotonic()
    for i,(x,y) in enumerate(loader(c,0,names),1):
        x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
        with autocast(c): losses,_=model(x,y); loss=sum(losses.values())/c['accumulation']
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite benchmark loss')
        scaler.scale(loss).backward()
        if i%c['accumulation']==0:
            scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'])
            if not torch.isfinite(norm) and not scaler.is_enabled(): raise RuntimeError('Nonfinite gradient in BF16/FP32')
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
            successful+=int(torch.isfinite(norm))
            times.append(time.monotonic()-step_start); step_start=time.monotonic()
    if successful<3: raise RuntimeError('测速阶段有效优化更新不足3次；请检查AMP/梯度，不启动正式训练')
    validation=evaluate(model,split()['val'][:4],c,'测速验证')
    per_epoch=math.ceil(math.ceil(len(split()['train'])/c['batch_size'])/c['accumulation'])
    val_count=len([e for e in range(1,c['epochs']+1) if e%c['validate_every_epochs']==0 or e==c['epochs']])
    # Exclude two warmup updates; estimate is not a guaranteed wall-clock duration.
    sec_update=float(np.median(times[2:])); sec_image=validation['seconds']/4
    hours=1.2*(sec_update*per_epoch*c['epochs']+sec_image*((1049+769)*val_count+c['test_count']))/3600
    report={'hardware':info,'seconds_per_update':sec_update,'seconds_per_validation_image':sec_image,'updates_per_epoch':per_epoch,'epochs':c['epochs'],'estimated_hours':hours,'peak_gib':torch.cuda.max_memory_allocated()/1024**3,'fingerprint':fingerprint(c)}
    save(run/'benchmark.json',report)
    if report['peak_gib']>c['max_peak_gib']: raise RuntimeError('Peak memory exceeds budget; lower crop or width explicitly in a new run')
    print(f'测速：{c["epochs"]}轮 + {val_count}次双验证 + 提交预测，预计 {hours:.1f} 小时（估算）',flush=True)

def train(c,run):
    gpu(); seed(c['seed']); identity=fingerprint(c); last=run/'last.pt'; model=Segmenter(c,initialize=not last.exists()).cuda()
    opt=optimizer(model,c); scaler=make_scaler(c); epoch=cursor=updates=0; best=-1.; elapsed=0.; cm=np.zeros((8,8),np.int64)
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False) # Only this project's own checkpoints.
        if state['fingerprint']!=identity: raise RuntimeError('配置/代码/划分已改变，禁止混用旧断点；用 --new-run 创建独立实验。')
        model.load_state_dict(state['model'],strict=True); opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        epoch,cursor,updates,best,elapsed=[state[k] for k in ['epoch','cursor','updates','best','elapsed']]
        cm=np.asarray(state['train_cm'],dtype=np.int64)
        random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy']); torch.set_rng_state(state['rng_cpu']); torch.cuda.set_rng_state_all(state['rng_cuda'])
        del state
        print(f'恢复：第 {epoch+1}/{c["epochs"]} 轮，已读取 {cursor} 张，更新 {updates}',flush=True)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(run/'tensorboard'),purge_step=updates if last.exists() else None)
    train_names=split()['train']; per_epoch=math.ceil(math.ceil(len(train_names)/c['batch_size'])/c['accumulation']); total=per_epoch*c['epochs']
    started=time.monotonic()
    def snapshot():
        atomic_checkpoint(last,{'fingerprint':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'cursor':cursor,'updates':updates,'best':best,'elapsed':elapsed+time.monotonic()-started,'train_cm':cm.tolist(),'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()})
    def log(obj):
        with (run/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n')
        save(run/'progress.json',obj)
    try:
        while epoch<c['epochs']:
            order=train_names.copy(); random.Random(c['seed']+epoch).shuffle(order)
            remaining=order[cursor:]; batches=math.ceil(len(remaining)/c['batch_size']); model.train(); opt.zero_grad(set_to_none=True)
            since=time.monotonic(); samples=0; loss_window=0.
            for idx,(x,y) in enumerate(loader(c,epoch,remaining)):
                # A partial final accumulation group is normalized by its actual batch count.
                group_start=(idx//c['accumulation'])*c['accumulation']; group_size=min(c['accumulation'],batches-group_start)
                progress=updates/max(1,total); warm=min(1.,(updates+1)/max(1,c['warmup_updates']))
                for group in opt.param_groups: group['lr']=c['lr']*group['lr_factor']*((1-progress)**.9)*warm
                x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
                need_pred=(updates+1)%c['train_miou_every_updates']==0
                with autocast(c): losses,pred=model(x,y,need_pred); loss=sum(losses.values())
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss, epoch={epoch+1}, cursor={cursor}; last.pt 保持有效。')
                scaler.scale(loss/group_size).backward(); loss_window+=float(loss.detach()); cursor+=len(x); samples+=len(x)
                if pred is not None: cm+=confusion(pred.argmax(1).cpu().numpy(),y.cpu().numpy())
                if (idx+1)%c['accumulation']==0 or idx+1==batches:
                    scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'])
                    if not torch.isfinite(norm) and not scaler.is_enabled(): raise RuntimeError('Nonfinite gradient in BF16/FP32; checkpoint preserved')
                    old_scale=scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    if not torch.isfinite(norm):
                        # GradScaler safely skipped the overflowing update; record, and abort persistent overflow.
                        print(f'AMP overflow: scale {old_scale} -> {scaler.get_scale()}',flush=True)
                        if scaler.get_scale()<1: raise RuntimeError('Persistent AMP overflow; resume checkpoint after diagnosis')
                    updates+=1
                    if updates%20==0 or cursor==len(order):
                        train_metric=metrics(cm)['miou']; seconds=time.monotonic()-since
                        obj={'stage':'train','epoch':epoch+1,'epochs':c['epochs'],'cursor':cursor,'images':len(order),'updates':updates,'loss':loss_window/(idx+1),'sampled_train_miou':train_metric,'lr':opt.param_groups[0]['lr'],'peak_gib':torch.cuda.max_memory_allocated()/1024**3,'epoch_train_remaining_minutes':(len(order)-cursor)*seconds/max(samples,1)/60,'amp_scale':scaler.get_scale()}
                        log(obj); miou=f'{train_metric:.2%}' if train_metric is not None else '等待采样'
                        print(f'训练 {epoch+1}/{c["epochs"]}轮 | {cursor}/{len(order)} | loss {obj["loss"]:.4f} | 采样训练 mIoU {miou} | 本轮训练剩余 {obj["epoch_train_remaining_minutes"]:.1f}分',flush=True)
                        writer.add_scalar('train/loss',obj['loss'],updates)
                        if train_metric is not None: writer.add_scalar('train/sampled_miou',train_metric,updates)
                    if updates%c['checkpoint_every_updates']==0: snapshot()
            snapshot() # If interrupted during validation, resume re-runs validation without repeating this epoch.
            if (epoch+1)%c['validate_every_epochs']==0 or epoch+1==c['epochs']:
                results={}
                for key,title in [('val','固定随机验证'),('domain_val','场景域验证')]:
                    results[key]=evaluate(model,split()[key],c,title,run/f'validation/epoch_{epoch+1:02d}_{key}.json')
                    writer.add_scalar(f'{key}/miou',results[key]['miou'],epoch+1)
                score=results['domain_val']['miou']; improved=score>best
                if improved:
                    best=score; atomic_checkpoint(run/'best.pt',{'model':model.state_dict(),'fingerprint':identity,'epoch':epoch+1,'metrics':results})
                    save(run/'best_metrics.json',{'epoch':epoch+1,**results})
                print(f'完整验证 | {epoch+1}/{c["epochs"]}轮 | Random {results["val"]["miou"]:.2%} | Domain {score:.2%} | 最佳 Domain {best:.2%}'+(' [NEW BEST]' if improved else ''),flush=True)
                print(' | '.join(f'{k}: {v:.2%}' if v is not None else f'{k}: N/A' for k,v in results['domain_val']['iou'].items()),flush=True)
                log({'stage':'validation','epoch':epoch+1,'epochs':c['epochs'],'best_domain_miou':best,'metrics':results})
            epoch+=1; cursor=0; cm=np.zeros((8,8),np.int64); snapshot()
        save(run/'training_complete.json',{'epochs':epoch,'best_domain_miou':best,'checkpoint':'best.pt'})
    finally: writer.close()
    return model

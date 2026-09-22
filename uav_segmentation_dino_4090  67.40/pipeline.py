"""One entry: audit -> timed GPU check -> train/resume -> best checkpoint -> checked ZIP."""
import argparse,contextlib,datetime,json,os,platform,subprocess,sys,traceback,urllib.request
from pathlib import Path
from aic.common import ROOT,config,save,read,sha,fingerprint

BASE_URL='https://dl.fbaipublicfiles.com/dinov2'

class Tee:
    def __init__(self,console,file): self.console,self.file=console,file
    def write(self,s): self.console.write(s); self.file.write(s); self.file.flush()
    def flush(self): self.console.flush(); self.file.flush()
    def isatty(self): return False

@contextlib.contextmanager
def lock(path):
    import fcntl
    with path.open('a') as f:
        try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('已有同一工程训练进程，禁止重复启动。')
        try: yield
        finally: fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def pretrained(c):
    url=f'{BASE_URL}/{c["backbone"]}/{c["backbone"]}_pretrain.pth'
    target=ROOT/c['pretrained']; target.parent.mkdir(exist_ok=True)
    if target.name!=c['backbone']+'_pretrain.pth': raise ValueError('Official backbone/weights mismatch')
    receipt=target.with_suffix('.json')
    if target.exists() and receipt.exists():
        recorded=read(receipt)
        if recorded['sha256']!=sha(target) or recorded['url']!=url: raise RuntimeError('Pretrained file changed since previous verified download')
        return
    if not target.exists():
        tmp=target.with_suffix('.partial'); print(f'Downloading official {c["backbone"]} weights (L: about 1.2GB)',flush=True)
        with urllib.request.urlopen(url,timeout=120) as src,tmp.open('wb') as dst:
            while True:
                chunk=src.read(8*1024*1024)
                if not chunk: break
                dst.write(chunk)
        minimum=1_000_000_000 if c['backbone']=='dinov2_vitl14' else 80_000_000
        if tmp.stat().st_size<minimum: raise RuntimeError('Incomplete pretrained download')
        tmp.replace(target)
    digest=sha(target)
    save(receipt,{'url':url,'sha256':digest,'bytes':target.stat().st_size,'training_data':'LVD-142M','usage':'backbone only','hash_note':'Local integrity hash, not independently published upstream digest'})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root'); p.add_argument('--config',default=str(ROOT/'configs/train.json'))
    p.add_argument('--new-run',action='store_true',help='创建新实验并记录为活动实验；默认恢复活动实验')
    p.add_argument('--check-only',action='store_true',help='仅做云端兼容性/前后向/推理测速')
    p.add_argument('--benchmark-child',action='store_true',help=argparse.SUPPRESS); p.add_argument('--run-dir',help=argparse.SUPPRESS)
    p.add_argument('--skip-budget-limit',action='store_true',help='明确接受超过配置时长预算的估算，继续训练')
    a=p.parse_args(); c=config(a.config,a.data_root)
    if a.benchmark_child:
        from aic.train import benchmark
        benchmark(c,Path(a.run_dir)); return
    if platform.system()!='Linux': raise RuntimeError('4090工程请上传Linux云端，运行 bash run_all.sh；不在3050本机启动。')
    from aic.train import gpu
    hardware=gpu(); print(json.dumps(hardware,ensure_ascii=False),flush=True)
    (ROOT/'runs').mkdir(exist_ok=True)
    with lock(ROOT/'runs/pipeline.lock'):
        active=ROOT/'runs/active.json'
        if a.new_run or not active.exists():
            name=datetime.datetime.now().strftime('dino_%Y%m%d_%H%M%S_%f'); save(active,{'run':name})
        name=read(active)['run']
        if Path(name).name!=name or name in ['.','..']: raise ValueError('Invalid run name')
        run=ROOT/'runs'/name; run.mkdir(exist_ok=True)
        with (run/'console.log').open('a',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            try:
                print(f'当前实验: {run}\n计划 {c["epochs"]} 轮，每 {c["validate_every_epochs"]} 轮完整验证，单模型输出 ZIP。',flush=True)
                identity=fingerprint(c)
                if (run/'identity.json').exists() and read(run/'identity.json')!=identity: raise RuntimeError('代码/配置改变；请保持原配置恢复，或使用 --new-run')
                save(run/'identity.json',identity); save(run/'config.json',c)
                from aic.precision import dtype
                precision=str(dtype(c))
                if (run/'environment.json').exists() and read(run/'environment.json')['precision']!=precision:
                    raise RuntimeError('Resolved precision changed; use original hardware or a new run')
                save(run/'environment.json',{'hardware':hardware,'python':sys.version,'platform':platform.platform(),'precision':precision})
                import shutil
                for folder in ['aic','uavseg','configs','scripts']:
                    shutil.copytree(ROOT/folder,run/'source'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
                shutil.copy2(ROOT/'pipeline.py',run/'source/pipeline.py')
                with (run/'pip-freeze.txt').open('w',encoding='utf-8') as f: subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
                from aic.data import audit
                save(run/'status.json',{'stage':'audit'}); audit(c,run/'audit.json')
                # An already finished run is idempotent: verify its deliverable, then return.
                finished=run/'submission/validation.json'; finished_zip=run/'submission/submission.zip'
                if not a.check_only and (run/'training_complete.json').exists() and finished.exists() and finished_zip.exists():
                    import zipfile
                    from aic.evaluate import validate_zip
                    report=read(finished)
                    if report['sha256']==sha(finished_zip) and report['checkpoint_sha256']==sha(run/'best.pt'):
                        with zipfile.ZipFile(Path(c['data_root'])/c['test_zip']) as source:
                            names=[Path(n).name for n in source.namelist() if n.lower().endswith('.png')]
                        validate_zip(finished_zip,names)
                        save(run/'status.json',{'stage':'completed','submission':str(finished_zip),'best_checkpoint':str(run/'best.pt')})
                        save(ROOT/'latest_submission.json',{'submission':str(finished_zip),'run':str(run),'checkpoint':str(run/'best.pt')})
                        print(f'已完成且提交检查通过，无需重跑：{finished_zip}',flush=True); return
                pretrained(c)
                pretrained_receipt=read((ROOT/c['pretrained']).with_suffix('.json'))
                if (run/'pretrained.json').exists() and read(run/'pretrained.json')['sha256']!=pretrained_receipt['sha256']:
                    raise RuntimeError('Pretrained weights changed within the same experiment')
                save(run/'pretrained.json',pretrained_receipt)
                receipt=run/'benchmark.json'
                if not receipt.exists() or read(receipt)['fingerprint']!=identity or read(receipt)['hardware']!=hardware:
                    save(run/'status.json',{'stage':'benchmark'})
                    # Independent process releases benchmark weights, optimizer and CUDA allocator before formal training.
                    command=[sys.executable,'-u',str(ROOT/'pipeline.py'),'--benchmark-child','--config',str(Path(a.config).resolve()),'--data-root',c['data_root'],'--run-dir',str(run)]
                    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as child:
                        for line in child.stdout: print(line,end='',flush=True)
                        if child.wait()!=0: raise RuntimeError(f'GPU 检查失败，退出码 {child.returncode}。若OOM：crop改336或width改32后 --new-run；没有启动正式训练。')
                estimate=read(receipt)['estimated_hours']
                if a.check_only:
                    save(run/'status.json',{'stage':'check_passed','estimated_hours':estimate}); print('云端检查通过；重新运行原命令（去掉 --check-only）开始训练。',flush=True); return
                if estimate>c['max_estimated_hours'] and not a.skip_budget_limit:
                    raise RuntimeError(f'预计 {estimate:.1f} 小时超过设定 {c["max_estimated_hours"]} 小时。接受费用请追加 --skip-budget-limit；或调整配置并 --new-run。')
                from aic.train import train
                from aic.evaluate import submit
                import torch
                save(run/'status.json',{'stage':'train','epochs':c['epochs']})
                model=train(c,run)
                checkpoint=run/'best.pt'; state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if state['fingerprint']!=identity: raise RuntimeError('Best checkpoint does not match experiment')
                model.load_state_dict(state['model'],strict=True); del state
                save(run/'status.json',{'stage':'predict'})
                path=submit(model,c,run,checkpoint)
                save(run/'status.json',{'stage':'completed','submission':str(path),'best_checkpoint':str(checkpoint)})
                save(ROOT/'latest_submission.json',{'submission':str(path),'run':str(run),'checkpoint':str(checkpoint)})
                print(f'全部完成。只上传：{path}',flush=True)
            except BaseException as e:
                if type(e).__name__=='OutOfMemoryError':
                    print('显存不足：先降低 crop（14的倍数，例如336）或width，再 --new-run。保持有效batch=8；未自动改模型。',flush=True)
                save(run/'status.json',{'stage':'failed','type':type(e).__name__,'error':str(e),'resume':'使用相同命令重启；不要删除 runs 或添加 --new-run。'})
                traceback.print_exc(); raise

if __name__=='__main__': main()

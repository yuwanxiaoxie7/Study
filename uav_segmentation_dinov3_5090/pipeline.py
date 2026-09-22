"""Preparation/check/benchmark/train/resume/predict; no implicit full-training sweep."""
import argparse,contextlib,datetime,json,os,platform,shutil,subprocess,sys,traceback
from pathlib import Path
from aic.common import ROOT,config,read,save,sha,fingerprint

class Tee:
    def __init__(self,a,b): self.a,self.b=a,b
    def write(self,s): self.a.write(s); self.b.write(s); self.b.flush()
    def flush(self): self.a.flush(); self.b.flush()
    def isatty(self): return False

@contextlib.contextmanager
def lock(path):
    import fcntl
    with path.open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process is using this project') from None
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

def space_check():
    free=shutil.disk_usage(ROOT).free/2**30
    print(f'Project disk free: {free:.1f} GiB',flush=True)
    if free<8: raise RuntimeError('Need at least 8 GiB free for checkpoints and outputs; no files deleted')

def run_benchmark(c,run,config_path):
    command=[sys.executable,'-u',str(ROOT/'pipeline.py'),'--benchmark-child','--run-dir',str(run),
             '--config',str(config_path),'--data-root',c['data_root']]
    result=subprocess.run(command)
    if result.returncode: raise RuntimeError('Benchmark failed. No full training started; inspect log / explicitly choose batch1 or smaller config.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root'); p.add_argument('--config',default=str(ROOT/'configs/train.json'))
    p.add_argument('--new-run',action='store_true'); p.add_argument('--check-only',action='store_true')
    p.add_argument('--prepare-only',action='store_true',help='Prepare/verify weights without GPU; setup handled by run_all.sh')
    p.add_argument('--compare-profiles',action='store_true',help='Benchmark selected config and fixed alternatives, no training and no auto-selection')
    p.add_argument('--skip-budget-limit',action='store_true')
    p.add_argument('--benchmark-child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--run-dir',help=argparse.SUPPRESS)
    args=p.parse_args(); c=config(args.config,args.data_root)
    from aic.weights import ensure_weights
    if args.benchmark_child:
        from aic.train import benchmark
        # Parent verified content; digest is local and no secret link enters experiment configs.
        c['weight_sha256']=read(Path(args.run_dir)/'weights.json')['sha256']
        benchmark(c,Path(args.run_dir)); return
    if platform.system()!='Linux': raise RuntimeError('Use Linux cloud for pipeline; local tests work on Windows/CPU')
    (ROOT/'runs').mkdir(exist_ok=True)
    with lock(ROOT/'runs/pipeline.lock'):
        if args.prepare_only:
            if args.new_run or args.compare_profiles: p.error('prepare-only does not create experiments')
            ensure_weights(c); print('Preparation complete; no GPU required, no training started.'); return
        if not c['data_root']: p.error('--data-root required')
        subprocess.run([sys.executable,str(ROOT/'scripts/check_gpu.py')],check=True)
        from aic.train import gpu
        hardware=gpu(); space_check()
        # Checks/download fixes occur before selecting run identity.
        c['weight_sha256']=ensure_weights(c)
        from aic.data import audit
        if args.compare_profiles:
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            folder=ROOT/'runs'/('profiles_'+stamp); folder.mkdir()
            audit(c,folder/'audit.json'); reports=[]
            variants=[('selected',{}),('crop512',{'crop':512,'tile':512,'stride':256}),
                      ('batch1',{'batch_size':1,'accumulation':8}),('no_checkpoint',{'activation_checkpoint':False})]
            for name,changes in variants:
                variant={**c,**changes}; destination=folder/name; destination.mkdir()
                save(destination/'config.json',variant); save(destination/'weights.json',{'sha256':c['weight_sha256']})
                try:
                    run_benchmark(variant,destination,destination/'config.json')
                    reports.append({'profile':name,'status':'passed',**read(destination/'benchmark.json')})
                except RuntimeError:
                    reports.append({'profile':name,'status':'failed; see console output'})
            save(folder/'comparison.json',reports)
            print(f'Comparison only; no training/automatic selection. Results: {folder}',flush=True); return
        active=ROOT/'runs/active.json'
        if args.new_run or not active.exists():
            save(active,{'run':datetime.datetime.now().strftime('dinov3_%Y%m%d_%H%M%S_%f')})
        name=read(active)['run']
        if Path(name).name!=name or name in ('.','..'): raise ValueError('Invalid run path')
        run=ROOT/'runs'/name; run.mkdir(exist_ok=True)
        with (run/'console.log').open('a',encoding='utf-8',buffering=1) as file,contextlib.redirect_stdout(Tee(sys.stdout,file)),contextlib.redirect_stderr(Tee(sys.stderr,file)):
            try:
                print(f'Experiment: {run}; {c["epochs"]} epochs, validate every {c["validate_every_epochs"]}',flush=True)
                identity=fingerprint(c)
                if (run/'identity.json').exists() and read(run/'identity.json')!=identity:
                    raise RuntimeError('Training identity changed: use original training config or --new-run. Download-only changes do not require a new run.')
                save(run/'identity.json',identity); save(run/'config.json',c); save(run/'weights.json',{'sha256':c['weight_sha256']})
                from aic.precision import dtype
                precision=str(dtype(c))
                if (run/'environment.json').exists() and read(run/'environment.json')['precision']!=precision:
                    raise RuntimeError('Precision changed; use original precision or --new-run')
                save(run/'environment.json',{'hardware':hardware,'precision':precision,'python':sys.version,'platform':platform.platform()})
                with (run/'pip-freeze.txt').open('w',encoding='utf-8') as f: subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
                # Audit operation versions separately; never persist signed URLs/environment secrets.
                ops={n:sha(ROOT/n) for n in ['pipeline.py','setup.sh','aic/weights.py']}
                with (run/'operations.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({'time':datetime.datetime.now().isoformat(),'files':ops})+'\n')
                if not (run/'source').exists():
                    for folder in ['aic','uavseg','configs','scripts','vendor','reference']:
                        shutil.copytree(ROOT/folder,run/'source'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pth','*.pt'))
                    for file in ['pipeline.py','run_all.sh','setup.sh','requirements.txt']: shutil.copy2(ROOT/file,run/'source'/file)
                save(run/'status.json',{'stage':'audit'}); audit(c,run/'audit.json')
                from aic.evaluate import validate_zip
                result=run/'submission/submission.zip'; report=run/'submission/validation.json'
                if not args.check_only and (run/'training_complete.json').exists() and result.exists() and report.exists():
                    import zipfile
                    recorded=read(report)
                    if recorded['sha256']==sha(result) and recorded['checkpoint_sha256']==sha(run/'best.pt'):
                        with zipfile.ZipFile(Path(c['data_root'])/c['test_zip']) as z: names=[Path(n).name for n in z.namelist() if n.lower().endswith('.png')]
                        validate_zip(result,names)
                        save(ROOT/'latest_submission.json',{'submission':str(result),'run':str(run)})
                        print(f'Already complete: {result}',flush=True); return
                receipt=run/'benchmark.json'
                if not receipt.exists() or read(receipt)['fingerprint']!=identity or read(receipt)['hardware']!=hardware:
                    save(run/'status.json',{'stage':'benchmark'})
                    # Resolved config retains weight content identity for the subprocess.
                    run_benchmark(c,run,run/'config.json')
                measured=read(receipt)
                if max(measured['peak_gib'],measured['reserved_peak_gib'])>min(c['max_peak_gib'],measured['memory_limit_gib']):
                    raise RuntimeError('Benchmark memory budget failed; choose an explicit new configuration')
                estimate=measured['estimated_hours']
                if args.check_only:
                    save(run/'status.json',{'stage':'check_passed','estimated_hours':estimate})
                    print(f'Checks passed, estimate {estimate:.1f} hours. Remove --check-only (and --new-run) to train.',flush=True); return
                if estimate>c['max_estimated_hours'] and not args.skip_budget_limit:
                    raise RuntimeError(f'Estimated {estimate:.1f} h exceeds {c["max_estimated_hours"]} h. Explicit acceptance: --skip-budget-limit')
                space_check()
                from aic.train import train
                from aic.evaluate import submit
                import torch
                save(run/'status.json',{'stage':'train'})
                model=train(c,run)
                state=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
                if state['fingerprint']!=identity: raise RuntimeError('Best checkpoint mismatch')
                model.load_state_dict(state['model'],strict=True); del state
                save(run/'status.json',{'stage':'predict'})
                path=submit(model,c,run,run/'best.pt')
                save(run/'status.json',{'stage':'completed','submission':str(path)})
                save(ROOT/'latest_submission.json',{'submission':str(path),'run':str(run),'checkpoint':str(run/'best.pt')})
                print(f'Complete. Submit prediction ZIP: {path}',flush=True)
            except BaseException as e:
                save(run/'status.json',{'stage':'failed','type':type(e).__name__,'error':str(e),'resume':'Same config/command without --new-run'})
                traceback.print_exc(); raise

if __name__=='__main__': main()

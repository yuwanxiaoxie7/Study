import argparse
import contextlib
import datetime
import sys
from pathlib import Path
import torch
from aic.common import ROOT, config, save, read, fingerprint, environment, lock, Tee, object_sha
from aic.data import audit, split
from aic.weights import ensure_weights
from aic.model import Segmenter
from aic.engine import seed_all, train, benchmark, checkpoint_load
from aic.evaluate import evaluate, submit


def main():
    parser = argparse.ArgumentParser(description='Single-model SegFormer, official test_2.zip only')
    parser.add_argument('--config', default=str(ROOT / 'configs/b4_baseline.json'))
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--stage', choices=['check','train','validate','predict','all'], default='check')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--run', default=str(ROOT / 'runs/b4_baseline'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    c = config(args.config); c['data_root'] = str(Path(args.data_root).resolve())
    if args.stop_after_updates is not None:
        if args.stop_after_updates < 1: parser.error('--stop-after-updates must be positive')
        c['stop_after_updates'] = args.stop_after_updates
    if args.cpu_threads < 1: parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    stage = 'check' if args.check_only else args.stage
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if stage in ('train','all') and device.type != 'cuda': raise RuntimeError('Full training requires CUDA; use --check-only --device cpu locally')
    run = Path(args.run).resolve()
    if stage == 'check':
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = ROOT / 'checks' / stamp
    else:
        output = run
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / '.running.lock'), (output / 'terminal.log').open('a', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('阶段:', stage, '| 设备:', device, '| 输出:', output, flush=True)
            env = environment(); save(output / 'environment.json', env)
            report = audit(c, output / 'audit.json')
            weight = ensure_weights(c); identity = fingerprint(c, report, weight)
            if stage != 'check':
                existing = run / 'identity.json'
                if existing.exists() and read(existing) != identity:
                    raise RuntimeError('Run identity changed; use another run directory')
                if stage in ('train','all'):
                    if (run / 'last.pt').exists() and not args.resume:
                        raise RuntimeError('Existing checkpoint: use --resume or a new --run')
                    if args.resume and not (run / 'last.pt').exists(): raise FileNotFoundError('Resume requested without last.pt')
                if stage in ('validate','predict') and not (run / 'best.pt').exists(): raise FileNotFoundError('No best.pt; first train and validate')
                save(existing, identity)
            save(output / 'resolved_config.json', c)
            seed_all(c['seed'])
            model = Segmenter(c).to(device)
            save(output / 'weight_loading.json', model.load_report)
            if stage == 'check':
                # Bounded real-architecture/pretrained forward; GPU benchmark uses the actual configured crop.
                model.eval()
                with torch.inference_mode():
                    logits = model(torch.zeros(1, 3, 64, 64, device=device))
                    if logits.shape != (1, 8, 64, 64) or not torch.isfinite(logits).all(): raise RuntimeError('Structural forward failed')
                result = benchmark(model, c, output / 'benchmark.json')
                save(output / 'check_result.json', {'passed': True, 'pretrained_forward': True,
                     'identity_sha256': object_sha(identity), 'gpu_measured': result['gpu_measured'],
                     'training_budget_pass': result['budget_pass'], 'full_training_started': False})
                print('检查完成，未启动完整训练。GPU 实测:', result['gpu_measured'], '训练预算通过:', result['budget_pass'], flush=True)
                return
            if stage in ('train','all'):
                result = benchmark(model, c, output / 'benchmark.json')
                if not result['budget_pass']:
                    raise RuntimeError('Measured preflight exceeds configured time/memory budget; inspect benchmark.json and choose a lighter explicit config')
                completed = train(model, c, run, identity, args.resume)
                if not completed or stage == 'train': return
            loaded = checkpoint_load(run / 'best.pt')
            if loaded['identity'] != identity: raise RuntimeError('Best checkpoint identity mismatch')
            model.load_state_dict(loaded['model'], strict=True); del loaded
            if stage in ('validate','all'):
                for key in ('val','domain_val'):
                    evaluate(model, split()[key], c, run / 'metrics' / ('best_' + key + '.json'))
            if stage in ('predict','all'):
                submit(model, c, run, run / 'best.pt', identity)


if __name__ == '__main__':
    main()

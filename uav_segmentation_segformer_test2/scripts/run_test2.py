"""Fixed B4-offset submission workflow. No model work in this wrapper."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command_for(data_root, run, check_only=False, predict_only=False):
    cmd = [sys.executable, str(ROOT/'pipeline.py'), '--config', str(ROOT/'configs/b4_offset.json'),
           '--data-root', str(data_root), '--run', str(run)]
    if check_only: return cmd + ['--check-only']
    if predict_only or (run/'completed.json').is_file():
        if not (run/'best.pt').is_file(): raise FileNotFoundError('Prediction requires best.pt')
        return cmd + ['--stage', 'predict']
    if (run/'best.pt').exists() and not (run/'last.pt').exists():
        raise RuntimeError('best.pt exists without last.pt; use --predict-only to explicitly use this model')
    cmd += ['--stage', 'all']
    if (run/'last.pt').is_file(): cmd.append('--resume')
    return cmd


def main():
    parser = argparse.ArgumentParser(description='Only B4 + OffSeg: train/resume, validate, test_2, submission ZIP')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--run', default=str(ROOT/'runs/b4_offset'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true', help='Checks and bounded GPU benchmark only')
    modes.add_argument('--predict-only', action='store_true', help='Use existing best.pt without training')
    parser.add_argument('--dry-run', action='store_true', help='Print command only; no setup/model execution')
    args = parser.parse_args(); run = Path(args.run).resolve()
    cmd = command_for(Path(args.data_root).resolve(), run, args.check_only, args.predict_only)
    print('Only model: MiT-B4 + OffSeg | run:', run, flush=True)
    print('Command:', subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run: return
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode: raise SystemExit(result.returncode)
    if args.check_only:
        print('Checks finished. No full training or submission generation requested.', flush=True)
        return
    output = run/'submission_test2/submission_test2.zip'
    receipt = output.parent/'validation.json'
    if not output.is_file() or not receipt.is_file():
        raise RuntimeError('No verified submission ZIP produced; inspect pipeline log')
    print('\nCompetition submission ZIP:', output, flush=True)
    print('Validation receipt:', receipt, flush=True)


if __name__ == '__main__': main()

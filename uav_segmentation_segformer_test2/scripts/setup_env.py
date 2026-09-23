"""Idempotent Linux cloud environment setup; invoked by setup.sh using system Python."""
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info < (3,10) or sys.version_info >= (3,13):
        raise SystemExit('Use Python 3.10-3.12 (recommended 3.11)')
    if os.name == 'nt': raise SystemExit('Cloud installer targets Linux. See README for local Windows checks.')
    python = ROOT / '.venv/bin/python'
    if not python.exists():
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(ROOT/'.venv')],check=True)
    torch_check = '''
import torch
from packaging.version import Version
assert Version(torch.__version__.split('+')[0]) >= Version('2.7.1')
assert torch.version.cuda and Version(torch.version.cuda) >= Version('12.8')
'''
    if subprocess.run([str(python),'-c',torch_check], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        subprocess.run([str(python),'-m','pip','install','--progress-bar','on','torch==2.7.1','torchvision==0.22.1','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
    versions = '''
from importlib.metadata import version
from pathlib import Path
for line in Path('requirements.txt').read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        name,wanted=line.strip().split('=='); assert version(name)==wanted, name
'''
    if subprocess.run([str(python),'-c',versions],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        subprocess.run([str(python),'-m','pip','install','--progress-bar','on','-r',str(ROOT/'requirements.txt')],check=True)
    subprocess.run([str(python),'-c',
        "import torch,transformers; from transformers import SegformerModel; print('Torch',torch.__version__,'CUDA',torch.version.cuda,'Transformers',transformers.__version__); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable; CPU checks only')"],check=True)


if __name__ == '__main__': main()

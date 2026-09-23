import contextlib
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation', 'Agricultural', 'Vehicle']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def config(path=None, seen=None):
    path = Path(path or ROOT / 'configs/b4_baseline.json').resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError('Circular config inheritance')
    seen.add(path)
    c = read(path)
    if 'extends' in c:
        c = {**config(path.parent / c.pop('extends'), seen), **c}
    for key in ['crop', 'batch_size', 'accumulation', 'epochs', 'tile', 'stride', 'benchmark_updates', 'validate_every', 'save_every', 'log_every', 'head_width', 'image_size', 'expected_test_count']:
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Invalid ' + key)
    if c['backbone'] not in ('b4', 'b5') or c['head'] not in ('mlp', 'fpn', 'ppm', 'msfe'):
        raise ValueError('Unsupported architecture')
    if c['head_norm'] not in ('gn', 'bn') or c['head_pool'] not in (1, 2):
        raise ValueError('Invalid head normalization/pooling')
    if c['head_width'] % 32 or c['workers'] < 0 or c['visualizations'] < 0:
        raise ValueError('Invalid width/workers/visualizations')
    if c['stride'] > c['tile'] or any(s < c['crop'] for s in c['resize_choices']):
        raise ValueError('Invalid stride or resize range')
    if c['precision'] not in ('auto', 'bf16', 'fp16', 'fp32') or c['inference'] not in ('slide', 'whole'):
        raise ValueError('Invalid precision/inference')
    for key in ['backbone_lr', 'head_lr', 'grad_clip', 'max_hours', 'max_peak_gib']:
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid ' + key)
    for key in ['dice_weight', 'lovasz_weight', 'sampling_probability', 'weight_decay']:
        if not 0 <= c[key] <= 1:
            raise ValueError('Invalid ' + key)
    if not 0 < c['warmup_fraction'] < 1:
        raise ValueError('Invalid warmup')
    return c


def fingerprint(c, audit, weight):
    operational = {'data_root', 'test_zip', 'expected_test_count', 'workers', 'benchmark_updates', 'max_hours', 'max_peak_gib', 'visualizations', 'stop_after_updates'}
    return {
        'config': {k: v for k, v in c.items() if k not in operational},
        'weights': weight['sha256'],
        'split': audit['split_sha256'], 'manifest': audit['manifest_sha256'],
        'code': {p.name: sha(p) for p in sorted((ROOT / 'aic').glob('*.py')) if p.name not in ('weights.py',)},
    }


def environment():
    from importlib.metadata import version
    import torch
    import transformers
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'numpy': version('numpy'), 'pillow': version('Pillow'),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_total_gib': torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None}


@contextlib.contextmanager
def lock(path):
    """OS lock: releases on crashes; no stale sentinel to manually delete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self): return False

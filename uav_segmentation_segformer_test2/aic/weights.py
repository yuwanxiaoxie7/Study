"""Pinned NVIDIA ImageNet weights, upstream LFS SHA256, resumable HTTPS download."""
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from .common import ROOT, read, save, sha, lock

SOURCES = {
    'b4': {'revision': '3844ddaa13d9ce98816bacb904688a5284f164af',
           'sha256': '1cddf0f9ed0b7f1639a8f5e339c177673685df3e2e3c9575a649eb15bcf68a55', 'bytes': 245780441},
    'b5': {'revision': '40357155205b036cf11b61f132d53d2f8861f170',
           'sha256': 'a389c0a604458fa205446fd08a2c01b74e6591a5da3e77de668c6ca5cbf75356', 'bytes': 328277459},
}


def weight_path(c):
    return ROOT / 'weights' / ('mit-' + c['backbone'] + '.bin')


def download(url, target, expected_sha, expected_bytes, attempts=4):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.partial')
    meta = part.with_suffix('.partial.json')
    if meta.exists() and read(meta).get('sha256') != expected_sha:
        raise RuntimeError('Partial download has different identity; move it aside explicitly')
    if part.exists() and not meta.exists():
        raise RuntimeError('Partial download has no provenance')
    save(meta, {'sha256': expected_sha, 'bytes': expected_bytes})
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected_bytes:
            if sha(part) != expected_sha:
                raise ValueError('Completed partial checksum mismatch; retained for inspection')
            part.replace(target)
            return
        if offset > expected_bytes:
            raise ValueError('Partial exceeds expected size')
        headers = {'User-Agent': 'AIC-SegFormer/1.0', 'Accept-Encoding': 'identity'}
        if offset: headers['Range'] = f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                if 'text/' in response.headers.get('Content-Type', ''):
                    raise ValueError('Weight URL returned text')
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or int(match[3]) != expected_bytes:
                        raise ValueError('Invalid resume response')
                elif response.status == 200:
                    offset = 0
                else:
                    raise ValueError('Unexpected HTTP response')
                start = time.monotonic(); last = 0.; done = offset
                with part.open('ab' if offset else 'wb') as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(chunk); done += len(chunk)
                        now = time.monotonic()
                        if now - last > 3 or done == expected_bytes:
                            rate = (done - offset) / max(now - start, .001)
                            print(f'权重下载 {done / expected_bytes:.1%} | {done / 2**20:.1f}/{expected_bytes / 2**20:.1f} MiB | {rate / 2**20:.2f} MiB/s', flush=True)
                            last = now
                if done != expected_bytes:
                    raise OSError('Incomplete file')
            if sha(part) != expected_sha:
                raise ValueError('Upstream SHA256 mismatch; partial retained')
            part.replace(target)
            return
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise RuntimeError(f'Weight server HTTP {exc.code}; no random-weight fallback') from None
        except (OSError, TimeoutError):
            pass
        if attempt + 1 < attempts:
            print('下载中断，保留已下载部分并重试', flush=True)
            time.sleep(2)
    raise RuntimeError('Weight download failed; partial retained')


def ensure_weights(c):
    info = SOURCES[c['backbone']]
    target = weight_path(c)
    url = f'https://huggingface.co/nvidia/mit-{c["backbone"]}/resolve/{info["revision"]}/pytorch_model.bin'
    with lock(target.with_suffix('.lock')):
        if not target.exists():
            download(url, target, info['sha256'], info['bytes'])
        if target.stat().st_size != info['bytes'] or sha(target) != info['sha256']:
            raise ValueError('Existing weight mismatch; will not overwrite silently')
        report = {**info, 'source': url, 'path': target.name, 'upstream_lfs_sha256_verified': True}
        save(target.with_suffix('.receipt.json'), report)
    print('权重复用/校验通过:', target.name, flush=True)
    return report


def load_backbone(backbone, path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    removed = sorted(k for k in state if k.startswith('classifier.'))
    if removed != ['classifier.bias', 'classifier.weight']:
        raise ValueError('Expected NVIDIA ImageNet classifier checkpoint')
    selected = {k[len('segformer.'):]: v for k, v in state.items() if k.startswith('segformer.')}
    if len(selected) + len(removed) != len(state):
        raise ValueError('Unexpected non-backbone parameters')
    backbone.load_state_dict(selected, strict=True)
    if any(not torch.isfinite(v).all() for v in selected.values() if v.is_floating_point()):
        raise ValueError('Nonfinite pretrained tensor')
    return {'backbone_keys': len(selected), 'coverage': 1.0, 'excluded': removed}

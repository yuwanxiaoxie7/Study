import io
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from .common import LABELS, read, save, sha, object_sha
from .data import tensor, encode, TestImages

PALETTE = np.array([[90,90,90], [220,50,50], [180,180,180], [30,90,220], [180,140,80], [30,130,50], [180,220,50], [240,180,20]], dtype=np.uint8)


def confusion(pred, target):
    pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
    valid = target != 255
    if not np.isin(pred, range(8)).all() or not np.isin(target[valid], range(8)).all():
        raise ValueError('Invalid encoded classes')
    return np.bincount(8 * target[valid].astype(np.int64) + pred[valid], minlength=64).reshape(8, 8)


def metrics(cm):
    def divide(num, den):
        return {n: float(a / b) if b else None for n, a, b in zip(LABELS, num, den)}
    tp = cm.diagonal(); union = cm.sum(0) + cm.sum(1) - tp
    iou = divide(tp, union); vals = [v for v in iou.values() if v is not None]
    return {'miou': float(np.mean(vals)) if vals else None, 'iou': iou,
            'precision': divide(tp, cm.sum(0)), 'recall': divide(tp, cm.sum(1)),
            'confusion': cm.tolist(), 'absent_policy': 'union=0 excluded; absent GT but false predictions => IoU=0'}


def precision(c, device):
    if device.type != 'cuda': return torch.float32
    setting = c['precision']
    if setting == 'auto': setting = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if setting == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported; explicitly select another precision')
    return {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[setting]


def autocast(c, device):
    dtype = precision(c, device)
    return torch.autocast(device_type=device.type, dtype=dtype if dtype != torch.float32 else torch.bfloat16, enabled=dtype != torch.float32)


def positions(length, tile, stride):
    if stride <= 0 or stride > tile: raise ValueError('Invalid stride')
    if length <= tile: return [0]
    return sorted(set(list(range(0, length - tile + 1, stride)) + [length - tile]))


@torch.inference_mode()
def predict(model, image, c):
    model.eval(); device = next(model.parameters()).device
    x = tensor(image).unsqueeze(0).to(device); h, w = x.shape[-2:]
    if c['inference'] == 'whole':
        with autocast(c, device): scores = model(x)[0].float()
    else:
        tile = c['tile']; ph, pw = max(h, tile), max(w, tile)
        x = F.pad(x, (0, pw - w, 0, ph - h), mode='replicate')
        scores = torch.zeros(8, ph, pw, device=device)
        weights = torch.zeros(ph, pw, device=device)
        hann = torch.hann_window(tile, periodic=False, device=device).clamp_min(.05)
        window = hann[:, None] * hann[None, :]
        for y in positions(ph, tile, c['stride']):
            for left in positions(pw, tile, c['stride']):
                with autocast(c, device): p = model(x[..., y:y+tile, left:left+tile])[0].float()
                scores[:, y:y+tile, left:left+tile] += p * window
                weights[y:y+tile, left:left+tile] += window
        if not (weights > 0).all(): raise RuntimeError('Uncovered pixels')
        scores = (scores / weights)[:, :h, :w]
    if not torch.isfinite(scores).all(): raise RuntimeError('Nonfinite prediction')
    return scores.argmax(0).byte().cpu().numpy()


def visualize(image, target, pred, output):
    valid = target != 255
    truth_rgb = PALETTE[np.where(valid, target, 0)].copy(); truth_rgb[~valid] = 0
    error = np.zeros_like(truth_rgb); error[valid & (target != pred)] = [255, 0, 0]
    error[~valid] = [80,80,80]
    strip = np.concatenate([np.asarray(image), truth_rgb, PALETTE[pred], error], axis=1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(output)


def evaluate(model, names, c, output=None):
    cm = np.zeros((8,8), np.int64); started = time.monotonic(); root = Path(c['data_root'])
    details = []
    for idx, name in enumerate(names, 1):
        with Image.open(root / c['images'] / name) as f: image = f.convert('RGB')
        with Image.open(root / c['masks'] / name) as f: target = encode(f)
        pred = predict(model, image, c); current = confusion(pred, target); cm += current
        tp = current.diagonal()
        details.append({'name': name, 'vehicle_fn': int(current[7].sum() - tp[7]),
                        'road_fn': int(current[2].sum() - tp[2]),
                        'vegetation_agricultural_confusion': int(current[5,6] + current[6,5])})
        if output and idx <= c['visualizations']:
            visualize(image, target, pred, Path(output).with_suffix('') / (name + '.comparison.png'))
        if idx % 50 == 0 or idx == len(names):
            print(f'验证 {idx}/{len(names)} | mIoU {metrics(cm)["miou"]} | 剩余 {(len(names)-idx)*(time.monotonic()-started)/idx/60:.1f} min', flush=True)
    result = {**metrics(cm), 'images': len(names), 'seconds': time.monotonic() - started,
              'diagnostic_note': 'Road FN measures missed road pixels, not topology/connectivity. Inspect fixed visualizations for fragmentation.',
              'per_image_diagnostics': details}
    if output: save(output, result)
    return result


def validate_png(raw, size):
    if raw[:8] != b'\x89PNG\r\n\x1a\n' or raw[24:26] != bytes([8, 0]):
        raise ValueError('Expected true 8-bit grayscale PNG, no palette')
    with Image.open(io.BytesIO(raw)) as im:
        im.load(); a = np.asarray(im)
        if im.mode != 'L' or im.size != (size, size) or a.ndim != 2 or a.min() < 1 or a.max() > 8:
            raise ValueError('Invalid prediction size/IDs; eight-class model emits IDs 1..8')


def validate_zip(path, names, size=1024):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(names) or set(z.namelist()) != set(names):
            raise ValueError('Submission names/count mismatch')
        if z.testzip() is not None: raise ValueError('ZIP checksum failed')
        for name in names: validate_png(z.read(name), size)
    return {'images': len(names), 'sha256': sha(path), 'bytes': Path(path).stat().st_size, 'errors': []}


def submit(model, c, run, checkpoint, identity):
    out = Path(run) / 'submission_test2'; png = out / 'png'
    png.mkdir(parents=True, exist_ok=True)
    with TestImages(c) as test:
        provenance = {'checkpoint_sha256': sha(checkpoint), 'input_sha256': sha(test.path),
                      'test_names_sha256': object_sha(test.names), 'training_identity_sha256': object_sha(identity),
                      'inference': {k:c[k] for k in ('inference','tile','stride','precision','image_size')},
                      'single_model': True, 'official_score': None}
        manifest = out / 'prediction_identity.json'
        if manifest.exists() and read(manifest) != provenance:
            raise RuntimeError('Prediction identity changed; use a separate run/output, do not mix results')
        if not manifest.exists() and any(png.iterdir()):
            raise RuntimeError('Unverified existing predictions')
        save(manifest, provenance)
        existing = {p.name for p in png.glob('*.png')}
        if existing - set(test.names): raise ValueError('Extra prediction PNGs')
        started = time.monotonic(); predicted = 0
        for idx, name in enumerate(test.names, 1):
            target = png / name
            if target.exists():
                try:
                    validate_png(target.read_bytes(), c['image_size']); continue
                except (ValueError, OSError):
                    pass  # Recompute only corrupt output under matching provenance.
            mask = predict(model, test.read(name), c) + np.uint8(1)
            temp = target.with_suffix('.partial.png'); Image.fromarray(mask).save(temp)
            validate_png(temp.read_bytes(), c['image_size']); temp.replace(target); predicted += 1
            if idx % 25 == 0 or idx == len(test.names):
                print(f'Test2 {idx}/{len(test.names)} | 剩余 {(len(test.names)-idx)*(time.monotonic()-started)/max(predicted,1)/60:.1f} min', flush=True)
        temporary = out / 'submission_test2.partial.zip'
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in test.names: z.write(png / name, name)
        report = {**validate_zip(temporary, test.names, c['image_size']), **provenance,
                  'config_sha256': object_sha(c)}
        final = out / 'submission_test2.zip'; temporary.replace(final)
        save(out / 'validation.json', report)
        (out / 'submission_test2.sha256').write_text(report['sha256'] + '  submission_test2.zip\n', encoding='ascii')
        print(f'复赛提交包检查通过: {final} | {report["bytes"]} bytes | SHA256 {report["sha256"]}', flush=True)
        return final

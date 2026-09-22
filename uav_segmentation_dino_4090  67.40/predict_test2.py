"""Predict round-two images with the original trained checkpoint; no training."""
import argparse
import io
import time
import zipfile
from pathlib import Path

from aic.common import ROOT, read, save, sha


def inspect_test(path):
    with zipfile.ZipFile(path) as source:
        entries = sorted(n for n in source.namelist() if n.lower().endswith('.png'))
        names = [Path(n).name for n in entries]
        if len(entries) != 1300 or len(set(names)) != 1300:
            raise ValueError('测试集2应包含1300张且文件名不重复的PNG')
        for entry in entries:
            raw = source.read(entry)
            if raw[:8] != b'\x89PNG\r\n\x1a\n' or len(raw) < 33:
                raise ValueError(f'Invalid PNG: {entry}')
            if (int.from_bytes(raw[16:20], 'big'), int.from_bytes(raw[20:24], 'big')) != (1024, 1024):
                raise ValueError(f'Unexpected dimensions: {entry}')
    return entries, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-zip', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'submission_test2/submission.zip')
    parser.add_argument('--check-only', action='store_true', help='只检查测试集，不需要GPU或权重')
    args = parser.parse_args()
    entries, names = inspect_test(args.test_zip)
    test_hash = sha(args.test_zip)
    print(f'测试集2检查通过：{len(names)}张，1024×1024；SHA256={test_hash}', flush=True)
    if args.check_only:
        return
    if args.checkpoint is None or not args.checkpoint.is_file():
        parser.error('请用 --checkpoint 指定初赛67.40分实验的 best.pt')
    if args.output.exists():
        raise FileExistsError(f'结果已存在，请另选 --output 路径：{args.output}')

    import numpy as np
    import torch
    from PIL import Image
    from aic.model import Segmenter
    from aic.evaluate import predict, validate_zip
    from aic.precision import dtype

    if not torch.cuda.is_available():
        raise RuntimeError('需要CUDA显卡；推荐使用原RTX 4090服务器')
    checkpoint_hash = sha(args.checkpoint)
    # This receipt ties the supplied checkpoint to the locally retained first-round ZIP.
    receipt = read(ROOT/'submission/validation.json')
    if checkpoint_hash != receipt['checkpoint_sha256']:
        raise ValueError('权重SHA256与保存的初赛提交记录不符，请提供对应best.pt')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    identity = state['fingerprint']
    c = dict(identity['config'])
    # Keep all original prediction settings and model code intact.
    for name, digest in identity['code'].items():
        if name.startswith(('aic/', 'uavseg/', 'vendor/')):
            if sha(ROOT/name) != digest:
                raise ValueError(f'源码与初赛checkpoint不一致：{name}')
    for key, filename in [('split', 'domain_split.json'), ('manifest', 'data_manifest.json')]:
        if sha(ROOT/'artifacts'/filename) != identity[key]:
            raise ValueError(f'原始数据记录不一致：{filename}')
    c.update(data_root=str(args.test_zip.resolve().parent), test_zip=args.test_zip.name)
    model = Segmenter(c, initialize=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.cuda().eval()
    print(f'已加载初赛权重；tile={c["tile"]}, stride={c["stride"]}, precision={dtype(c)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial.zip')
    started = time.monotonic()
    with zipfile.ZipFile(args.test_zip) as source, zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as target:
        for i, (entry, name) in enumerate(zip(entries, names), 1):
            with source.open(entry) as f, Image.open(f) as image:
                mask = predict(model, image.convert('RGB'), c) + np.uint8(1)
            if mask.min() < 1 or mask.max() > 8:
                raise ValueError(f'Invalid class IDs: {name}')
            buffer = io.BytesIO()
            Image.fromarray(mask).save(buffer, format='PNG')
            target.writestr(name, buffer.getvalue())
            if i % 25 == 0 or i == len(names):
                remaining = (len(names)-i)*(time.monotonic()-started)/i/60
                print(f'测试集2预测 {i}/{len(names)}，预计剩余 {remaining:.1f} 分钟', flush=True)
    report = validate_zip(temporary, names)
    report.update(checkpoint_sha256=checkpoint_hash, test_zip_sha256=test_hash,
                  single_model=True, format='8-bit grayscale PNG, 1024x1024, IDs 1..8',
                  config=c, precision=str(dtype(c)), seconds=time.monotonic()-started)
    temporary.replace(args.output)
    save(args.output.with_suffix('.validation.json'), report)
    print(f'检查通过，只提交：{args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()

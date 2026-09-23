import ast
import io
import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
import numpy as np
import torch
from PIL import Image, ImageEnhance
from .common import ROOT, LABELS, read, save, sha, object_sha


def encode(mask):
    a = np.asarray(mask)
    if a.ndim != 2 or not np.isin(a, np.arange(9)).all():
        raise ValueError('Expected raw grayscale labels 0..8')
    return np.where(a == 0, 255, a.astype(np.int64) - 1)


def tensor(im):
    x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy().transpose(2, 0, 1) / 255.)
    return (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def split():
    s = read(ROOT / 'artifacts/domain_split.json')
    groups = [s[k] for k in ('train', 'val', 'domain_val')]
    if [len(x) for x in groups] != [5178, 1049, 769]:
        raise ValueError('Fixed split count changed')
    if len(set(sum(groups, []))) != 6996:
        raise ValueError('Duplicate/leaked split entries')
    return s


class TestImages:
    def __init__(self, c):
        path = Path(c['test_zip'])
        self.path = path if path.is_absolute() else Path(c['data_root']) / path
        self.zip = zipfile.ZipFile(self.path)
        entries = [e for e in self.zip.infolist() if not e.is_dir() and e.filename.lower().endswith('.png')]
        self.items = {}
        try:
            for e in entries:
                p = PurePosixPath(e.filename.replace('\\', '/'))
                if p.is_absolute() or '..' in p.parts or ':' in p.name:
                    raise ValueError('Unsafe archive filename')
                if p.name in self.items: raise ValueError('Duplicate test basename: ' + p.name)
                self.items[p.name] = e.filename
            if len(self.items) != c['expected_test_count']:
                raise ValueError(f'Test2 count mismatch: {len(self.items)} != {c["expected_test_count"]}')
        except BaseException:
            self.zip.close(); raise
        self.names = sorted(self.items)
        self.c = c

    def read(self, name):
        with self.zip.open(self.items[name]) as stream, Image.open(stream) as im:
            if im.mode != 'RGB' or im.size != (self.c['image_size'],) * 2:
                raise ValueError('Invalid test RGB/size: ' + name)
            im.load()
            return im.copy()

    def close(self): self.zip.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def audit(c, out):
    root = Path(c['data_root'])
    tree = ast.parse((root / 'Label.txt').read_text(encoding='utf-8-sig'))
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LABEL_NAMES' for t in n.targets)]
    if values != [dict(enumerate(['Ignore'] + LABELS))]:
        raise ValueError('Label.txt contract changed')
    s = split(); rows = read(ROOT / 'artifacts/data_manifest.json')
    names = {r['name'] for r in rows}
    if len(rows) != 6996 or names != set(s['train'] + s['val'] + s['domain_val']):
        raise ValueError('Manifest/split mismatch')
    for sub in (c['images'], c['masks']):
        actual = {p.name for p in (root / sub).glob('*.png')}
        if actual != names: raise ValueError('Training filenames/count mismatch: ' + sub)
    cache_path = ROOT / '.cache/data_audit.json'
    cache = read(cache_path) if cache_path.exists() else {}
    verified = {}
    exact = defaultdict(list); near = defaultdict(list)
    membership = {n: k for k in ('train', 'val', 'domain_val') for n in s[k]}
    for idx, row in enumerate(rows, 1):
        name = row['name']
        for sub, key in ((c['images'], 'image_sha256'), (c['masks'], 'mask_sha256')):
            path = root / sub / name; stat = path.stat()
            signature = [stat.st_size, stat.st_mtime_ns, row[key], c['image_size']]
            old = cache.get(str(path.resolve()), {})
            if old.get('signature') == signature:
                entry = old
            else:
                if sha(path) != row[key]: raise ValueError('Data SHA256 mismatch: ' + str(path))
                with Image.open(path) as im:
                    im.load()
                    if im.size != (c['image_size'],) * 2: raise ValueError('Bad training size')
                    entry = {'signature': signature}
                    if key == 'mask_sha256':
                        if im.mode != 'L': raise ValueError('Training mask must be grayscale')
                        encode(im)
                        if np.bincount(np.asarray(im).ravel(), minlength=9).tolist() != row['pixels']:
                            raise ValueError('Manifest pixel count mismatch')
                    else:
                        if im.mode != 'RGB': raise ValueError('Training image must be RGB')
                        small = np.asarray(im.convert('L').resize((9, 8)))
                        entry['dhash'] = np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()
            verified[str(path.resolve())] = entry
            if key == 'image_sha256': near[entry['dhash']].append(name)
        exact[row['image_sha256']].append(name)
        if idx % 250 == 0: print(f'数据检查 {idx}/{len(rows)}', flush=True)
    save(cache_path, verified)
    duplicates = [v for v in exact.values() if len(v) > 1]
    cross = [v for v in duplicates if len({membership[n] for n in v}) > 1]
    with TestImages(c) as test:
        for idx, name in enumerate(test.names, 1):
            test.read(name)
            if idx % 250 == 0: print(f'复赛图像检查 {idx}/{len(test.names)}', flush=True)
        report = {'train': 5178, 'val': 1049, 'domain_val': 769, 'test_count': len(test.names),
                  'test_sha256': sha(test.path), 'test_names_sha256': object_sha(test.names),
                  'test_archive': test.path.name, 'split_sha256': sha(ROOT / 'artifacts/domain_split.json'),
                  'manifest_sha256': sha(ROOT / 'artifacts/data_manifest.json'),
                  'exact_duplicates': duplicates, 'cross_split_exact_duplicates': cross,
                  'dhash_collisions': [v for v in near.values() if len(v) > 1],
                  'near_duplicate_note': 'Exact perceptual-hash collisions are review candidates, not proof; adjacent non-overlapping tiles cannot be ruled out without provenance.',
                  'cache_note': 'Reuses full decoded/hash audit only when path,size,mtime,expected hash and size contract match.'}
    save(out, report)
    if cross: print('警告：发现跨划分相同图像，见 audit.json；固定划分未改动', flush=True)
    return report


def class_weights(c):
    if not c['class_weighting']: return None
    chosen = set(split()['train'])
    counts = np.asarray([r['pixels'][1:] for r in read(ROOT / 'artifacts/data_manifest.json') if r['name'] in chosen], dtype=float).sum(0)
    if (counts <= 0).any(): raise ValueError('Training split misses a class')
    weights = np.clip(np.sqrt(np.median(counts) / counts), .5, 3.)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


class Crops(torch.utils.data.Dataset):
    def __init__(self, c, epoch, names):
        self.c, self.epoch, self.names = c, epoch, names
        original = split()['train']
        self.indices = {n: i for i, n in enumerate(original)}
        rows = read(ROOT / 'artifacts/data_manifest.json'); chosen = set(original)
        self.pools = {k: [r['name'] for r in rows if r['name'] in chosen and r['pixels'][k] > 0] for k in (3, 5, 7, 8)}

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        c = self.c; name = self.names[i]
        rng = random.Random(c['seed'] + self.epoch * 1000003 + self.indices[name])
        target_class = None
        if rng.random() < c['sampling_probability']:
            target_class = rng.choice([k for k, v in self.pools.items() if v])
            name = rng.choice(self.pools[target_class])
        root = Path(c['data_root'])
        with Image.open(root / c['images'] / name) as f: image = f.convert('RGB')
        with Image.open(root / c['masks'] / name) as f: mask = f.copy()
        side = rng.choice(c['resize_choices']); crop = c['crop']
        image = image.resize((side, side), Image.Resampling.BILINEAR)
        mask = mask.resize((side, side), Image.Resampling.NEAREST)
        positions = np.argwhere(np.asarray(mask) == target_class) if target_class is not None else []
        if len(positions):
            y, x = positions[rng.randrange(len(positions))]
            x, y = int(np.clip(x - crop // 2, 0, side - crop)), int(np.clip(y - crop // 2, 0, side - crop))
        else: x, y = rng.randint(0, side - crop), rng.randint(0, side - crop)
        image, mask = image.crop((x, y, x + crop, y + crop)), mask.crop((x, y, x + crop, y + crop))
        for op in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM):
            if rng.random() < .5: image, mask = image.transpose(op), mask.transpose(op)
        for _ in range(rng.randrange(4)):
            image, mask = image.transpose(Image.Transpose.ROTATE_90), mask.transpose(Image.Transpose.ROTATE_90)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(.8, 1.2))
        image = ImageEnhance.Color(image).enhance(rng.uniform(.9, 1.1))
        return tensor(image), torch.from_numpy(encode(mask))

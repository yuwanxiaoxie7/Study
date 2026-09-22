"""Stdlib-only source package with an explicit allowlist."""
import ast,hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
required=['pipeline.py','setup.sh','run_all.sh','configs/train.json','README.md','PAPER_RESEARCH.md','LOCAL_CHECKS.md','SERVER_CHECKLIST.md','CHANGES.md']
for name in required:
    if not (root/name).is_file(): raise FileNotFoundError(name)
selected=[]
for folder in ['aic','uavseg','scripts','tests','vendor','configs','artifacts','reference']:
    for p in (root/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ['.py','.json','.md','.txt'] or p.name=='LICENSE'):
            if p.suffix=='.py': ast.parse(p.read_text(encoding='utf-8-sig'))
            selected.append(p)
selected+=list(root.glob('*.md'))+list(root.glob('*.sh'))+list(root.glob('*.bat'))+[root/'pipeline.py',root/'requirements.txt']
target=root/(root.name+'_cloud.zip'); tmp=target.with_suffix('.partial.zip')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(selected)): z.write(p,root.name+'/'+p.relative_to(root).as_posix())
with zipfile.ZipFile(tmp) as z:
    if z.testzip() is not None: raise RuntimeError('ZIP CRC failure')
tmp.replace(target)
print(json.dumps({'package':str(target),'files':len(set(selected)),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()},ensure_ascii=False))
print('Upload and extract on Linux Python 3.10 / RTX 4090. Then:')
print('Round 2 train and predict: bash run_test2.sh --data-root /path/to/dataset')
print('Training (only when intended): bash run_all.sh --data-root /path/to/dataset')

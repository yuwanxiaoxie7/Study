"""No GPU or weights needed; verify syntax/config and pinned official source integrity."""
import ast,json,sys,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root))
from aic.common import config,read
for directory in ['aic','uavseg','scripts','tests','vendor']:
    for file in (root/directory).rglob('*.py'): ast.parse(file.read_text(encoding='utf-8-sig'))
for file in (root/'configs').glob('*.json'): config(file)
record=read(root/'reference/dinov3_source.json')
for name,digest in record['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('Vendor changed: '+name)
print('Syntax, configs and pinned DINOv3 source hashes passed')

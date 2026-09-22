"""Maintainer tool: record reviewed official source snapshot, never called automatically by setup."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
record={'repository':'https://github.com/facebookresearch/dinov3','commit':'6876159a11b4df116f30f667f8c9888617df0751',
        'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((root/'vendor/dinov3').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
(root/'reference/dinov3_source.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Recorded pinned source:',len(record['files']))

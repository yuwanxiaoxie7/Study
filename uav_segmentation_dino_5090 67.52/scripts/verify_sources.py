"""Verify immutable upstream vendor files; experiment configs remain editable."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in sorted((root/'vendor').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    target=root/'reference/source_hashes.json'
    if target.exists():
        saved=json.loads(target.read_text(encoding='utf-8'))
        if saved!=records: raise RuntimeError('Source hashes changed: regenerate only after reviewing edits')
    else: target.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Source hashes verified:',len(records))
if __name__=='__main__':main()

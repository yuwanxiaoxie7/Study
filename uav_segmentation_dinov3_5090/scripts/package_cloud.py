import ast,hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files=[]
for folder in ['aic','uavseg','scripts','tests','vendor','configs','artifacts','reference']:
    for p in (root/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and (folder=='vendor' or p.suffix in ['.py','.json','.md','.txt']):
            if p.name.startswith('weights_'): continue
            if p.suffix=='.py': ast.parse(p.read_text(encoding='utf-8-sig'))
            files.append(p)
files+=list(root.glob('*.md'))+list(root.glob('*.sh'))+[root/'pipeline.py',root/'requirements.txt']
target=root/(root.name+'_cloud.zip'); partial=target.with_suffix('.partial.zip')
with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)): z.write(p,root.name+'/'+p.relative_to(root).as_posix())
with zipfile.ZipFile(partial) as z:
    assert z.testzip() is None
partial.replace(target)
print(json.dumps({'package':str(target),'files':len(set(files)),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))

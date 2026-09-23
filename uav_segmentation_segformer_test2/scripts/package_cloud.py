"""Package only reviewable source/config/split files; no data, weights, venv or runs."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    folders = ('aic','configs','scripts','artifacts','third_party')
    files = [p for directory in folders for p in (ROOT/directory).rglob('*')
             if p.is_file() and p.suffix in ('.py','.json','.md','.txt') and '__pycache__' not in p.parts]
    files += [ROOT/name for name in ('pipeline.py','requirements.txt','setup.sh','run_all.sh','.gitignore',
              'README.md','RESEARCH.md','THIRD_PARTY.md','LOCAL_VERIFICATION.json') if (ROOT/name).is_file()]
    files = sorted(set(files))
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out = ROOT/'dist'; out.mkdir(exist_ok=True)
    target=out/'uav_segmentation_segformer_test2_cloud.zip'
    temp=target.with_suffix('.partial.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p, ROOT.name+'/'+p.relative_to(ROOT).as_posix())
        z.writestr(ROOT.name+'/SOURCE_SHA256.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip(): raise RuntimeError('Cloud ZIP CRC failed')
        if len(z.namelist()) != len(files)+1: raise RuntimeError('Cloud ZIP duplicate entry')
        for name in z.namelist():
            if any(x in name.split('/') for x in ('.venv','weights','runs','.cache')): raise RuntimeError('Unexpected payload')
    temp.replace(target)
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.sha256').write_text(digest+'  '+target.name+'\n',encoding='ascii')
    print(json.dumps({'path':str(target),'files':len(files)+1,'bytes':target.stat().st_size,'sha256':digest},indent=2))


if __name__ == '__main__': main()

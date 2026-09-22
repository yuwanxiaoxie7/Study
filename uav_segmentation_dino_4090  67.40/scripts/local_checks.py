import ast,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from aic.common import config
count=0
for folder in ['aic','uavseg','scripts','tests','vendor']:
    for file in (root/folder).rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file)); count+=1
ast.parse((root/'pipeline.py').read_text(encoding='utf-8'))
for file in (root/'configs').glob('*.json'): config(file,str(root))
print(json.dumps({'python_syntax_files':count+1,'config_parse':'passed'}))

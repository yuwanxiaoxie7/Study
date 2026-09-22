"""No third-party dependency needed: fail before installing packages."""
import argparse,os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--data-root',default=os.environ.get('AIC_DATA_ROOT'))
p.add_argument('--config',default='configs/train.json')
a,_=p.parse_known_args()
if not a.data_root: p.error('--data-root or AIC_DATA_ROOT is required')
for sub in ['train/train/images','train/train/masks','Label.txt','test_2.zip']:
    if not (Path(a.data_root)/sub).exists(): raise FileNotFoundError(Path(a.data_root)/sub)
if not Path(a.config).is_file(): raise FileNotFoundError(a.config)

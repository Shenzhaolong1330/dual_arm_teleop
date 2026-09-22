"""Upload the seven explicitly selected cleaned datasets and verify every byte."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

BASE = Path('/home/deepcybo/.cache/huggingface/lerobot/franka_dual_arm')
DEST = '/user/users/szl/dataset/dual_franka'
SSH = ['ssh', '-p', '30295', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'dev@10.30.13.100']
NAMES = [
    'insert_tube_rack_nonhard_20260828_merged_cleaned_v04',
    'insert_tube_rack_nonhard_20260831_merged_cleaned_v04',
    'insert_tube_rack_nonhard_20260901_merged_cleaned_v04',
    'insert_tube_rack_nonhard_20260902_merged_cleaned_v04',
    'insert_tube_rack_hard_20260903_merged_cleaned_v04',
    'insert_tube_rack_hard_20260904_merged_cleaned_v04',
    'insert_tube_rack_hard_20260910_merged_cleaned_v02',
]
REPORT = Path(__file__).resolve().parents[2] / 'reports/transfers/cleaned_20260917'


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in NAMES:
        root = BASE/name
        assert (root/'meta/info.json').is_file(), root
        for path in sorted(root.rglob('*')):
            if path.is_symlink(): raise ValueError(f'Refusing symlink: {path}')
            if not path.is_file(): continue
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024), b''):digest.update(block)
            manifest[str(path.relative_to(BASE))]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
    (REPORT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('SOURCE_HASHED',len(manifest),'files',sum(v['bytes'] for v in manifest.values()),'bytes',flush=True)
    started=time.time()
    # Exact targets only. Refuse existing files instead of overwriting remote work.
    sender=subprocess.Popen(['tar','-C',str(BASE),'-cf','-',*NAMES],stdout=subprocess.PIPE)
    receiver=subprocess.Popen(SSH+[f'tar --keep-old-files -xf - -C {DEST}'],stdin=sender.stdout)
    sender.stdout.close()
    receiving=receiver.wait()
    sending=sender.wait()
    if sending or receiving:raise RuntimeError(f'Transfer failed: tar={sending}, ssh={receiving}; partial files retained')
    print('TRANSFER_COMPLETE',round(time.time()-started,1),'seconds; verifying SHA-256',flush=True)
    remote_code = '''import hashlib,json,sys
from pathlib import Path
base=Path(DEST)
errors=[]
for name,expected in MANIFEST.items():
 p=base/name
 if not p.is_file():errors.append([name,'missing']);continue
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
 if p.stat().st_size!=expected['bytes'] or digest.hexdigest()!=expected['sha256']:errors.append([name,'hash or size mismatch'])
actual={str(p.relative_to(base)) for name in NAMES for p in (base/name).rglob('*') if p.is_file()}
if actual!=set(MANIFEST):errors.append(['file set mismatch',sorted(actual-set(MANIFEST))])
datasets={name:json.loads((base/name/'meta/info.json').read_text()) for name in NAMES}
result={'destination':str(base),'files':len(MANIFEST),'bytes':sum(v['bytes'] for v in MANIFEST.values()),'sha256_all_match':not errors,'errors':errors,'datasets':{n:{k:i[k] for k in ('total_episodes','total_frames','fps')} for n,i in datasets.items()}}
print(json.dumps(result,indent=2))
sys.exit(bool(errors))
'''
    source='DEST='+repr(DEST)+'\nNAMES='+repr(NAMES)+'\nMANIFEST='+repr(manifest)+'\n'+remote_code
    checked=subprocess.run(SSH+['python3 -'],input=source,text=True,capture_output=True)
    (REPORT/'remote_verification.json').write_text(checked.stdout)
    print(checked.stdout,flush=True)
    if checked.returncode:raise RuntimeError(checked.stderr or 'Remote verification failed')


if __name__=='__main__':main()

"""Remove only the four inspected failed derived datasets authorized by the user."""
import argparse
import json
import shutil
from pathlib import Path

BASE = Path('/home/deepcybo/.cache/huggingface/lerobot/franka_dual_arm')
TARGETS = {
    'insert_tube_rack_20260828_merged_cleaned_v01': 2,
    'insert_tube_rack_20260831_merged_cleaned_v01': 1,
    'insert_tube_rack_20260831_merged_cleaned_v02': 4,
    'insert_tube_rack_hard_20260903_merged_cleaned_v01': 4,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    audit = []
    for name, expected in TARGETS.items():
        path = BASE / name
        if not path.exists(): continue
        assert path.parent == BASE and path.resolve() == path and not path.is_symlink()
        assert '_merged_cleaned_' in name and 'wrong_cam' not in name
        info = json.loads((path/'meta/info.json').read_text())
        assert info['total_episodes'] == expected
        merged = BASE / (name.split('_merged_cleaned_')[0] + '_merged_v01')
        source_info = json.loads((merged/'meta/info.json').read_text())
        assert source_info['total_episodes'] > expected
        provenance = json.loads((merged/'meta/merge_summary.json').read_text())
        assert all((Path(s['root'])/'meta/info.json').is_file() for s in provenance['sources'])
        audit.append({'path':str(path),'failed_episodes':expected,'expected_episodes':source_info['total_episodes'],
                      'metadata':info,'operation':'permanent deletion of failed derived output'})
    log = Path(__file__).resolve().parents[1]/'config/generated_daily_20260907/reviewed_v03/deleted_failed_outputs.json'
    if args.delete and audit:
        log.write_text(json.dumps(audit,indent=2))
        for entry in audit:
            shutil.rmtree(entry['path'])
    print(json.dumps({'deleted':args.delete,'targets':[{k:v for k,v in a.items() if k != 'metadata'} for a in audit]},indent=2))

if __name__ == '__main__': main()

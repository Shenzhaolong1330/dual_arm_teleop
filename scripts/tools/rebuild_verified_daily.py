"""Rebuild six audited collection days from raw, then verify lineage and videos."""
import argparse
import copy
import fcntl
import json
import logging
import re
import time
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from merge_lerobot_datasets import merge_lerobot_datasets
from dataset_integrity import repair_data_references
from fast_preprocess_dataset import _materialize_job, _plan_job, _DataFileCache, _output_episode_table

BASE = Path('/home/deepcybo/.cache/huggingface/lerobot/franka_dual_arm')
CONFIG = Path(__file__).resolve().parents[1] / 'config/generated_daily_20260907/reviewed_v04'
DATES = ['20260828', '20260831', '20260901', '20260902', '20260903', '20260904']

def read_info(root):
    return json.loads((root / 'meta/info.json').read_text())

def data_table(root):
    return pa.concat_tables([pq.read_table(p) for p in sorted((root / 'data').rglob('*.parquet'))], promote_options='default')

def episode_rows(root):
    rows = []
    for p in sorted((root / 'meta/episodes').rglob('*.parquet')):
        rows.extend(pq.read_table(p).to_pylist())
    return sorted(rows, key=lambda r: r['episode_index'])

def picture(root, info, row, key, local_index):
    path = root / info['video_path'].format(video_key=key, chunk_index=row[f'videos/{key}/chunk_index'], file_index=row[f'videos/{key}/file_index'])
    ts = row[f'videos/{key}/from_timestamp'] + local_index / info['fps']
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(int(round(ts / float(stream.time_base))), stream=stream, backward=True)
        for frame in container.decode(stream):
            actual = float(frame.pts * frame.time_base)
            if actual >= ts - 1e-4:
                if abs(actual - ts) > 1e-4:
                    raise ValueError(f'Video timestamp mismatch: {path} {ts} -> {actual}')
                return frame.to_ndarray(format='rgb24')
    raise ValueError(f'No frame at {ts}: {path}')

def prepare():
    CONFIG.mkdir(parents=True, exist_ok=True)
    old = yaml.safe_load((CONFIG.parent / 'fast_clean_daily_final.yaml').read_text())['preprocess_dataset']
    jobs = []
    for date in DATES:
        difficulty = 'hard' if date in ('20260903', '20260904') else 'nonhard'
        raw_prefix = 'insert_tube_rack_' + ('hard_' if difficulty == 'hard' else '') + date
        raw = [p for p in sorted(BASE.iterdir()) if re.fullmatch(re.escape(raw_prefix) + r'_v\d+', p.name)
               and (p / 'meta/info.json').exists() and read_info(p)['total_episodes'] > 0 and read_info(p)['total_frames'] > 0]
        assert raw
        stem = f'insert_tube_rack_{difficulty}_{date}'
        merged = BASE / (stem + '_merged_v03')
        cleaned = BASE / (stem + '_merged_cleaned_v04')
        merge_cfg = {'source': {'parent_dir': str(BASE), 'repo_id_prefix': 'franka_dual_arm', 'datasets': [p.name for p in raw],
                                'strict_schema': True, 'strict_robot_type': True},
                     'output': {'parent_dir': str(BASE), 'dataset_name': merged.name, 'repo_id_prefix': 'franka_dual_arm', 'overwrite': False}, 'dry_run': False}
        clean_cfg = {k: copy.deepcopy(v) for k, v in old.items() if k != 'jobs'}
        clean_cfg['source'] = {'root': str(merged), 'repo_id': 'franka_dual_arm/' + merged.name}
        clean_cfg['output'] = {'root': str(cleaned), 'repo_id': 'franka_dual_arm/' + cleaned.name,
                               'video_encoding': {'codec': 'h264', 'preset': 'veryfast', 'crf': 23, 'gop': 30}}
        clean_cfg['fast_stream'] = {'video_workers': 3, 'ffmpeg_threads': 4, 'verify_video_frames': True,
                                    'single_video_per_camera': True, 'stats_mode': 'recompute_visual_and_full_decode'}
        clean_cfg['gripper_events'] = {'enabled': True, 'change_threshold': 0.0001,
                                      'units': 'meters', 'include_observation_state': True,
                                      'require_observation_state': True, 'require_gripper_features': True,
                                      'keep_radius_frames': 15}
        (CONFIG / f'merge_{date}.yaml').write_text(yaml.safe_dump({'merge_datasets': merge_cfg}, sort_keys=False))
        (CONFIG / f'clean_{date}.yaml').write_text(yaml.safe_dump({'preprocess_dataset': clean_cfg}, sort_keys=False))
        jobs.append((date, difficulty, raw, merged, cleaned, merge_cfg, clean_cfg))
    return jobs

def verify_lineage(raw, merged, cleaned, cfg):
    mi, ci = read_info(merged), read_info(cleaned)
    mt, ct = data_table(merged), data_table(cleaned)
    me, ce = episode_rows(merged), episode_rows(cleaned)
    offset = ep_offset = 0
    comparisons = []
    for root in raw:
        ri, rt, re = read_info(root), data_table(root), episode_rows(root)
        segment = mt.slice(offset, len(rt))
        for key in ['action', 'observation.state', 'timestamp', 'frame_index']:
            assert rt[key].combine_chunks().equals(segment[key].combine_chunks()), (root, key)
        assert np.array_equal(segment['episode_index'].to_numpy(), rt['episode_index'].to_numpy() + ep_offset)
        for ep in sorted({0, len(re) // 2, len(re) - 1}):
            for local in sorted({0, re[ep]['length'] // 2, re[ep]['length'] - 1}):
                for key, ft in ri['features'].items():
                    if ft['dtype'] != 'video': continue
                    a = picture(root, ri, re[ep], key, local)
                    b = picture(merged, mi, me[ep_offset + ep], key, local)
                    assert np.array_equal(a, b), (root, ep, local, key, 'merge pixel mismatch')
                    comparisons.append((ep_offset + ep, key))
        offset += len(rt)
        ep_offset += len(re)
    assert offset == len(mt) == mi['total_frames'] and ep_offset == mi['total_episodes'] == ci['total_episodes'] == len(ce)
    _, _, _, _, plans, _ = _plan_job(cfg, None)
    cache = _DataFileCache(merged, mi)
    cursor = 0
    for row, plan, outrow in zip(me, plans, ce, strict=True):
        expect = _output_episode_table(cache.episode_table(row), plan, cursor, ci['fps'])
        actual = ct.slice(cursor, plan.output_length)
        for key in expect.column_names:
            assert expect[key].combine_chunks().equals(actual[key].combine_chunks()), (row['episode_index'], key, 'clean row mismatch')
        assert outrow['length'] == plan.output_length
        assert outrow['dataset_from_index'] == cursor
        cursor += plan.output_length
        assert outrow['dataset_to_index'] == cursor
    assert cursor == len(ct) == ci['total_frames']
    psnr_min = float('inf')
    count = 0
    for ep, key in sorted(set(comparisons)):
        plan = plans[ep]
        # Include first/last retained frames and both sides of a deleted span.
        positions = {0, plan.output_length // 2, plan.output_length - 1}
        gaps = np.flatnonzero(np.diff(plan.keep_indices) > 1)
        if len(gaps): positions.update((int(gaps[0]), int(gaps[0] + 1)))
        for idx in sorted(positions):
            a = picture(merged, mi, me[ep], key, int(plan.keep_indices[idx])).astype(float)
            b = picture(cleaned, ci, ce[ep], key, idx).astype(float)
            mse = np.mean((a - b)**2)
            psnr = 99 if mse == 0 else 10 * np.log10(255**2 / mse)
            if psnr < 28: raise ValueError(f'Unexpected clean image difference ep={ep} {key} index={idx} PSNR={psnr}')
            psnr_min = min(psnr_min, psnr)
            count += 1
    return {'raw_to_merge_numeric_rows_verified': len(mt), 'raw_to_merge_image_comparisons': len(comparisons),
            'merge_to_clean_numeric_rows_verified': len(ct), 'merge_to_clean_image_comparisons': count,
            'min_image_psnr_db': psnr_min}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--date', action='append')
    args = parser.parse_args()
    jobs = prepare()
    if args.date: jobs = [j for j in jobs if j[0] in args.date]
    report_path = CONFIG / 'verified_report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for date, difficulty, raw, merged, cleaned, merge_cfg, cfg in jobs:
        dry = copy.deepcopy(merge_cfg)
        dry['dry_run'] = True
        merge_lerobot_datasets(dry)
        if args.prepare_only: continue
        started = time.time()
        print(f'[DAY START] {date}', flush=True)
        if not merged.exists(): merge_lerobot_datasets(merge_cfg)
        repair_data_references(merged)
        if not cleaned.exists():
            logging.getLogger('fast_preprocess_dataset').setLevel(logging.INFO)
            _materialize_job(cfg, None, False)
        repair_data_references(cleaned)
        logging.getLogger('fast_preprocess_dataset').setLevel(logging.WARNING)
        verified = verify_lineage(raw, merged, cleaned, cfg)
        mi, ci = read_info(merged), read_info(cleaned)
        manifest = json.loads((cleaned / 'meta/fast_preprocess_manifest.json').read_text())
        assert 'full_decode_verification' in manifest
        entry = {'date': date, 'difficulty': difficulty, 'sources': [p.name for p in raw],
                 'episodes': ci['total_episodes'], 'before_frames': mi['total_frames'], 'after_frames': ci['total_frames'],
                 'fps': ci['fps'], 'merged': str(merged), 'cleaned': str(cleaned), 'verification': verified,
                 'full_decode_verification': manifest['full_decode_verification'], 'elapsed_s': time.time() - started}
        entry['gripper_protection'] = manifest['gripper_protection']
        # Independent day workers share the checkpoint without lost updates.
        with (CONFIG / '.report.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            report[date] = entry
            temporary = report_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(sorted(report.items())), indent=2))
            temporary.replace(report_path)
        print('[DAY VERIFIED] ' + json.dumps(entry), flush=True)

if __name__ == '__main__':
    main()

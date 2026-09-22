"""Physical LeRobot v3 validation and metadata repair for derived outputs."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def repair_data_references(root):
    """Resolve episode references from physical contiguous parquet index ranges."""
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    from lerobot.datasets.utils import get_hf_features_from_features, load_info
    schema = get_hf_features_from_features(load_info(root)['features']).arrow_schema
    ranges = []
    for path in sorted((root / 'data').rglob('*.parquet')):
        table = pq.read_table(path)
        indices = table['index'].to_numpy()
        if not table.schema.equals(schema, check_metadata=True):
            cast = table.cast(schema)
            # Reject any numeric conversion that changes recorded values.
            for name in table.column_names:
                if not np.array_equal(np.asarray(table[name].to_pylist()), np.asarray(cast[name].to_pylist())):
                    raise ValueError(f'Lossy schema repair for {path}: {name}')
            tmp = path.with_suffix('.schema.parquet')
            pq.write_table(cast, tmp, compression='snappy')
            tmp.replace(path)
        assert len(indices) and np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
        ranges.append((int(indices[0]), int(indices[-1]) + 1,
                       int(path.parent.name.split('-')[-1]), int(path.stem.split('-')[-1])))
    assert sum(end - start for start, end, _, _ in ranges) == info['total_frames']
    changed = 0
    for path in sorted((root / 'meta/episodes').rglob('*.parquet')):
        table = pq.read_table(path)
        chunks, files = [], []
        for row in table.to_pylist():
            matches = [(c, f) for start, end, c, f in ranges
                       if start <= row['dataset_from_index'] < row['dataset_to_index'] <= end]
            if len(matches) != 1:
                raise ValueError(f'Cannot resolve episode {row["episode_index"]} in {root}')
            c, f = matches[0]
            changed += (c, f) != (row['data/chunk_index'], row['data/file_index'])
            chunks.append(c)
            files.append(f)
        for key, vals in [('data/chunk_index', chunks), ('data/file_index', files)]:
            idx = table.schema.get_field_index(key)
            table = table.set_column(idx, table.schema.field(idx), pa.array(vals, type=table.schema.field(idx).type))
        tmp = path.with_suffix('.repair.parquet')
        pq.write_table(table, tmp, compression='snappy')
        tmp.replace(path)
    return changed


def _hist_stats(hist, frame_count):
    levels = np.arange(256, dtype=np.float64)
    counts = hist.sum(axis=1)
    mean = (hist * levels).sum(axis=1) / counts
    std = np.sqrt(np.maximum(0, (hist * levels**2).sum(axis=1) / counts - mean**2))
    stat = {'min': (hist > 0).argmax(axis=1),
            'max': 255 - (hist[:, ::-1] > 0).argmax(axis=1), 'mean': mean, 'std': std}
    for q in (.01, .1, .5, .9, .99):
        stat[f'q{int(q * 100):02}'] = np.array([
            np.searchsorted(np.cumsum(h), q * (h.sum() - 1) + 1) for h in hist])
    stat = {k: (np.asarray(v) / 255).reshape(3, 1, 1) for k, v in stat.items()}
    stat['count'] = np.array([frame_count], dtype=np.int64)
    return stat


def decode_and_visual_stats(path, episodes, key, fps):
    """Decode every frame and verify PTS; sample 16 images/episode for visual stats."""
    sample_to_episode = {}
    histograms = {}
    sample_counts = {}
    expected = 0
    for row in episodes:
        start = int(round(row[f'videos/{key}/from_timestamp'] * fps))
        end = int(round(row[f'videos/{key}/to_timestamp'] * fps))
        assert start == expected and end - start == row['length']
        expected = end
        ep = row['episode_index']
        histograms[ep] = np.zeros((3, 256), dtype=np.int64)
        sample_counts[ep] = 0
        for index in np.unique(np.rint(np.linspace(start, end - 1, min(16, end - start))).astype(int)):
            sample_to_episode[int(index)] = ep
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 2
        for frame in container.decode(stream):
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - count / fps) > 1e-4:
                raise ValueError(f'Bad PTS in {path} at frame {count}: {frame.pts}')
            if count in sample_to_episode:
                ep = sample_to_episode[count]
                pixels = frame.to_ndarray(format='rgb24')[::2, ::2]
                for c in range(3):
                    histograms[ep][c] += np.bincount(pixels[:, :, c].ravel(), minlength=256)
                sample_counts[ep] += 1
            count += 1
    if count != expected:
        raise ValueError(f'Decoded {count} frames, expected {expected}: {path}')
    return dict(path=str(path), frames=count, duration_s=count/fps, pts_verified=True), {
        ep: _hist_stats(hist, sample_counts[ep]) for ep, hist in histograms.items()}


def verify_all_videos_and_stats(root, info, rows):
    root = Path(root)
    jobs = []
    for key, ft in info['features'].items():
        if ft['dtype'] != 'video':
            continue
        groups = {}
        for row in rows:
            loc = (row[f'videos/{key}/chunk_index'], row[f'videos/{key}/file_index'])
            groups.setdefault(loc, []).append(row)
        for (chunk, file), eps in groups.items():
            path = root / info['video_path'].format(video_key=key, chunk_index=chunk, file_index=file)
            jobs.append((key, path, eps))
    reports, stats = [], {row['episode_index']: {} for row in rows}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(key, pool.submit(decode_and_visual_stats, path, eps, key, info['fps']))
                   for key, path, eps in jobs]
        for key, future in futures:
            report, epstats = future.result()
            reports.append(report)
            for ep, value in epstats.items():
                stats[ep][key] = value
    return reports, stats

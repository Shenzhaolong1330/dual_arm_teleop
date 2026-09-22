"""Publish a checked daily index and isolate obsolete derived datasets."""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch
import yaml
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rebuild_verified_daily import BASE, CONFIG, DATES, read_info


def duration(frames, fps=30):
    millis = round(frames * 1000 / fps)
    hours, rest = divmod(millis, 3600000)
    minutes, rest = divmod(rest, 60000)
    seconds, ms = divmod(rest, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{ms:03}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--organize', action='store_true')
    args = parser.parse_args()
    report = json.loads((CONFIG / 'verified_report.json').read_text())
    assert sorted(report) == DATES, 'All six days must pass before organizing any old output'
    for date, entry in report.items():
        root = Path(entry['cleaned'])
        info = read_info(root)
        assert entry['after_frames'] == info['total_frames'] and entry['episodes'] == info['total_episodes']
        d = LeRobotDataset('franka_dual_arm/' + root.name, root=root, video_backend='torchcodec')
        for i in [0, len(d)//3, len(d)//2, len(d)-1]:
            item = d[i]
            assert item['action'].dtype == torch.float32
            for key in d.meta.video_keys:
                assert tuple(item[key].shape) == (3, 240, 424)
        subset = LeRobotDataset('franka_dual_arm/' + root.name, root=root, episodes=[0, info['total_episodes']-1], video_backend='torchcodec')
        subset[0]
        subset[len(subset)-1]
        # This checkout's loader currently reads all parquet rows even when
        # episodes= is provided. Record that library limitation honestly;
        # it is not evidence of corrupt daily data or working subset selection.
        selected_rows = [r for p in (root/'meta/episodes').rglob('*.parquet')
                         for r in pq.read_table(p, columns=['episode_index','length','dataset_from_index','dataset_to_index']).to_pylist()
                         if r['episode_index'] in (0, info['total_episodes']-1)]
        entry['loader_episode_subset_filters_rows'] = len(subset) == sum(r['length'] for r in selected_rows)
        for row in selected_rows:
            assert int(d[row['dataset_from_index']]['episode_index']) == row['episode_index']
            assert int(d[row['dataset_to_index']-1]['episode_index']) == row['episode_index']
        merged_root = Path(entry['merged'])
        merged_dataset = LeRobotDataset('franka_dual_arm/' + merged_root.name, root=merged_root, video_backend='torchcodec')
        assert len(merged_dataset) == entry['before_frames']
        for i in [0, len(merged_dataset)//2, len(merged_dataset)-1]:
            item = merged_dataset[i]
            assert item['action'].dtype == torch.float32
            assert len([key for key in merged_dataset.meta.video_keys if key in item]) == 3
        entry['loader_verified'] = True
        entry['short_episodes'] = [r for p in (root/'meta/episodes').rglob('*.parquet')
                                   for r in pq.read_table(p, columns=['episode_index','length']).to_pylist() if r['length'] <= 10]
        entry['episodes_over_10_frames'] = entry['episodes'] - len(entry['short_episodes'])
        entry['frames_over_10_frames'] = entry['after_frames'] - sum(r['length'] for r in entry['short_episodes'])
        manifest_path = root / 'meta/fast_preprocess_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        cfg = yaml.safe_load((CONFIG / f'clean_{date}.yaml').read_text())['preprocess_dataset']
        expected_rules = {k: cfg[k] for k in ('static_trim','gripper_events','action_smoothing')}
        assert manifest['cleaning_rules'] == expected_rules, 'Do not relabel old outputs with new cleaning rules'
        assert manifest['gripper_protection'] == entry['gripper_protection']
        assert entry['gripper_protection']['all_protected_frames_retained']
        video_checks = manifest['full_decode_verification']
        assert len(video_checks) == len(d.meta.video_keys)
        assert all(v['frames'] == entry['after_frames'] and v['pts_verified'] for v in video_checks)
        for video in manifest['full_decode_verification']:
            video['path'] = video['path'].replace(root.name + '.fast-staging', root.name)
        entry['full_decode_verification'] = manifest['full_decode_verification']
        manifest['raw_sources'] = entry['sources']
        manifest['lineage_verification'] = entry['verification']
        manifest_path.write_text(json.dumps(manifest, indent=2))
        (root / 'meta/VERIFIED.json').write_text(json.dumps(entry, indent=2))
        text = f'''# {date} {entry['difficulty']} — 已验收日合并清洗集

记录条数：{entry['episodes']} episodes。FPS：30。
清洗前：{entry['before_frames']:,} 帧 / {duration(entry['before_frames'])}。
清洗后：{entry['after_frames']:,} 帧 / {duration(entry['after_frames'])}。

三路相机各一个完整 MP4，每路的时长均为上述清洗后时长；三路是同步观测，不能将它们的时长相加。
动作、状态、索引与保留掩码逐行核对通过；所有视频逐帧解码及 PTS 检查通过，并抽查原始到合并、合并到清洗的画面对齐。
仅压缩持续近零动作段，保留边界和夹爪事件上下文；没有按任务成功率筛选 episode。
夹爪：任一侧 action 或 observation.state 宽度变化 >= 0.0001 米（0.1 毫米），保留变化两端及前后各 15 帧。
双臂：action 平移分量合并范数 < 0.001、旋转分量合并范数 < 0.005，且不在夹爪保护区的持续静止段，首尾各保留 5 帧。
相较未保护夹爪的 Cartesian-only 规则，本日恢复 {entry['gripper_protection']['restored_vs_cartesian_only_frames']:,} 帧。
图像统计从清洗后视频重新抽样计算（每条最多 16 帧、空间步长 2），不复用原始图像统计。
hard/nonhard 来自原始采集目录命名。
清洗规则与完整来源清单见 meta/fast_preprocess_manifest.json；验收见 meta/VERIFIED.json。
'''
        (root / 'README.md').write_text(text)
        (merged_root / 'README.md').write_text(
            f'# {date} {entry["difficulty"]} 日合并集（未清洗）\n\n'
            f'{entry["episodes"]} 条，{entry["before_frames"]:,} 帧，30 FPS，总时长 {duration(entry["before_frames"])}。\n\n'
            '本目录保留了全部有效原始采集帧。视频可能按文件大小分片，每路相机所有分片之和才是整日时长。\n'
            f'对应已验收日清洗集：{root.name}（每路相机一个完整视频）。\n'
            '有效原始采集及合并来源见 meta/merge_summary.json；wrong_cam 与空集已排除。\n')
        print('[LOADER VERIFIED]', date, entry['episodes'], duration(entry['after_frames']), flush=True)

    archive = BASE / '_archive_daily_pre_v04_20260907'
    obsolete = []
    for root in sorted(BASE.iterdir()):
        if not root.is_dir() or not root.name.startswith('insert_tube_rack_'):
            continue
        if root.name == 'insert_tube_rack_20260828_20260901_merged_E1037_v01':
            obsolete.append(root)
            continue
        if not re.search(r'_merged(?:_cleaned)?_v0[12]$', root.name) and not re.search(r'_merged_cleaned_v03$', root.name) and not re.search(r'_(?:nonhard|hard)_\d{8}_cleaned_v02$', root.name):
            continue
        if not any(date in root.name for date in DATES) or 'wrong_cam' in root.name:
            continue
        obsolete.append(root)
    history = []
    for root in obsolete:
        target = archive / root.name
        if target.exists(): raise FileExistsError(target)
        history.append({'original': str(root), 'archive': str(target), 'reason': 'obsolete derived dataset; superseded by gripper-aware cleaned v04 and verified merged v03'})
    if args.organize:
        archive.mkdir(exist_ok=True)
        old_manifest = archive / 'archive_manifest.json'
        previous_history = json.loads(old_manifest.read_text()) if old_manifest.exists() else []
        for item in history:
            Path(item['original']).rename(item['archive'])
        (archive / 'README.md').write_text('# 历史派生数据归档\n\n包含已被夹爪保护修正后的 cleaned_v04 替代的旧派生版本。旧 clean 的 0.5 夹爪阈值单位错误，未有效保护夹爪事件，不应再用于训练或日统计。原始采集未移动。所有归档均可按 archive_manifest.json 的 original 路径移回。四个确认失败的半成品已按用户要求删除，删除清单见数据根目录 DELETED_FAILED_OUTPUTS.json。\n')
        old_manifest.write_text(json.dumps(previous_history + history, indent=2, ensure_ascii=False))
    lines = ['# 每日数据集：2026-09-07 重建验收', '', '原始采集全部保留；wrong_cam 和零 episode 目录全部排除。训练与清洗后统计请使用下面列出的 merged_cleaned_v04；merged_v03 是验收后的未清洗日合并源。旧派生版本已移入 _archive_daily_pre_v04_20260907。', '',
             '| 日期 | 难度 | 条数 | 清洗前时长 | 清洗后有效时长 | 删除时长 |', '|---|---|---:|---:|---:|---:|']
    for date, e in report.items():
        lines.append(f'| {date} | {e["difficulty"]} | {e["episodes"]} | {duration(e["before_frames"])} | {duration(e["after_frames"])} | {duration(e["before_frames"]-e["after_frames"])} |')
    before = sum(e['before_frames'] for e in report.values())
    after = sum(e['after_frames'] for e in report.values())
    episodes = sum(e['episodes'] for e in report.values())
    restored = sum(e['gripper_protection']['restored_vs_cartesian_only_frames'] for e in report.values())
    lines += [f'| 合计 | | {episodes} | {duration(before)} | {duration(after)} | {duration(before-after)} |', '',
              '时长按实际帧数 / 30 计算；三路相机不重复计时。条数是保留的采集 episode 数，不等于经人工确认的任务成功条数。', '',
              f'修正夹爪保护后，相较旧规则共恢复 {restored:,} 帧（{duration(restored)}）。原始帧未改动，仅重新计算保留掩码并输出。', '',
              '## 最终目录', '']
    for date, e in report.items():
        lines += [f'- {date} 日合并：`{Path(e["merged"]).name}`', f'- {date} 日合并清洗：`{Path(e["cleaned"]).name}`']
    lines += ['', '## 审查发现与处理', '',
              '- 旧 8/28 cleaned_v01 仅 2 条、96.200 秒；旧 8/31 cleaned_v01 仅 1 条、30.967 秒；旧 8/31 merged_cleaned_v02 仅 4 条、115.500 秒；旧 hard 9/3 cleaned_v01 仅 4 条、103.133 秒。四个失败/中断半成品已按用户要求删除，约 475 MiB；原始数据完整保留。',
              '- LeRobot 原先按大小切分视频，单个 MP4 不代表整日时长。本次清洗集每路相机合为一个完整视频。',
              '- 修复旧合并逻辑造成的 parquet 文件引用错误，以及 float32 定长数组被改写为 double 变长数组的问题。',
              '- 新结果补齐每条 episode 的统计与元数据索引；重新计算清洗后图像采样统计，并实际解码每一帧检查时间戳。',
              '- 原始到日合并的动作/状态逐行一致；日合并到 clean 的所有列与保留掩码逐行一致。各原始批次开头、中间、结尾及删帧边界作画面对照。',
              '- 清洗规则：双臂 action 平移分量合并范数 < 0.001、旋转分量合并范数 < 0.005，且不在夹爪保护区，才作为静止候选；连续至少 10 帧，首尾各保留 5 帧；不平滑动作。',
              '- 夹爪保护已修正：左右夹爪 action 或 observation.state 宽度相邻帧变化达到 0.0001 米（0.1 毫米），保留变化两端及前后各 15 帧；按 episode 独立计算。数据宽度最大约 0.085 米，旧阈值 0.5 米使旧版本保护未生效。',
              '- 新保留掩码是旧 Cartesian-only 掩码的超集，所有夹爪保护帧均保留。各日保护帧及恢复帧计数见 DAILY_DATASETS_VERIFIED.json 的 gripper_protection。未仅因夹爪处于闭合位置而无限保留静止等待。',
              '- 非运动 episode 仍按边界保留规则保留少量帧；本次没有擅自按成功率或短 episode 删除整条记录。',
              '- 跨日 E1037 合并集是前三天的历史派生副本，已归档，不计入每日统计。', '',
              '- 加载器注意：当前本机 LeRobot 的 episodes=[...] 参数不会过滤已存在的全部 parquet 行；选择少量 episode 训练时应显式筛选索引。本次日统计按实际文件计算，不依赖该参数，也未修改上游加载器。', '',
              '配置与复现入口：`scripts/config/generated_daily_20260907/reviewed_v04/`、`scripts/tools/rebuild_verified_daily.py`。',
              '原始 wrong_cam 四个目录共 187 条、193,335 帧、01:47:24.500，已排除且保留。']
    lines += ['', '## 极短 episode 复核清单（保留但标记）', '',
              '以下为清洗后不超过 10 帧的记录。它们没有被自动视作失败采集删除；总条数和总时长包含这些记录。', '']
    for date, e in report.items():
        if e['short_episodes']:
            lines.append(f'- {date}: ' + ', '.join(f"episode {r['episode_index']} ({r['length']} 帧)" for r in e['short_episodes']))
    (BASE / 'DAILY_DATASETS_INDEX.md').write_text('\n'.join(lines) + '\n')
    (CONFIG / 'DATASET_REVIEW.md').write_text('\n'.join(lines) + '\n')
    (BASE / 'DAILY_DATASETS_VERIFIED.json').write_text(json.dumps(report, indent=2))
    deletion_log = CONFIG.parent / 'reviewed_v03/deleted_failed_outputs.json'
    if deletion_log.exists():
        shutil.copy2(deletion_log, BASE / 'DELETED_FAILED_OUTPUTS.json')
    initial_audit = Path('/tmp/daily_video_audit.json')
    if initial_audit.exists() and not (BASE / 'INITIAL_DAILY_AUDIT.json').exists():
        shutil.copy2(initial_audit, BASE / 'INITIAL_DAILY_AUDIT.json')
    (CONFIG / 'verified_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'episodes':episodes,'before_frames':before,'after_frames':after,'before':duration(before),'after':duration(after),'archived':len(history) if args.organize else 0}))

if __name__ == '__main__':
    main()

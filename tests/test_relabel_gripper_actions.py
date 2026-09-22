"""Offline detection, Parquet round-trip and publication failure tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tools import relabel_gripper_actions as tool
from scripts.utils.dataset_utils import validate_merge_semantics


class PlateauTests(unittest.TestCase):
    def mask(self, values, **kwargs):
        return tool.closed_plateau_mask(np.asarray(values, dtype=np.float32), 30, tool.Parameters(**kwargs))

    def test_closing_plateau_only_including_first_window(self):
        values = [.085, .06, .03] + [.015] * 20 + [.03, .06] + [.085] * 20
        expected = np.zeros(len(values), dtype=bool)
        expected[3:23] = True
        np.testing.assert_array_equal(self.mask(values), expected)

    def test_opening_and_initial_small_aperture_are_untouched(self):
        self.assertFalse(self.mask([.015] * 20 + [.03] * 20 + [.085] * 20).any())
        self.assertFalse(self.mask([.015] * 100).any())

    def test_short_pause_and_episode_boundary(self):
        self.assertFalse(self.mask([.085, .06, .03] + [.015] * 14 + [.085] * 20).any())
        self.assertFalse(self.mask([.085] * 20 + [.015] * 8).any())
        self.assertFalse(self.mask([.015] * 8 + [.085] * 20).any())

    def test_last_frame_and_exact_duration(self):
        result = self.mask([.085, .03] + [.015] * 15)
        self.assertEqual(result.sum(), 15)
        self.assertTrue(result[-1])
        self.assertEqual(self.mask([.085, .03] + [.015] * 6, stable_seconds=.2).sum(), 6)

    def test_open_cutoff_is_strict_in_float32(self):
        for plateau in (.084, .0845, .085):
            # Prior >2 mm decline ensures this tests the cutoff, not direction detection.
            self.assertFalse(self.mask([.09] + [plateau] * 20).any())
        self.assertEqual(self.mask([.09] + [.0839] * 20).sum(), 20)

    def test_jitter_and_unstable_widths(self):
        self.assertEqual(self.mask([.085, .03] + [.015, .0153] * 15).sum(), 30)
        self.assertFalse(self.mask([.085, .03] + [.015, .016] * 15).any())

    def test_multiple_cycles_and_cumulative_drop(self):
        cycle = [.085, .06, .03] + [.015] * 20 + [.03, .06, .085]
        mask = self.mask(cycle * 2)
        self.assertEqual(mask.sum(), 40)
        ramp = np.linspace(.085, .015, 141).tolist()
        mask = self.mask(ramp + [.015] * 20)
        self.assertTrue(mask[-20:].all())
        self.assertFalse(mask[:130].any())

    def test_opening_confirmation_backdates_to_trough(self):
        widths = [.085, .03] + [.015] * 20 + [.0153] * 20 + [.018] * 20
        mask = self.mask(widths)
        self.assertEqual(mask.sum(), 20)
        self.assertFalse(mask[22:].any())

    def test_invalid_parameters_or_widths(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                self.mask([.085] + [.015] * 18 + [value])
        for kwargs in ({'stable_seconds': 0}, {'closing_drop': -1}, {'open_tolerance': .085},
                       {'opening_rise': np.nan}, {'stable_range': .002}):
            with self.assertRaises(ValueError):
                self.mask([.085] * 20, **kwargs)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.output = self.root / "old", self.root / "new"
        (self.source / 'data/chunk-000').mkdir(parents=True)
        (self.source / 'meta/episodes/chunk-000').mkdir(parents=True)
        (self.source / 'videos/camera/chunk-000').mkdir(parents=True)
        # Deliberately move one gripper away from the last two columns.
        self.names = [f"arm_{i}" for i in range(14)]
        self.names[2], self.names[13] = tool.GRIPPER_NAMES
        self.features = {'action': {'dtype': 'float32', 'shape': [14], 'names': self.names},
                         'observation.state': {'dtype': 'float32', 'shape': [14], 'names': self.names}}
        self.info = {'codebase_version': 'v3.0', 'fps': 30, 'total_frames': 60, 'total_episodes': 2,
                     'features': self.features, 'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'}
        tool.write_json(self.source / 'meta/info.json', self.info)
        self.original = []
        rows = []
        vector_type = pa.list_(pa.float32(), 14)
        for ep in range(2):
            actions = np.full((30, 14), .123, dtype=np.float32)
            actions[:, 2] = [.085, .06, .03] + [.015] * 20 + [.03, .06] + [.085] * 5
            actions[:, 13] = .025 if ep == 0 else actions[:, 2]
            self.original.append(actions)
            array = pa.FixedSizeListArray.from_arrays(pa.array(actions.reshape(-1)), 14)
            state = pa.array(np.full((30, 14), .015, dtype=np.float32).tolist(), type=vector_type)
            table = pa.table({'action': array, 'observation.state': state, 'episode_index': [ep] * 30,
                              'index': list(range(ep * 30, (ep + 1) * 30)), 'frame_index': list(range(30)),
                              'timestamp': np.arange(30, dtype=np.float32) / 30,
                              'task_index': [0] * 30}).replace_schema_metadata({b'huggingface': b'unchanged metadata'})
            pq.write_table(table, self.source / f'data/chunk-000/file-{ep:03d}.parquet')
            row = {'episode_index': ep, 'length': 30, 'dataset_from_index': ep * 30, 'dataset_to_index': (ep + 1) * 30,
                   'data/chunk_index': 0, 'data/file_index': ep, 'videos/camera/from_timestamp': ep,
                   'videos/camera/to_timestamp': ep + 1, 'stats/observation.state/mean': [.015] * 14}
            row.update({f'stats/action/{key}': value for key, value in tool.action_stats(actions).items()})
            rows.append(row)
        self.meta = self.source / 'meta/episodes/chunk-000/file-000.parquet'
        pq.write_table(pa.Table.from_pylist(rows), self.meta)
        tool.write_json(self.source / 'meta/stats.json', {'action': tool.action_stats(np.concatenate(self.original)),
                                                        'observation.state': {'mean': [.015] * 14},
                                                        'observation.images.camera': {'count': [60]}})
        (self.source / 'videos/camera/chunk-000/file-000.mp4').write_bytes(b'opaque-video-copy-test' * 100)

    def test_dry_run_and_round_trip_preserve_source_state_arms_and_videos(self):
        before = tool.inventory(self.source)
        report = tool.relabel_dataset(self.source, self.output, tool.Parameters(), dry_run=True)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name('new.relabel-staging').exists())
        self.assertEqual(report['modified_frames_per_gripper'], {'left_gripper_width': 40, 'right_gripper_width': 20})
        self.assertEqual(report['modified_episodes_any_gripper'], 2)
        applied = tool.relabel_dataset(self.source, self.output, tool.Parameters())
        self.assertEqual(applied['intervals'], report['intervals'])
        self.assertEqual(tool.inventory(self.source), before)
        self.assertEqual(tool.read_json(self.output / 'meta/info.json')[tool.SEMANTICS_KEY], tool.SEMANTICS)
        outputs = []
        for ep in range(2):
            relative = Path(f'data/chunk-000/file-{ep:03d}.parquet')
            source, output = pq.read_table(self.source / relative), pq.read_table(self.output / relative)
            self.assertTrue(source.schema.equals(output.schema, check_metadata=True))
            for name in source.column_names:
                if name != 'action':
                    self.assertTrue(source[name].equals(output[name]))
            values = tool.action_matrix(output['action'], 14)
            expected = self.original[ep].copy()
            expected[3:23, 2] = 0
            if ep == 1:
                expected[3:23, 13] = 0
            np.testing.assert_array_equal(values, expected)
            outputs.append(values)
        metadata = pq.read_table(self.output / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
        for ep, row in enumerate(metadata):
            for key, value in tool.action_stats(outputs[ep]).items():
                np.testing.assert_allclose(row[f'stats/action/{key}'], value)
        stats = tool.read_json(self.output / 'meta/stats.json')
        self.assertEqual(stats['action'], tool.action_stats(np.concatenate(outputs)))
        source_stats = tool.read_json(self.source / 'meta/stats.json')
        for key in source_stats.keys() - {'action'}:
            self.assertEqual(stats[key], source_stats[key])
        video = Path('videos/camera/chunk-000/file-000.mp4')
        self.assertEqual((self.source / video).read_bytes(), (self.output / video).read_bytes())
        self.assertNotEqual((self.source / video).stat().st_ino, (self.output / video).stat().st_ino)
        with self.assertRaises(ValueError):
            validate_merge_semantics({}, {tool.SEMANTICS_KEY: tool.SEMANTICS})
        with self.assertRaises(ValueError):
            validate_merge_semantics({tool.SEMANTICS_KEY: 'command_target_width_v1'}, {tool.SEMANTICS_KEY: tool.SEMANTICS})

    def test_refuse_overlap_overwrite_and_repeat_relabel(self):
        for output in (self.source, self.source / 'nested', self.root):
            with self.assertRaises((ValueError, FileExistsError)):
                tool.relabel_dataset(self.source, output, tool.Parameters())
        self.output.mkdir()
        (self.output / 'keep').write_text('keep')
        with self.assertRaises(FileExistsError):
            tool.relabel_dataset(self.source, self.output, tool.Parameters())
        self.assertEqual((self.output / 'keep').read_text(), 'keep')
        self.info[tool.SEMANTICS_KEY] = tool.SEMANTICS
        tool.write_json(self.source / 'meta/info.json', self.info)
        with self.assertRaisesRegex(ValueError, 'unmarked legacy'):
            tool.relabel_dataset(self.source, self.root / 'another', tool.Parameters())

    def test_invalid_schema_nonfinite_and_episode_indices_fail_before_output(self):
        original_info = dict(self.info)
        self.info['features'] = {'action': {'shape': [1], 'names': ['missing']}}
        tool.write_json(self.source / 'meta/info.json', self.info)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            tool.relabel_dataset(self.source, self.output, tool.Parameters())
        tool.write_json(self.source / 'meta/info.json', original_info)
        path = self.source / 'data/chunk-000/file-000.parquet'
        original = pq.read_table(path)
        bad = self.original[0].copy()
        bad[10, 2] = np.nan
        pq.write_table(tool.replace_action(original, bad), path)
        with self.assertRaisesRegex(ValueError, 'finite'):
            tool.relabel_dataset(self.source, self.output, tool.Parameters())
        bad_indices = original.set_column(original.schema.get_field_index('frame_index'), 'frame_index', pa.array([0] * 30))
        pq.write_table(bad_indices, path)
        with self.assertRaisesRegex(ValueError, 'indices disagree'):
            tool.relabel_dataset(self.source, self.output, tool.Parameters())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_name('new.relabel-staging').exists())

    def test_failed_copy_validation_does_not_publish(self):
        before = tool.inventory(self.source)
        copy2 = tool.shutil.copy2
        def corrupt_video(src, dst):
            result = copy2(src, dst)
            if Path(src).suffix == '.mp4':
                Path(dst).write_bytes(b'corrupt')
            return result
        with patch.object(tool.shutil, 'copy2', side_effect=corrupt_video):
            with self.assertRaisesRegex(ValueError, 'Unmodified file differs'):
                tool.relabel_dataset(self.source, self.output, tool.Parameters())
        self.assertFalse(self.output.exists())
        self.assertEqual(tool.inventory(self.source), before)


if __name__ == '__main__':
    unittest.main()

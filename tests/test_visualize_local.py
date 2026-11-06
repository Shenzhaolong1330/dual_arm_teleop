"""Local path errors should not trigger Hub requests; episode rows are relative."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.core import run_visualize as visualize


class LocalVisualizeTests(unittest.TestCase):
    def test_missing_metadata_suggests_exact_name_with_space_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            actual = base / "flexiv_dual_arm/snack _20261106_v02"
            (actual / "meta").mkdir(parents=True)
            (actual / "meta/info.json").write_text("{}")
            with patch.object(visualize, "HF_LEROBOT_HOME", base):
                with self.assertRaisesRegex(FileNotFoundError, "snack _20261106_v02"):
                    visualize.resolve_dataset_root("flexiv_dual_arm/snack_20261106_v02", None)
                self.assertEqual(visualize.resolve_dataset_root("flexiv_dual_arm/snack _20261106_v02", None), actual)

    def test_explicit_root_and_opt_in_hub_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(FileNotFoundError, "meta/info.json"):
                visualize.resolve_dataset_root("team/task", root)
            self.assertEqual(visualize.resolve_dataset_root("team/task", root, allow_hub_download=True), root)

    def test_filtered_episode_sampler_uses_relative_rows(self):
        dataset = SimpleNamespace(hf_dataset={"episode_index": [2, 2, 2]})
        sampler = visualize.EpisodeSampler(dataset, 2)
        self.assertEqual(list(sampler), [0, 1, 2])
        self.assertEqual(len(sampler), 3)
        with self.assertRaisesRegex(ValueError, "Episode 3"):
            visualize.EpisodeSampler(dataset, 3)

    def test_unfiltered_episode_sampler_selects_only_requested_rows(self):
        dataset = SimpleNamespace(hf_dataset={"episode_index": [0, 0, 1, 2, 2]})
        self.assertEqual(list(visualize.EpisodeSampler(dataset, 2)), [3, 4])


if __name__ == "__main__":
    unittest.main()

"""Robot selection must also select the default dataset namespace."""
from pathlib import Path
import re
import tempfile
import unittest

import yaml

from scripts.core.run_record import RecordConfig
from scripts.utils.dataset_utils import generate_dataset_name


SOURCE = (Path(__file__).resolve().parents[1] / "scripts/config/record_cfg.yaml").read_text()


def selected_record(robot_type):
    source = re.sub(r'^robot_type: &robot_type .+$',
                    f'robot_type: &robot_type "{robot_type}"', SOURCE, flags=re.MULTILINE)
    return yaml.safe_load(source)["record"]


class RecordRepoIdTests(unittest.TestCase):
    def test_top_level_robot_selection_updates_repo_and_default_path(self):
        for robot_type in ("nero_dual_arm", "franka_dual_arm", "flexiv_dual_arm"):
            with self.subTest(robot_type=robot_type):
                config = RecordConfig(selected_record(robot_type))
                self.assertEqual(config.repo_id, f"{robot_type}/B-CARE")
                self.assertEqual(config.dataset_path.parts[-2:], (robot_type, "B-CARE"))

    def test_explicit_repo_and_dataset_path_remain_supported(self):
        raw = selected_record("flexiv_dual_arm")
        raw.update(repo_id="my_team/custom_task", dataset_path="/tmp/explicit_dataset")
        config = RecordConfig(raw)
        self.assertEqual(config.repo_id, "my_team/custom_task")
        self.assertEqual(config.dataset_path, Path("/tmp/explicit_dataset"))

    def test_generated_names_and_version_search_use_selected_robot_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            for robot_type in ("nero_dual_arm", "flexiv_dual_arm"):
                config = RecordConfig(selected_record(robot_type))
                config.dataset_path = Path(temp) / config.repo_id
                name, version = generate_dataset_name(config)
                self.assertTrue(name.startswith(f"{robot_type}/B-CARE_"))
                self.assertEqual(version, "v01")
                (Path(temp) / name).mkdir()
                next_name, next_version = generate_dataset_name(config)
                self.assertTrue(next_name.startswith(f"{robot_type}/B-CARE_"))
                self.assertEqual(next_version, "v02")


if __name__ == "__main__":
    unittest.main()

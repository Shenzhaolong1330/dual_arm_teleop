import unittest
from unittest.mock import patch

import numpy as np

from scripts.tools.split_label_dataset import (
    LabelResult,
    OpenRouterLabeler,
    RawSegment,
    _merge_segments,
    _normalize_label,
    _stable_state_events,
    _template_instruction,
    split_episode,
)


def cfg(**segmentation):
    base = {
        "_fps": 10,
        "segmentation": {
            "open_threshold": 0.75,
            "closed_threshold": 0.25,
            "debounce_frames": 2,
            "close_is_low": True,
            "context_before_sec": 0.2,
            "context_after_sec": 0.2,
            "min_segment_sec": 0.5,
            "merge_gap_sec": 0.2,
            "motion_fallback_enabled": True,
            "motion_context_sec": 0.1,
            "min_motion_sec": 0.2,
            "motion_translation_threshold": 0.01,
            "motion_rotation_threshold": 0.01,
        },
    }
    base["segmentation"].update(segmentation)
    return base


class SplitLabelDatasetTest(unittest.TestCase):
    def test_stable_events_ignore_single_frame_noise(self):
        values = np.array([1, 1, 0.1, 1, 1, 0, 0, 0, 1, 1], dtype=np.float32)
        events = _stable_state_events(
            values,
            "left",
            open_threshold=0.75,
            closed_threshold=0.25,
            debounce_frames=2,
        )
        self.assertEqual([(event.event, event.frame) for event in events], [("close", 5), ("open", 8)])

    def test_split_open_close_open_segment(self):
        left = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=np.float32)
        grippers = left[:, None]
        segments = split_episode(0, grippers, ["left"], None, [], cfg())
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.active_arm, "left")
        self.assertEqual((segment.core_start, segment.core_end), (3, 7))
        self.assertEqual((segment.start, segment.end), (1, 10))

    def test_short_segments_are_filtered(self):
        left = np.array([1, 1, 0, 0, 1, 1], dtype=np.float32)
        segments = split_episode(0, left[:, None], ["left"], None, [], cfg(min_segment_sec=1.0))
        self.assertEqual(segments, [])

    def test_overlapping_left_right_segments_merge_to_both(self):
        left = RawSegment(
            parent_episode=3,
            segment_id=-1,
            stage_id="",
            active_arm="left",
            start=0,
            end=10,
            core_start=2,
            core_end=8,
            close_frames={"left": 2},
            open_frames={"left": 8},
            source_segments=[{"active_arm": "left"}],
        )
        right = RawSegment(
            parent_episode=3,
            segment_id=-1,
            stage_id="",
            active_arm="right",
            start=9,
            end=16,
            core_start=11,
            core_end=14,
            close_frames={"right": 11},
            open_frames={"right": 14},
            source_segments=[{"active_arm": "right"}],
        )
        merged = _merge_segments([left, right], 3, cfg())
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].active_arm, "both")
        self.assertEqual((merged[0].start, merged[0].end), (0, 16))

    def test_motion_fallback_when_no_gripper(self):
        actions = np.zeros((12, 3), dtype=np.float32)
        actions[4:8, 0] = 0.02
        segments = split_episode(
            0,
            None,
            [],
            actions,
            ["left_delta_ee_pose.x", "left_delta_ee_pose.y", "left_delta_ee_pose.z"],
            cfg(),
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].active_arm, "motion")

    def test_template_and_label_normalization_always_have_three_variants(self):
        segment = RawSegment(
            parent_episode=1,
            segment_id=2,
            stage_id="stage",
            active_arm="left",
            start=0,
            end=10,
            core_start=2,
            core_end=8,
        )
        fallback = _template_instruction(segment, "Pick objects into the basket", 5)
        self.assertEqual(len(fallback.variants), 3)

        raw = {
            "canonical_instruction": "Pick up the visible object.",
            "stage_label": "pick_object",
            "object": "object",
            "target": "basket",
            "confidence": 0.2,
            "variants": ["Lift the object."],
            "needs_review": False,
        }
        label = _normalize_label(raw, fallback, min_confidence=0.55)
        self.assertIsInstance(label, LabelResult)
        self.assertTrue(label.needs_review)
        self.assertEqual(len(label.variants), 3)

    def test_openrouter_api_key_prefers_cfg_then_env(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
            labeler = OpenRouterLabeler({"openrouter": {"api_key": "cfg-key"}}, cache=None)
            self.assertEqual(labeler.api_key, "cfg-key")

            fallback_labeler = OpenRouterLabeler({"openrouter": {"api_key": ""}}, cache=None)
            self.assertEqual(fallback_labeler.api_key, "env-key")


if __name__ == "__main__":
    unittest.main()

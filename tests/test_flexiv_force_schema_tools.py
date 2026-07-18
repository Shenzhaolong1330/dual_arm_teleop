from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from robots.dual_flexiv_rizon4s.flexiv_state_schema import (
    FLEXIV_SCHEMA_INFO_KEY,
    FLEXIV_STATE_DIM,
    build_flexiv_state_schema,
    flexiv_action_names,
    flexiv_state_names,
    propagate_flexiv_dataset_schema,
)
from scripts.core.run_visualize import log_named_vector
from scripts.tools import merge_lerobot_datasets as merge_tool
from scripts.tools import preprocess_dataset
from scripts.tools import split_label_dataset


class _FakeRerun:
    class Scalars:
        def __init__(self, value):
            self.value = value

    def __init__(self):
        self.logs = []

    def log(self, path, scalar):
        self.logs.append((path, scalar.value))


def _features():
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (FLEXIV_STATE_DIM,),
            "names": flexiv_state_names(),
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": flexiv_action_names(),
        },
    }


def _flexiv_info(features):
    return {
        "robot_type": "flexiv_dual_arm",
        "features": features,
        FLEXIV_SCHEMA_INFO_KEY: build_flexiv_state_schema(zero_ft_sensor_on_connect=False),
    }


def test_preprocess_motion_mask_ignores_force_noise_by_default_but_allows_explicit_opt_in():
    names = flexiv_state_names()
    states = np.zeros((4, FLEXIV_STATE_DIM), dtype=np.float32)
    states[1:, 34:] = np.arange(1, 15, dtype=np.float32)
    cfg = {"static_trim": {"state_rate": {"include_force": False}}}

    motion = preprocess_dataset._state_rate_motion_mask(states, names, 30.0, cfg)
    assert motion is not None
    assert not motion.any()

    opt_in = {"static_trim": {"state_rate": {"include_force": True}}}
    force_motion = preprocess_dataset._state_rate_motion_mask(states, names, 30.0, opt_in)
    assert force_motion is not None
    assert force_motion.any()


def test_preprocess_gripper_events_prefer_real_state_over_action_command_noise():
    names = flexiv_state_names()
    actions = np.zeros((4, 14), dtype=np.float32)
    actions[1:, -2:] = 1.0
    states = np.zeros((4, FLEXIV_STATE_DIM), dtype=np.float32)
    cfg = {"gripper_events": {"enabled": True, "change_threshold": 0.5}}

    action_noise = preprocess_dataset._gripper_event_mask(
        actions,
        flexiv_action_names(),
        states,
        names,
        cfg,
    )
    assert not action_noise.any()

    states[2:, 16] = 1.0
    state_event = preprocess_dataset._gripper_event_mask(
        actions,
        flexiv_action_names(),
        states,
        names,
        cfg,
    )
    assert state_event.any()


def test_split_motion_filter_excludes_force_noise_and_gripper_force_is_not_width():
    names = flexiv_state_names()
    states = np.zeros((4, FLEXIV_STATE_DIM), dtype=np.float32)
    states[1:, 34:] = np.arange(1, 15, dtype=np.float32)
    cfg = {"_fps": 30.0, "action_filter": {"state_rate": {"include_force": False}}}

    activity = split_label_dataset._state_rate_activity_mask(states, names, cfg)
    assert activity is not None
    assert not activity.any()
    assert split_label_dataset._gripper_indices(names, prefer_state=True) == [16, 33]

    opt_in = {"_fps": 30.0, "action_filter": {"state_rate": {"include_force": True}}}
    force_activity = split_label_dataset._state_rate_activity_mask(states, names, opt_in)
    assert force_activity is not None
    assert force_activity.any()


def test_offline_visualizer_logs_all_48_metadata_named_dimensions():
    rerun = _FakeRerun()
    values = np.arange(FLEXIV_STATE_DIM, dtype=np.float32)

    paths = log_named_vector(rerun, "observation.state", values, flexiv_state_names())

    assert len(paths) == FLEXIV_STATE_DIM
    assert len(rerun.logs) == FLEXIV_STATE_DIM
    assert paths[-14:] == tuple(
        f"observation.state/{name}" for name in flexiv_state_names()[34:]
    )


def test_rewrite_frame_helpers_preserve_full_state_and_action_vectors():
    features = _features()
    source = SimpleNamespace(features=features)
    state = np.arange(FLEXIV_STATE_DIM, dtype=np.float32)
    action = np.arange(14, dtype=np.float32)
    item = {"observation.state": state, "action": action, "task": "task"}

    preprocess_frame = preprocess_dataset._frame_from_source_item(source, item, action)
    split_frame = split_label_dataset._frame_from_source_item(source, item, "new task")
    merge_frame = merge_tool._frame_from_source_item(source, item)

    for frame in (preprocess_frame, split_frame, merge_frame):
        np.testing.assert_array_equal(frame["observation.state"], state)
        np.testing.assert_array_equal(frame["action"], action)
        assert frame["observation.state"].shape == (FLEXIV_STATE_DIM,)


def test_rewrite_metadata_propagation_keeps_v3_contract(tmp_path):
    features = _features()
    source_info = _flexiv_info(features)
    output_root = tmp_path / "rewritten"
    (output_root / "meta").mkdir(parents=True)
    (output_root / "meta" / "info.json").write_text(
        json.dumps({"robot_type": "flexiv_dual_arm", "features": features}),
        encoding="utf-8",
    )
    output = SimpleNamespace(
        features=features,
        meta=SimpleNamespace(
            root=output_root,
            info={"robot_type": "flexiv_dual_arm", "features": features},
        ),
    )

    schema = propagate_flexiv_dataset_schema(
        source_info,
        output,
        source_features=features,
        output_features=features,
        source="test rewrite",
    )

    assert schema == source_info[FLEXIV_SCHEMA_INFO_KEY]
    assert output.meta.info[FLEXIV_SCHEMA_INFO_KEY]["state_dim"] == 48
    persisted = json.loads((output_root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert persisted[FLEXIV_SCHEMA_INFO_KEY] == source_info[FLEXIV_SCHEMA_INFO_KEY]

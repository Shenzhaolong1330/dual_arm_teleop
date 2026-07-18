import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.dual_flexiv_rizon4s.flexiv_dual_arm import (
    FlexivDualArm,
    _apply_delta_to_pose7,
    _pose7_to_absolute_xyz_rot6d,
)
from robots.dual_flexiv_rizon4s.flexiv_state_schema import (
    FLEXIV_ACTION_DIM,
    FLEXIV_STATE_DIM,
    FLEXIV_STATE_SCHEMA,
    STATE_FORCE_FIELDS,
    build_flexiv_state_schema,
    flexiv_action_names,
    flexiv_kinematic_state_names,
    flexiv_state_names,
    persist_flexiv_checkpoint_schema,
    persist_flexiv_dataset_schema,
    validate_flexiv_checkpoint,
    validate_flexiv_dataset_schema,
)


def _pose7(rotation: Rotation, position=(0.1, -0.2, 0.3)) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    return np.asarray((*position, quat_xyzw[3], *quat_xyzw[:3]), dtype=float)


def _feature_map(state_dim=FLEXIV_STATE_DIM, state_names=None, action_dim=FLEXIV_ACTION_DIM):
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": flexiv_state_names() if state_names is None else state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": flexiv_action_names(),
        },
    }


def test_identity_quaternion_uses_explicit_column_0_column_1_order():
    _, rotation_6d = _pose7_to_absolute_xyz_rot6d(_pose7(Rotation.identity()))
    np.testing.assert_array_equal(rotation_6d, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_quaternion_is_normalized_and_zero_norm_is_rejected():
    pose = _pose7(Rotation.from_euler("x", 0.4))
    scaled = pose.copy()
    scaled[3:] *= 7.0
    np.testing.assert_allclose(
        _pose7_to_absolute_xyz_rot6d(pose)[1],
        _pose7_to_absolute_xyz_rot6d(scaled)[1],
        rtol=0.0,
        atol=1e-12,
    )
    invalid = pose.copy()
    invalid[3:] = 0.0
    with pytest.raises(ValueError, match="near-zero norm"):
        _pose7_to_absolute_xyz_rot6d(invalid)


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_known_rotations_match_matrix_column_convention(axis):
    rotation = Rotation.from_euler(axis, 0.73)
    _, actual = _pose7_to_absolute_xyz_rot6d(_pose7(rotation))
    matrix = rotation.as_matrix()
    expected = np.concatenate((matrix[:, 0], matrix[:, 1]))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_quaternion_sign_does_not_change_rotation_6d():
    pose = _pose7(Rotation.from_euler("xyz", [0.3, -0.7, 1.2]))
    negated = pose.copy()
    negated[3:] *= -1.0
    np.testing.assert_allclose(
        _pose7_to_absolute_xyz_rot6d(pose)[1],
        _pose7_to_absolute_xyz_rot6d(negated)[1],
        rtol=0.0,
        atol=1e-12,
    )


def test_pi_crossing_is_continuous_in_rotation_6d():
    axis = np.array([0.3, -0.4, 0.5], dtype=float)
    axis /= np.linalg.norm(axis)
    epsilon = 1e-7
    before = Rotation.from_rotvec(axis * (np.pi - epsilon))
    after = Rotation.from_rotvec(axis * (np.pi + epsilon))
    distance = np.linalg.norm(
        _pose7_to_absolute_xyz_rot6d(_pose7(before))[1]
        - _pose7_to_absolute_xyz_rot6d(_pose7(after))[1]
    )
    assert distance < 1e-5


def test_rotation_6d_is_finite_and_orthonormal_for_mock_rdk_state():
    _, rotation_6d = _pose7_to_absolute_xyz_rot6d(_pose7(Rotation.from_euler("zyx", [1.1, -0.5, 0.2])))
    c0, c1 = rotation_6d[:3], rotation_6d[3:]
    assert np.isfinite(rotation_6d).all()
    np.testing.assert_allclose(np.linalg.norm(c0), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(c1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(c0, c1), 0.0, atol=1e-12)


class _MockRdkStates:
    def __init__(self, pose7, wrench=(1, -2, 3, -4, 5, -6)):
        self.q = np.arange(7, dtype=float)
        self.tcp_pose = pose7
        self.ext_wrench_in_tcp_raw = np.asarray(wrench, dtype=float)


class _MockRdkRobot:
    def __init__(self, pose7, wrench=(1, -2, 3, -4, 5, -6)):
        self._states = _MockRdkStates(pose7, wrench)
        self.states_calls = 0

    def states(self):
        self.states_calls += 1
        return self._states


class _MockRdkGripperStates:
    def __init__(self, width, force):
        self.width = width
        self.force = force
        self.is_moving = False


class _MockRdkGripper:
    def __init__(self, width, force):
        self._states = _MockRdkGripperStates(width, force)
        self.states_calls = 0

    def states(self):
        self.states_calls += 1
        return self._states


def _mock_observation_robot():
    robot = FlexivDualArm.__new__(FlexivDualArm)
    robot._left_robot_lock = threading.Lock()
    robot._right_robot_lock = threading.Lock()
    robot._num_joints_per_arm = 7
    robot._cached_left_pose7 = np.zeros(7, dtype=float)
    robot._cached_right_pose7 = np.zeros(7, dtype=float)
    robot.config = SimpleNamespace(
        use_gripper=True,
        gripper_min_width=0.0,
        gripper_max_open=0.1,
        save_rgbd_timestamps=False,
        timing_debug=False,
    )
    robot._left_gripper_cmd = 1.0
    robot._right_gripper_cmd = 1.0
    robot._left_gripper_width = None
    robot._right_gripper_width = None
    robot._left_gripper = _MockRdkGripper(0.04, -7.5)
    robot._right_gripper = _MockRdkGripper(0.05, 8.25)
    robot._left_robot = _MockRdkRobot(_pose7(Rotation.from_euler("z", 0.4)), (1, 2, 3, 4, 5, 6))
    robot._right_robot = _MockRdkRobot(_pose7(Rotation.from_euler("x", -0.2)), (-10, -20, -30, -40, -50, -60))
    robot.cameras = {}
    robot._prev_observation = None
    robot._timing_debug_counts = {}
    return robot


def test_mock_rdk_observation_has_absolute_xyz_rotation6d_and_no_rotvec_fields():
    robot = FlexivDualArm.__new__(FlexivDualArm)
    robot._left_robot_lock = threading.Lock()
    robot._num_joints_per_arm = 7
    robot._cached_left_pose7 = np.zeros(7, dtype=float)
    robot.config = SimpleNamespace(
        use_gripper=True,
        gripper_min_width=0.0,
        gripper_max_open=0.1,
    )
    robot._left_gripper_cmd = 1.0
    robot._left_gripper = _MockRdkGripper(0.04, -7.5)
    robot._left_gripper_width = None

    observation = {}
    force_values = robot._add_arm_observation(
        observation,
        "left",
        _MockRdkRobot(_pose7(Rotation.from_euler("z", 0.4))),
    )

    assert list(observation) == flexiv_state_names()[:17]
    assert not any(name.startswith("left_ee_pose.r") for name in observation)
    assert all(np.isfinite(value) for value in observation.values())
    assert force_values == ((1.0, -2.0, 3.0, -4.0, 5.0, -6.0), -7.5)


def test_v3_state_names_are_exactly_legacy_34d_then_force_tail():
    names = flexiv_state_names()
    assert FLEXIV_STATE_DIM == 48
    assert len(names) == 48
    assert names[:34] == flexiv_kinematic_state_names()
    assert names[34:] == list(STATE_FORCE_FIELDS)
    assert names[34:] == [
        "left_ee_ext_wrench_in_tcp_raw.fx",
        "left_ee_ext_wrench_in_tcp_raw.fy",
        "left_ee_ext_wrench_in_tcp_raw.fz",
        "left_ee_ext_wrench_in_tcp_raw.mx",
        "left_ee_ext_wrench_in_tcp_raw.my",
        "left_ee_ext_wrench_in_tcp_raw.mz",
        "left_gripper_force",
        "right_ee_ext_wrench_in_tcp_raw.fx",
        "right_ee_ext_wrench_in_tcp_raw.fy",
        "right_ee_ext_wrench_in_tcp_raw.fz",
        "right_ee_ext_wrench_in_tcp_raw.mx",
        "right_ee_ext_wrench_in_tcp_raw.my",
        "right_ee_ext_wrench_in_tcp_raw.mz",
        "right_gripper_force",
    ]


def test_full_observation_is_48d_and_reads_each_side_snapshot_once():
    robot = _mock_observation_robot()
    robot._is_connected = True

    observation = robot.get_observation()
    raw_values = np.asarray([observation[name] for name in flexiv_state_names()], dtype=np.float64)
    values = raw_values.astype(np.float32)

    # This is the old v2 value contract reconstructed from the same RDK
    # snapshots: the v3 tail must not alter any of the original 34 values.
    left_position, left_rotation_6d = _pose7_to_absolute_xyz_rot6d(
        robot._left_robot._states.tcp_pose
    )
    right_position, right_rotation_6d = _pose7_to_absolute_xyz_rot6d(
        robot._right_robot._states.tcp_pose
    )
    legacy_v2_values = np.concatenate(
        (
            robot._left_robot._states.q,
            left_position,
            left_rotation_6d,
            [robot._gripper_state_norm_from_width(robot._left_gripper._states.width)],
            robot._right_robot._states.q,
            right_position,
            right_rotation_6d,
            [robot._gripper_state_norm_from_width(robot._right_gripper._states.width)],
        )
    )

    assert values.shape == (48,)
    assert list(observation)[:34] == flexiv_kinematic_state_names()
    assert list(observation)[34:] == list(STATE_FORCE_FIELDS)
    np.testing.assert_allclose(raw_values[:34], legacy_v2_values, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(raw_values[34:], np.array([1, 2, 3, 4, 5, 6, -7.5, -10, -20, -30, -40, -50, -60, 8.25]))
    assert robot._left_robot.states_calls == 1
    assert robot._right_robot.states_calls == 1
    assert robot._left_gripper.states_calls == 1
    assert robot._right_gripper.states_calls == 1


def test_observation_features_and_action_contract_are_48d_and_14d():
    robot = _mock_observation_robot()
    robot.config.control_mode = "oculus"
    assert list(robot.observation_features) == flexiv_state_names()
    assert len(robot.observation_features) == 48
    assert list(robot.action_features) == flexiv_action_names()
    assert len(robot.action_features) == 14


@pytest.mark.parametrize(
    "bad_wrench",
    [
        None,
        (1, 2, 3),
        "123456",
        (1, 2, 3, 4, 5, float("nan")),
        (1, 2, 3, 4, 5, float("inf")),
    ],
)
def test_bad_wrench_fails_fast_without_reusing_previous_observation(bad_wrench):
    robot = _mock_observation_robot()
    robot._is_connected = True
    robot._prev_observation = {"left_joint_1.pos": 123.0}
    if bad_wrench is None:
        del robot._left_robot._states.ext_wrench_in_tcp_raw
    else:
        robot._left_robot._states.ext_wrench_in_tcp_raw = bad_wrench

    with pytest.raises(ValueError, match="ext_wrench_in_tcp_raw|not finite|length 6"):
        robot.get_observation()


def test_missing_or_nonfinite_gripper_force_fails_fast_and_signed_zero_is_preserved():
    robot = _mock_observation_robot()
    robot._is_connected = True
    robot._left_gripper._states.force = -3.75
    observation = robot.get_observation()
    assert observation["left_gripper_force"] == -3.75

    del robot._left_gripper._states.force
    with pytest.raises(ValueError, match=r"gripper.states\(\).force|force.*missing"):
        robot.get_observation()

    robot = _mock_observation_robot()
    robot._is_connected = True
    robot._right_gripper._states.force = float("nan")
    with pytest.raises(ValueError, match="force.*not finite"):
        robot.get_observation()


def test_action_features_remain_the_existing_14d_delta_rotvec_contract():
    robot = FlexivDualArm.__new__(FlexivDualArm)
    robot._num_joints_per_arm = 7
    robot.config = SimpleNamespace(control_mode="oculus", use_gripper=True)
    assert list(robot.action_features) == flexiv_action_names()
    assert len(robot.action_features) == FLEXIV_ACTION_DIM


def test_delta_rotvec_execution_helper_still_applies_incremental_rotation():
    current = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=float)
    delta = np.array([0.01, -0.02, 0.03, 0.0, 0.0, np.pi / 2.0], dtype=float)
    target = _apply_delta_to_pose7(current, delta)
    np.testing.assert_allclose(target[:3], current[:3] + delta[:3])
    expected_rotation = Rotation.from_rotvec(delta[3:])
    actual_rotation = Rotation.from_quat([target[4], target[5], target[6], target[3]])
    np.testing.assert_allclose(actual_rotation.as_matrix(), expected_rotation.as_matrix(), atol=1e-12)


def test_dataset_schema_persists_v3_metadata_and_rejects_legacy_34d_and_28d(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {"robot_type": "flexiv_dual_arm", "features": {}}
    schema = persist_flexiv_dataset_schema(root, info, zero_ft_sensor_on_connect=True)
    loaded_info = json.loads((root / "meta" / "info.json").read_text())
    assert loaded_info["robot_state_schema"] == schema
    assert schema["state_schema"] == FLEXIV_STATE_SCHEMA == "flexiv_abs_rot6d_raw_force_v3"
    assert schema["state_dim"] == 48
    assert schema["action_dim"] == 14
    assert schema["state_names"] == flexiv_state_names()
    assert schema["wrench_source"] == "robot.states().ext_wrench_in_tcp_raw"
    assert schema["wrench_frame"] == "tcp"
    assert schema["wrench_order"] == ["fx", "fy", "fz", "mx", "my", "mz"]
    assert schema["wrench_units"] == ["N", "N", "N", "Nm", "Nm", "Nm"]
    assert schema["gripper_force_source"] == "gripper.states().force"
    assert schema["gripper_force_unit"] == "N"
    assert schema["gripper_force_sign_convention"] == "preserve_raw_signed_value"
    assert schema["software_filter"] == "none"
    assert schema["zero_ft_sensor_on_connect"] is True

    validate_flexiv_dataset_schema(
        loaded_info,
        _feature_map(),
        source="new dataset",
    )

    with pytest.raises(ValueError, match="state_dim=34|v2/34D"):
        validate_flexiv_dataset_schema(
            loaded_info,
            _feature_map(state_dim=34, state_names=flexiv_kinematic_state_names()),
            source="legacy v2 resume dataset",
        )

    legacy_names = [
        f"legacy_{index}" for index in range(28)
    ]
    with pytest.raises(ValueError, match="state_dim=28|v1/28D"):
        validate_flexiv_dataset_schema(
            loaded_info,
            _feature_map(state_dim=28, state_names=legacy_names),
            source="legacy resume dataset",
        )


def test_checkpoint_schema_validation_rejects_old_shape_and_requires_metadata(tmp_path):
    old = tmp_path / "old" / "pretrained_model"
    old.mkdir(parents=True)
    (old / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"shape": [28]}},
                "output_features": {"action": {"shape": [14]}},
            }
        )
    )
    with pytest.raises(ValueError, match="checkpoint state_dim=28|old 28D"):
        validate_flexiv_checkpoint(old, source="old checkpoint")

    current = tmp_path / "current" / "pretrained_model"
    current.mkdir(parents=True)
    (current / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"shape": [48]}},
                "output_features": {"action": {"shape": [14]}},
            }
        )
    )
    with pytest.raises(ValueError, match="no persisted Flexiv schema metadata"):
        validate_flexiv_checkpoint(current, source="unannotated checkpoint")

    persist_flexiv_checkpoint_schema(current)
    validate_flexiv_checkpoint(current, source="current checkpoint")


def test_checkpoint_schema_rejects_old_v2_34d_even_with_action_14d(tmp_path):
    old = tmp_path / "old_v2" / "pretrained_model"
    old.mkdir(parents=True)
    (old / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"shape": [34]}},
                "output_features": {"action": {"shape": [14]}},
            }
        )
    )
    with pytest.raises(ValueError, match="state_dim=34|v2/34D|48D"):
        validate_flexiv_checkpoint(old, source="old v2 checkpoint")


def test_checkpoint_schema_rejects_wrong_v3_feature_names(tmp_path):
    checkpoint = tmp_path / "wrong_names" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {
                        "shape": [48],
                        "names": ["wrong_state_name"] * 48,
                    }
                },
                "output_features": {"action": {"shape": [14]}},
            }
        )
    )
    persist_flexiv_checkpoint_schema(checkpoint)
    with pytest.raises(ValueError, match="feature-name schema mismatch|v3 order"):
        validate_flexiv_checkpoint(checkpoint, source="wrong-name checkpoint")

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
    flexiv_action_names,
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
    def __init__(self, pose7):
        self.q = np.arange(7, dtype=float)
        self.tcp_pose = pose7


class _MockRdkRobot:
    def __init__(self, pose7):
        self._states = _MockRdkStates(pose7)

    def states(self):
        return self._states


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
    robot._left_gripper = None

    observation = {}
    robot._add_arm_observation(
        observation,
        "left",
        _MockRdkRobot(_pose7(Rotation.from_euler("z", 0.4))),
    )

    assert list(observation) == flexiv_state_names()[:17]
    assert not any(name.startswith("left_ee_pose.r") for name in observation)
    assert all(np.isfinite(value) for value in observation.values())


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


def test_dataset_schema_persists_and_rejects_legacy_28d(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {"robot_type": "flexiv_dual_arm", "features": {}}
    schema = persist_flexiv_dataset_schema(root, info)
    loaded_info = json.loads((root / "meta" / "info.json").read_text())
    assert loaded_info["robot_state_schema"] == schema

    validate_flexiv_dataset_schema(
        loaded_info,
        _feature_map(),
        source="new dataset",
    )

    legacy_names = [
        name.replace("ee_rotation_6d.c0x", "ee_pose.rx")
        for name in flexiv_state_names()[:28]
    ]
    with pytest.raises(ValueError, match="state_dim=28|old 28D"):
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
                "input_features": {"observation.state": {"shape": [34]}},
                "output_features": {"action": {"shape": [14]}},
            }
        )
    )
    with pytest.raises(ValueError, match="no persisted Flexiv schema metadata"):
        validate_flexiv_checkpoint(current, source="unannotated checkpoint")

    persist_flexiv_checkpoint_schema(current)
    validate_flexiv_checkpoint(current, source="current checkpoint")

"""Hardware-free contract, basis-integrity, and initial-posture regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.primeu.action_codec import ActionCodec, decode_finger
from examples.primeu.contract import (
    CAMERAS, STATE_SLICES, expected_action_slices, export_contract,
    load_checkpoint_contract, load_dataset_contract,
)


def make_basis(side):
    names = ["index", "middle", "ring", "little"] if side == "left" else ["thumb", "index", "middle", "ring", "little"]
    fingers = []
    for number, name in enumerate(names, start=2 if side == "left" else 1):
        fingers.append({
            "name": name, "degree": 2, "coordinate_range": [-1.0, 1.0],
            "joint_names": [f"{side}_finger{number}_link{j}_joint" for j in range(1, 5)],
            "joint_lower": [-2] * 4, "joint_upper": [2] * 4,
            "clip_joint_limits": True,
            "coefficients": [[0.1 * number] * 4, [0.2, 0.3, 0.4, 0.5], [0.01] * 4],
        })
    return {
        "kind": "chebyshev_taskspace", "version": 1, "hand": side,
        "latent_order": (["thumb_j1", "thumb_j2", "thumb_j3", "thumb_j4"] if side == "left" else ["thumb_synergy"]) +
                        [f"{name}_synergy" for name in ["index", "middle", "ring", "little"]],
        "fingers": fingers,
    }


def make_dataset(root, dim):
    meta = root / "meta"
    meta.mkdir(parents=True)
    slices = expected_action_slices(dim)
    info = {"features": {"observation.state": {"shape": [60]}, "action": {"shape": [dim]},
                         **{f"observation.images.{name}": {"dtype": "video"} for name in CAMERAS}}}
    modality = {"state": STATE_SLICES, "action": slices,
                "video": {name: {"original_key": f"observation.images.{name}"} for name in CAMERAS}}
    layout = {"action_dim": dim, "state_dim": 60, "action_slices": slices, "state_slices": STATE_SLICES,
              "raw_body_actions": "absolute joint targets; arms become relative in GR00T processor",
              "locked_body_absolute_target": [-999] * 33}
    for side in ("left", "right"):
        if f"{side}_hand_synergy" in slices:
            name = f"{side}_basis.json"
            layout[f"{side}_hand_basis"] = name
            (meta / name).write_text(json.dumps(make_basis(side)))
    for name, payload in (("info", info), ("modality", modality), ("action_layout", layout)):
        (meta / f"{name}.json").write_text(json.dumps(payload))
    return root


class ContractCodecTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def codec(self, dim):
        root = make_dataset(self.root / str(dim), dim)
        return ActionCodec(load_dataset_contract(root), root / "meta")

    def test_all_layouts_decode_with_correct_absolute_arm_slices(self):
        for dim in (60, 33, 27, 15, 12):
            with self.subTest(dim=dim):
                codec = self.codec(dim)
                actions = np.arange(3 * dim, dtype=np.float32).reshape(3, dim) * 0.001
                state = np.arange(60, dtype=np.float32) + 100
                result = codec.decode(actions, state)
                self.assertEqual(result.shape, (3, 60))
                self.assertEqual(result.dtype, np.float32)
                for group, bounds in codec.contract.action_slices.items():
                    if "synergy" not in group:
                        target = STATE_SLICES[group]
                        np.testing.assert_array_equal(result[:, target["start"]:target["end"]],
                                                      actions[:, bounds["start"]:bounds["end"]])
                for group in codec.contract.held_groups:
                    bounds = STATE_SLICES[group]
                    np.testing.assert_array_equal(result[:, bounds["start"]:bounds["end"]],
                                                  np.tile(state[bounds["start"]:bounds["end"]], (3, 1)))
                if dim == 60:
                    np.testing.assert_array_equal(result, actions)

    def test_unpredicted_groups_hold_initial_pose_until_reset(self):
        for dim in (27, 15, 12):
            with self.subTest(dim=dim):
                codec = self.codec(dim)
                state = np.arange(60, dtype=np.float32)
                action = np.zeros((2, dim), dtype=np.float32)
                first = codec.decode(action, state)
                state += 1000  # caller mutation must not mutate latched state
                second = codec.decode(action, state)
                np.testing.assert_array_equal(first, second)
                codec.reset()
                third = codec.decode(action, state)
                for group in codec.contract.held_groups:
                    bounds = STATE_SLICES[group]
                    np.testing.assert_array_equal(third[:, bounds["start"]:bounds["end"]],
                                                  first[:, bounds["start"]:bounds["end"]] + 1000)

    def test_invalid_observation_or_action_never_latches_posture(self):
        codec = self.codec(15)
        actions = np.zeros((2, 15))
        for state in (np.zeros(59), np.full(60, np.nan), np.full(60, np.inf)):
            with self.assertRaises(ValueError):
                codec.decode(actions, state)
        for bad in (np.zeros((0, 15)), np.zeros((2, 12)), np.zeros(15), np.full((2, 15), np.inf)):
            with self.assertRaises(ValueError):
                codec.decode(bad, np.ones(60))
        result = codec.decode(actions, np.full(60, 7.0))
        np.testing.assert_array_equal(result[:, 27:], np.full((2, 33), 7.0))

    def test_chebyshev_source_math_coordinate_and_joint_clipping(self):
        curve = make_basis("right")["fingers"][0]
        coordinates = np.array([-2.0, -0.7, 0.0, 0.3, 2.0])
        clipped = coordinates.clip(-1, 1)
        coeff = np.asarray(curve["coefficients"])
        expected = coeff[0] + clipped[:, None] * coeff[1] + (2 * clipped[:, None] ** 2 - 1) * coeff[2]
        np.testing.assert_allclose(decode_finger(coordinates, curve), expected, atol=1e-15)
        curve["joint_upper"] = [0.15] * 4
        np.testing.assert_allclose(decode_finger(coordinates, curve), expected.clip(-2, 0.15), atol=1e-15)

    def test_thumb_absolute_values_and_independent_hand_coordinates(self):
        codec = self.codec(33)
        actions = np.zeros((2, 33), dtype=np.float32)
        actions[:, 7:11] = [0.9, -0.8, 0.7, -0.6]
        actions[:, 11:15] = -0.5
        actions[:, 22:27] = 0.6
        decoded = codec.decode(actions, np.zeros(60))
        np.testing.assert_array_equal(decoded[:, 7:11], actions[:, 7:11])
        left_index = decode_finger(np.array([-0.5, -0.5]), codec.bases["left"]["fingers"][0])
        right_index = decode_finger(np.array([0.6, 0.6], np.float32), codec.bases["right"]["fingers"][1])
        np.testing.assert_allclose(decoded[:, 11:15], left_index, atol=1e-7)
        np.testing.assert_allclose(decoded[:, 38:42], right_index, atol=1e-7)

    def test_export_load_and_reject_basis_tampering(self):
        dataset = make_dataset(self.root / "data", 27)
        checkpoint = self.root / "checkpoint"
        exported = export_contract(dataset, checkpoint)
        self.assertEqual(load_checkpoint_contract(checkpoint), exported)
        self.assertEqual(ActionCodec.from_directory(checkpoint).contract, exported)
        basis = checkpoint / "primeu" / exported.basis_files["left"]
        basis.write_text(basis.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_checkpoint_contract(checkpoint)

    def test_reject_layout_reorder_basis_side_and_nonfinite_coefficients(self):
        dataset = make_dataset(self.root / "data", 33)
        path = dataset / "meta/modality.json"
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed["action"]["left_arm"], changed["action"]["right_arm"] = changed["action"]["right_arm"], changed["action"]["left_arm"]
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "field order"):
            load_dataset_contract(dataset)
        path.write_text(json.dumps(original))
        path = dataset / "meta/left_basis.json"
        basis = json.loads(path.read_text())
        for mutate in (lambda item: item.update(hand="right"),
                       lambda item: item["fingers"][0]["coefficients"][0].__setitem__(0, float("nan"))):
            changed = copy.deepcopy(basis)
            mutate(changed)
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_dataset_contract(dataset)

    def test_export_refuses_changed_dataset_normalization_fingerprint(self):
        dataset = make_dataset(self.root / "data", 60)
        checkpoint = self.root / "checkpoint"
        export_contract(dataset, checkpoint)
        (dataset / "meta/stats.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "different checkpoint contract"):
            export_contract(dataset, checkpoint)


if __name__ == "__main__":
    unittest.main()

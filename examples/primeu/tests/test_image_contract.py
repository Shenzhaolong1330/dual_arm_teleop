"""Full-resolution camera contracts without a GPU, model allocation, or robot."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.primeu import client
from examples.primeu.policy import camera_image_contract, validate_processed_image_shapes
from examples.primeu.server import PrimeuPolicyAdapter


NATIVE_SIZES = {"head": [720, 1280], "left_wrist": [480, 640], "right_wrist": [480, 640]}


def features(sizes):
    return {f"observation.images.{camera}": SimpleNamespace(shape=(3, *size)) for camera, size in sizes.items()}


def test_native_camera_shapes_have_explicit_contract():
    contract = camera_image_contract(features(NATIVE_SIZES))
    assert contract == {"image_sizes": NATIVE_SIZES, "image_size": None}
    # Shape-only objects ensure this test does not allocate full-resolution tensors.
    batch = {key: SimpleNamespace(shape=(1, *feature.shape)) for key, feature in features(NATIVE_SIZES).items()}
    validate_processed_image_shapes(batch, contract["image_sizes"])


def test_uniform_camera_contract_retains_legacy_field():
    sizes = {camera: [256, 256] for camera in NATIVE_SIZES}
    assert camera_image_contract(features(sizes)) == {"image_sizes": sizes, "image_size": [256, 256]}


@pytest.mark.parametrize("camera", NATIVE_SIZES)
def test_postprocessor_wrong_camera_size_rejected(camera):
    batch = {key: SimpleNamespace(shape=(1, *feature.shape)) for key, feature in features(NATIVE_SIZES).items()}
    batch[f"observation.images.{camera}"] = SimpleNamespace(shape=(1, 3, 256, 256))
    with pytest.raises(ValueError, match=f"Processed {camera} image shape"):
        validate_processed_image_shapes(batch, NATIVE_SIZES)


@pytest.fixture
def cfg():
    path = Path(__file__).parents[1] / "configs/robot.yaml"
    result = yaml.safe_load(path.read_text())
    result["server"].pop("expected_image_size", None)
    result["server"]["expected_image_sizes"] = copy.deepcopy(NATIVE_SIZES)
    return result


def write_config(tmp_path, cfg):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_client_accepts_full_resolution_config(tmp_path, cfg):
    loaded = client.load_config(write_config(tmp_path, cfg))
    assert loaded["server"]["expected_image_sizes"] == NATIVE_SIZES
    assert "expected_image_size" not in loaded["server"]


def test_client_rejects_conflicting_image_modes(tmp_path, cfg):
    cfg["server"]["expected_image_size"] = [256, 256]
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.load_config(write_config(tmp_path, cfg))


@pytest.mark.parametrize("sizes", [
    {}, {"head": [720, 1280]},
    {**NATIVE_SIZES, "head": [720, 0]},
    {**NATIVE_SIZES, "left_wrist": [False, 640]},
    {**NATIVE_SIZES, "right_wrist": [480, 640.0]},
])
def test_client_rejects_invalid_camera_sizes(tmp_path, cfg, sizes):
    cfg["server"]["expected_image_sizes"] = sizes
    with pytest.raises(ValueError, match="[Ii]mage.*size"):
        client.load_config(write_config(tmp_path, cfg))


@pytest.mark.parametrize("mismatch", [False, True])
def test_client_checks_each_camera_in_handshake(monkeypatch, cfg, mismatch):
    server = cfg["server"]
    contract = {
        "protocol": "primeu_lerobot_v1", "model_family": server["expected_model_family"],
        "state_dim": 60, "action_dim": 60, "model_action_dim": server["expected_model_action_dim"],
        "joint_groups": client.JOINT_DIMS, "joint_order": list(client.JOINT_DIMS),
        "camera_keys": client.CAMERAS, "action_representation": "absolute_joint_target",
        "action_horizon": server["expected_horizon"], "basis_sha256": server["expected_basis_sha256"],
        **camera_image_contract(features(NATIVE_SIZES)),
    }
    if mismatch:
        contract["image_sizes"]["head"] = [256, 256]
    policy = SimpleNamespace(contract=contract, horizon=server["expected_horizon"])
    adapter = PrimeuPolicyAdapter(policy)
    monkeypatch.setattr(client, "rpc", lambda settings, endpoint: adapter.dispatch({"endpoint": endpoint}))
    if mismatch:
        with pytest.raises(ValueError, match="expected LeRobot"):
            client.check_server(cfg)
    else:
        assert client.check_server(cfg)["contract"]["image_sizes"] == NATIVE_SIZES

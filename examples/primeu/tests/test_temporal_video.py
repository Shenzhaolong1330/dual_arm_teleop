"""Explicit single-frame temporal video queries must keep the requested time axis."""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from lerobot.datasets import lerobot_dataset as dataset_module


@pytest.mark.parametrize(("delta_indices", "timestamps", "shape"), [
    (None, [0.0], (3, 8, 12)),
    ({"observation.images.head": [0]}, [0.0], (1, 3, 8, 12)),
    ({"observation.images.head": [-1, 0]}, [0.0, 1 / 30], (2, 3, 8, 12)),
    ({"action": [0, 1]}, [0.0], (3, 8, 12)),
])
def test_video_temporal_axis_and_episode_offset(monkeypatch, delta_indices, timestamps, shape):
    key = "observation.images.head"
    seen = []

    def decode(path, query, tolerance, backend):
        seen.append((path, query, tolerance, backend))
        return torch.zeros((len(query), 3, 8, 12), dtype=torch.float32)

    monkeypatch.setattr(dataset_module, "decode_video_frames", decode)
    dataset = object.__new__(dataset_module.LeRobotDataset)
    dataset.delta_indices = delta_indices
    dataset.root = Path("/unused/robot-free-fixture")
    dataset.tolerance_s = 0.01
    dataset.video_backend = "torchcodec"
    dataset.meta = SimpleNamespace(
        episodes=[{f"videos/{key}/from_timestamp": 2.0}],
        get_video_file_path=lambda episode, camera: Path("videos/head.mp4"),
    )
    result = dataset._query_videos({key: timestamps}, ep_idx=0)
    assert tuple(result[key].shape) == shape
    assert seen == [(dataset.root / "videos/head.mp4", [2.0 + t for t in timestamps], 0.01, "torchcodec")]


def test_single_frame_diffusion_collation_keeps_camera_channels_separate(monkeypatch):
    keys = [f"observation.images.{camera}" for camera in ("head", "left_wrist", "right_wrist")]
    monkeypatch.setattr(dataset_module, "decode_video_frames",
                        lambda *args: torch.zeros((1, 3, 8, 12), dtype=torch.float32))
    dataset = object.__new__(dataset_module.LeRobotDataset)
    dataset.delta_indices = {key: [0] for key in keys}
    dataset.root, dataset.tolerance_s, dataset.video_backend = Path("/unused"), 0.01, "torchcodec"
    dataset.meta = SimpleNamespace(
        episodes=[{f"videos/{key}/from_timestamp": 0.0 for key in keys}],
        get_video_file_path=lambda episode, camera: Path("fixture.mp4"),
    )
    sample = dataset._query_videos({key: [0.0] for key in keys}, 0)
    sample["observation.state"] = torch.zeros((1, 60))
    batch = torch.utils.data.default_collate([sample, sample])
    images = torch.stack([batch[key] for key in keys], dim=-4)
    assert tuple(batch["observation.state"].shape) == (2, 1, 60)
    assert tuple(images.shape) == (2, 1, 3, 3, 8, 12)
    # Native Diffusion collapses B,T,N before the shared RGB encoder.
    flattened = images.flatten(0, 2)
    assert tuple(flattened.shape) == (6, 3, 8, 12)

"""Offline PRIMEU dataset replay: direct, ZMQ, or fixed-seed comparison; no robot SDK."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .wire import CAMERAS, JOINT_GROUPS, PROTOCOL, from_runtime, rpc, to_runtime


def _path(value, config_path):
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.parent / path


def sample_to_runtime(sample):
    """LeRobot decodes CHW float RGB in [0,1]; restore lossless uint8 wire RGB."""
    state = sample["observation.state"].detach().cpu().numpy()
    physical = {"observation.state": state, "task": sample["task"]}
    for key in CAMERAS:
        name = f"observation.images.{key}"
        image = sample[name].detach().cpu().numpy()
        if (image.ndim != 3 or image.shape[0] != 3 or not np.isfinite(image).all()
                or image.min() < 0 or image.max() > 1):
            raise ValueError(f"Decoded {name} must be finite CHW RGB in [0,1]")
        physical[name] = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
    return to_runtime(physical)


def _response_actions(response, horizon):
    if not isinstance(response, (list, tuple)) or len(response) != 2:
        raise ValueError("Unexpected policy response")
    grouped, _ = response
    parts = []
    for name, (start, end) in JOINT_GROUPS.items():
        part = np.asarray(grouped[name])
        if part.dtype != np.float32 or part.shape != (1, horizon, end - start) or not np.isfinite(part).all():
            raise ValueError(f"Invalid response group {name}")
        parts.append(part[0])
    return np.concatenate(parts, axis=-1)


def run(config, config_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    mode = config.get("mode", "compare")
    if mode not in ("direct", "remote", "compare"):
        raise ValueError("mode must be direct, remote or compare")
    frames = config.get("frame_indices", [0])
    if not isinstance(frames, list) or not frames or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in frames):
        raise ValueError("frame_indices must be a nonempty list of nonnegative integers")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 - len(frames):
        raise ValueError("Invalid replay seed")
    root = _path(config["dataset_root"], config_path).resolve(strict=True)
    dataset = LeRobotDataset(config.get("repo_id", f"primeu/{root.name}"), root=root,
                             video_backend="torchcodec", download_videos=False)
    if max(frames) >= len(dataset):
        raise ValueError("Replay frame index is outside the dataset")
    direct, remote_contract = None, None
    if mode != "remote":
        from .policy import NativePrimeuPolicy
        direct = NativePrimeuPolicy(_path(config["checkpoint"], config_path), config.get("device", "cuda"))
    host, port, timeout_ms = config.get("host", "127.0.0.1"), config.get("port", 5559), config.get("timeout_ms", 30000)
    if mode != "direct":
        ping = rpc(host, port, "ping", timeout_ms=timeout_ms)
        remote_contract = ping["contract"]
        if ping.get("status") != "ok" or remote_contract.get("protocol") != PROTOCOL:
            raise ValueError("Wrong server protocol")
        if direct is not None:
            for key in ("contract_sha256", "basis_sha256", "model_family", "model_action_dim", "action_horizon"):
                if remote_contract.get(key) != direct.contract[key]:
                    raise ValueError(f"Remote/direct contract mismatch: {key}")
    contract = direct.contract if direct is not None else remote_contract
    records, previous_episode = [], None
    tolerance = float(config.get("atol", 1e-5))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("atol must be finite and nonnegative")
    for ordinal, index in enumerate(frames):
        sample = dataset[index]
        episode = int(sample["episode_index"])
        if episode != previous_episode:
            if direct is not None:
                direct.reset()
            if mode != "direct":
                reset = rpc(host, port, "reset", timeout_ms=timeout_ms)
                if reset.get("status") != "ok" or reset.get("robot_motion") is not False:
                    raise ValueError("Unexpected reset response")
            previous_episode = episode
        observation = sample_to_runtime(sample)
        record = {"frame_index": index, "episode_index": episode, "seed": seed + ordinal}
        direct_actions = remote_actions = None
        if direct is not None:
            start = time.monotonic()
            direct_actions = direct.infer(from_runtime(observation), seed=seed + ordinal)
            record["direct_ms"] = (time.monotonic() - start) * 1000
        if mode != "direct":
            start = time.monotonic()
            response = rpc(host, port, "get_action", {"observation": observation, "options": {"seed": seed + ordinal}}, timeout_ms)
            record["round_trip_ms"] = (time.monotonic() - start) * 1000
            remote_actions = _response_actions(response, contract["action_horizon"])
            record["server_inference_ms"] = float(response[1]["inference_ms"])
        if mode == "compare":
            record["max_absolute_difference"] = float(np.max(np.abs(direct_actions - remote_actions)))
            if record["max_absolute_difference"] > tolerance:
                raise AssertionError(f"Direct/ZMQ mismatch at dataset frame {index}: {record['max_absolute_difference']} > {tolerance}")
        actions = direct_actions if direct_actions is not None else remote_actions
        if actions.shape != (contract["action_horizon"], 60) or not np.isfinite(actions).all():
            raise ValueError("Invalid replay output")
        record["output_shape"] = list(actions.shape)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {"mode": mode, "robot_initialized": False, "dataset_root": str(root),
              "contract": contract, "comparison_atol": tolerance, "frames": records}
    destination = _path(config["report"], config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    path = args.config.expanduser().resolve(strict=True)
    run(yaml.safe_load(path.read_text()), path)


if __name__ == "__main__":
    main()

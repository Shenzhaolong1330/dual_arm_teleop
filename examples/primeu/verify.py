"""Offline real-data checkpoint verification, without robot SDKs or robot actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from examples.primeu.contract import STATE_SLICES, load_dataset_contract
from examples.primeu.policy import NativePrimeuPolicy
from examples.primeu.train import load_config


def current_frame(value, feature):
    """Select the present frame from native [B,...] or [B,T,...] observations."""
    dimensions = len(feature.shape)
    if value.ndim == dimensions + 1:
        return value[:1]
    if value.ndim == dimensions + 2:
        return value[:1, -1]
    raise ValueError(f"Unexpected observation rank {tuple(value.shape)} for {feature.shape}")


def verify(config_path, checkpoint, device="cuda"):
    from scripts.core.run_train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset

    torch.set_num_threads(4)
    resolved = load_config(config_path)
    cfg = TrainPipelineConfig(resolved["train"])
    dataset = make_dataset(cfg)
    dataset_contract = load_dataset_contract(resolved["train"]["dataset"]["root"])
    adapter = NativePrimeuPolicy(checkpoint, device=device)
    if adapter.codec.contract.contract_sha256 != dataset_contract.contract_sha256:
        raise AssertionError("Checkpoint dataset/normalization/basis contract does not match the configured dataset")
    if adapter.model_family != cfg.policy.type:
        raise AssertionError("Checkpoint policy type differs from the training configuration")
    model, pre, post = adapter.model, adapter.pre, adapter.post
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    boundary_checks = 0
    for episode in dataset.meta.episodes:
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        for index in (start, end - 1):
            indices, _ = dataset._get_query_indices(index, int(episode["episode_index"]))
            if any(i < start or i >= end for values in indices.values() for i in values):
                raise AssertionError(f"Temporal window crossed episode {episode['episode_index']}")
            boundary_checks += 1

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    original_actions = batch["action"].clone()
    raw_current = {key: current_frame(batch[key], feature).clone()
                   for key, feature in model.config.input_features.items()}
    second_real_state = current_frame(
        batch["observation.state"][1:2], model.config.input_features["observation.state"]
    )[0].clone().cpu().float().numpy()
    normalized = pre(batch)
    recovered = post(normalized["action"])
    roundtrip_error = (recovered.detach().cpu() - original_actions).abs().max().item()
    if roundtrip_error > 1e-5:
        raise AssertionError(f"Native normalization roundtrip error: {roundtrip_error}")
    for name, value in normalized.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise AssertionError(f"Non-finite processed batch: {name}")
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise AssertionError(f"Non-finite checkpoint parameter: {name}")

    # Native ACT computes its VAE posterior only in training mode. Restore buffers
    # after this no-grad loss check so inference still uses the saved checkpoint.
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    model.train()
    with torch.no_grad():
        torch.manual_seed(42)
        loss, _ = model(normalized)
    model.eval()
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    del buffers
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise AssertionError("Non-finite or non-scalar reloaded forward loss")

    observation = {"observation.state": raw_current["observation.state"][0].cpu().float().numpy()}
    serving_batch = {"observation.state": raw_current["observation.state"]}
    for camera in dataset_contract.cameras.values():
        image = raw_current[camera][0].cpu()
        if image.ndim != 3 or image.shape[0] != 3 or image.min() < 0 or image.max() > 1:
            raise AssertionError(f"Unexpected decoded RGB tensor: {camera} {tuple(image.shape)}")
        pixels = image.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        observation[camera] = pixels
        serving_batch[camera] = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        if not torch.allclose(serving_batch[camera][0], image, atol=1e-7, rtol=0):
            raise AssertionError(f"Decoded RGB is not equivalent to uint8 wire RGB: {camera}")

    serving_normalized = pre(serving_batch)
    preprocessing_errors = {}
    for name, feature in model.config.input_features.items():
        expected = current_frame(normalized[name], feature)
        error = (expected - serving_normalized[name]).abs().max().item()
        preprocessing_errors[name] = error
        if error > 1e-6:
            raise AssertionError(f"Training/serving processor mismatch for {name}: {error}")

    outputs, timings = [], []
    for _ in range(2):
        adapter.reset()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = adapter.infer(observation, seed=123)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        if output.shape != (adapter.horizon, 60) or not np.isfinite(output).all():
            raise AssertionError(f"Invalid expanded physical output: {output.shape}")
        outputs.append(output)
    repeat_error = float(np.max(np.abs(outputs[0] - outputs[1])))
    if repeat_error > 1e-5:
        raise AssertionError(f"Fixed-seed reset inference changed by {repeat_error}")

    # Use two different real feedback vectors to ensure later observations cannot
    # silently move held joints. Synthetic drift additionally exercises stationary
    # dataset joints (e.g. head/waist) whose real feedback may be identical.
    second_state = second_real_state
    first_state = observation["observation.state"]
    zeros = np.zeros((2, adapter.action_dim), dtype=np.float32)
    codec = adapter.codec
    codec.reset()
    first = codec.decode(zeros, first_state)
    second = codec.decode(zeros, second_state)
    drifted = codec.decode(zeros, second_state + 0.1)
    codec.reset()
    reset = codec.decode(zeros, second_state)
    for group in dataset_contract.held_groups:
        bounds = STATE_SLICES[group]
        sl = slice(bounds["start"], bounds["end"])
        np.testing.assert_array_equal(first[:, sl], second[:, sl])
        np.testing.assert_array_equal(first[:, sl], drifted[:, sl])
        np.testing.assert_array_equal(reset[:, sl], np.tile(second_state[sl], (len(reset), 1)))
        np.testing.assert_array_equal(outputs[0][:, sl], np.tile(first_state[sl], (adapter.horizon, 1)))
    adapter.reset()

    return {
        "checkpoint": str(Path(checkpoint).resolve()), "policy": adapter.model_family,
        "state_dim": 60, "model_action_dim": adapter.action_dim,
        "model_action_shape": [1, adapter.horizon, adapter.action_dim],
        "wire_action_shape": [1, adapter.horizon, 60],
        "episodes": dataset.num_episodes, "frames": dataset.num_frames,
        "episode_boundary_checks": boundary_checks,
        "batch_shapes": {name: list(value.shape) for name, value in normalized.items() if torch.is_tensor(value)},
        "reloaded_forward_loss": float(loss.detach().cpu()),
        "normalization_roundtrip_max_error": roundtrip_error,
        "training_serving_preprocessing_max_errors": preprocessing_errors,
        "fixed_seed_max_difference": repeat_error, "inference_seconds": timings,
        "physical_actions_finite": True, "all_parameters_finite": True,
        "physical_action_min": float(outputs[0].min()), "physical_action_max": float(outputs[0].max()),
        "physical_action_mean": float(outputs[0].mean()),
        "hold_groups_verified": dataset_contract.held_groups,
        "hold_reset_verified": True, "robot_motion": False,
        "contract_sha256": dataset_contract.contract_sha256,
        "basis_sha256": dataset_contract.basis_sha256,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.device(device).type == "cuda" else None,
        "processor_steps": [type(step).__name__ for step in pre.steps],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = verify(args.config, args.checkpoint, args.device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

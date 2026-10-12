"""Load a native LeRobot checkpoint and its saved PRIMEU contract/processors."""

from pathlib import Path

import numpy as np

from .wire import CAMERAS, JOINT_GROUPS, PROTOCOL


def camera_image_contract(image_features):
    """Describe checkpoint sizes without requiring all camera views to match."""
    sizes = {}
    for camera in CAMERAS:
        shape = tuple(image_features[f"observation.images.{camera}"].shape)
        if len(shape) != 3 or shape[0] != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError(f"Invalid checkpoint image shape for {camera}: {shape}")
        sizes[camera] = list(shape[-2:])
    unique_sizes = {tuple(size) for size in sizes.values()}
    # Keep the v1 uniform-size field for existing clients. Mixed sizes require
    # the explicit per-camera contract instead of an arbitrary first camera.
    uniform_size = list(next(iter(unique_sizes))) if len(unique_sizes) == 1 else None
    return {"image_sizes": sizes, "image_size": uniform_size}


def validate_processed_image_shapes(batch, image_sizes):
    """Validate after saved resize/crop, preserving legacy raw-camera inputs."""
    for camera, size in image_sizes.items():
        key = f"observation.images.{camera}"
        actual = tuple(batch[key].shape)
        expected = (1, 3, *size)
        if actual != expected:
            raise ValueError(f"Processed {camera} image shape {actual} does not match checkpoint {expected}")


class NativePrimeuPolicy:
    def __init__(self, checkpoint, device="cuda", seed=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        from .action_codec import ActionCodec
        from .contract import load_checkpoint_contract

        self.checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        data_contract = load_checkpoint_contract(self.checkpoint)
        self.codec = ActionCodec(data_contract, self.checkpoint / "primeu")
        config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        if config.type not in ("act", "diffusion"):
            raise ValueError(f"Unsupported PRIMEU policy: {config.type}")
        if config.n_obs_steps != 1:
            raise ValueError("PRIMEU single-frame service requires n_obs_steps=1; legacy two-frame checkpoints remain training-only")
        expected_images = {f"observation.images.{key}" for key in CAMERAS}
        if set(config.input_features) != expected_images | {"observation.state"}:
            raise ValueError("Checkpoint must consume exactly three PRIMEU cameras and physical 60D state")
        if tuple(config.input_features["observation.state"].shape) != (60,):
            raise ValueError("Checkpoint state shape must be [60]")
        if set(config.output_features) != {"action"} or tuple(config.output_features["action"].shape) != (data_contract.action_dim,):
            raise ValueError("Checkpoint action shape disagrees with the saved PRIMEU contract")
        image_contract = camera_image_contract(config.image_features)
        self.image_sizes = image_contract["image_sizes"]
        config.device = str(device)
        # Complete weights are in model.safetensors. Do not re-download an ImageNet
        # backbone only to overwrite it immediately with the checkpoint weights.
        config.pretrained_backbone_weights = None
        self.model = get_policy_class(config.type).from_pretrained(
            self.checkpoint, config=config, local_files_only=True, strict=True,
        ).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        self.horizon = config.chunk_size if config.type == "act" else config.n_action_steps
        self.action_dim, self.model_family = data_contract.action_dim, config.type
        self.device = torch.device(device)
        if seed is not None:
            self._seed(seed)
        self.contract = {
            "protocol": PROTOCOL, "model_family": config.type,
            "state_dim": 60, "action_dim": 60, "model_state_dim": 60,
            "model_action_dim": self.action_dim,
            "joint_groups": {key: b - a for key, (a, b) in JOINT_GROUPS.items()},
            "joint_order": list(JOINT_GROUPS), "camera_keys": list(CAMERAS),
            "action_representation": "absolute_joint_target",
            "action_semantics": "absolute_joint_targets", "action_horizon": self.horizon,
            **image_contract, "checkpoint": str(self.checkpoint),
            "basis_sha256": data_contract.basis_sha256,
            "contract_sha256": data_contract.contract_sha256,
            "unpredicted_joints": "hold_first_valid_observation_until_reset",
            "image_processing": "checkpoint_native_processor",
        }

    @staticmethod
    def _seed(seed):
        import torch
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        torch.manual_seed(seed)

    def reset(self):
        self.model.reset()
        self.codec.reset()

    def infer(self, observation, seed=None):
        """Generate a fresh physical chunk; never consume actions from a prior request."""
        import torch

        if seed is not None:
            self._seed(seed)
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        batch = {"observation.state": torch.from_numpy(state.copy()).unsqueeze(0)}
        for camera in CAMERAS:
            key = f"observation.images.{camera}"
            image = np.asarray(observation[key])
            batch[key] = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        with torch.inference_mode():
            normalized = self.pre(batch)
            validate_processed_image_shapes(normalized, self.image_sizes)
            self.model.reset()
            if self.model_family == "act":
                predicted = self.model.predict_action_chunk(dict(normalized))
            else:
                # Native select_action populates image/observation queues before
                # generating once. Draining exactly one fresh queue preserves its
                # crop/camera handling without depending on private queue internals.
                predicted = torch.stack([
                    self.model.select_action(dict(normalized)) for _ in range(self.horizon)
                ], dim=1)
            physical = self.post(predicted).detach().cpu().float().numpy()
        expected = (1, self.horizon, self.action_dim)
        if physical.shape != expected or not np.isfinite(physical).all():
            raise ValueError(f"Invalid physical model output: {physical.shape}, expected {expected}")
        return self.codec.decode(physical[0], state)

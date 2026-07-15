#!/usr/bin/env python

from __future__ import annotations

import argparse
import builtins
import datetime as dt
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any, Dict

import yaml

try:
    from scripts.utils.training_device import (
        TrainingDeviceConfig,
        apply_cuda_visible_devices_from_config_path,
        apply_cuda_visible_devices_from_train_cfg,
        log_training_device_state,
        setup_training_device,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.utils.training_device import (
        TrainingDeviceConfig,
        apply_cuda_visible_devices_from_config_path,
        apply_cuda_visible_devices_from_train_cfg,
        log_training_device_state,
        setup_training_device,
    )

from scripts.core.policy_config_utils import (
    build_policy_config,
    load_policy_yaml,
    resolve_policy_config_path,
    self_test_policy_config_loader,
)


def _default_train_cfg_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "train_cfg.yaml"


def _default_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_project_root() -> Path:
    return _default_scripts_dir().parent


def _extract_config_path_from_argv(argv: list[str] | None = None) -> Path:
    argv = list(sys.argv[1:] if argv is None else argv)
    option_names = {"--config", "--config-path", "--train-cfg"}
    for index, arg in enumerate(argv):
        if arg in option_names and index + 1 < len(argv):
            return Path(argv[index + 1])
        for option_name in option_names:
            prefix = f"{option_name}="
            if arg.startswith(prefix):
                return Path(arg[len(prefix) :])
    return _default_train_cfg_path()


_EARLY_CUDA_VISIBLE_DEVICES = apply_cuda_visible_devices_from_config_path(_extract_config_path_from_argv())

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim import OptimizerConfig
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.hub import HubMixin
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging
from robots.dual_flexiv_rizon4s.flexiv_state_schema import (
    build_flexiv_state_schema,
    persist_flexiv_checkpoint_schema,
    validate_flexiv_checkpoint,
    validate_flexiv_dataset_schema,
)

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

TRAIN_CONFIG_NAME = "train_config.json"


def _is_video_decode_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(
        marker in text
        for marker in (
            "InvalidDataError",
            "Invalid data found when processing input",
            "Could not push packet to decoder",
            "avcodec_send_packet",
        )
    )


class RetryOnVideoDecodeErrorDataset(torch.utils.data.Dataset):
    """Replace rare unreadable video samples with nearby valid samples during training."""

    def __init__(self, dataset: torch.utils.data.Dataset, max_retries: int = 64, log_limit: int = 20):
        self.dataset = dataset
        self.max_retries = max(1, int(max_retries))
        self.log_limit = max(0, int(log_limit))
        self._logged_failures = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def _fallback_indices(self, idx: int):
        length = len(self)
        for offset in range(1, self.max_retries + 1):
            yield (idx + offset) % length
            yield (idx - offset) % length

    def __getitem__(self, idx: int) -> Any:
        idx = int(idx)
        try:
            return self.dataset[idx]
        except Exception as exc:
            if not _is_video_decode_error(exc):
                raise
            original_exc = exc

        for attempt, fallback_idx in enumerate(self._fallback_indices(idx), start=1):
            try:
                item = self.dataset[fallback_idx]
            except Exception as exc:
                if _is_video_decode_error(exc):
                    continue
                raise
            if self._logged_failures < self.log_limit:
                logging.warning(
                    "Skipped unreadable video sample idx=%s; using fallback idx=%s after %d attempt(s).",
                    idx,
                    fallback_idx,
                    attempt,
                )
                self._logged_failures += 1
            return item

        raise RuntimeError(
            f"Failed to replace unreadable video sample idx={idx} after {self.max_retries} retries."
        ) from original_exc


def _validate_local_pretrained_path(pretrained_path: str | Path | None) -> None:
    """Fail early when an absolute local checkpoint path is misspelled."""
    if not pretrained_path:
        return

    raw_path = str(pretrained_path)
    path = Path(raw_path).expanduser()
    is_local_reference = path.is_absolute() or raw_path.startswith(("~", ".")) or path.exists()
    if not is_local_reference:
        return

    if not path.is_dir():
        raise FileNotFoundError(
            "Local pretrained_path does not exist or is not a directory: "
            f"{path}\n"
            "Expected a checkpoint directory containing config.json and model.safetensors. "
            "For example: .../checkpoints/010000/pretrained_model"
        )

    missing = [name for name in ("config.json", "model.safetensors") if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Local pretrained_path is missing required file(s): {missing}\n"
            f"Path: {path}"
        )


def _resolve_resume_checkpoint_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        raise ValueError("`train.output_dir` is required when `train.resume: true`.")

    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            "Cannot resume training because the checkpoints directory does not exist: "
            f"{checkpoints_dir}"
        )

    last_checkpoint = checkpoints_dir / "last"
    if last_checkpoint.exists():
        checkpoint_dir = last_checkpoint.resolve()
    else:
        numbered_checkpoints = sorted(
            (
                path
                for path in checkpoints_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )
        if not numbered_checkpoints:
            raise FileNotFoundError(
                "Cannot resume training because no numbered checkpoints were found in: "
                f"{checkpoints_dir}"
            )
        checkpoint_dir = numbered_checkpoints[-1]

    pretrained_dir = checkpoint_dir / "pretrained_model"
    training_state_dir = checkpoint_dir / "training_state"
    if not pretrained_dir.is_dir() or not training_state_dir.is_dir():
        raise FileNotFoundError(
            "Resume checkpoint is incomplete. Expected both `pretrained_model` and "
            f"`training_state` under: {checkpoint_dir}"
        )

    return checkpoint_dir


def run_act_dagger_from_train_cfg(train_cfg: Dict[str, Any]) -> None:
    """
    Internal ACT training helper for the round-based DAgger controller.

    The public DAgger flow is scripts.core.run_dagger_rounds. This helper
    only adapts an already aggregated standard LeRobot dataset into the normal
    ACT training path.

    Expected in-memory extension:
    train:
      policy:
        type: act_dagger
        ...
      dagger:
        dataset: {...}
        training: {...}    # optional
    """
    apply_cuda_visible_devices_from_train_cfg(train_cfg)

    dagger_section = train_cfg.get("dagger")
    if dagger_section is None:
        raise ValueError(
            "When train.policy.type='act_dagger', a 'train.dagger' section is required. "
            "Use scripts.core.run_dagger_rounds for the supported DAgger flow."
        )

    # Import round-Dagger dataset helpers lazily to avoid impacting BC-only startup.
    from lerobot.policies.dagger.configuration_dagger import DAggerDatasetConfig
    from lerobot.policies.dagger.dataset import ensure_aggregated_dataset_ready

    policy_cfg = dict(train_cfg.get("policy", {}))
    # Round DAgger still trains the underlying ACT policy.
    policy_cfg["type"] = "act"
    # Backward compatibility for users who may set `policy.path`.
    if "path" in policy_cfg and "pretrained_path" not in policy_cfg:
        policy_cfg["pretrained_path"] = policy_cfg["path"]
    policy_cfg.pop("path", None)

    dagger_dataset_cfg = dagger_section.get("dataset")
    if dagger_dataset_cfg is None:
        raise ValueError("train.dagger.dataset is required for the act_dagger round-training helper.")

    dagger_training_cfg = dict(dagger_section.get("training", {}))
    dagger_training_cfg.setdefault("rounds", 1)
    dagger_training_cfg.setdefault("steps_per_round", train_cfg.get("steps", 10_000))
    dagger_training_cfg.setdefault("batch_size", train_cfg.get("batch_size", 8))
    dagger_training_cfg.setdefault("num_workers", train_cfg.get("num_workers", 4))
    dagger_training_cfg.setdefault("log_freq", train_cfg.get("log_freq", 100))
    dagger_training_cfg.setdefault("save_checkpoint", train_cfg.get("save_checkpoint", True))

    aggregated_repo_id, aggregated_root = ensure_aggregated_dataset_ready(
        DAggerDatasetConfig(**dagger_dataset_cfg)
    )

    act_train_cfg = dict(train_cfg)
    act_train_cfg["policy"] = policy_cfg
    act_train_cfg["dataset"] = {
        "repo_id": aggregated_repo_id,
        "root": str(aggregated_root),
    }
    act_train_cfg["steps"] = (
        int(dagger_training_cfg["rounds"]) * int(dagger_training_cfg["steps_per_round"])
    )
    act_train_cfg["batch_size"] = dagger_training_cfg["batch_size"]
    act_train_cfg["num_workers"] = dagger_training_cfg["num_workers"]
    act_train_cfg["log_freq"] = dagger_training_cfg["log_freq"]
    act_train_cfg["save_checkpoint"] = dagger_training_cfg["save_checkpoint"]

    train_cfg_obj = TrainPipelineConfig(act_train_cfg)
    run_train(train_cfg_obj)


class TrainPipelineConfig(HubMixin):
    def __init__(self, cfg: Dict[str, Any]):
        dataset = cfg["dataset"]
        # env = cfg["env"]
        policy = cfg["policy"]
        eval = cfg["eval"]
        wandb = cfg["wandb"]
        self.training: TrainingDeviceConfig = TrainingDeviceConfig.from_mapping(cfg.get("training"))
        self.requested_policy_device: str | None = (
            str(policy["device"]) if policy.get("device") is not None else None
        )
    
        dataset_kwargs: dict[str, Any] = {
            "repo_id": dataset["repo_id"],
            "root": dataset.get("root"),
        }
        for key in ("episodes", "revision", "use_imagenet_stats", "video_backend", "streaming"):
            if key in dataset:
                dataset_kwargs[key] = dataset[key]
        self.dataset: DatasetConfig = DatasetConfig(**dataset_kwargs)

        # self.env: envs.EnvConfig | None = envs.EnvConfig(
        #     env_name = env["env_name"],
        #     env_type = env["env_type"],
        #     env_kwargs = env["env_kwargs"],
        # )
        self.env = None

        self.policy_type = str(policy["type"]).strip().lower()
        self.policy_config_path = resolve_policy_config_path(
            policy,
            scripts_dir=_default_scripts_dir(),
            project_root=_default_project_root(),
            mode="train",
        )
        policy_yaml = load_policy_yaml(self.policy_config_path)
        self.policy = build_policy_config(
            self.policy_type,
            policy_yaml,
            legacy_policy_dict=policy,
            legacy_source_name="train_cfg.yaml",
            config_path=self.policy_config_path,
            mode="train",
        )
        if policy.get("pretrained_path") is None and self.policy.pretrained_path is not None:
            logging.warning(
                "train policy yaml sets pretrained_path=%s. Training will initialize from this "
                "checkpoint unless you set pretrained_path: null in a training-specific policy yaml.",
                self.policy.pretrained_path,
            )
        if (
            "temporal_ensemble_coeff" not in policy
            and getattr(self.policy, "temporal_ensemble_coeff", None) is not None
        ):
            logging.warning(
                "train policy yaml sets temporal_ensemble_coeff=%s. This is usually a record-time "
                "inference setting; use a training-specific policy yaml if training should differ.",
                self.policy.temporal_ensemble_coeff,
            )

        # Set `dir` to where you would like to save all of the run outputs. If you run another training session
        # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
        self.output_dir: Path | None = Path(cfg["output_dir"]) if cfg["output_dir"] else None
        self.job_name: str | None = cfg["job_name"]
        # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
        # `dir` is the directory of an existing run with at least one checkpoint in it.
        # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
        # regardless of what's provided with the training command at the time of resumption.
        self.resume: bool = cfg["resume"]
        # `seed` is used for training (eg: model initialization, dataset shuffling)
        # AND for the evaluation environments.
        self.seed: int | None = cfg["seed"]
        # Number of workers for the dataloader.
        self.num_workers: int = cfg["num_workers"]
        self.batch_size: int = cfg["batch_size"]
        self.skip_bad_video_samples: bool = bool(dataset.get("skip_bad_video_samples", False))
        self.bad_video_sample_retries: int = int(dataset.get("bad_video_sample_retries", 64))
        self.dagger_sampling: dict[str, Any] = dict(cfg.get("dagger_sampling", {"enabled": False}))
        self.steps: int = cfg["steps"]
        self.eval_freq: int = cfg["eval_freq"]
        self.log_freq: int = cfg["log_freq"]
        self.save_checkpoint: bool = cfg["save_checkpoint"]
        self.save_freq: int = cfg["save_freq"]
        self.use_policy_training_preset: bool = cfg["use_policy_training_preset"]
        self.optimizer: OptimizerConfig | None = None
        self.scheduler: LRSchedulerConfig | None = None

        self.eval: EvalConfig = EvalConfig(
            n_episodes = eval["n_episodes"],
            batch_size = eval["batch_size"]
        )

        self.wandb: WandBConfig = WandBConfig(
            enable = wandb["enable"],
            project = wandb["project"],
            entity = wandb.get("entity"),  # 使用 get 方法，允许键不存在
            notes = wandb.get("notes"),    # 使用 get 方法，允许键不存在
            run_id = wandb.get("run_id"),  # 使用 get 方法，允许键不存在
            mode = wandb["mode"]
        )

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):

        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            checkpoint_dir = _resolve_resume_checkpoint_dir(self.output_dir)
            self.policy.pretrained_path = checkpoint_dir / "pretrained_model"
            self.checkpoint_path = checkpoint_dir

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset:
            self.optimizer = self.optimizer or self.policy.get_optimizer_preset()
            self.scheduler = self.scheduler or self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        """将配置对象转换为可序列化的字典"""
        result = {}
        for key, value in self.__dict__.items():
            # 跳过私有属性
            if key.startswith('_'):
                continue
                
            # 处理特殊类型的属性
            if value is None:
                result[key] = None
            elif isinstance(value, (str, int, float, bool)):
                result[key] = value
            elif isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [self._serialize_item(item) for item in value]
            elif isinstance(value, dict):
                result[key] = {k: self._serialize_item(v) for k, v in value.items()}
            elif hasattr(value, 'to_dict'):
                # 如果属性有 to_dict 方法，调用它
                result[key] = value.to_dict()
            else:
                # 对于其他对象，使用安全的序列化方法
                result[key] = self._serialize_item(value)
        return result

    def _serialize_item(self, item):
        """安全地序列化单个项目"""
        if item is None:
            return None
        elif isinstance(item, (str, int, float, bool)):
            return item
        elif isinstance(item, Path):
            return str(item)
        elif isinstance(item, (list, tuple)):
            return [self._serialize_item(i) for i in item]
        elif isinstance(item, dict):
            return {k: self._serialize_item(v) for k, v in item.items()}
        elif hasattr(item, 'to_dict'):
            return item.to_dict()
        elif hasattr(item, '__dict__'):
            # 对于复杂对象，只序列化基本属性
            return self._serialize_simple_object(item)
        else:
            # 最后手段：返回字符串表示
            return str(item)
    
    def _serialize_simple_object(self, obj):
        """安全地序列化简单对象，避免循环引用"""
        result = {}
        for attr_name, attr_value in obj.__dict__.items():
            # 跳过私有属性和复杂对象
            if attr_name.startswith('_'):
                continue
            
            try:
                # 只序列化基本类型
                if isinstance(attr_value, (str, int, float, bool, type(None))):
                    result[attr_name] = attr_value
                elif isinstance(attr_value, Path):
                    result[attr_name] = str(attr_value)
            except:
                # 如果序列化失败，跳过该属性
                continue
        
        # 如果没有任何可序列化的属性，返回类型名称
        if not result:
            return f"<{obj.__class__.__name__}>"
        
        return result

    def _save_pretrained(self, save_directory: Path) -> None:
        # 使用手动实现的 to_dict 方法保存配置
        config_dict = self.to_dict()
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

    @classmethod
    def from_pretrained(
        cls: builtins.type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)



def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def run_train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()
    device_state = setup_training_device(
        cfg.training,
        policy_device=cfg.requested_policy_device or cfg.policy.device,
    )
    cfg.policy.device = device_state.final_device.type

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            cpu=device_state.final_device.type == "cpu",
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        if _EARLY_CUDA_VISIBLE_DEVICES.warning:
            logging.warning(_EARLY_CUDA_VISIBLE_DEVICES.warning)
        if accelerator.device.type != device_state.final_device.type:
            logging.warning(
                "Accelerator selected device '%s' while resolved training device was '%s'. "
                "Using accelerator device for model and batch placement.",
                accelerator.device,
                device_state.final_device,
            )
        cfg.policy.device = accelerator.device.type
        log_training_device_state(device_state)
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device for the model and the preprocessor so policy and batch tensors agree.
    device = accelerator.device
    cfg.policy.device = device.type

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    flexiv_schema = None
    if getattr(dataset.meta, "robot_type", None) == "flexiv_dual_arm":
        validate_flexiv_dataset_schema(
            dataset.meta.info,
            dataset.features,
            source=f"training dataset {cfg.dataset.repo_id}",
        )
        validate_flexiv_checkpoint(
            cfg.policy.pretrained_path,
            source="Flexiv training checkpoint",
        )
        flexiv_schema = build_flexiv_state_schema()

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    _validate_local_pretrained_path(cfg.policy.pretrained_path)
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"dataset.video_backend={cfg.dataset.video_backend}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dagger_sampling_cfg = getattr(cfg, "dagger_sampling", {"enabled": False})
    if isinstance(dagger_sampling_cfg, dict) and dagger_sampling_cfg.get("enabled", False):
        if sampler is not None:
            logging.warning(
                "DAgger source-aware sampler requested, but another sampler is already active; "
                "keeping the existing sampler."
            )
        elif cfg.dataset.streaming:
            logging.warning(
                "DAgger source-aware sampler requested, but streaming datasets do not support "
                "WeightedRandomSampler; keeping default DataLoader sampling."
            )
        else:
            from scripts.core.dagger_sampling import (
                build_source_weighted_sampler_for_dataset,
                format_sampling_stats,
            )

            sampling_result = build_source_weighted_sampler_for_dataset(dataset, dagger_sampling_cfg)
            logging.info(format_sampling_stats(sampling_result.stats))
            if sampling_result.sampler is not None:
                sampler = sampling_result.sampler
                shuffle = False

    dataset_for_loader = dataset
    if cfg.skip_bad_video_samples:
        logging.warning(
            "skip_bad_video_samples=true: unreadable video samples will be replaced with nearby valid samples."
        )
        dataset_for_loader = RetryOnVideoDecodeErrorDataset(
            dataset,
            max_retries=cfg.bad_video_sample_retries,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset_for_loader,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    logged_first_batch = False
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        if is_main_process and not logged_first_batch:
            logging.info("Loading first training batch...")
        batch = next(dl_iter)
        if is_main_process and not logged_first_batch:
            logging.info("First training batch loaded in %.2fs", time.perf_counter() - start_time)
            logging.info("Preprocessing first training batch...")
            preprocess_start_time = time.perf_counter()
        batch = preprocessor(batch)
        if is_main_process and not logged_first_batch:
            logging.info(
                "First training batch preprocessed in %.2fs",
                time.perf_counter() - preprocess_start_time,
            )
            logging.info("Running first optimization step...")
            first_update_start_time = time.perf_counter()
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        if is_main_process and not logged_first_batch:
            logging.info("First optimization step finished in %.2fs", time.perf_counter() - first_update_start_time)
            logged_first_batch = True

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_warmup_log_step = 0 < step <= 5
        is_log_step = is_main_process and (
            is_warmup_log_step or (cfg.log_freq > 0 and step % cfg.log_freq == 0)
        )
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                if flexiv_schema is not None:
                    persist_flexiv_checkpoint_schema(checkpoint_dir / "pretrained_model")
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("âˆ‘rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def _build_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description="Train a LeRobot policy with the custom dual-arm config.")
    arg_parser.add_argument(
        "--config",
        "--config-path",
        dest="config_path",
        type=Path,
        default=_default_train_cfg_path(),
        help="Path to train_cfg.yaml.",
    )
    arg_parser.add_argument(
        "--dry-run-policy-config",
        action="store_true",
        help="Load train_cfg.yaml and the referenced policy yaml, build the policy config, then exit.",
    )
    arg_parser.add_argument(
        "--self-test-policy-config",
        action="store_true",
        help="Run minimal in-process checks for the shared policy config loader, then exit.",
    )
    return arg_parser


def _load_train_cfg_yaml(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "train" not in cfg:
        raise ValueError(f"Train config must contain a top-level `train` mapping: {cfg_path}")
    return cfg


def dry_run_policy_config(cfg_path: Path) -> TrainPipelineConfig:
    cfg = _load_train_cfg_yaml(cfg_path)
    train_cfg = TrainPipelineConfig(cfg["train"])
    logging.info("====== [TRAIN POLICY CONFIG DRY-RUN] OK ======")
    logging.info("policy.type: %s", train_cfg.policy_type)
    logging.info("policy.config_path: %s", train_cfg.policy_config_path)
    logging.info("policy.config_class: %s", type(train_cfg.policy).__name__)
    logging.info("policy.device: %s", train_cfg.policy.device)
    logging.info("policy.pretrained_path: %s", train_cfg.policy.pretrained_path)
    return train_cfg


def main():
    args = _build_arg_parser().parse_args()
    if args.self_test_policy_config:
        self_test_policy_config_loader()
        return

    cfg_path = args.config_path
    if args.dry_run_policy_config:
        dry_run_policy_config(cfg_path)
        return

    cfg = _load_train_cfg_yaml(cfg_path)
    train_section = cfg["train"]
    apply_cuda_visible_devices_from_train_cfg(train_section)
    policy_type = train_section.get("policy", {}).get("type")

    if policy_type == "act_dagger":
        raise ValueError(
            "Direct robot-train with train.policy.type='act_dagger' is deprecated. "
            "Use `robot-dagger` or `python -m scripts.core.run_dagger_rounds` for DAgger. "
            "The round controller "
            "will call the internal ACT training helper itself."
        )

    train_cfg = TrainPipelineConfig(train_section)
    run_train(train_cfg)


if __name__ == "__main__":
    main()

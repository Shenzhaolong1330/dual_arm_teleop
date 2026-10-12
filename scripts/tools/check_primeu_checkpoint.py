"""Offline checkpoint/processor check using real PRIMEU observations; no robot SDK."""
import argparse
import json
from pathlib import Path
import time

import torch
import yaml
from torch.utils.data import DataLoader

from scripts.core.run_train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cfg = TrainPipelineConfig(yaml.safe_load(args.config.read_text())["train"])
    dataset = make_dataset(cfg)
    boundary_checks = 0
    for ep in dataset.meta.episodes:
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        for idx in (start, end-1):
            indices, _ = dataset._get_query_indices(idx, ep["episode_index"])
            if any(i < start or i >= end for values in indices.values() for i in values):
                raise AssertionError("Temporal window crossed an episode boundary")
            boundary_checks += 1
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    policy = get_policy_class(cfg.policy.type).from_pretrained(args.checkpoint, local_files_only=True)
    policy.to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    normalized = pre(batch)
    recovered = post(normalized["action"])
    roundtrip_error = (recovered.cpu() - batch["action"]).abs().max().item()
    if roundtrip_error > 1e-5:
        raise AssertionError(f"Normalization roundtrip error {roundtrip_error}")
    for name, value in normalized.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise AssertionError(f"Non-finite processed input {name}")
    for name, param in policy.named_parameters():
        if not torch.isfinite(param).all():
            raise AssertionError(f"Non-finite parameter {name}")
    # ACT's native forward loss requires its training-mode VAE posterior.
    policy.train()
    with torch.no_grad():
        torch.manual_seed(42)
        loss, _ = policy(normalized)
    policy.eval()
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite reloaded loss")

    # Reconstruct a single physical observation; temporal queues are managed by the policy.
    obs = {k: v[:1].clone() for k, v in batch.items() if k in policy.config.input_features}
    if cfg.policy.type == "diffusion":
        obs = {k: v[:, -1] for k, v in obs.items()}
    observation = pre(obs)
    actions, elapsed = [], []
    for _ in range(2):
        policy.reset()
        torch.manual_seed(123)
        torch.cuda.synchronize()
        start = time.perf_counter()
        if cfg.policy.type == "act":
            result = policy.predict_action_chunk(dict(observation))
        else:
            result = torch.stack([policy.select_action(dict(observation))
                                  for _ in range(policy.config.n_action_steps)], dim=1)
        result = post(result)
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        if not torch.isfinite(result).all() or result.shape[-1] != 60:
            raise AssertionError(f"Invalid physical action output {result.shape}")
        actions.append(result.cpu())
    reproducibility_error = (actions[0] - actions[1]).abs().max().item()
    if reproducibility_error > 1e-5:
        raise AssertionError(f"Reset/fixed-seed inference changed by {reproducibility_error}")
    report = dict(checkpoint=str(args.checkpoint), policy=cfg.policy.type,
                  episodes=dataset.num_episodes, frames=dataset.num_frames,
                  episode_boundary_checks=boundary_checks,
                  batch_shapes={k: list(v.shape) for k, v in normalized.items() if torch.is_tensor(v)},
                  normalized_loss=float(loss), normalization_roundtrip_max_error=roundtrip_error,
                  action_shape=list(actions[0].shape), physical_actions_finite=True,
                  fixed_seed_max_difference=reproducibility_error, inference_seconds=elapsed,
                  trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
                  peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  processor_steps=[type(s).__name__ for s in pre.steps])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

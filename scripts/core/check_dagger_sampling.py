#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.core.dagger_sampling import (
    build_source_weighted_sampler_from_index,
    dataset_counts,
    load_source_index,
    source_index_path_for_dataset,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect DAgger source-aware sampling weights.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to an aggregated LeRobot dataset.")
    parser.add_argument(
        "--source-index",
        type=Path,
        default=None,
        help="Path to dagger_source_index.json. Defaults to <dataset>/dagger_source_index.json.",
    )
    parser.add_argument("--dagger-sample-ratio", type=float, default=0.4)
    parser.add_argument(
        "--on-missing-source",
        choices=["disable_sampler", "treat_as_seed", "error"],
        default="disable_sampler",
    )
    parser.add_argument("--replacement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draws", type=int, default=1000)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    source_index_path = args.source_index or source_index_path_for_dataset(args.dataset)
    counts = dataset_counts(args.dataset)
    source_index = load_source_index(source_index_path) if source_index_path.is_file() else None
    sampling_cfg = {
        "enabled": True,
        "strategy": "source_weighted",
        "dagger_sample_ratio": args.dagger_sample_ratio,
        "source_index_path": str(source_index_path),
        "on_missing_source": args.on_missing_source,
        "replacement": args.replacement,
    }
    result = build_source_weighted_sampler_from_index(
        num_samples=counts["total_frames"],
        source_index=source_index,
        sampling_cfg=sampling_cfg,
    )
    stats = result.stats
    print(f"dataset: {args.dataset}")
    print(f"source_index: {source_index_path}")
    print(f"dataset length: {counts['total_frames']}")
    print(f"episodes: {counts['total_episodes']}")
    print(f"sampler enabled: {stats.get('sampler_enabled')}")
    print(f"seed sample count: {stats.get('seed_samples')}")
    print(f"dagger sample count: {stats.get('dagger_samples')}")
    print(f"seed weight: {stats.get('seed_per_sample_weight')}")
    print(f"dagger weight: {stats.get('dagger_per_sample_weight')}")
    print(f"target dagger ratio: {stats.get('dagger_sample_ratio')}")
    print(f"estimated dagger ratio: {stats.get('estimated_dagger_ratio')}")
    print(f"replacement: {stats.get('replacement')}")
    print(f"disabled reason: {stats.get('disabled_reason')}")

    if result.weights is not None and result.dagger_mask is not None and args.draws > 0:
        import torch

        draw_count = min(args.draws, max(1, len(result.weights)))
        sampled_indices = torch.multinomial(result.weights, num_samples=draw_count, replacement=True)
        empirical_ratio = float(result.dagger_mask[sampled_indices].double().mean().item())
        print(f"empirical dagger ratio ({draw_count} draws): {empirical_ratio}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

TELEOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(TELEOP_ROOT))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    print("dagger sampling tests skipped: torch is not installed in this Python environment")
    raise SystemExit(0)

from scripts.core.dagger_sampling import build_source_weighted_sampler_from_index


def _sampling_cfg(ratio: float = 0.4, on_missing_source: str = "disable_sampler") -> dict:
    return {
        "enabled": True,
        "strategy": "source_weighted",
        "dagger_sample_ratio": ratio,
        "on_missing_source": on_missing_source,
        "replacement": True,
    }


def _source_index(seed_samples: int, dagger_samples: int) -> dict:
    frame_ranges = []
    if seed_samples:
        frame_ranges.append({"from": 0, "to": seed_samples, "source": "seed"})
    if dagger_samples:
        frame_ranges.append(
            {
                "from": seed_samples,
                "to": seed_samples + dagger_samples,
                "source": "dagger",
                "round_id": 1,
            }
        )
    return {"version": 1, "frame_ranges": frame_ranges}


def test_source_weighted_probability_mass() -> None:
    result = build_source_weighted_sampler_from_index(
        num_samples=1100,
        source_index=_source_index(seed_samples=1000, dagger_samples=100),
        sampling_cfg=_sampling_cfg(0.4),
    )
    assert result.sampler is not None
    assert result.weights is not None
    assert result.dagger_mask is not None
    assert result.stats["seed_samples"] == 1000
    assert result.stats["dagger_samples"] == 100
    assert abs(float(result.weights[result.dagger_mask].sum().item()) - 0.4) < 1e-9
    assert abs(float(result.weights[~result.dagger_mask].sum().item()) - 0.6) < 1e-9
    assert abs(float(result.stats["estimated_dagger_ratio"]) - 0.4) < 1e-9


def test_missing_source_disables_sampler() -> None:
    result = build_source_weighted_sampler_from_index(
        num_samples=100,
        source_index=None,
        sampling_cfg=_sampling_cfg(0.4, on_missing_source="disable_sampler"),
    )
    assert result.sampler is None
    assert result.stats["sampler_enabled"] is False
    assert "missing" in result.stats["disabled_reason"]


def test_zero_dagger_samples_does_not_crash() -> None:
    result = build_source_weighted_sampler_from_index(
        num_samples=100,
        source_index=_source_index(seed_samples=100, dagger_samples=0),
        sampling_cfg=_sampling_cfg(0.4, on_missing_source="disable_sampler"),
    )
    assert result.sampler is None
    assert result.stats["sampler_enabled"] is False
    assert "zero DAgger samples" in result.stats["disabled_reason"]


def test_disabled_config_keeps_sampler_off() -> None:
    result = build_source_weighted_sampler_from_index(
        num_samples=1100,
        source_index=_source_index(seed_samples=1000, dagger_samples=100),
        sampling_cfg={"enabled": False},
    )
    assert result.sampler is None
    assert result.stats["sampler_enabled"] is False


if __name__ == "__main__":
    test_source_weighted_probability_mass()
    test_missing_source_disables_sampler()
    test_zero_dagger_samples_does_not_crash()
    test_disabled_config_keeps_sampler_off()
    print("dagger sampling tests passed")

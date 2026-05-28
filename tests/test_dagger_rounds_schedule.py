from __future__ import annotations

import logging
import sys
from pathlib import Path

TELEOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(TELEOP_ROOT))

from scripts.core.run_dagger_rounds import (
    resolve_effective_train_step_mode,
    resolve_train_steps_schedule,
    should_stop_dagger,
)


def _state(round_id: int, frames: int, episodes: int, interventions: int = 1) -> dict:
    return {
        "round_id": round_id,
        "exported_frames": frames,
        "exported_episodes": episodes,
        "intervention_count": interventions,
    }


def _schedule(patience: int = 2) -> dict:
    return {
        "early_stop": {
            "enabled": True,
            "patience": patience,
            "stop_if_no_new_data": True,
            "stop_if_export_too_small": False,
            "stop_if_low_expert_ratio": False,
        }
    }


def _export_summary(export_mode: str, frames: int = 100, episodes: int = 3, **stats: int) -> dict:
    summary = {
        "export_mode": export_mode,
        "exported_frames": frames,
        "exported_episodes": episodes,
        "exported_dataset": {
            "total_frames": frames,
            "total_episodes": episodes,
        },
    }
    summary.update(stats)
    return summary


def _train_steps_schedule(mode: str = "auto_by_export_mode") -> dict:
    return {
        "train_steps": {
            "mode": mode,
            "min_steps": 500,
            "max_steps": 3000,
            "steps_per_exported_frame": 4,
            "steps_per_exported_episode": 300,
            "steps_per_full_episode": 300,
            "steps_per_recovery_segment": 200,
            "steps_per_recovery_frame": 3,
            "intervention_segments_mode": "hybrid",
            "full_success_episode_mode": "by_exported_episodes",
            "hybrid_mode": "by_export_subtype",
            "decay_per_round": 1.0,
        }
    }


def test_no_exported_data_respects_patience() -> None:
    triggered, reason = should_stop_dagger(
        [],
        _state(1, 0, 0),
        _schedule(patience=2),
    )

    assert not triggered
    assert reason is None


def test_no_exported_data_stops_after_patience_rounds() -> None:
    triggered, reason = should_stop_dagger(
        [_state(1, 0, 0)],
        _state(2, 0, 0),
        _schedule(patience=2),
    )

    assert triggered
    assert "2 consecutive round(s)" in reason


def test_auto_by_export_mode_selects_intervention_hybrid() -> None:
    cfg = _train_steps_schedule()["train_steps"]

    assert resolve_effective_train_step_mode("intervention_segments", cfg) == "hybrid"


def test_auto_by_export_mode_selects_full_episode_by_episode() -> None:
    cfg = _train_steps_schedule()["train_steps"]

    assert resolve_effective_train_step_mode("full_success_episode", cfg) == "by_exported_episodes"


def test_auto_by_export_mode_selects_hybrid_by_subtype() -> None:
    cfg = _train_steps_schedule()["train_steps"]

    assert resolve_effective_train_step_mode("hybrid", cfg) == "by_export_subtype"


def test_by_export_subtype_components_and_clamp() -> None:
    resolved = resolve_train_steps_schedule(
        round_id=2,
        exported_summary=_export_summary(
            "hybrid",
            frames=1200,
            episodes=7,
            full_episode_count=5,
            recovery_segment_count=2,
            recovery_frames=100,
        ),
        base_train_steps=10_000,
        schedule_cfg=_train_steps_schedule(),
    )

    assert resolved["train_steps_effective_mode"] == "by_export_subtype"
    assert resolved["resolved_train_steps"] == 2200
    assert resolved["train_steps_components"]["full_steps"] == 1500
    assert resolved["train_steps_components"]["recovery_steps"] == 700
    assert resolved["train_steps_components"]["raw_steps"] == 2200
    assert resolved["train_steps_components"]["clamped_steps"] == 2200


def test_by_export_subtype_missing_summary_falls_back_to_episodes(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        resolved = resolve_train_steps_schedule(
            round_id=2,
            exported_summary=_export_summary("hybrid", frames=1200, episodes=4),
            base_train_steps=10_000,
            schedule_cfg=_train_steps_schedule(),
        )

    assert resolved["train_steps_effective_mode"] == "by_export_subtype"
    assert resolved["resolved_train_steps"] == 1200
    assert resolved["train_steps_components"]["fallback_mode"] == "by_exported_episodes"
    assert "by_export_subtype requested but subtype summary is missing" in caplog.text

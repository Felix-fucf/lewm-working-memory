from __future__ import annotations

import json
from pathlib import Path

from analysis.select_occlusion_v3_winner import RUNS, select


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_audit(path: Path, nmse: float, probe: float = 0.10) -> None:
    _write(
        path / "memory_ablation.json",
        {
            "final_reappearance": {
                "normal": nmse,
                "normalized_mse": {"normal": nmse},
                "shuffled_hidden_relative_increase": 0.08,
                "reset_each_step_relative_increase": 0.04,
                "no_memory_relative_increase": 0.05,
            },
            "bootstrap_final_reappearance": {
                "shuffled_hidden": {"ci95_lower": 0.02, "ci95_upper": 0.12}
            },
        },
    )
    _write(
        path / "effective_rank_report.json",
        {
            "global_rank": {
                "target_emb": {"covariance": {"effective_rank": 30.0, "top1_variance_ratio": 0.2}},
                "pred_emb": {"covariance": {"effective_rank": 25.0, "top1_variance_ratio": 0.25}},
            },
            "prediction_metrics": {
                "groups": {"final_reappearance": {"mean_cosine_error": nmse / 10.0}}
            },
        },
    )
    _write(
        path / "position_probe.json",
        {
            "scopes": {"final_reappearance": {"x": {"r2": probe}, "y": {"r2": probe}}},
            "angle_scopes": {"final_reappearance": {"mean_r2": probe}},
        },
    )
    _write(path / "matched_pair_report.json", {"mean_delta_alignment_cosine": 0.2})


def test_v3_selector_requires_material_improvement(tmp_path: Path) -> None:
    values = {"N1": 1.00, "N2": 0.94, "N3": 0.92}
    for label, definition in RUNS.items():
        for dataset in ("original_motion", "motion15x"):
            _write_audit(
                tmp_path / "runs" / definition["run_id"] / "audit" / f"validation_{dataset}",
                values[label],
            )
    report = select(tmp_path)
    # N2 is within five percent of N3, so the simpler preregistered tie order wins.
    assert report["selected"]["label"] == "N2"


def test_v3_selector_rejects_latent_gain_with_probe_loss(tmp_path: Path) -> None:
    values = {"N1": 1.00, "N2": 0.80, "N3": 1.04}
    for label, definition in RUNS.items():
        for dataset in ("original_motion", "motion15x"):
            _write_audit(
                tmp_path / "runs" / definition["run_id"] / "audit" / f"validation_{dataset}",
                values[label],
                probe=-0.20 if label == "N2" else 0.10,
            )
    report = select(tmp_path)
    assert not report["candidates"]["N2"]["eligible_for_replication"]
    assert report["selected"]["label"] == "N1"

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.select_occlusion_winner import select


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _audit_payload(shuffle: float) -> tuple[dict, dict]:
    ablation = {
        "final_reappearance": {
            "normal": 1.70,
            "target_variance": 1.0,
            "normalized_mse": {"normal": 1.70},
            "shuffled_hidden_relative_increase": shuffle,
            "reset_each_step_relative_increase": 0.02,
            "no_memory_relative_increase": 0.03,
        },
        "bootstrap_final_reappearance": {
            "shuffled_hidden": {"ci95_lower": 0.01, "ci95_upper": 0.09}
        },
    }
    rank = {
        "global_rank": {
            "target_emb": {"covariance": {"effective_rank": 30.0, "top1_variance_ratio": 0.2}},
            "pred_emb": {"covariance": {"effective_rank": 25.0, "top1_variance_ratio": 0.25}},
        }
    }
    return ablation, rank


def test_dfinal_wins_one_percentage_point_tie(tmp_path: Path) -> None:
    scores = {"dshort": 0.03, "dmid": 0.060, "dfinal": 0.055}
    labels = {"dshort": "R1", "dmid": "R2", "dfinal": "R3"}
    for variant, score in scores.items():
        _write(
            tmp_path / "dataset_audit" / f"primitive_v2_{variant}.audit.json",
            {"all_required_acceptance_pass": True},
        )
        audit_dir = (
            tmp_path / "runs" / f"{labels[variant]}_{variant}_L0_s3072"
            / "audit" / "validation_own"
        )
        ablation, rank = _audit_payload(score)
        _write(audit_dir / "memory_ablation.json", ablation)
        _write(audit_dir / "effective_rank_report.json", rank)
        _write(
            tmp_path / "runs" / "C0_d0_L0_s3072" / "audit"
            / f"validation_{variant}" / "memory_ablation.json",
            {"final_reappearance": {"normalized_mse": {"normal": 1.60}}},
        )

    report = select(argparse.Namespace(campaign_root=tmp_path))
    assert report["selected"] == "dfinal"
    assert "one percentage point" in report["selection_reason"]


def test_candidate_fails_same_dataset_corrected_c0_ten_percent_gate(tmp_path: Path) -> None:
    for variant in ("dshort", "dmid", "dfinal"):
        label = {"dshort": "R1", "dmid": "R2", "dfinal": "R3"}[variant]
        _write(
            tmp_path / "dataset_audit" / f"primitive_v2_{variant}.audit.json",
            {"all_required_acceptance_pass": True},
        )
        audit = tmp_path / "runs" / f"{label}_{variant}_L0_s3072" / "audit" / "validation_own"
        ablation, rank = _audit_payload(0.08)
        ablation["final_reappearance"]["normalized_mse"]["normal"] = (
            1.11 if variant == "dshort" else 1.05
        )
        _write(audit / "memory_ablation.json", ablation)
        _write(audit / "effective_rank_report.json", rank)
        _write(
            tmp_path / "runs" / "C0_d0_L0_s3072" / "audit"
            / f"validation_{variant}" / "memory_ablation.json",
            {"final_reappearance": {"normalized_mse": {"normal": 1.0}}},
        )
    report = select(argparse.Namespace(campaign_root=tmp_path))
    assert not report["candidates"]["dshort"]["eligible"]
    assert report["candidates"]["dmid"]["eligible"]

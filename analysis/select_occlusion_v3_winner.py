#!/usr/bin/env python3
"""Select the V3 configuration after the N1/N2/N3 validation audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = {
    "N1": {"run_id": "N1_original_motion_L1_s3072", "train_dataset": "original_motion", "loss": "L1"},
    "N2": {"run_id": "N2_original_motion_L2_s3072", "train_dataset": "original_motion", "loss": "L2"},
    "N3": {"run_id": "N3_motion15x_L2_s3072", "train_dataset": "motion15x", "loss": "L2"},
}
EVAL_DATASETS = ("original_motion", "motion15x")
TIE_ORDER = ("N1", "N2", "N3")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_metrics(audit: Path) -> dict[str, Any]:
    ablation = _load(audit / "memory_ablation.json")
    rank = _load(audit / "effective_rank_report.json")
    probe = _load(audit / "position_probe.json")
    pair = _load(audit / "matched_pair_report.json")
    final = ablation["final_reappearance"]
    bootstrap = ablation["bootstrap_final_reappearance"]["shuffled_hidden"]
    target_rank = rank["global_rank"]["target_emb"]["covariance"]
    prediction_rank = rank["global_rank"]["pred_emb"]["covariance"]
    prediction_metrics = rank["prediction_metrics"]["groups"]["final_reappearance"]
    position = probe["scopes"]["final_reappearance"]
    angle = probe.get("angle_scopes", {}).get("final_reappearance", {})
    return {
        "normalized_mse": float(final["normalized_mse"]["normal"]),
        "raw_mse": float(final["normal"]),
        "cosine_error": float(prediction_metrics["mean_cosine_error"]),
        "hidden_shuffle_relative_increase": float(final["shuffled_hidden_relative_increase"]),
        "shuffle_ci95_lower": float(bootstrap["ci95_lower"]),
        "shuffle_ci95_upper": float(bootstrap["ci95_upper"]),
        "reset_relative_increase": float(final["reset_each_step_relative_increase"]),
        "no_memory_relative_increase": float(final["no_memory_relative_increase"]),
        "target_effective_rank": float(target_rank["effective_rank"]),
        "prediction_effective_rank": float(prediction_rank["effective_rank"]),
        "target_top1_variance_ratio": float(target_rank["top1_variance_ratio"]),
        "prediction_top1_variance_ratio": float(prediction_rank["top1_variance_ratio"]),
        "probe_r2_x": float(position["x"]["r2"]),
        "probe_r2_y": float(position["y"]["r2"]),
        "probe_angle_mean_r2": float(angle.get("mean_r2", -1.0)),
        "matched_pair_delta_alignment": float(pair.get("mean_delta_alignment_cosine", -1.0)),
    }


def _run_metrics(campaign: Path, label: str) -> dict[str, Any]:
    definition = RUNS[label]
    evaluations = {}
    for dataset in EVAL_DATASETS:
        audit = campaign / "runs" / definition["run_id"] / "audit" / f"validation_{dataset}"
        evaluations[dataset] = _eval_metrics(audit)
    macro_nmse = sum(value["normalized_mse"] for value in evaluations.values()) / len(evaluations)
    macro_cosine = sum(value["cosine_error"] for value in evaluations.values()) / len(evaluations)
    own = evaluations[definition["train_dataset"]]
    return {
        **definition,
        "evaluations": evaluations,
        "macro_normalized_mse": macro_nmse,
        "macro_cosine_error": macro_cosine,
        "own_validation": own,
    }


def select(campaign: Path) -> dict[str, Any]:
    metrics = {label: _run_metrics(campaign, label) for label in TIE_ORDER}
    baseline = metrics["N1"]["evaluations"]["original_motion"]
    candidates: dict[str, Any] = {}
    for label in TIE_ORDER:
        current = metrics[label]
        original = current["evaluations"]["original_motion"]
        own = current["own_validation"]
        rank_ok = all(
            evaluation["target_effective_rank"] >= 10.0
            and evaluation["prediction_effective_rank"] >= 10.0
            and evaluation["target_top1_variance_ratio"] <= 0.50
            and evaluation["prediction_top1_variance_ratio"] <= 0.50
            for evaluation in current["evaluations"].values()
        )
        probe_ok = (
            original["probe_r2_x"] >= baseline["probe_r2_x"] - 0.02
            and original["probe_r2_y"] >= baseline["probe_r2_y"] - 0.02
            and original["probe_angle_mean_r2"] >= baseline["probe_angle_mean_r2"] - 0.02
        )
        performance_ok = original["normalized_mse"] <= 1.10 * baseline["normalized_mse"]
        causal_checks = {
            "hidden_shuffle_at_least_5_percent": own["hidden_shuffle_relative_increase"] >= 0.05,
            "shuffle_ci95_lower_above_zero": own["shuffle_ci95_lower"] > 0.0,
            "reset_direction_positive": own["reset_relative_increase"] > 0.0,
            "no_memory_direction_positive": own["no_memory_relative_increase"] > 0.0,
        }
        eligibility = {
            "representation_healthy": rank_ok,
            "original_motion_nmse_within_110_of_N1": performance_ok,
            "physical_probe_drop_within_002": probe_ok,
        }
        candidates[label] = {
            "metrics": current,
            "eligibility": eligibility,
            "eligible_for_replication": bool(all(eligibility.values())),
            "causal_checks": causal_checks,
            "causal_signal": bool(all(causal_checks.values()) and rank_ok and performance_ok),
        }

    eligible = [label for label in TIE_ORDER if candidates[label]["eligible_for_replication"]]
    if not eligible:
        selected = None
        reason = "No run passed representation, original-motion performance, and physical-probe gates."
    else:
        best_nmse = min(candidates[label]["metrics"]["macro_normalized_mse"] for label in eligible)
        within_five_percent = [
            label
            for label in eligible
            if candidates[label]["metrics"]["macro_normalized_mse"] <= best_nmse * 1.05
        ]
        selected = next(label for label in TIE_ORDER if label in within_five_percent)
        reason = (
            "Lowest macro final-reappearance NMSE; changes below five percent use the "
            "preregistered simpler-configuration tie order N1, N2, N3."
        )
        if not any(candidates[label]["causal_signal"] for label in eligible):
            reason += (
                " No run passed the causal-memory gate; the selected run is replicated "
                "as the healthiest negative/tentative result, not as positive evidence."
            )

    selected_config = None
    if selected is not None:
        selected_config = {
            "label": selected,
            "dataset": RUNS[selected]["train_dataset"],
            "loss": RUNS[selected]["loss"],
        }
    return {
        "selection_policy": {
            "primary": "macro final-reappearance normalized MSE on original_motion and motion15x",
            "cosine_error_role": "diagnostic only; never sufficient because L2 can move targets",
            "material_improvement": 0.05,
            "tie_order": list(TIE_ORDER),
        },
        "candidates": candidates,
        "selected": selected_config,
        "selection_reason": reason,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = select(args.campaign_root.resolve())
    output = args.output or args.campaign_root / "selection_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    print(json.dumps({"selected": report["selected"], "reason": report["selection_reason"]}, indent=2))


if __name__ == "__main__":
    main()

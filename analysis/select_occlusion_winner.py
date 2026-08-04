#!/usr/bin/env python3
"""Select Dshort/Dmid/Dfinal against corrected same-variant C0 audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = ("dshort", "dmid", "dfinal")
RUN_LABEL = {"dshort": "R1", "dmid": "R2", "dfinal": "R3"}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_health(report: dict[str, Any]) -> dict[str, Any]:
    target = report["global_rank"]["target_emb"]["covariance"]
    prediction = report["global_rank"]["pred_emb"]["covariance"]
    checks = {
        "target_effective_rank_at_least_10": target["effective_rank"] >= 10.0,
        "prediction_effective_rank_at_least_10": prediction["effective_rank"] >= 10.0,
        "target_top1_at_most_050": target["top1_variance_ratio"] <= 0.50,
        "prediction_top1_at_most_050": prediction["top1_variance_ratio"] <= 0.50,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "target_effective_rank": target["effective_rank"],
        "prediction_effective_rank": prediction["effective_rank"],
        "target_top1_variance_ratio": target["top1_variance_ratio"],
        "prediction_top1_variance_ratio": prediction["top1_variance_ratio"],
    }


def _run_metrics(
    campaign: Path,
    run_id: str,
    variant: str,
) -> dict[str, Any]:
    audit = campaign / "runs" / run_id / "audit" / "validation_own"
    ablation = _load(audit / "memory_ablation.json")
    rank = _load(audit / "effective_rank_report.json")
    final = ablation["final_reappearance"]
    bootstrap = ablation["bootstrap_final_reappearance"]["shuffled_hidden"]
    health = _rank_health(rank)
    candidate_nmse = float(final["normalized_mse"]["normal"])
    c0_audit = (
        campaign / "runs" / "C0_d0_L0_s3072" / "audit"
        / f"validation_{variant}" / "memory_ablation.json"
    )
    c0_final = _load(c0_audit)["final_reappearance"]
    c0_nmse = float(c0_final["normalized_mse"]["normal"])
    performance_ratio = candidate_nmse / max(c0_nmse, 1.0e-12)
    causal_checks = {
        "shuffle_relative_increase_at_least_005": (
            final["shuffled_hidden_relative_increase"] >= 0.05
        ),
        "shuffle_bootstrap_ci_lower_above_zero": bootstrap["ci95_lower"] > 0.0,
        "reset_direction_positive": final["reset_each_step_relative_increase"] > 0.0,
        "no_memory_direction_positive": final["no_memory_relative_increase"] > 0.0,
        "representation_healthy": health["pass"],
        "nmse_within_110_of_corrected_c0_same_variant": performance_ratio <= 1.10,
    }
    return {
        "run_id": run_id,
        "final_reappearance_mse": final["normal"],
        "final_reappearance_normalized_mse": candidate_nmse,
        "corrected_c0_same_variant_normalized_mse": c0_nmse,
        "nmse_ratio_to_corrected_c0_same_variant": performance_ratio,
        "shuffled_hidden_relative_increase": final["shuffled_hidden_relative_increase"],
        "shuffle_ci95_lower": bootstrap["ci95_lower"],
        "shuffle_ci95_upper": bootstrap["ci95_upper"],
        "reset_each_step_relative_increase": final["reset_each_step_relative_increase"],
        "no_memory_relative_increase": final["no_memory_relative_increase"],
        "rank_health": health,
        "causal_checks": causal_checks,
        "causal_signal": all(causal_checks.values()),
    }


def select(args: argparse.Namespace) -> dict[str, Any]:
    campaign = args.campaign_root.resolve()
    candidates: dict[str, dict[str, Any]] = {}

    for variant in VARIANTS:
        run_id = f"{RUN_LABEL[variant]}_{variant}_L0_s3072"
        dataset = _load(campaign / "dataset_audit" / f"primitive_v2_{variant}.audit.json")
        metrics = _run_metrics(campaign, run_id, variant)
        eligibility = {
            "dataset_acceptance": dataset["all_required_acceptance_pass"],
            "representation_healthy": metrics["rank_health"]["pass"],
            "nmse_within_110_of_corrected_c0": (
                metrics["nmse_ratio_to_corrected_c0_same_variant"] <= 1.10
            ),
        }
        candidates[variant] = {
            "dataset": dataset,
            "metrics": metrics,
            "eligibility": eligibility,
            "eligible": all(eligibility.values()),
        }

    eligible = [name for name in VARIANTS if candidates[name]["eligible"]]
    if not eligible:
        selected = None
        reason = "No candidate passed dataset, collapse, and performance gates."
    else:
        ranked = sorted(
            eligible,
            key=lambda name: candidates[name]["metrics"]["shuffled_hidden_relative_increase"],
            reverse=True,
        )
        selected = ranked[0]
        if "dfinal" in eligible:
            best_score = candidates[selected]["metrics"]["shuffled_hidden_relative_increase"]
            final_score = candidates["dfinal"]["metrics"]["shuffled_hidden_relative_increase"]
            if best_score - final_score < 0.01:
                selected = "dfinal"
                reason = "Dfinal is within one percentage point and wins the coverage tie-break."
            else:
                reason = "Largest hidden-shuffle relative error increase among eligible candidates."
        else:
            reason = "Largest hidden-shuffle relative error increase among eligible candidates."

    seed_reports: list[dict[str, Any]] = []
    if selected:
        patterns = (
            f"R1_{selected}_L0_s3072" if selected == "dshort" else
            f"R2_{selected}_L0_s3072" if selected == "dmid" else
            f"R3_{selected}_L0_s3072",
            f"R4_{selected}_L0_s4096",
            f"R5_{selected}_L0_s5120",
        )
        for run_id in patterns:
            report_path = campaign / "runs" / run_id / "audit" / "validation_own" / "memory_ablation.json"
            if report_path.is_file():
                seed_reports.append(_run_metrics(campaign, run_id, selected))

    return {
        "baseline_policy": "corrected C0 evaluated on the same variant validation split",
        "operational_noncollapse_rule": {
            "effective_rank_min": 10.0,
            "top1_variance_ratio_max": 0.50,
            "note": "A diagnostic gate, not evidence of memory by itself.",
        },
        "candidates": candidates,
        "selected": selected,
        "selection_reason": reason,
        "selected_seed_reports": seed_reports,
        "loss_experiment_status": "deferred_until_after_core_six_runs",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = select(args)
    output = args.output or args.campaign_root / "selection_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "selected": report["selected"],
        "selection_reason": report["selection_reason"],
        "loss_experiment_status": report["loss_experiment_status"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

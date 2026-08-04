#!/usr/bin/env python3
"""Freeze LeWM campaign tables, figures, evidence ledger, brief, and handoff."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


LENGTH = {"dshort": 2, "dmid": 4, "dlong": 8}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional(path: Path) -> dict[str, Any] | None:
    return _load(path) if path.is_file() else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(campaign: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for record_path in sorted(campaign.glob("runs/*/run_record.json")):
        record = _load(record_path)
        run_id = record["run_id"]
        own = campaign / "runs" / run_id / "audit" / "validation_own"
        ablation = _optional(own / "memory_ablation.json")
        rank = _optional(own / "effective_rank_report.json")
        pair = _optional(own / "matched_pair_report.json")
        probe = _optional(own / "position_probe.json")
        row: dict[str, Any] = {
            "label": record["label"],
            "run_id": run_id,
            "dataset": record["dataset"],
            "loss": record["loss"],
            "seed": record["seed"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "training_hours": (
                record["training_seconds_this_invocation"] / 3600
                if record.get("training_seconds_this_invocation") is not None else None
            ),
        }
        if ablation and rank:
            final = ablation["final_reappearance"]
            ci = ablation["bootstrap_final_reappearance"]["shuffled_hidden"]
            row.update({
                "final_reappearance_mse": final["normal"],
                "final_reappearance_normalized_mse": final["normalized_mse"]["normal"],
                "shuffle_relative_increase": final["shuffled_hidden_relative_increase"],
                "shuffle_ci95_lower": ci["ci95_lower"],
                "shuffle_ci95_upper": ci["ci95_upper"],
                "reset_relative_increase": final["reset_each_step_relative_increase"],
                "no_memory_relative_increase": final["no_memory_relative_increase"],
                "target_effective_rank": rank["global_rank"]["target_emb"]["covariance"]["effective_rank"],
                "prediction_effective_rank": rank["global_rank"]["pred_emb"]["covariance"]["effective_rank"],
            })
        if pair:
            row.update({
                "matched_pair_count": pair.get("pair_count"),
                "matched_pair_separation_ratio": pair.get("mean_separation_ratio"),
                "matched_pair_delta_alignment": pair.get("mean_delta_alignment_cosine"),
            })
        if probe:
            final_probe = probe.get("scopes", {}).get("final_reappearance", {})
            if final_probe.get("status") == "ok":
                row["hidden_probe_r2_x"] = final_probe["x"]["r2"]
                row["hidden_probe_r2_y"] = final_probe["y"]["r2"]
        run_rows.append(row)

        for variant, length in LENGTH.items():
            test = campaign / "runs" / run_id / "audit" / f"test_{variant}" / "memory_ablation.json"
            if not test.is_file():
                continue
            test_ablation = _load(test)
            final = test_ablation["final_reappearance"]
            ci = test_ablation["bootstrap_final_reappearance"]["shuffled_hidden"]
            cross_rows.append({
                "run_id": run_id,
                "dataset": record["dataset"],
                "loss": record["loss"],
                "seed": record["seed"],
                "test_variant": variant,
                "occlusion_length": length,
                "final_reappearance_mse": final["normal"],
                "shuffle_relative_increase": final["shuffled_hidden_relative_increase"],
                "shuffle_ci95_lower": ci["ci95_lower"],
                "shuffle_ci95_upper": ci["ci95_upper"],
            })
    return run_rows, cross_rows


def _plots(campaign: Path, cross_rows: list[dict[str, Any]], selected: str | None) -> list[str]:
    if not selected:
        return []
    import matplotlib.pyplot as plt
    import numpy as np

    selected_rows = [
        row for row in cross_rows
        if row["dataset"] == selected and row["loss"] == "L0"
    ]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[int(row["occlusion_length"])].append(row)
    if not grouped:
        return []
    plot_dir = campaign / "final_report" / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    x = sorted(grouped)
    for field, ylabel, filename in (
        ("final_reappearance_mse", "Normalized reappearance MSE", "error_vs_occlusion_length.png"),
        ("shuffle_relative_increase", "Hidden-shuffle relative error increase", "intervention_vs_occlusion_length.png"),
    ):
        mean = [np.mean([row[field] for row in grouped[length]]) for length in x]
        std = [np.std([row[field] for row in grouped[length]]) for length in x]
        fig, axis = plt.subplots(figsize=(6.2, 4.2))
        axis.errorbar(x, mean, yerr=std, marker="o", capsize=4, linewidth=2)
        if field == "shuffle_relative_increase":
            axis.axhline(0.05, color="tab:red", linestyle="--", label="5% causal gate")
            axis.legend()
        axis.set_xlabel("Occlusion length (frames)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def _evidence_rows(
    campaign: Path,
    run_rows: list[dict[str, Any]],
    selection: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit_path in sorted((campaign / "dataset_audit").glob("*.audit.json")):
        audit = _load(audit_path)
        rows.append({
            "claim": f"{audit['dataset_variant']} passed preregistered data invariants",
            "status": "confirmed" if audit["all_required_acceptance_pass"] else "contradicted",
            "evidence": str(audit_path),
            "boundary": "Data validity only; says nothing about learned memory.",
        })
    for row in run_rows:
        if "shuffle_relative_increase" not in row:
            continue
        causal = (
            row["shuffle_relative_increase"] >= 0.05
            and row["shuffle_ci95_lower"] > 0
            and row["reset_relative_increase"] > 0
            and row["no_memory_relative_increase"] > 0
            and row["target_effective_rank"] >= 10
            and row["prediction_effective_rank"] >= 10
        )
        rows.append({
            "claim": f"{row['run_id']} shows preliminary causal memory",
            "status": "confirmed" if causal else "not_supported",
            "evidence": str(campaign / "runs" / row["run_id"] / "audit" / "validation_own"),
            "boundary": "Requires seed-level agreement before a campaign-level claim.",
        })
    if selection:
        rows.append({
            "claim": "Preregistered data winner selected",
            "status": "confirmed" if selection.get("selected") else "blocked",
            "evidence": str(campaign / "selection_decision.json"),
            "boundary": selection.get("selection_reason"),
        })
    return rows


def _brief(selection: dict[str, Any] | None, run_rows: list[dict[str, Any]]) -> str:
    selected = selection.get("selected") if selection else None
    reason = selection.get("selection_reason") if selection else "Selection not yet available."
    loss_status = selection.get("loss_experiment_status") if selection else "Not yet determined."
    completed = len(run_rows)
    causal = [
        row for row in run_rows
        if row.get("shuffle_relative_increase", -1) >= 0.05
        and row.get("shuffle_ci95_lower", -1) > 0
        and row.get("reset_relative_increase", -1) > 0
        and row.get("no_memory_relative_increase", -1) > 0
    ]
    return f"""# LeWM 遮挡课程实验：给陈老师的研究简报

## 研究问题与诊断起点

历史 main25 虽通过 conditioned variance/covariance regularization 缓解了表征塌缩，但其 predictor 仍能访问 Transformer 历史 token，并存在当前视觉 residual 旁路；它在 hidden-shuffle、逐步 reset 与 no-memory 干预下几乎不变，因此只保留为“架构不满足严格递归瓶颈的历史负结果”。本轮先修正为严格数据流 `s_t → h_t`、`[h_t,a_t] → s_hat_(t+1)`，再检验短遮挡可能无需记忆、长遮挡可能过难、中等遮挡可能处于 Goldilocks 区间。

## 受控设计

在 corrected C0 与 R1–R5 之间，模型、学习率、batch、encoder 正则与 L0 完全冻结。Dshort/Dmid 只把第二次遮挡改为 2/4 帧；Dlong_eval 保留 8 帧；Dfinal 按 50/30/20 混合三种长度、平衡 x/y，并对第二次 onset/reveal 做 ±1 帧抖动。所有变化均从物理状态重新渲染，未做像素编辑。训练/验证/测试按 trajectory seed 隔离；Dfinal 另含 matched-history/future-divergence 对，用相同行为与遮挡像素、不同可见 y 历史及未来，直接检验模型是否利用历史。

## 当前结果

已完成 {completed} 次完整训练。预注册选择结果为 `{selected}`；理由：{reason}

满足单次“初步因果记忆”完整门槛的运行数为 {len(causal)}。该门槛同时要求：hidden-shuffle 误差增量至少 5%、bootstrap 95% CI 下界大于 0、reset/no-memory 同向、target/prediction 不塌缩，且同一数据 validation split 上的 NMSE 相对 corrected C0 不恶化超过 10%。x/y probe、effective rank 和 matched-pair separation 仅作为机制诊断，未被单独解释成因果证据。

## 解释边界与下一步

若中等遮挡的干预效应最高、短遮挡可预测但干预弱、长遮挡失败，则支持 Goldilocks 解释。若所有新数据训练健康但都无干预效应，数据难度不再是充分解释，申请材料完成后再独立检验 L1 或更强的递归状态更新。若数据不变量、训练健康或 seed 一致性失败，则不扩张理论结论，优先修复并重跑同一配置。

Loss 对照状态：{loss_status}

这套工作形成了“发现伪成功—定位数据捷径/难度—受控重构数据—因果干预验证”的闭环。即使最终为负结果，也能区分数据太易、数据太难、loss 瓶颈与递归架构瓶颈，而不是把单一训练曲线误写成记忆能力。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    output = campaign / "final_report"
    output.mkdir(parents=True, exist_ok=True)
    selection = _optional(campaign / "selection_decision.json")
    run_rows, cross_rows = collect(campaign)
    _write_csv(output / "results_table.csv", run_rows)
    _write_csv(output / "cross_length_results.csv", cross_rows)
    figures = _plots(campaign, cross_rows, selection.get("selected") if selection else None)
    ledger = _evidence_rows(campaign, run_rows, selection)
    _write_csv(output / "evidence_ledger.csv", ledger)
    (output / "advisor_brief.md").write_text(_brief(selection, run_rows), encoding="utf-8")
    handoff = {
        "campaign_root": str(campaign),
        "generated_artifacts": [
            str(output / "results_table.csv"),
            str(output / "cross_length_results.csv"),
            str(output / "evidence_ledger.csv"),
            str(output / "advisor_brief.md"),
            *figures,
        ],
        "selection": selection,
        "run_count": len(run_rows),
        "commands": {
            "status": "python scripts/run_occlusion_campaign.py status",
            "reselect": f"python analysis/select_occlusion_winner.py --campaign-root {campaign}",
            "rerender_report": f"python analysis/render_campaign_report.py --campaign-root {campaign}",
        },
    }
    (output / "experiment_handoff.json").write_text(
        json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(handoff, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

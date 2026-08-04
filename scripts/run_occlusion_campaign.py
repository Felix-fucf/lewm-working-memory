#!/usr/bin/env python3
"""Single-GPU orchestration for the preregistered LeWM occlusion campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence


VARIANTS = ("dshort", "dmid", "dfinal")
EVAL_VARIANTS = ("dshort", "dmid", "dlong", "dfinal")
DATA_CONFIG = {name: f"primitive_v2_{name}" for name in EVAL_VARIANTS}
SCREEN_RUNS = (
    ("C0", "d0", 3072),
    ("R1", "dshort", 3072),
    ("R2", "dmid", 3072),
    ("R3", "dfinal", 3072),
)
REPLICATION_RUNS = (("R4", 4096), ("R5", 5120))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path(__file__).resolve().parent.parent
        self.world = args.world_root.resolve()
        self.datasets = self.world / "datasets"
        self.v2_dir = self.datasets / "primitive_v2"
        self.results = self.world / "results" / args.campaign_name
        self.checkpoints = self.world / ".stable-wm" / "checkpoints"
        self.source = (
            args.source_dataset.resolve()
            if args.source_dataset
            else self.datasets / "primitive_v1" / "formal_v1" / "primitive_v1_mixed_train.h5"
        )
        self.env = os.environ.copy()
        self.env["WORLD_MODEL_ROOT"] = str(self.world)
        self.env["STABLEWM_HOME"] = str(self.world / ".stable-wm")
        self.env["LOCAL_DATASET_DIR"] = str(self.datasets)
        self.env["LOCAL_ONLY_DATASET"] = "1"

    def execute(self, command: Sequence[str], log_path: Path, label: str) -> float:
        rendered = " ".join(str(item) for item in command)
        print(f"[{label}] {rendered}", flush=True)
        if self.args.dry_run:
            return 0.0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{_now()}] START {rendered}\n")
            log.flush()
            result = subprocess.run(
                list(command),
                # Preserve the execution root used by the archived main25 run.
                # The code itself may live under world-model/repos/.
                cwd=self.world,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            duration = time.monotonic() - started
            log.write(f"[{_now()}] END returncode={result.returncode} seconds={duration:.3f}\n")
        if result.returncode != 0:
            raise RuntimeError(f"{label} failed; inspect {log_path}")
        return duration

    def prepare(self) -> None:
        if not self.source.is_file() and not self.args.dry_run:
            raise FileNotFoundError(self.source)
        expected = [self.v2_dir / f"primitive_v2_{name}.h5" for name in EVAL_VARIANTS]
        expected_manifests = [
            self.v2_dir / f"primitive_v2_{name}.manifest.json"
            for name in EVAL_VARIANTS
        ]
        datasets_current = False
        if (
            not self.args.dry_run
            and all(path.is_file() for path in [*expected, *expected_manifests])
        ):
            import h5py

            policies = []
            for path in expected:
                with h5py.File(path, "r") as dataset:
                    policies.append(str(dataset.attrs.get("frame_loss_weight_policy", "")))
            datasets_current = all(
                policy == "phase_relative_sum_normalized_v2" for policy in policies
            )
        needs_rebuild = self.args.overwrite or not datasets_current
        if (
            not self.args.dry_run
            and needs_rebuild
            and self.v2_dir.exists()
            and any(self.v2_dir.iterdir())
        ):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            quarantine = self.datasets / f"_quarantine_primitive_v2_{stamp}"
            files = [path for path in self.v2_dir.rglob("*") if path.is_file()]
            archive_manifest = {
                "created_utc": _now(),
                "reason": "superseded absolute-timestep frame-loss weights",
                "source_directory": str(self.v2_dir),
                "quarantine_directory": str(quarantine),
                "files": {
                    str(path.relative_to(self.v2_dir)): _sha256(path) for path in files
                },
            }
            _write_json(self.results / "dataset_archive_manifest.json", archive_manifest)
            shutil.move(str(self.v2_dir), str(quarantine))
            print(f"[archive datasets] {self.v2_dir} -> {quarantine}", flush=True)
        elif self.args.dry_run:
            print("[archive datasets] archive primitive_v2 only when policy is legacy", flush=True)

        build = [
            self.args.python,
            str(self.repo / "schema_tools" / "build_occlusion_curriculum.py"),
            "--source", str(self.source),
            "--output-dir", str(self.v2_dir),
            "--variants", *EVAL_VARIANTS,
            "--seed", "20260728",
            "--pair-fraction", "0.20",
        ]
        if self.args.overwrite:
            build.append("--overwrite")
        if not needs_rebuild:
            print("[build datasets] current phase-relative datasets already exist; skip", flush=True)
        else:
            self.execute(build, self.results / "logs" / "prepare.log", "build datasets")

        audit_inputs = [
            self.source,
            *[self.v2_dir / f"primitive_v2_{name}.h5" for name in EVAL_VARIANTS],
        ]
        audit = [
            self.args.python,
            str(self.repo / "schema_tools" / "audit_occlusion_dataset.py"),
            *[str(path) for path in audit_inputs],
            "--output-dir", str(self.results / "dataset_audit"),
            "--enforce",
        ]
        self.execute(audit, self.results / "logs" / "prepare.log", "audit datasets")
        montage = [
            self.args.python,
            str(self.repo / "schema_tools" / "render_occlusion_samples.py"),
            str(self.v2_dir / "primitive_v2_dfinal.h5"),
            str(self.results / "dataset_audit" / "dfinal_samples.png"),
            "--rows", "4",
        ]
        self.execute(montage, self.results / "logs" / "prepare.log", "render Dfinal samples")
        if self.args.dry_run:
            return
        manifest_files = [
            self.repo / "train_oot.py",
            self.repo / "module_oot.py",
            self.repo / "config" / "train" / "lewm_occlusion.yaml",
            self.repo / "config" / "train" / "lewm_occlusion_l1.yaml",
            self.repo / "config" / "train" / "model" / "oot_leaky_bottleneck_ln_zeroh0.yaml",
            *[
                self.repo / "config" / "train" / "data" / f"primitive_v2_{name}.yaml"
                for name in EVAL_VARIANTS
            ],
            self.repo / "configs" / "occlusion_curriculum" / "campaign.yaml",
            self.repo / "schema_tools" / "build_occlusion_curriculum.py",
            self.repo / "schema_tools" / "audit_occlusion_dataset.py",
            self.repo / "schema_tools" / "render_occlusion_samples.py",
            self.repo / "analysis" / "metrics.py",
            self.repo / "analysis" / "audit_effective_rank.py",
            self.repo / "analysis" / "select_occlusion_winner.py",
            self.repo / "analysis" / "render_campaign_report.py",
            self.repo / "scripts" / "run_occlusion_campaign.py",
        ]
        manifest = {
            "created_utc": _now(),
            "campaign": self.args.campaign_name,
            "world_root": str(self.world),
            "source_dataset": str(self.source),
            "source_sha256": _sha256(self.source),
            "files": {str(path.relative_to(self.repo)): _sha256(path) for path in manifest_files},
            "datasets": {
                name: json.loads(
                    (self.v2_dir / f"primitive_v2_{name}.manifest.json").read_text(encoding="utf-8")
                )
                for name in EVAL_VARIANTS
            },
        }
        _write_json(self.results / "campaign_manifest.json", manifest)

    def _train_command(
        self,
        run_id: str,
        variant: str,
        seed: int,
        loss: str,
        smoke: bool,
    ) -> list[str]:
        config_name = "lewm_occlusion" if loss == "L0" else "lewm_occlusion_l1"
        if variant == "d0":
            data_config = "primitive_v1_mixed_train"
        else:
            data_config = DATA_CONFIG[variant]
        command = [
            self.args.python,
            str(self.repo / "train_oot.py"),
            "--config-name", config_name,
            f"data={data_config}",
            f"seed={seed}",
            f"output_model_name={run_id}",
            f"subdir={run_id}",
            f"checkpoint.dirpath={self.results / 'runs' / run_id / 'best_checkpoints'}",
        ]
        if variant == "d0":
            command.extend(["split.assignment_key=null", "+split.stratify_key=motion_type"])
        if smoke:
            command.append("+trainer.max_steps=1000")
            if variant != "d0":
                command.extend([
                    "checkpoint.enabled=false",
                    "checkpoint.export_weights=false",
                ])
        return command

    def smoke(self) -> None:
        timings: dict[str, object] = {"created_utc": _now(), "runs": {}}
        for label, variant, seed in SCREEN_RUNS:
            run_id = f"smoke_{label}_{variant}_L0_s{seed}"
            duration = self.execute(
                self._train_command(run_id, variant, seed, "L0", smoke=True),
                self.results / "smoke" / f"{run_id}.log",
                run_id,
            )
            timings["runs"][variant] = {"seconds_1000_steps": duration}
            if variant == "d0" and not self.args.dry_run:
                exported = sorted((self.checkpoints / run_id).glob("weights_epoch_*.pt"))
                if not exported:
                    raise FileNotFoundError(
                        f"Corrected C0 smoke did not export a checkpoint under "
                        f"{self.checkpoints / run_id}"
                    )
                strict_load = [
                    self.args.python,
                    str(self.repo / "analysis" / "audit_effective_rank.py"),
                    "--checkpoint", str(exported[-1]),
                    "--output-dir", str(self.results / "smoke" / "c0_strict_load"),
                    "--dataset-root", str(self.datasets),
                    "--config-name", "lewm_occlusion",
                    "--data-config", "primitive_v1_mixed_train",
                    "--model-config", "oot_leaky_bottleneck_ln_zeroh0",
                    "--seed", str(seed),
                    "--legacy-random-split",
                    "--max-batches", "1",
                    "--bootstrap-samples", "100",
                ]
                self.execute(
                    strict_load,
                    self.results / "smoke" / "c0_strict_load.log",
                    "corrected C0 checkpoint strict-load",
                )
        if not self.args.dry_run:
            observed = [item["seconds_1000_steps"] for item in timings["runs"].values()]
            median = float(sorted(observed)[len(observed) // 2])
            projected_hours = median * 135_000 / 1000 / 3600
            timings["median_seconds_1000_steps"] = median
            timings["projected_25_epoch_hours"] = projected_hours
            timings["budget_branch"] = (
                "C0-R5" if projected_hours <= 28
                else "C0-R4_R5_only_if_before_Aug8" if projected_hours <= 38
                else "C0-R4_two_seed_tentative"
            )
            _write_json(self.results / "smoke" / "timing_estimate.json", timings)

    def audit_checkpoint(
        self,
        run_id: str,
        checkpoint: Path,
        seed: int,
        include_reserved_test: bool = False,
    ) -> None:
        evaluations: list[tuple[str, str | None, int]] = [("validation_own", None, 1)]
        own_variant = re.search(r"_(dshort|dmid|dfinal|d0)_", run_id)
        if own_variant:
            evaluations[0] = ("validation_own", own_variant.group(1), 1)
        if own_variant and own_variant.group(1) == "d0":
            for variant in VARIANTS:
                evaluations.append((f"validation_{variant}", variant, 1))
        if include_reserved_test:
            for variant in EVAL_VARIANTS:
                evaluations.append((f"test_{variant}", variant, 2))

        for label, variant, split_value in evaluations:
            if variant is None:
                continue
            output_dir = self.results / "runs" / run_id / "audit" / label
            report = output_dir / "memory_ablation.json"
            if report.is_file() and not self.args.overwrite:
                print(f"[{run_id}] skip existing {label}")
                continue
            command = [
                self.args.python,
                str(self.repo / "analysis" / "audit_effective_rank.py"),
                "--checkpoint", str(checkpoint),
                "--output-dir", str(output_dir),
                "--dataset-root", str(self.datasets),
                "--config-name", "lewm_occlusion",
                "--data-config", (
                    "primitive_v1_mixed_train" if variant == "d0" else DATA_CONFIG[variant]
                ),
                "--model-config", "oot_leaky_bottleneck_ln_zeroh0",
                "--seed", str(seed),
                "--split-value", str(split_value),
            ]
            if variant == "d0":
                command.append("--legacy-random-split")
            self.execute(
                command,
                self.results / "runs" / run_id / f"audit_{label}.log",
                f"{run_id} {label}",
            )

    def full_run(self, label: str, variant: str, seed: int, loss: str) -> None:
        run_id = f"{label}_{variant}_{loss}_s{seed}"
        run_dir = self.results / "runs" / run_id
        checkpoint = self.checkpoints / run_id / "weights_epoch_25.pt"
        record_path = run_dir / "run_record.json"
        started = _now()
        if checkpoint.is_file() and not self.args.overwrite:
            print(f"[{run_id}] epoch-25 checkpoint exists; skip training")
            duration = None
        else:
            duration = self.execute(
                self._train_command(run_id, variant, seed, loss, smoke=False),
                run_dir / "train.log",
                run_id,
            )
        if self.args.dry_run:
            return
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training ended without {checkpoint}")
        log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
        health_failures = re.findall(
            r"(?i)(CUDA out of memory|Traceback \(most recent call last\)|\bloss[^\n]{0,30}\bnan\b)",
            log_text,
        )
        record = {
            "label": label,
            "run_id": run_id,
            "dataset": variant,
            "loss": loss,
            "seed": seed,
            "epochs": 25,
            "model_config": "oot_leaky_bottleneck_ln_zeroh0",
            "strict_recurrent_bottleneck": True,
            "started_utc": started,
            "completed_utc": _now(),
            "training_seconds_this_invocation": duration,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "health_failures": health_failures,
        }
        _write_json(record_path, record)
        if health_failures:
            raise RuntimeError(f"Training health check failed for {run_id}: {health_failures[:3]}")
        self.audit_checkpoint(run_id, checkpoint, seed)

    def screen(self) -> None:
        for label, variant, seed in SCREEN_RUNS:
            self.full_run(label, variant, seed, "L0")

    def replicate(self) -> None:
        if self.args.selected not in VARIANTS:
            raise ValueError("replicate requires --selected dshort|dmid|dfinal")
        runs = REPLICATION_RUNS
        timing_path = self.results / "smoke" / "timing_estimate.json"
        if timing_path.is_file() and not self.args.force_budget:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            if timing.get("budget_branch") == "C0-R4_two_seed_tentative":
                runs = REPLICATION_RUNS[:1]
                print("[budget] >38h projection: run R4 only; conclusion is two-seed tentative")
            elif (
                timing.get("budget_branch") == "C0-R4_R5_only_if_before_Aug8"
                and date.today() >= date(2026, 8, 8)
            ):
                runs = REPLICATION_RUNS[:1]
                print("[budget] R5 start gate closed on 2026-08-08")
        for label, seed in runs:
            self.full_run(label, self.args.selected, seed, "L0")

    def status(self) -> None:
        records = []
        for record in sorted(self.results.glob("runs/*/run_record.json")):
            records.append(json.loads(record.read_text(encoding="utf-8")))
        payload = {
            "campaign": self.args.campaign_name,
            "world_root": str(self.world),
            "completed_runs": records,
            "completed_count": len(records),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    def report(self) -> None:
        selection = [
            self.args.python,
            str(self.repo / "analysis" / "select_occlusion_winner.py"),
            "--campaign-root", str(self.results),
        ]
        self.execute(selection, self.results / "logs" / "report.log", "select winner")
        if not self.args.dry_run:
            for record_path in sorted(self.results.glob("runs/*/run_record.json")):
                record = json.loads(record_path.read_text(encoding="utf-8"))
                self.audit_checkpoint(
                    str(record["run_id"]),
                    Path(record["checkpoint"]),
                    int(record["seed"]),
                    include_reserved_test=True,
                )
        render = [
            self.args.python,
            str(self.repo / "analysis" / "render_campaign_report.py"),
            "--campaign-root", str(self.results),
        ]
        self.execute(render, self.results / "logs" / "report.log", "render report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "smoke", "screen", "replicate", "status", "report"),
    )
    parser.add_argument(
        "--world-root",
        type=Path,
        default=Path(os.environ.get("WORLD_MODEL_ROOT", ".")),
    )
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--campaign-name", default="occlusion_curriculum_20260728")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--selected", choices=VARIANTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--force-budget",
        action="store_true",
        help="Override wall-clock/deadline gates after an explicit manual decision.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign = Campaign(args)
    getattr(campaign, args.stage)()


if __name__ == "__main__":
    main()

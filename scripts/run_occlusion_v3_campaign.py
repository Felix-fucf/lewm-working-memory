#!/usr/bin/env python3
"""Run the V3 five-reveal, stop-gradient, and motion-amplitude campaign."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


CAMPAIGN_NAME = "occlusion_v3_reveal5_20260731"
DATA_CONFIG = {
    "original_motion": "primitive_v3_h4_reveal5_original_motion",
    "motion15x": "primitive_v3_h4_reveal5_motion15x",
}
LOSS_CONFIG = {
    "L1": "lewm_occlusion_v3_l1",
    "L2": "lewm_occlusion_v3_l2",
}
SCREEN_RUNS = (
    ("N1", "original_motion", "L1", 3072),
    ("N2", "original_motion", "L2", 3072),
    ("N3", "motion15x", "L2", 3072),
)
REPLICATION_RUNS = (("N4", 4096), ("N5", 5120))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )


class V3Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path(__file__).resolve().parent.parent
        self.world = args.world_root.resolve()
        self.datasets = self.world / "datasets"
        self.v3_dir = self.datasets / "primitive_v3"
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
        rendered = " ".join(str(value) for value in command)
        print(f"[{label}] {rendered}", flush=True)
        if self.args.dry_run:
            return 0.0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"\n[{_now()}] START {rendered}\n")
            log.flush()
            result = subprocess.run(
                list(command),
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

    def _dataset_paths(self) -> dict[str, Path]:
        return {
            name: self.v3_dir / f"primitive_v3_h4_reveal5_{name}.h5"
            for name in DATA_CONFIG
        }

    def _datasets_current(self) -> bool:
        if self.args.dry_run:
            return False
        paths = self._dataset_paths()
        manifests = [path.with_suffix(".manifest.json") for path in paths.values()]
        pair_manifest = self.v3_dir / "primitive_v3_pair_manifest.json"
        if not all(path.is_file() for path in [*paths.values(), *manifests, pair_manifest]):
            return False
        import h5py

        for path in paths.values():
            with h5py.File(path, "r") as dataset:
                if (
                    str(dataset.attrs.get("dataset_version", "")) != "primitive_v3"
                    or int(dataset.attrs.get("completed", 0)) != 1
                    or int(dataset.attrs.get("first_occlusion_randomized", 1)) != 0
                    or str(dataset.attrs.get("frame_loss_weight_policy", ""))
                    != "phase_relative_random_onset_zero_fixed_reveal_sum_normalized_v3_1"
                    or str(dataset.attrs.get("object_orientation_policy", ""))
                    != "cos_sin_from_object_angle_v1"
                    or dataset["object_orientation"].shape
                    != (dataset["object_angle"].shape[0], 2)
                ):
                    return False
        return True

    def prepare(self) -> None:
        if not self.source.is_file() and not self.args.dry_run:
            raise FileNotFoundError(self.source)
        current = self._datasets_current()
        if (
            not self.args.dry_run
            and (self.args.overwrite or not current)
            and self.v3_dir.exists()
            and any(self.v3_dir.iterdir())
        ):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            quarantine = self.datasets / f"_quarantine_primitive_v3_{stamp}"
            files = [path for path in self.v3_dir.rglob("*") if path.is_file()]
            _write_json(
                self.results / "dataset_archive_manifest.json",
                {
                    "created_utc": _now(),
                    "reason": "partial, superseded, or explicitly overwritten primitive_v3 build",
                    "source_directory": str(self.v3_dir),
                    "quarantine_directory": str(quarantine),
                    "files": {str(path.relative_to(self.v3_dir)): _sha256(path) for path in files},
                },
            )
            shutil.move(str(self.v3_dir), str(quarantine))
            print(f"[archive datasets] {self.v3_dir} -> {quarantine}", flush=True)
        elif self.args.dry_run:
            print("[archive datasets] archive only an incomplete or explicitly overwritten V3 build")

        if current and not self.args.overwrite:
            print("[build datasets] completed primitive_v3 pair already exists; skip", flush=True)
        else:
            build = [
                self.args.python,
                str(self.repo / "schema_tools" / "build_occlusion_v3.py"),
                "--source", str(self.source),
                "--output-dir", str(self.v3_dir),
                "--seed", "20260731",
                "--pair-fraction", "0.20",
                "--minimum-eligible-episodes", "16000",
            ]
            if self.args.overwrite:
                build.append("--overwrite")
            self.execute(build, self.results / "logs" / "prepare.log", "build V3 datasets")

        paths = self._dataset_paths()
        audit = [
            self.args.python,
            str(self.repo / "schema_tools" / "audit_occlusion_v3.py"),
            "--original", str(paths["original_motion"]),
            "--motion15x", str(paths["motion15x"]),
            "--output-dir", str(self.results / "dataset_audit"),
            "--enforce",
        ]
        self.execute(audit, self.results / "logs" / "prepare.log", "audit V3 datasets")
        for name, path in paths.items():
            montage = [
                self.args.python,
                str(self.repo / "schema_tools" / "render_occlusion_samples.py"),
                str(path),
                str(self.results / "dataset_audit" / f"{name}_samples.png"),
                "--rows", "5",
            ]
            self.execute(
                montage,
                self.results / "logs" / "prepare.log",
                f"render {name} samples",
            )
        if self.args.dry_run:
            return
        manifest_files = [
            self.repo / "module_oot.py",
            self.repo / "train_oot.py",
            self.repo / "config" / "train" / "lewm_occlusion_v3_l1.yaml",
            self.repo / "config" / "train" / "lewm_occlusion_v3_l2.yaml",
            *[
                self.repo / "config" / "train" / "data" / f"primitive_v3_h4_reveal5_{name}.yaml"
                for name in DATA_CONFIG
            ],
            self.repo / "configs" / "occlusion_v3" / "campaign.yaml",
            self.repo / "schema_tools" / "build_occlusion_v3.py",
            self.repo / "schema_tools" / "audit_occlusion_v3.py",
            self.repo / "analysis" / "audit_effective_rank.py",
            self.repo / "analysis" / "compare_v3_latent_drift.py",
            self.repo / "analysis" / "select_occlusion_v3_winner.py",
            self.repo / "scripts" / "run_occlusion_v3_campaign.py",
        ]
        _write_json(
            self.results / "campaign_manifest.json",
            {
                "created_utc": _now(),
                "campaign": self.args.campaign_name,
                "source_dataset": str(self.source),
                "source_sha256": _sha256(self.source),
                "files": {
                    str(path.relative_to(self.repo)): _sha256(path) for path in manifest_files
                },
                "dataset_pair": json.loads(
                    (self.v3_dir / "primitive_v3_pair_manifest.json").read_text(encoding="utf-8")
                ),
            },
        )

    def _train_command(
        self,
        run_id: str,
        dataset: str,
        loss: str,
        seed: int,
        smoke: bool,
    ) -> list[str]:
        command = [
            self.args.python,
            str(self.repo / "train_oot.py"),
            "--config-name", LOSS_CONFIG[loss],
            f"data={DATA_CONFIG[dataset]}",
            f"seed={seed}",
            f"output_model_name={run_id}",
            f"subdir={run_id}",
            f"checkpoint.dirpath={self.results / 'runs' / run_id / 'best_checkpoints'}",
        ]
        if smoke:
            command.append("+trainer.max_steps=1000")
        return command

    def audit_checkpoint(
        self,
        run_id: str,
        checkpoint: Path,
        loss: str,
        seed: int,
        max_batches: int = 0,
        bootstrap_samples: int = 10_000,
        split_value: int = 1,
    ) -> None:
        for dataset in DATA_CONFIG:
            output = self.results / "runs" / run_id / "audit" / (
                f"validation_{dataset}" if split_value == 1 else f"test_{dataset}"
            )
            report = output / "memory_ablation.json"
            if report.is_file() and not self.args.overwrite:
                print(f"[{run_id}] skip existing {output.name}", flush=True)
                continue
            command = [
                self.args.python,
                str(self.repo / "analysis" / "audit_effective_rank.py"),
                "--checkpoint", str(checkpoint),
                "--output-dir", str(output),
                "--dataset-root", str(self.datasets),
                "--config-name", LOSS_CONFIG[loss],
                "--data-config", DATA_CONFIG[dataset],
                "--model-config", "oot_leaky_bottleneck_ln_zeroh0",
                "--seed", str(seed),
                "--split-value", str(split_value),
                "--bootstrap-samples", str(bootstrap_samples),
            ]
            if max_batches:
                command.extend(["--max-batches", str(max_batches)])
            self.execute(
                command,
                self.results / "runs" / run_id / f"audit_{output.name}.log",
                f"{run_id} {output.name}",
            )

    def smoke(self) -> None:
        timings: dict[str, object] = {"created_utc": _now(), "runs": {}}
        for label, dataset, loss, seed in SCREEN_RUNS:
            run_id = f"smoke_{label}_{dataset}_{loss}_s{seed}"
            duration = self.execute(
                self._train_command(run_id, dataset, loss, seed, smoke=True),
                self.results / "smoke" / f"{run_id}.log",
                run_id,
            )
            timings["runs"][label] = {"seconds_1000_steps": duration}
            if self.args.dry_run:
                continue
            exported = sorted((self.checkpoints / run_id).glob("weights_epoch_*.pt"))
            if not exported:
                raise FileNotFoundError(f"Smoke did not export a strict-load checkpoint: {run_id}")
            self.audit_checkpoint(
                run_id,
                exported[-1],
                loss,
                seed,
                max_batches=1,
                bootstrap_samples=100,
            )
        if self.args.dry_run:
            return
        audit = json.loads(
            (self.results / "dataset_audit" / "original_motion.audit.json").read_text(encoding="utf-8")
        )
        train_episodes = int(audit["split"]["counts"]["0"])
        projected_steps = ((train_episodes + 3) // 4) * 25
        observed = [float(value["seconds_1000_steps"]) for value in timings["runs"].values()]
        median = sorted(observed)[len(observed) // 2]
        timings["train_episodes"] = train_episodes
        timings["projected_steps_25_epochs"] = projected_steps
        timings["median_seconds_1000_steps"] = median
        timings["projected_25_epoch_hours"] = median * projected_steps / 1000.0 / 3600.0
        _write_json(self.results / "smoke" / "timing_estimate.json", timings)

    def full_run(self, label: str, dataset: str, loss: str, seed: int) -> None:
        run_id = f"{label}_{dataset}_{loss}_s{seed}"
        run_dir = self.results / "runs" / run_id
        checkpoint = self.checkpoints / run_id / "weights_epoch_25.pt"
        started = _now()
        if checkpoint.is_file() and not self.args.overwrite:
            print(f"[{run_id}] epoch-25 checkpoint exists; skip training", flush=True)
            duration = None
        else:
            duration = self.execute(
                self._train_command(run_id, dataset, loss, seed, smoke=False),
                run_dir / "train.log",
                run_id,
            )
        if self.args.dry_run:
            return
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training ended without {checkpoint}")
        log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
        failures = re.findall(
            r"(?i)(CUDA out of memory|Traceback \(most recent call last\)|\bloss[^\n]{0,30}\bnan\b)",
            log_text,
        )
        _write_json(
            run_dir / "run_record.json",
            {
                "label": label,
                "run_id": run_id,
                "dataset": dataset,
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
                "health_failures": failures,
            },
        )
        if failures:
            raise RuntimeError(f"Training health check failed for {run_id}: {failures[:3]}")
        self.audit_checkpoint(run_id, checkpoint, loss, seed)

    def screen(self) -> None:
        for label, dataset, loss, seed in SCREEN_RUNS:
            self.full_run(label, dataset, loss, seed)

    def select(self) -> None:
        n1 = self.results / "runs" / "N1_original_motion_L1_s3072" / "audit" / "validation_original_motion" / "anchor_embeddings.npz"
        for label, run_id in (
            ("N2", "N2_original_motion_L2_s3072"),
            ("N3", "N3_motion15x_L2_s3072"),
        ):
            candidate = self.results / "runs" / run_id / "audit" / "validation_original_motion" / "anchor_embeddings.npz"
            command = [
                self.args.python,
                str(self.repo / "analysis" / "compare_v3_latent_drift.py"),
                "--reference", str(n1),
                "--candidate", str(candidate),
                "--output", str(self.results / "selection" / f"N1_vs_{label}_latent_drift.json"),
            ]
            self.execute(command, self.results / "logs" / "select.log", f"N1 vs {label} CKA")
        command = [
            self.args.python,
            str(self.repo / "analysis" / "select_occlusion_v3_winner.py"),
            "--campaign-root", str(self.results),
        ]
        self.execute(command, self.results / "logs" / "select.log", "select V3 winner")

    def replicate(self) -> None:
        decision_path = self.results / "selection_decision.json"
        if not decision_path.is_file() and not self.args.dry_run:
            raise FileNotFoundError("Run the select stage before replicate")
        if self.args.dry_run:
            print(
                "[dry-run] replication uses dataset/loss from selection_decision.json; "
                "showing the N1-shaped command as a placeholder",
                flush=True,
            )
            selected = {"dataset": "original_motion", "loss": "L1"}
        else:
            selected = json.loads(decision_path.read_text(encoding="utf-8"))["selected"]
            if selected is None:
                raise RuntimeError("No configuration passed the replication eligibility gates")
        for label, seed in REPLICATION_RUNS:
            self.full_run(label, str(selected["dataset"]), str(selected["loss"]), seed)

    def status(self) -> None:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.results.glob("runs/*/run_record.json"))
        ]
        print(json.dumps({
            "campaign": self.args.campaign_name,
            "completed_count": len(records),
            "completed_runs": records,
        }, indent=2, ensure_ascii=False))

    def report(self) -> None:
        decision = self.results / "selection_decision.json"
        if not decision.is_file() and not self.args.dry_run:
            self.select()
        if not self.args.dry_run:
            for record_path in sorted(self.results.glob("runs/*/run_record.json")):
                record = json.loads(record_path.read_text(encoding="utf-8"))
                self.audit_checkpoint(
                    str(record["run_id"]),
                    Path(record["checkpoint"]),
                    str(record["loss"]),
                    int(record["seed"]),
                    split_value=2,
                )
        print(f"V3 reports are under {self.results}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "smoke", "screen", "select", "replicate", "status", "report"),
    )
    parser.add_argument(
        "--world-root",
        type=Path,
        default=Path(os.environ.get("WORLD_MODEL_ROOT", ".")),
    )
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--campaign-name", default=CAMPAIGN_NAME)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign = V3Campaign(args)
    getattr(campaign, args.stage)()


if __name__ == "__main__":
    main()

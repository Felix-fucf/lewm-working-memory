#!/usr/bin/env python3
"""Audit OOT embedding spectra and explicit Leaky-RNN memory use.

This script is intentionally standalone: it reconstructs the validation split
used by ``train_oot.py``, loads an exported ``weights_epoch_N.pt`` state dict,
collects full-validation latents, and writes rank/ablation reports without
resuming training or changing the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor

try:
    from analysis.metrics import (
        normalize_mse,
        target_variance_from_sufficient_statistics,
    )
except ImportError:  # Direct script execution.
    from metrics import normalize_mse, target_variance_from_sufficient_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-validation effective-rank diagnostics and "
            "Leaky-RNN ablations for an OOT checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("LOCAL_DATASET_DIR", "datasets")),
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--history-size", type=int, default=21)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 means the complete validation split.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--config-name",
        default="lewm",
    )
    parser.add_argument(
        "--data-config",
        default="primitive_v1_mixed_train",
    )
    parser.add_argument(
        "--model-config",
        default="oot_leaky_bottleneck_ln_zeroh0",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Episode bootstrap draws for final-reappearance intervention CIs.",
    )
    parser.add_argument(
        "--probe-alpha",
        type=float,
        default=1.0e-3,
        help="Ridge penalty for held-out x/y probes of memory_hidden.",
    )
    parser.add_argument(
        "--split-value",
        type=int,
        default=1,
        help="Preassigned split_id to evaluate: 1=validation, 2=reserved test.",
    )
    parser.add_argument(
        "--legacy-random-split",
        action="store_true",
        help="Use the original seeded/stratified split for D0, which has no split_id.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: move_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def compose_config(
    repo_root: Path,
    args: argparse.Namespace,
) -> DictConfig:
    config_dir = (repo_root / "config" / "train").resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")

    overrides = [
        f"data={args.data_config}",
        f"model={args.model_config}",
        f"seed={args.seed}",
        f"history_size={args.history_size}",
        f"num_preds={args.num_preds}",
        f"loader.batch_size={args.batch_size}",
    ]

    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        cfg = compose(
            config_name=args.config_name,
            overrides=overrides,
        )
    if args.legacy_random_split:
        with open_dict(cfg):
            cfg.split.assignment_key = None
            cfg.split.stratify_key = "motion_type"
    return cfg


def build_validation_loader(
    cfg: DictConfig,
    args: argparse.Namespace,
    helpers: Any,
) -> tuple[torch.utils.data.DataLoader, int, Path]:
    dataset_cfg = OmegaConf.to_container(
        cfg.data.dataset,
        resolve=True,
    )
    if not isinstance(dataset_cfg, dict):
        raise TypeError("cfg.data.dataset did not resolve to a dict")

    configured_name = Path(str(dataset_cfg.pop("name")))
    if configured_name.is_absolute():
        dataset_path = configured_name
    else:
        dataset_path = (args.dataset_root / configured_name).resolve()

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"[dataset] {dataset_path}")
    dataset = swm.data.load_dataset(
        str(dataset_path),
        transform=None,
        cache_dir=None,
        **dataset_cfg,
    )

    from utils import get_column_normalizer, get_img_preprocessor

    transforms = [
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=cfg.img_size,
        )
    ]

    configured_metadata = set(
        helpers.cfg_select(cfg, "data.metadata_keys", [])
    )
    metadata_keys = helpers.DEFAULT_METADATA_KEYS | configured_metadata

    for column in cfg.data.dataset.keys_to_load:
        if str(column).startswith("pixels"):
            continue
        if column in metadata_keys:
            continue
        try:
            transforms.append(
                get_column_normalizer(
                    dataset,
                    column,
                    column,
                )
            )
        except Exception as exc:
            print(
                f"[normalizer warning] {column!r}: "
                f"{type(exc).__name__}: {exc}"
            )

    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip
            * dataset.get_dim("action")
        )

    dataset.transform = spt.data.transforms.Compose(*transforms)

    split_generator = torch.Generator().manual_seed(args.seed)
    stratify_key = helpers.cfg_select(cfg, "split.stratify_key", None)
    assignment_key = helpers.cfg_select(cfg, "split.assignment_key", None)

    if assignment_key:
        validation_value = int(args.split_value)
        with h5py.File(dataset_path, "r") as file:
            if assignment_key not in file:
                raise KeyError(
                    f"split.assignment_key={assignment_key!r} is absent from H5"
                )
            episode_offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
            assignments = np.asarray(
                file[str(assignment_key)][episode_offsets]
            ).reshape(-1)
        val_indices = np.flatnonzero(
            assignments == validation_value
        ).tolist()
        if not val_indices:
            raise RuntimeError(
                f"No validation episodes for {assignment_key}={validation_value}"
            )
        val_set = torch.utils.data.Subset(dataset, val_indices)
        split_kind = f"preassigned:{assignment_key}={validation_value}"
    elif stratify_key:
        with h5py.File(dataset_path, "r") as file:
            if stratify_key not in file:
                raise KeyError(
                    f"split.stratify_key={stratify_key!r} is absent from H5"
                )
            episode_offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
            episode_labels = np.asarray(
                file[str(stratify_key)][episode_offsets]
            ).reshape(-1)

        rng = np.random.default_rng(args.seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        for label in np.unique(episode_labels):
            indices = np.flatnonzero(episode_labels == label)
            rng.shuffle(indices)
            n_train = int(round(float(cfg.train_split) * len(indices)))
            train_indices.extend(indices[:n_train].tolist())
            val_indices.extend(indices[n_train:].tolist())
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)
        split_kind = f"stratified:{stratify_key}"
    else:
        _, val_set = spt.data.random_split(
            dataset,
            lengths=[cfg.train_split, 1.0 - cfg.train_split],
            generator=split_generator,
        )
        split_kind = "seeded_random_split"

    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    if not isinstance(loader_kwargs, dict):
        raise TypeError("cfg.loader did not resolve to a dict")
    loader_kwargs["batch_size"] = args.batch_size
    loader_kwargs["num_workers"] = args.num_workers
    loader_kwargs.pop("shuffle", None)
    loader_kwargs.pop("drop_last", None)

    if args.num_workers == 0:
        loader_kwargs.pop("prefetch_factor", None)
        loader_kwargs.pop("persistent_workers", None)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        **loader_kwargs,
        shuffle=False,
        drop_last=False,
    )

    print(
        f"[split] {split_kind}; validation episodes={len(val_set)}; "
        f"batches={len(val_loader)}"
    )
    return val_loader, len(val_set), dataset_path


def extract_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    if isinstance(checkpoint, Mapping):
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
        for key in ("state_dict", "model", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                state = dict(candidate)
                if state and all(name.startswith("model.") for name in state):
                    state = {
                        name.removeprefix("model."): value
                        for name, value in state.items()
                    }
                return state
    raise TypeError(
        "Unsupported checkpoint structure; expected a model state dict or a "
        "mapping containing state_dict/model/model_state_dict."
    )


def build_model(
    cfg: DictConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model.to(device=device)
    model.eval()
    print(
        f"[model] {type(model).__name__}; "
        f"parameters={sum(p.numel() for p in model.parameters()):,}; "
        f"device={device}"
    )
    return model


def as_bool_mask(mask: Optional[Tensor], reference: Tensor) -> Tensor:
    if mask is None:
        return torch.zeros_like(reference, dtype=torch.bool)
    return mask > 0.5


def build_group_masks(
    masks: Dict[str, Optional[Tensor]],
) -> Dict[str, Tensor]:
    valid = as_bool_mask(masks.get("valid"), masks["valid"])
    groups: Dict[str, Tensor] = {"all": valid}
    for name in (
        "visible",
        "visible_non_reappearance",
        "occluded",
        "reappearance",
        "first_reappearance",
        "final_reappearance",
    ):
        value = masks.get(name)
        if value is not None:
            groups[name] = (value > 0.5) & valid

    phase = masks.get("phase_id")
    if phase is not None:
        for phase_id in range(5):
            groups[f"phase_{phase_id}"] = (phase == phase_id) & valid
    return groups


def memory_embedding_from_hidden(
    model: torch.nn.Module,
    context_emb: Tensor,
    hidden: Tensor,
) -> Tensor:
    if getattr(model, "strict_recurrent_bottleneck", False):
        return hidden
    memory_delta = model.memory_to_embedding(hidden)
    if model.memory_residual:
        result = context_emb + model.memory_residual_scale * memory_delta
    else:
        result = memory_delta
    return model.memory_output_norm(result)


def no_delta_memory_embedding(
    model: torch.nn.Module,
    context_emb: Tensor,
) -> Tensor:
    if getattr(model, "strict_recurrent_bottleneck", False):
        return torch.zeros_like(context_emb)
    if model.memory_residual:
        base = context_emb
    else:
        base = torch.zeros_like(context_emb)
    return model.memory_output_norm(base)


class MSEAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.counts: Dict[str, float] = defaultdict(float)
        self.target_counts: Dict[str, int] = defaultdict(int)
        self.target_sums: Dict[str, Tensor] = {}
        self.target_square_sums: Dict[str, float] = defaultdict(float)

    def update(
        self,
        errors: Dict[str, Tensor],
        groups: Dict[str, Tensor],
        target: Tensor,
    ) -> None:
        for group_name, mask in groups.items():
            mask_f = mask.to(dtype=torch.float32)
            count = float(mask_f.sum().item())
            if count <= 0:
                continue
            self.counts[group_name] += count
            selected_target = target[mask].detach().to(device="cpu", dtype=torch.float64)
            self.target_counts[group_name] += int(selected_target.shape[0])
            target_sum = selected_target.sum(dim=0)
            if group_name not in self.target_sums:
                self.target_sums[group_name] = target_sum
            else:
                self.target_sums[group_name] += target_sum
            self.target_square_sums[group_name] += float(
                selected_target.square().sum().item()
            )
            for variant, error in errors.items():
                self.sums[variant][group_name] += float(
                    (error.float() * mask_f).sum().item()
                )

    def finalize(self) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for group, count in sorted(self.counts.items()):
            values = {
                variant: self.sums[variant][group] / count
                for variant in sorted(self.sums)
            }
            target_count = self.target_counts[group]
            target_sum = self.target_sums[group]
            embedding_dim = int(target_sum.numel())
            target_variance = target_variance_from_sufficient_statistics(
                target_count,
                embedding_dim,
                self.target_square_sums[group],
                float(target_sum.square().sum().item()),
            )
            normalized = {
                variant: normalize_mse(value, target_variance)
                for variant, value in values.items()
            }
            normal = values.get("normal", 0.0)
            report[group] = {
                "count": int(count),
                **values,
                "raw_mse": values,
                "target_variance": target_variance,
                "normalized_mse": normalized,
            }
            if normal > 0:
                for variant, value in values.items():
                    if variant == "normal":
                        continue
                    report[group][f"{variant}_relative_increase"] = (
                        value / normal - 1.0
                    )
        return report


def covariance_spectrum(
    features: Tensor,
    standardize_dimensions: bool,
) -> Dict[str, Any]:
    if features.ndim != 2:
        raise ValueError(f"Expected (N,D), got {tuple(features.shape)}")
    n, d = features.shape
    if n < 2:
        return {
            "num_samples": int(n),
            "embedding_dim": int(d),
            "error": "fewer than two samples",
        }

    x = features.to(device="cpu", dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    variances = x.square().sum(dim=0) / float(n - 1)
    inactive = int((variances <= 1.0e-12).sum().item())

    if standardize_dimensions:
        x = x / variances.clamp_min(1.0e-12).sqrt()

    covariance = (x.T @ x) / float(n - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1.0e-30)
    probabilities = eigenvalues / total

    entropy = -(
        probabilities
        * probabilities.clamp_min(1.0e-30).log()
    ).sum()
    effective_rank = entropy.exp()
    participation_ratio = total.square() / (
        eigenvalues.square().sum().clamp_min(1.0e-30)
    )
    stable_rank = total / eigenvalues[0].clamp_min(1.0e-30)
    cumulative = probabilities.cumsum(dim=0)
    dimensions_95 = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(0.95, dtype=cumulative.dtype),
        ).item()
        + 1
    )
    numerical_rank = int(
        (eigenvalues > eigenvalues[0] * 1.0e-6).sum().item()
    )

    return {
        "num_samples": int(n),
        "embedding_dim": int(d),
        "standardized_dimensions": bool(standardize_dimensions),
        "inactive_dimensions": inactive,
        "effective_rank": float(effective_rank.item()),
        "participation_ratio": float(participation_ratio.item()),
        "stable_rank": float(stable_rank.item()),
        "top1_variance_ratio": float(probabilities[0].item()),
        "top5_variance_ratio": float(probabilities[:5].sum().item()),
        "top10_variance_ratio": float(probabilities[:10].sum().item()),
        "dimensions_for_95pct": dimensions_95,
        "numerical_rank_1e-6": numerical_rank,
        "mean_dimension_std": float(variances.sqrt().mean().item()),
        "min_dimension_std": float(variances.sqrt().min().item()),
        "max_dimension_std": float(variances.sqrt().max().item()),
        "largest_eigenvalues": [
            float(value)
            for value in eigenvalues[:10].tolist()
        ],
    }


def rank_report(features: Tensor) -> Dict[str, Any]:
    return {
        "covariance": covariance_spectrum(
            features,
            standardize_dimensions=False,
        ),
        "correlation": covariance_spectrum(
            features,
            standardize_dimensions=True,
        ),
    }


def append_csv_rows(
    rows: list[Dict[str, Any]],
    scope: str,
    feature: str,
    report: Dict[str, Any],
) -> None:
    for spectrum_name in ("covariance", "correlation"):
        stats = report[spectrum_name]
        row: Dict[str, Any] = {
            "scope": scope,
            "feature": feature,
            "spectrum": spectrum_name,
        }
        for key, value in stats.items():
            if key == "largest_eigenvalues":
                row[key] = json.dumps(value)
            else:
                row[key] = value
        rows.append(row)


def _target_aligned(
    batch: Mapping[str, Tensor],
    key: str,
    num_preds: int,
    context_length: int,
) -> Optional[Tensor]:
    value = batch.get(key)
    if value is None:
        return None
    value = value[:, num_preds:num_preds + context_length]
    return value


def _episode_bootstrap(
    errors: Mapping[str, np.ndarray],
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    normal = np.asarray(errors["normal"], dtype=np.float64)
    report: Dict[str, Any] = {"episode_count": int(normal.size)}
    if normal.size == 0:
        return report

    rng = np.random.default_rng(seed)
    for variant in ("reset_each_step", "no_memory"):
        alternative = np.asarray(errors[variant], dtype=np.float64)
        observed = float(alternative.mean() / max(normal.mean(), 1.0e-12) - 1.0)
        draws: list[np.ndarray] = []
        remaining = samples
        while remaining > 0:
            chunk = min(remaining, 512)
            indices = rng.integers(0, normal.size, size=(chunk, normal.size))
            normal_means = normal[indices].mean(axis=1)
            alternative_means = alternative[indices].mean(axis=1)
            draws.append(
                alternative_means / np.maximum(normal_means, 1.0e-12) - 1.0
            )
            remaining -= chunk
        bootstrap = np.concatenate(draws)
        report[variant] = {
            "relative_increase": observed,
            "ci95_lower": float(np.quantile(bootstrap, 0.025)),
            "ci95_upper": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_samples": int(samples),
        }
    permutation_keys = sorted(
        key for key in errors if key.startswith("shuffled_hidden_perm_")
    )
    if permutation_keys:
        matrix = np.stack(
            [np.asarray(errors[key], dtype=np.float64) for key in permutation_keys]
        )
        observed = float(matrix.mean() / max(normal.mean(), 1.0e-12) - 1.0)
        draws = []
        remaining = samples
        while remaining > 0:
            chunk = min(remaining, 512)
            episode_indices = rng.integers(
                0, normal.size, size=(chunk, normal.size)
            )
            permutation_indices = rng.integers(
                0, matrix.shape[0], size=(chunk, matrix.shape[0])
            )
            normal_means = normal[episode_indices].mean(axis=1)
            alternatives = np.empty(chunk, dtype=np.float64)
            for draw in range(chunk):
                sampled = matrix[permutation_indices[draw]][:, episode_indices[draw]]
                alternatives[draw] = sampled.mean()
            draws.append(
                alternatives / np.maximum(normal_means, 1.0e-12) - 1.0
            )
            remaining -= chunk
        bootstrap = np.concatenate(draws)
        report["shuffled_hidden"] = {
            "relative_increase": observed,
            "ci95_lower": float(np.quantile(bootstrap, 0.025)),
            "ci95_upper": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_samples": int(samples),
            "permutation_seeds": len(permutation_keys),
            "resampling_unit": "episode_and_permutation",
        }
    return report


def _ridge_probe(
    x: np.ndarray,
    y: np.ndarray,
    episode_id: np.ndarray,
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    unique_episodes = np.unique(episode_id)
    if unique_episodes.size < 5 or x.shape[0] < 10:
        return {
            "status": "insufficient_data",
            "samples": int(x.shape[0]),
            "episodes": int(unique_episodes.size),
        }

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_episodes)
    split = max(1, min(unique_episodes.size - 1, int(round(0.8 * unique_episodes.size))))
    train_episodes = unique_episodes[:split]
    train_mask = np.isin(episode_id, train_episodes)
    test_mask = ~train_mask

    x_train = x[train_mask].astype(np.float64)
    x_test = x[test_mask].astype(np.float64)
    y_train = y[train_mask].astype(np.float64)
    y_test = y[test_mask].astype(np.float64)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    x_train = (x_train - mean) / np.maximum(std, 1.0e-6)
    x_test = (x_test - mean) / np.maximum(std, 1.0e-6)
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    gram = x_train.T @ x_train
    penalty = np.eye(gram.shape[0], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, x_train.T @ y_train)
    prediction = x_test @ weights
    mse = ((prediction - y_test) ** 2).mean(axis=0)
    denominator = ((y_test - y_test.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ((prediction - y_test) ** 2).sum(axis=0) / np.maximum(
        denominator,
        1.0e-12,
    )
    return {
        "status": "ok",
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "train_episodes": int(train_episodes.size),
        "test_episodes": int(unique_episodes.size - train_episodes.size),
        "x": {"mse": float(mse[0]), "r2": float(r2[0])},
        "y": {"mse": float(mse[1]), "r2": float(r2[1])},
    }


def _position_probe_report(
    features: Mapping[str, Tensor],
    groups: Mapping[str, Tensor],
    alpha: float,
    seed: int,
) -> Dict[str, Any]:
    x = features["memory_hidden"].numpy()
    y = features["object_position"].numpy()
    episode_id = features["episode_id"].numpy().reshape(-1)
    report: Dict[str, Any] = {
        "interpretation": "Decodability diagnostic only; not causal-memory evidence.",
        "ridge_alpha": float(alpha),
        "scopes": {},
    }
    for offset, scope in enumerate(("all", "occluded", "final_reappearance")):
        if scope not in groups:
            continue
        selection = groups[scope].numpy().astype(bool)
        report["scopes"][scope] = _ridge_probe(
            x[selection],
            y[selection],
            episode_id[selection],
            alpha,
            seed + offset,
        )
    if "object_velocity" in features:
        velocity = features["object_velocity"].numpy()
        report["velocity_scopes"] = {}
        for offset, scope in enumerate(("all", "occluded", "final_reappearance")):
            if scope not in groups:
                continue
            selection = groups[scope].numpy().astype(bool)
            report["velocity_scopes"][scope] = _ridge_probe(
                x[selection],
                velocity[selection],
                episode_id[selection],
                alpha,
                seed + 100 + offset,
            )
    if "object_angle" in features:
        angle = features["object_angle"].numpy().reshape(-1)
        finite = angle[np.isfinite(angle)]
        radians = angle if finite.size and np.max(np.abs(finite)) <= 2.0 * np.pi + 0.1 else np.radians(angle)
        circular = np.stack((np.sin(radians), np.cos(radians)), axis=1)
        report["angle_scopes"] = {}
        for offset, scope in enumerate(("all", "occluded", "final_reappearance")):
            if scope not in groups:
                continue
            selection = groups[scope].numpy().astype(bool)
            angle_probe = _ridge_probe(
                x[selection],
                circular[selection],
                episode_id[selection],
                alpha,
                seed + 200 + offset,
            )
            if angle_probe.get("status") == "ok":
                angle_probe["sin"] = angle_probe.pop("x")
                angle_probe["cos"] = angle_probe.pop("y")
                angle_probe["mean_r2"] = 0.5 * (
                    angle_probe["sin"]["r2"] + angle_probe["cos"]["r2"]
                )
            report["angle_scopes"][scope] = angle_probe
    return report


def _matched_pair_report(
    features: Mapping[str, Tensor],
    groups: Mapping[str, Tensor],
) -> Dict[str, Any]:
    if "pair_id" not in features or "pair_role" not in features:
        return {"pair_count": 0, "status": "pair_metadata_absent"}
    selection = groups.get("final_reappearance")
    if selection is None:
        return {"pair_count": 0, "status": "final_reappearance_mask_absent"}

    selected = selection.numpy().astype(bool)
    pair_id = features["pair_id"].numpy().reshape(-1)[selected]
    pair_role = features["pair_role"].numpy().reshape(-1)[selected]
    target = features["target_emb"].numpy()[selected]
    prediction = features["pred_emb"].numpy()[selected]
    rows: list[Dict[str, float]] = []
    for current_pair in sorted(int(item) for item in np.unique(pair_id) if item >= 0):
        members = np.flatnonzero(pair_id == current_pair)
        role0 = members[pair_role[members] == 0]
        role1 = members[pair_role[members] == 1]
        if role0.size != 1 or role1.size != 1:
            continue
        target_delta = target[role1[0]] - target[role0[0]]
        prediction_delta = prediction[role1[0]] - prediction[role0[0]]
        target_norm = float(np.linalg.norm(target_delta))
        prediction_norm = float(np.linalg.norm(prediction_delta))
        cosine = float(
            np.dot(target_delta, prediction_delta)
            / max(target_norm * prediction_norm, 1.0e-12)
        )
        rows.append({
            "target_separation": target_norm,
            "prediction_separation": prediction_norm,
            "separation_ratio": prediction_norm / max(target_norm, 1.0e-12),
            "delta_alignment_cosine": cosine,
        })
    if not rows:
        return {"pair_count": 0, "status": "no_complete_pairs_in_validation"}
    return {
        "pair_count": len(rows),
        "status": "ok",
        "mean_target_separation": float(np.mean([row["target_separation"] for row in rows])),
        "mean_prediction_separation": float(np.mean([row["prediction_separation"] for row in rows])),
        "mean_separation_ratio": float(np.mean([row["separation_ratio"] for row in rows])),
        "mean_delta_alignment_cosine": float(np.mean([row["delta_alignment_cosine"] for row in rows])),
    }


@torch.inference_mode()
def collect_validation(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    args: argparse.Namespace,
    helpers: Any,
    device: torch.device,
) -> tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]:
    feature_chunks: Dict[str, list[Tensor]] = defaultdict(list)
    group_chunks: Dict[str, list[Tensor]] = defaultdict(list)
    mse_accumulator = MSEAccumulator()
    episode_final_errors: Dict[str, list[np.ndarray]] = defaultdict(list)
    batches_processed = 0
    episodes_processed = 0

    for batch_idx, cpu_batch in enumerate(val_loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        batch = move_to_device(cpu_batch, device)
        clean_batch = dict(batch)
        clean_batch["action"] = torch.nan_to_num(
            batch["action"],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        encoded = model.encode(clean_batch)
        emb = encoded["emb"]
        act_emb = encoded["act_emb"]
        context_length = int(cfg.history_size)
        num_preds = int(cfg.num_preds)

        context_emb = emb[:, :context_length]
        current_action = act_emb[:, :context_length]
        target_emb = emb[:, num_preds:num_preds + context_length]

        input_valid_mask = helpers.first_existing(
            clean_batch,
            ["valid_mask"],
        )
        if input_valid_mask is not None:
            input_valid_mask = helpers.squeeze_last_singleton(input_valid_mask)
            input_valid_mask = input_valid_mask[:, :context_length]

        input_reset_mask = helpers.first_existing(
            clean_batch,
            ["reset_mask", "episode_start"],
        )
        if input_reset_mask is not None:
            input_reset_mask = helpers.squeeze_last_singleton(input_reset_mask)
            input_reset_mask = input_reset_mask[:, :context_length]

        previous_action = helpers.build_previous_action_embedding(
            current_action,
            reset_mask=input_reset_mask,
        )

        memory_emb, memory_hidden, _ = model.remember(
            context_emb,
            previous_action,
            valid_mask=input_valid_mask,
            reset_mask=input_reset_mask,
        )
        pred_normal = model.predict(memory_emb, current_action)

        batch_size = context_emb.shape[0]
        reset_every_step = torch.ones(
            context_emb.shape[:2],
            device=device,
            dtype=context_emb.dtype,
        )
        zero_initial = torch.zeros(
            batch_size,
            model.memory.hidden_dim,
            device=device,
            dtype=context_emb.dtype,
        )
        reset_memory_emb, _, _ = model.remember(
            context_emb,
            previous_action,
            initial_state=zero_initial,
            valid_mask=input_valid_mask,
            reset_mask=reset_every_step,
        )
        pred_reset = model.predict(reset_memory_emb, current_action)

        no_memory_state = no_delta_memory_embedding(model, context_emb)
        pred_no_memory = model.predict(no_memory_state, current_action)

        shuffled_predictions: list[Tensor] = []
        permutation_seed_values = [11, 29, 47, 71, 97]
        for permutation_seed in permutation_seed_values:
            if batch_size > 1:
                generator = torch.Generator(device="cpu").manual_seed(
                    args.seed + 600_000 + batch_idx * 1_003 + permutation_seed
                )
                permutation = torch.randperm(batch_size, generator=generator).to(device)
                if torch.equal(permutation, torch.arange(batch_size, device=device)):
                    permutation = permutation.roll(1)
                shuffled_state = memory_embedding_from_hidden(
                    model,
                    context_emb,
                    memory_hidden[permutation],
                )
            else:
                shuffled_state = no_memory_state
            shuffled_predictions.append(model.predict(shuffled_state, current_action))

        # Compatibility-only roll(1) diagnostic; never used as the main score.
        roll1 = torch.arange(batch_size, device=device).roll(1)
        roll1_state = (
            memory_embedding_from_hidden(model, context_emb, memory_hidden[roll1])
            if batch_size > 1
            else no_memory_state
        )
        pred_roll1 = model.predict(roll1_state, current_action)

        step_errors = {
            "normal": (pred_normal - target_emb).square().mean(dim=-1),
            "reset_each_step": (
                pred_reset - target_emb
            ).square().mean(dim=-1),
            "no_memory": (
                pred_no_memory - target_emb
            ).square().mean(dim=-1),
            "shuffled_hidden_roll1": (
                pred_roll1 - target_emb
            ).square().mean(dim=-1),
        }
        shuffled_errors = []
        for permutation_index, prediction in enumerate(shuffled_predictions):
            error = (prediction - target_emb).square().mean(dim=-1)
            step_errors[f"shuffled_hidden_perm_{permutation_index}"] = error
            shuffled_errors.append(error)
        step_errors["shuffled_hidden"] = torch.stack(shuffled_errors).mean(dim=0)

        masks = helpers.get_sequence_masks(
            clean_batch,
            num_preds,
            context_length,
            step_errors["normal"],
            cfg,
        )
        groups = build_group_masks(masks)
        mse_accumulator.update(step_errors, groups, target_emb)

        final_mask = groups.get("final_reappearance")
        if final_mask is not None:
            denominator = final_mask.float().sum(dim=1)
            present = denominator > 0
            for variant, error in step_errors.items():
                per_episode = (
                    (error.float() * final_mask.float()).sum(dim=1)
                    / denominator.clamp_min(1.0)
                )
                episode_final_errors[variant].append(
                    per_episode[present].detach().cpu().numpy()
                )

        valid = groups["all"]
        feature_chunks["target_emb"].append(
            target_emb[valid].detach().float().cpu()
        )
        feature_chunks["pred_emb"].append(
            pred_normal[valid].detach().float().cpu()
        )
        feature_chunks["context_emb"].append(
            context_emb[valid].detach().float().cpu()
        )
        feature_chunks["memory_emb"].append(
            memory_emb[valid].detach().float().cpu()
        )
        feature_chunks["memory_hidden"].append(
            memory_hidden[valid].detach().float().cpu()
        )

        position = _target_aligned(
            clean_batch,
            "object_position",
            num_preds,
            context_length,
        )
        if position is None:
            raise KeyError("object_position is required for the x/y probe")
        feature_chunks["object_position"].append(
            position[valid].detach().float().cpu()
        )
        for state_key in ("object_velocity", "object_angle"):
            state_value = _target_aligned(
                clean_batch,
                state_key,
                num_preds,
                context_length,
            )
            if state_value is not None:
                feature_chunks[state_key].append(
                    state_value[valid].detach().float().cpu()
                )
        for metadata_key in ("episode_id", "pair_id", "pair_role"):
            metadata_value = _target_aligned(
                clean_batch,
                metadata_key,
                num_preds,
                context_length,
            )
            if metadata_value is None:
                fill = torch.full(
                    valid.shape,
                    -1,
                    dtype=torch.int64,
                    device=device,
                )
                metadata_value = fill
            metadata_value = helpers.squeeze_last_singleton(metadata_value)
            feature_chunks[metadata_key].append(
                metadata_value[valid].detach().long().cpu()
            )

        valid_f = valid.to(dtype=target_emb.dtype).unsqueeze(-1)
        episode_denominator = valid_f.sum(dim=1).clamp_min(1.0)
        feature_chunks["target_episode_mean"].append(
            (
                (target_emb * valid_f).sum(dim=1)
                / episode_denominator
            ).detach().float().cpu()
        )

        for group_name, group_mask in groups.items():
            group_chunks[group_name].append(
                group_mask[valid].detach().cpu()
            )

        batches_processed += 1
        episodes_processed += batch_size
        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            print(
                f"[audit] batch {batch_idx + 1}/{len(val_loader)}; "
                f"episodes={episodes_processed}; vectors="
                f"{sum(x.shape[0] for x in feature_chunks['target_emb'])}"
            )

    if not feature_chunks["target_emb"]:
        raise RuntimeError("No validation embeddings were collected")

    features = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in feature_chunks.items()
    }
    group_vectors = {
        name: torch.cat(chunks, dim=0).bool()
        for name, chunks in group_chunks.items()
    }
    metadata = {
        "batches_processed": batches_processed,
        "episodes_processed": episodes_processed,
        "vectors_processed": int(features["target_emb"].shape[0]),
        "ablation_mse": mse_accumulator.finalize(),
        "episode_final_errors": {
            name: np.concatenate(chunks).tolist()
            for name, chunks in episode_final_errors.items()
        },
    }
    return features, group_vectors, metadata


def write_reports(
    args: argparse.Namespace,
    cfg: DictConfig,
    features: Dict[str, Tensor],
    groups: Dict[str, Tensor],
    metadata: Dict[str, Any],
    dataset_path: Path,
    validation_episodes: int,
) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_final_errors = metadata.pop("episode_final_errors")
    report: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(dataset_path),
        "seed": args.seed,
        "validation_episodes_expected": validation_episodes,
        **metadata,
        "global_rank": {},
        "group_rank": {},
    }
    csv_rows: list[Dict[str, Any]] = []

    for name in (
        "target_emb",
        "pred_emb",
        "context_emb",
        "memory_emb",
        "memory_hidden",
        "target_episode_mean",
    ):
        print(f"[spectrum] global/{name}: {tuple(features[name].shape)}")
        stats = rank_report(features[name])
        report["global_rank"][name] = stats
        append_csv_rows(csv_rows, "global", name, stats)

    target = features["target_emb"]
    prediction = features["pred_emb"]
    cosine_similarity = torch.nn.functional.cosine_similarity(
        prediction,
        target,
        dim=1,
        eps=1.0e-8,
    )
    report["prediction_metrics"] = {
        "global": {
            "mse": float((prediction - target).square().mean().item()),
            "mean_cosine_similarity": float(cosine_similarity.mean().item()),
            "mean_cosine_error": float((1.0 - cosine_similarity).mean().item()),
        },
        "groups": {},
    }
    for group_name, selection in sorted(groups.items()):
        if group_name == "all":
            continue
        count = int(selection.sum().item())
        if count < 2:
            continue
        print(f"[spectrum] group/{group_name}: {count} vectors")
        group_report: Dict[str, Any] = {}
        for feature_name, matrix in (
            ("target_emb", target),
            ("pred_emb", prediction),
        ):
            stats = rank_report(matrix[selection])
            group_report[feature_name] = stats
            append_csv_rows(csv_rows, group_name, feature_name, stats)
        report["group_rank"][group_name] = group_report
        group_cosine = cosine_similarity[selection]
        report["prediction_metrics"]["groups"][group_name] = {
            "count": count,
            "mse": float((prediction[selection] - target[selection]).square().mean().item()),
            "mean_cosine_similarity": float(group_cosine.mean().item()),
            "mean_cosine_error": float((1.0 - group_cosine).mean().item()),
        }

    gaussian_generator = torch.Generator().manual_seed(args.seed + 900_001)
    gaussian = torch.randn(
        target.shape,
        generator=gaussian_generator,
        dtype=torch.float32,
    )
    print(f"[spectrum] baseline/gaussian: {tuple(gaussian.shape)}")
    report["gaussian_baseline"] = rank_report(gaussian)
    append_csv_rows(
        csv_rows,
        "baseline",
        "gaussian",
        report["gaussian_baseline"],
    )

    target_erank = report["global_rank"]["target_emb"]["covariance"][
        "effective_rank"
    ]
    gaussian_erank = report["gaussian_baseline"]["covariance"][
        "effective_rank"
    ]
    report["target_effective_rank_fraction_of_gaussian"] = (
        target_erank / gaussian_erank
    )

    json_path = args.output_dir / "effective_rank_report.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)

    csv_path = args.output_dir / "effective_rank_spectra.csv"
    fieldnames = sorted({key for row in csv_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    ablation_path = args.output_dir / "memory_ablation.json"
    ablation_report = dict(report["ablation_mse"])
    strict_bottleneck = str(cfg.model.get("_target_", "")).endswith(
        "StrictRecurrentOOTJEPA"
    )
    ablation_report["audit_protocol"] = {
        "strict_recurrent_bottleneck": strict_bottleneck,
        "normal": "raw recurrent hidden h_t",
        "shuffled_hidden_permutation_seeds": [11, 29, 47, 71, 97],
        "shuffled_hidden_primary": "five-permutation episode-level aggregate",
        "shuffled_hidden_roll1": "compatibility diagnostic only",
        "reset_each_step": "zero hidden before every recurrent update",
        "no_memory": "h_t=0; predictor receives only E(a_t)",
        "bootstrap_unit": "episode_and_permutation",
    }
    ablation_report["bootstrap_final_reappearance"] = _episode_bootstrap(
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in episode_final_errors.items()
        },
        samples=args.bootstrap_samples,
        seed=args.seed + 700_001,
    )
    with ablation_path.open("w", encoding="utf-8") as file:
        json.dump(
            ablation_report,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    probe_report = _position_probe_report(
        features,
        groups,
        alpha=args.probe_alpha,
        seed=args.seed + 800_001,
    )
    with (args.output_dir / "position_probe.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(probe_report, file, indent=2, ensure_ascii=False, allow_nan=False)

    pair_report = _matched_pair_report(features, groups)
    with (args.output_dir / "matched_pair_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(pair_report, file, indent=2, ensure_ascii=False, allow_nan=False)

    # Deterministic, bounded anchor set for cross-checkpoint representation
    # drift.  CKA is computed later and is diagnostic rather than a success
    # criterion by itself.
    anchor_count = min(4096, int(target.shape[0]))
    np.savez_compressed(
        args.output_dir / "anchor_embeddings.npz",
        target_emb=target[:anchor_count].numpy(),
        pred_emb=prediction[:anchor_count].numpy(),
        episode_id=features["episode_id"][:anchor_count].numpy(),
    )

    config_path = args.output_dir / "audit_resolved_config.yaml"
    OmegaConf.save(cfg, config_path)

    print("\n===== KEY RESULTS =====")
    print(f"target effective rank: {target_erank:.6f}")
    print(f"gaussian effective rank: {gaussian_erank:.6f}")
    print(
        "target/gaussian effective-rank ratio: "
        f"{report['target_effective_rank_fraction_of_gaussian']:.6f}"
    )
    target_cov = report["global_rank"]["target_emb"]["covariance"]
    print(f"target top-1 variance ratio: {target_cov['top1_variance_ratio']:.6f}")
    print(f"target top-5 variance ratio: {target_cov['top5_variance_ratio']:.6f}")
    print(f"target dimensions for 95%: {target_cov['dimensions_for_95pct']}")
    print("\nMemory ablation, reappearance:")
    print(json.dumps(
        report["ablation_mse"].get("reappearance", {}),
        indent=2,
        ensure_ascii=False,
    ))
    print(f"\nReports written to: {args.output_dir.resolve()}")
    return report


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.num_preds != 1:
        raise ValueError("This audit currently requires --num-preds=1")

    import train_oot as helpers

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    cfg = compose_config(repo_root, args)
    val_loader, validation_episodes, dataset_path = build_validation_loader(
        cfg,
        args,
        helpers,
    )
    device = torch.device(args.device)
    model = build_model(cfg, args.checkpoint, device)
    features, groups, metadata = collect_validation(
        model,
        val_loader,
        cfg,
        args,
        helpers,
        device,
    )
    write_reports(
        args,
        cfg,
        features,
        groups,
        metadata,
        dataset_path,
        validation_episodes,
    )


if __name__ == "__main__":
    main()

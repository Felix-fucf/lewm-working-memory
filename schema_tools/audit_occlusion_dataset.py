#!/usr/bin/env python3
"""Audit primitive occlusion datasets before any full training is allowed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EPISODE_LENGTH = 22
FINAL_REVEAL_INDEX = 18
OBJECT_RADIUS = 0.155
FRAME_LOSS_WEIGHT_SUM = 20.2
FRAME_LOSS_WEIGHT_POLICY = "phase_relative_sum_normalized_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_values(file: h5py.File, key: str) -> np.ndarray:
    offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
    return np.asarray(file[key][offsets])


def _reshape_frames(file: h5py.File, key: str) -> np.ndarray:
    values = np.asarray(file[key][:])
    return values.reshape((len(file["ep_offset"]), EPISODE_LENGTH, *values.shape[1:]))


def _loss_weight_report(file: h5py.File, reveal: np.ndarray) -> dict[str, Any]:
    policy = str(file.attrs.get("frame_loss_weight_policy", "legacy_or_unspecified"))
    if "frame_loss_weight" not in file:
        return {"available": False, "policy": policy}
    weights = _reshape_frames(file, "frame_loss_weight").astype(np.float64)
    weights = weights.reshape(weights.shape[:2])
    sums = weights.sum(axis=1)
    reveal_weights = weights[np.arange(len(weights)), reveal]
    mean_by_reveal = {
        str(int(index)): float(reveal_weights[reveal == index].mean())
        for index in np.unique(reveal)
    }
    means = list(mean_by_reveal.values())
    reveal_ratio = max(means) / max(min(means), 1.0e-12)
    return {
        "available": True,
        "policy": policy,
        "episode_sum_min": float(sums.min()),
        "episode_sum_max": float(sums.max()),
        "episode_sums_20_2": bool(
            np.all(np.abs(sums - FRAME_LOSS_WEIGHT_SUM) <= 1.0e-5)
        ),
        "final_reveal_weight_min": float(reveal_weights.min()),
        "final_reveal_weight_at_least_2_5": bool(np.all(reveal_weights >= 2.5)),
        "mean_final_reveal_weight_by_index": mean_by_reveal,
        "mean_reveal_weight_ratio": float(reveal_ratio),
        "mean_reveal_weight_ratio_at_most_1_25": bool(reveal_ratio <= 1.25),
        "final_reappearance_target_index_aligned": bool(
            np.all(reveal_weights == weights[np.arange(len(weights)), reveal])
        ),
    }


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = 1e-3,
) -> np.ndarray:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    train = (x_train - x_mean) / x_std
    test = (x_test - x_mean) / x_std
    train = np.concatenate([train, np.ones((len(train), 1))], axis=1)
    test = np.concatenate([test, np.ones((len(test), 1))], axis=1)
    gram = train.T @ train
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * alpha
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(gram + regularizer, train.T @ y_train)
    return test @ weights


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    numerator = np.sum((y_true - y_pred) ** 2, axis=0)
    values = 1.0 - numerator / np.maximum(denominator, 1e-12)
    return [float(value) for value in np.atleast_1d(values)]


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _split_masks(file: h5py.File, num_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    if "split_id" in file:
        split = _episode_values(file, "split_id").reshape(-1)
        train = split == 0
        test = split == 2
        if np.any(train) and np.any(test):
            return train, test
    indices = np.arange(num_episodes)
    return indices % 5 != 0, indices % 5 == 0


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or left.std() < 1.0e-12 or right.std() < 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _distribution_and_correlation_report(file: h5py.File) -> dict[str, Any]:
    num_episodes = len(file["ep_offset"])
    phase = _reshape_frames(file, "phase_id")
    lengths = _reshape_frames(file, "phase_lengths")[:, 0]
    action = _reshape_frames(file, "action").astype(np.float64)
    position = _reshape_frames(file, "object_position").astype(np.float64)
    velocity = _reshape_frames(file, "object_velocity").astype(np.float64)
    reveal = lengths[:, :4].sum(axis=1).astype(np.int64)
    onset = lengths[:, :3].sum(axis=1).astype(np.int64)
    final = position[np.arange(num_episodes), reveal]
    near_boundary = np.any(
        (position <= OBJECT_RADIUS + 0.02)
        | (position >= 1.0 - OBJECT_RADIUS - 0.02),
        axis=2,
    )
    boundary_count = near_boundary.sum(axis=1)
    action_mean = action.mean(axis=1)
    action_std = action.std(axis=1)

    correlations: dict[str, dict[str, float]] = {}
    predictors = {
        "occlusion_onset": onset,
        "occlusion_length": lengths[:, 3],
        "reveal_index": reveal,
        "boundary_frame_count": boundary_count,
    }
    for axis in range(action.shape[2]):
        predictors[f"action_mean_{axis}"] = action_mean[:, axis]
        predictors[f"action_std_{axis}"] = action_std[:, axis]
    for name, predictor in predictors.items():
        correlations[name] = {
            "future_x": _safe_corr(predictor, final[:, 0]),
            "future_y": _safe_corr(predictor, final[:, 1]),
        }

    return {
        "phase_frame_counts": {
            str(value): int(np.sum(phase == value)) for value in range(5)
        },
        "phase_frame_fractions": {
            str(value): float(np.mean(phase == value)) for value in range(5)
        },
        "position": _summary(position.reshape(-1, 2)),
        "velocity": _summary(velocity.reshape(-1, velocity.shape[-1])),
        "action": _summary(action.reshape(-1, action.shape[-1])),
        "boundary": {
            "episode_fraction_with_event": float(np.mean(boundary_count > 0)),
            "frame_fraction": float(np.mean(near_boundary)),
            "frames_per_episode": _summary(boundary_count),
        },
        "future_correlations": correlations,
    }


def _current_input_ambiguity_report(file: h5py.File) -> dict[str, Any]:
    """Quantify whether an occluded current input maps to divergent futures.

    The signature deliberately excludes visible history: it is the blank
    observation regime, rounded current action, phase, and time-to-reveal.  A
    high collision/divergence rate shows why a no-memory solution is
    underidentified at the current frame.  Matched renderer pairs provide the
    stronger exact controlled test in Dfinal.
    """
    num_episodes = len(file["ep_offset"])
    lengths = _reshape_frames(file, "phase_lengths")[:, 0]
    action = _reshape_frames(file, "action").astype(np.float64)
    position = _reshape_frames(file, "object_position").astype(np.float64)
    reveal = lengths[:, :4].sum(axis=1).astype(np.int64)
    onset = lengths[:, :3].sum(axis=1).astype(np.int64)
    probe_t = reveal - 1
    current_action = action[np.arange(num_episodes), probe_t]
    future = position[np.arange(num_episodes), reveal]

    groups: dict[tuple[Any, ...], list[int]] = {}
    for episode in range(num_episodes):
        signature = tuple(np.round(current_action[episode], 2).tolist()) + (
            int(reveal[episode] - probe_t[episode]),
            int(lengths[episode, 3]),
        )
        groups.setdefault(signature, []).append(episode)

    collisions = [indices for indices in groups.values() if len(indices) >= 2]
    divergent = [
        indices
        for indices in collisions
        if np.max(np.ptp(future[indices], axis=0)) >= 0.10
    ]
    collision_episodes = sum(len(indices) for indices in collisions)
    divergent_episodes = sum(len(indices) for indices in divergent)
    return {
        "signature": "rounded_action_at_last_hidden_frame+time_to_reveal+occlusion_length",
        "rounding_decimals": 2,
        "collision_groups": len(collisions),
        "divergent_future_groups": len(divergent),
        "episode_fraction_in_collision_groups": collision_episodes / max(num_episodes, 1),
        "episode_fraction_in_divergent_groups": divergent_episodes / max(num_episodes, 1),
    }


def _structural_invariants(file: h5py.File) -> dict[str, Any]:
    num_episodes = len(file["ep_offset"])
    lengths = _reshape_frames(file, "phase_lengths")[:, 0]
    phase = _reshape_frames(file, "phase_id")
    visible = _reshape_frames(file, "object_visible")
    reappearance = _reshape_frames(file, "is_reappearance")
    final_reappearance = _reshape_frames(file, "is_final_reappearance")
    expected_phase = np.stack([
        np.concatenate([
            np.full(int(length), phase_id, dtype=phase.dtype)
            for phase_id, length in enumerate(schedule)
        ])
        for schedule in lengths
    ])
    expected_visible = np.isin(expected_phase, (0, 2, 4)).astype(visible.dtype)
    expected_reappearance = np.zeros_like(reappearance)
    expected_reappearance[:, 1:] = (
        (expected_visible[:, 1:] == 1) & (expected_visible[:, :-1] == 0)
    )
    expected_final = np.zeros_like(final_reappearance)
    reveal = lengths[:, :4].sum(axis=1).astype(np.int64)
    expected_final[np.arange(num_episodes), reveal] = 1

    reset_ok = True
    if "episode_start" in file:
        reset = _reshape_frames(file, "episode_start")
        reset_ok = bool(np.all(reset[:, 0] == 1) and np.all(reset[:, 1:] == 0))
    valid_ok = True
    if "valid_mask" in file:
        valid_ok = bool(np.all(_reshape_frames(file, "valid_mask") == 1))

    action_alignment: bool | None = None
    if "trajectory_id" in file and "source_dataset" in file.attrs:
        source_path = Path(str(file.attrs["source_dataset"]))
        if source_path.is_file():
            trajectory = _episode_values(file, "trajectory_id").reshape(-1).astype(np.int64)
            output_action = _reshape_frames(file, "action")
            with h5py.File(source_path, "r") as source:
                source_action = _reshape_frames(source, "action")
                action_alignment = bool(np.array_equal(output_action, source_action[trajectory]))

    checks = {
        "phase_id_matches_phase_lengths": bool(np.array_equal(phase, expected_phase)),
        "visibility_matches_phase": bool(np.array_equal(visible, expected_visible)),
        "reappearance_masks_match_visibility": bool(
            np.array_equal(reappearance, expected_reappearance)
            and np.array_equal(final_reappearance, expected_final)
        ),
        "episode_reset_only_at_t0": reset_ok,
        "valid_mask_all_one": valid_ok,
        "action_alignment_to_source_trajectory": action_alignment,
    }
    required = [value for value in checks.values() if value is not None]
    return {"checks": checks, "all_pass": bool(all(required))}


def _baseline_report(file: h5py.File) -> dict[str, Any]:
    num_episodes = len(file["ep_offset"])
    phase_lengths = _reshape_frames(file, "phase_lengths")[:, 0]
    action = _reshape_frames(file, "action").astype(np.float64)
    position = _reshape_frames(file, "object_position").astype(np.float64)
    velocity = _reshape_frames(file, "object_velocity").astype(np.float64)
    reveal = phase_lengths[:, :4].sum(axis=1).astype(np.int64)
    onset = phase_lengths[:, :3].sum(axis=1).astype(np.int64)
    target = position[np.arange(num_episodes), reveal]

    time_phase = np.concatenate(
        [phase_lengths.astype(np.float64), onset[:, None], reveal[:, None]],
        axis=1,
    )
    action_only = action.reshape(num_episodes, -1)
    no_memory = np.concatenate([time_phase, action_only], axis=1)

    oracle_rows = []
    for episode in range(num_episodes):
        last_visible = int(onset[episode] - 1)
        oracle_rows.append(
            np.concatenate(
                [
                    position[episode, last_visible],
                    velocity[episode, last_visible],
                    action[episode, last_visible:reveal[episode]].reshape(-1),
                ]
            )
        )
    max_width = max(len(row) for row in oracle_rows)
    oracle = np.zeros((num_episodes, max_width), dtype=np.float64)
    for index, row in enumerate(oracle_rows):
        oracle[index, : len(row)] = row

    train, test = _split_masks(file, num_episodes)
    report: dict[str, Any] = {}
    for name, features in (
        ("time_phase", time_phase),
        ("action_only", action_only),
        ("no_memory_time_action", no_memory),
        ("kinematic_oracle", oracle),
    ):
        prediction = _ridge_predict(features[train], target[train], features[test])
        report[name] = {
            "test_episodes": int(np.sum(test)),
            "r2_x": _r2(target[test], prediction)[0],
            "r2_y": _r2(target[test], prediction)[1],
            "mse": float(np.mean((target[test] - prediction) ** 2)),
        }
    return report


def _pair_report(file: h5py.File) -> dict[str, Any]:
    if "pair_id" not in file or "pair_role" not in file:
        return {"pair_count": 0, "all_invariants_pass": True, "nuisance_auc": None}
    pair_id = _episode_values(file, "pair_id").reshape(-1)
    pair_role = _episode_values(file, "pair_role").reshape(-1)
    valid_pairs = sorted(int(value) for value in np.unique(pair_id) if value >= 0)
    if not valid_pairs:
        return {"pair_count": 0, "all_invariants_pass": True, "nuisance_auc": None}

    action = _reshape_frames(file, "action")
    position = _reshape_frames(file, "object_position")
    phase = _reshape_frames(file, "phase_id")
    pixels = file["pixels"]
    split = _episode_values(file, "split_id").reshape(-1) if "split_id" in file else None
    lengths = _reshape_frames(file, "phase_lengths")[:, 0]
    reveal = lengths[:, :4].sum(axis=1).astype(np.int64)

    failures: list[str] = []
    nuisance_features: list[np.ndarray] = []
    nuisance_labels: list[int] = []
    nuisance_groups: list[int] = []
    future_y_delta: list[float] = []

    for current_pair in valid_pairs:
        members = np.flatnonzero(pair_id == current_pair)
        if len(members) != 2 or set(pair_role[members].tolist()) != {0, 1}:
            failures.append(f"pair {current_pair}: expected roles 0/1")
            continue
        first = int(members[np.argmin(pair_role[members])])
        second = int(members[np.argmax(pair_role[members])])
        if not np.array_equal(action[first], action[second]):
            failures.append(f"pair {current_pair}: actions differ")
        if not np.allclose(position[first, :, 0], position[second, :, 0], atol=1e-7):
            failures.append(f"pair {current_pair}: x trajectory differs")
        if not np.array_equal(phase[first], phase[second]):
            failures.append(f"pair {current_pair}: phase timing differs")
        if split is not None and split[first] != split[second]:
            failures.append(f"pair {current_pair}: split leakage")

        hidden = phase[first] == 3
        first_pixels = np.asarray(
            pixels[first * EPISODE_LENGTH : (first + 1) * EPISODE_LENGTH]
        )
        second_pixels = np.asarray(
            pixels[second * EPISODE_LENGTH : (second + 1) * EPISODE_LENGTH]
        )
        if not np.array_equal(first_pixels[hidden], second_pixels[hidden]):
            failures.append(f"pair {current_pair}: hidden pixels differ")
        visible_history = phase[first] == 2
        if np.allclose(
            position[first, visible_history, 1], position[second, visible_history, 1], atol=1e-6
        ):
            failures.append(f"pair {current_pair}: visible y history is identical")
        delta = float(abs(
            position[first, reveal[first], 1] - position[second, reveal[second], 1]
        ))
        future_y_delta.append(delta)
        if delta < 0.10:
            failures.append(f"pair {current_pair}: future y separation {delta:.4f} < 0.10")

        for member in (first, second):
            feature = np.concatenate(
                [lengths[member].astype(np.float64), action[member].mean(axis=0), action[member].std(axis=0)]
            )
            nuisance_features.append(feature)
            nuisance_labels.append(int(pair_role[member]))
            nuisance_groups.append(current_pair)

    x = np.asarray(nuisance_features, dtype=np.float64)
    y = np.asarray(nuisance_labels, dtype=np.int8)
    groups = np.asarray(nuisance_groups, dtype=np.int64)
    train = np.asarray([int(group) % 5 != 0 for group in groups])
    test = ~train
    nuisance_auc = None
    if np.any(train) and np.any(test):
        score = _ridge_predict(x[train], y[train, None], x[test]).reshape(-1)
        nuisance_auc = float(_roc_auc(y[test], score))

    return {
        "pair_count": len(valid_pairs),
        "all_invariants_pass": not failures,
        "failures": failures[:100],
        "future_y_delta_median": float(np.median(future_y_delta)) if future_y_delta else None,
        "nuisance_auc": nuisance_auc,
    }


def audit_dataset(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    with h5py.File(path, "r") as file:
        episode_length = int(file.attrs["episode_length"])
        if episode_length != EPISODE_LENGTH:
            raise ValueError(f"Expected 22 frames, found {episode_length}")
        num_episodes = len(file["ep_offset"])
        lengths = _reshape_frames(file, "phase_lengths")[:, 0]
        position = _reshape_frames(file, "object_position").astype(np.float64)
        reveal = lengths[:, :4].sum(axis=1).astype(np.int64)
        onset = lengths[:, :3].sum(axis=1).astype(np.int64)
        final_position = position[np.arange(num_episodes), reveal]
        sigma_all = position.reshape(-1, 2).std(axis=0)
        sigma_final = final_position.std(axis=0)
        ratio_all = float(sigma_all[0] / max(sigma_all[1], 1e-12))
        ratio = float(sigma_final[0] / max(sigma_final[1], 1e-12))
        unique_schedules, counts = np.unique(lengths, axis=0, return_counts=True)
        schedule_counts = {
            str(tuple(int(value) for value in schedule)): int(count)
            for schedule, count in zip(unique_schedules, counts)
        }

        split_report: dict[str, Any] = {"available": "split_id" in file}
        split_disjoint = True
        if "split_id" in file and "trajectory_id" in file:
            split_id = _episode_values(file, "split_id").reshape(-1)
            trajectory = _episode_values(file, "trajectory_id").reshape(-1)
            conflicts = 0
            for value in np.unique(trajectory):
                if len(np.unique(split_id[trajectory == value])) != 1:
                    conflicts += 1
            split_disjoint = conflicts == 0
            split_report.update(
                counts={str(value): int(np.sum(split_id == value)) for value in (0, 1, 2)},
                trajectory_conflicts=conflicts,
            )

        pair = _pair_report(file)
        baseline = _baseline_report(file)
        distributions = _distribution_and_correlation_report(file)
        ambiguity = _current_input_ambiguity_report(file)
        structural = _structural_invariants(file)
        loss_weights = _loss_weight_report(file, reveal)
        is_dfinal = str(file.attrs.get("dataset_variant", "")).endswith("dfinal")
        reveal_from_mask = np.argmax(
            _reshape_frames(file, "is_final_reappearance"), axis=1
        )
        length_proportions = {
            int(value): float(np.mean(lengths[:, 3] == value))
            for value in np.unique(lengths[:, 3])
        }
        acceptance = {
            "reveal_metadata_matches_mask": bool(np.array_equal(reveal, reveal_from_mask)),
            "reveal_policy_valid": bool(
                set(reveal.tolist()).issubset({17, 18, 19})
                and ((len(np.unique(onset)) >= 2) if is_dfinal else np.all(reveal == 18))
            ),
            "xy_std_ratio_all_in_range": 0.8 <= ratio_all <= 1.25 if is_dfinal else None,
            "xy_std_ratio_final_in_range": 0.8 <= ratio <= 1.25 if is_dfinal else None,
            "curriculum_mix_50_30_20": bool(
                abs(length_proportions.get(2, 0.0) - 0.50) <= 1.0 / max(num_episodes, 1)
                and abs(length_proportions.get(4, 0.0) - 0.30) <= 1.0 / max(num_episodes, 1)
                and abs(length_proportions.get(8, 0.0) - 0.20) <= 1.0 / max(num_episodes, 1)
            ) if is_dfinal else None,
            "pair_invariants": pair["all_invariants_pass"] if is_dfinal else None,
            "nuisance_auc_at_most_055": (
                pair["nuisance_auc"] is not None and pair["nuisance_auc"] <= 0.55
            ) if is_dfinal else None,
            "trajectory_splits_disjoint": split_disjoint,
            "structural_invariants": structural["all_pass"],
            "phase_relative_weight_policy": (
                loss_weights["policy"] == FRAME_LOSS_WEIGHT_POLICY
            ) if str(file.attrs.get("dataset_version", "")) == "primitive_v2" else None,
            "frame_weight_sum_20_2": (
                loss_weights.get("episode_sums_20_2", False)
            ) if str(file.attrs.get("dataset_version", "")) == "primitive_v2" else None,
            "final_reveal_weight_at_least_2_5": (
                loss_weights.get("final_reveal_weight_at_least_2_5", False)
            ) if str(file.attrs.get("dataset_version", "")) == "primitive_v2" else None,
            "reveal_weight_ratio_at_most_1_25": (
                loss_weights.get("mean_reveal_weight_ratio_at_most_1_25", False)
            ) if str(file.attrs.get("dataset_version", "")) == "primitive_v2" else None,
            "frame_weight_target_alignment": (
                loss_weights.get("final_reappearance_target_index_aligned", False)
            ) if str(file.attrs.get("dataset_version", "")) == "primitive_v2" else None,
        }
        required = [value for value in acceptance.values() if value is not None]

        report = {
            "path": str(path.resolve()),
            "sha256": _sha256(path) if include_sha256 else None,
            "dataset_variant": str(file.attrs.get("dataset_variant", "unknown")),
            "num_episodes": num_episodes,
            "episode_length": episode_length,
            "phase_schedule_counts": schedule_counts,
            "second_occlusion_lengths": sorted(set(int(value) for value in lengths[:, 3])),
            "reveal_index_counts": {
                str(int(value)): int(np.sum(reveal == value))
                for value in np.unique(reveal)
            },
            "occlusion_onset_counts": {
                str(int(value)): int(np.sum(onset == value))
                for value in np.unique(onset)
            },
            "position_distribution": {
                "sigma_x_all": float(sigma_all[0]),
                "sigma_y_all": float(sigma_all[1]),
                "sigma_x_final": float(sigma_final[0]),
                "sigma_y_final": float(sigma_final[1]),
                "sigma_x_over_sigma_y_all": ratio_all,
                "sigma_x_over_sigma_y_final": ratio,
            },
            "split": split_report,
            "matched_pairs": pair,
            "shortcut_and_oracle_baselines": baseline,
            "distribution_and_future_correlation": distributions,
            "current_input_ambiguity": ambiguity,
            "structural_invariants": structural,
            "frame_loss_weights": loss_weights,
            "acceptance": acceptance,
            "all_required_acceptance_pass": bool(all(required)),
        }
    return report


def write_summary_csv(reports: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset_variant",
        "num_episodes",
        "occlusion_lengths",
        "sigma_x_final",
        "sigma_y_final",
        "sigma_x_over_sigma_y_all",
        "sigma_x_over_sigma_y_final",
        "pair_count",
        "nuisance_auc",
        "time_phase_r2_x",
        "time_phase_r2_y",
        "action_r2_x",
        "action_r2_y",
        "oracle_r2_x",
        "oracle_r2_y",
        "acceptance_pass",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            distribution = report["position_distribution"]
            pairs = report["matched_pairs"]
            baselines = report["shortcut_and_oracle_baselines"]
            writer.writerow(
                {
                    "dataset_variant": report["dataset_variant"],
                    "num_episodes": report["num_episodes"],
                    "occlusion_lengths": json.dumps(report["second_occlusion_lengths"]),
                    "sigma_x_final": distribution["sigma_x_final"],
                    "sigma_y_final": distribution["sigma_y_final"],
                    "sigma_x_over_sigma_y_all": distribution["sigma_x_over_sigma_y_all"],
                    "sigma_x_over_sigma_y_final": distribution["sigma_x_over_sigma_y_final"],
                    "pair_count": pairs["pair_count"],
                    "nuisance_auc": pairs["nuisance_auc"],
                    "time_phase_r2_x": baselines["time_phase"]["r2_x"],
                    "time_phase_r2_y": baselines["time_phase"]["r2_y"],
                    "action_r2_x": baselines["action_only"]["r2_x"],
                    "action_r2_y": baselines["action_only"]["r2_y"],
                    "oracle_r2_x": baselines["kinematic_oracle"]["r2_x"],
                    "oracle_r2_y": baselines["kinematic_oracle"]["r2_y"],
                    "acceptance_pass": report["all_required_acceptance_pass"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for dataset in args.datasets:
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        report = audit_dataset(dataset, include_sha256=not args.skip_sha256)
        reports.append(report)
        output = args.output_dir / f"{dataset.stem}.audit.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, indent=2, allow_nan=False))
        if args.enforce and not report["all_required_acceptance_pass"]:
            raise RuntimeError(f"Dataset acceptance failed: {dataset}")
    write_summary_csv(reports, args.output_dir / "dataset_audit_summary.csv")


if __name__ == "__main__":
    main()

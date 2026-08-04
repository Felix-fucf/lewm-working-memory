#!/usr/bin/env python3
"""Enforce the paired primitive_v3 dataset contract before training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

try:
    from .audit_occlusion_dataset import (
        _baseline_report,
        _distribution_and_correlation_report,
        _reshape_frames,
        _sha256,
    )
    from .build_occlusion_v3 import (
        BOUNDARY_POLICY,
        FIRST_OCCLUSION_POLICY,
        FRAME_LOSS_WEIGHT_POLICY,
        FRAME_LOSS_WEIGHT_SUM,
        MOTION_SCALE,
        ORIENTATION_POLICY,
        V3_SCHEDULES,
        _orientation_vectors,
        _scale_angles,
        _scale_positions,
        phase_vectors_v3,
    )
    from .primitive_common import EPISODE_LENGTH, OBJECT_RADIUS
except ImportError:  # Direct execution on AutoDL.
    from audit_occlusion_dataset import (
        _baseline_report,
        _distribution_and_correlation_report,
        _reshape_frames,
        _sha256,
    )
    from build_occlusion_v3 import (
        BOUNDARY_POLICY,
        FIRST_OCCLUSION_POLICY,
        FRAME_LOSS_WEIGHT_POLICY,
        FRAME_LOSS_WEIGHT_SUM,
        MOTION_SCALE,
        ORIENTATION_POLICY,
        V3_SCHEDULES,
        _orientation_vectors,
        _scale_angles,
        _scale_positions,
        phase_vectors_v3,
    )
    from primitive_common import EPISODE_LENGTH, OBJECT_RADIUS


def _episode_values(file: h5py.File, key: str) -> np.ndarray:
    offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
    return np.asarray(file[key][offsets])


def _schedule_report(lengths: np.ndarray, split: np.ndarray) -> dict[str, Any]:
    expected = {tuple(schedule) for schedule in V3_SCHEDULES}
    observed = {tuple(int(value) for value in row) for row in lengths}
    per_split: dict[str, Any] = {}
    balanced = True
    for split_value in (0, 1, 2):
        selection = split == split_value
        counts = [int(np.sum(np.all(lengths[selection] == schedule, axis=1))) for schedule in V3_SCHEDULES]
        difference = max(counts) - min(counts) if counts else 0
        balanced = balanced and difference <= 1 and all(value > 0 for value in counts)
        per_split[str(split_value)] = {
            "counts": {str(schedule): count for schedule, count in zip(V3_SCHEDULES, counts)},
            "max_minus_min": difference,
        }
    reveal = lengths[:, :4].sum(axis=1)
    return {
        "observed_schedules": sorted(str(value) for value in observed),
        "exact_schedule_set": observed == expected,
        "first_occlusion_fixed_onset_3_length_4": bool(
            np.all(lengths[:, 0] == 3) and np.all(lengths[:, 1] == 4)
        ),
        "second_occlusion_fixed_length_4": bool(np.all(lengths[:, 3] == 4)),
        "reveal_counts": {str(value): int(np.sum(reveal == value)) for value in range(15, 20)},
        "per_split": per_split,
        "per_split_balanced_within_one": bool(balanced),
    }


def _structure_report(file: h5py.File, lengths: np.ndarray) -> dict[str, Any]:
    phase = _reshape_frames(file, "phase_id")
    visible = _reshape_frames(file, "object_visible")
    reappearance = _reshape_frames(file, "is_reappearance")
    final = _reshape_frames(file, "is_final_reappearance")
    vectors = [phase_vectors_v3(row) for row in lengths]
    checks = {
        "phase_id_matches_schedule": bool(
            np.array_equal(phase, np.stack([item["phase_id"] for item in vectors]))
        ),
        "visibility_matches_schedule": bool(
            np.array_equal(visible, np.stack([item["object_visible"] for item in vectors]))
        ),
        "reappearance_masks_match_schedule": bool(
            np.array_equal(reappearance, np.stack([item["is_reappearance"] for item in vectors]))
            and np.array_equal(final, np.stack([item["is_final_reappearance"] for item in vectors]))
        ),
    }
    if "episode_start" in file:
        reset = _reshape_frames(file, "episode_start")
        checks["episode_reset_only_at_t0"] = bool(
            np.all(reset[:, 0] == 1) and np.all(reset[:, 1:] == 0)
        )
    if "valid_mask" in file:
        checks["valid_mask_all_one"] = bool(np.all(_reshape_frames(file, "valid_mask") == 1))
    return {"checks": checks, "all_pass": bool(all(checks.values()))}


def _weight_report(file: h5py.File, lengths: np.ndarray) -> dict[str, Any]:
    weights = _reshape_frames(file, "frame_loss_weight").reshape(len(lengths), EPISODE_LENGTH)
    onset = lengths[:, :3].sum(axis=1).astype(np.int64)
    reveal = lengths[:, :4].sum(axis=1).astype(np.int64)
    sums = weights.sum(axis=1)
    onset_weights = weights[np.arange(len(weights)), onset]
    reveal_weights = weights[np.arange(len(weights)), reveal]
    reveal_means = [float(reveal_weights[reveal == value].mean()) for value in range(15, 20)]
    return {
        "policy": str(file.attrs.get("frame_loss_weight_policy", "")),
        "episode_sum_min": float(sums.min()),
        "episode_sum_max": float(sums.max()),
        "episode_sums_20_2": bool(np.all(np.abs(sums - FRAME_LOSS_WEIGHT_SUM) <= 1.0e-5)),
        "random_second_onset_weights_zero": bool(np.all(onset_weights == 0.0)),
        "final_reveal_weight_min": float(reveal_weights.min()),
        "final_reveal_weight_at_least_2_5": bool(np.all(reveal_weights >= 2.5)),
        "reveal_weight_mean_ratio": max(reveal_means) / max(min(reveal_means), 1.0e-12),
        "prediction_target_alignment": (
            "train_oot.align_target_time(frame_loss_weight) uses frame t+1 weight for prediction t"
        ),
    }


def _boundary_report(file: h5py.File) -> dict[str, Any]:
    position = _reshape_frames(file, "object_position").astype(np.float64)
    radius = float(file.attrs.get("object_radius", OBJECT_RADIUS))
    contact = np.any(
        (position <= radius + 1.0e-7) | (position >= 1.0 - radius - 1.0e-7),
        axis=2,
    )
    near = np.any(
        (position <= radius + 0.02) | (position >= 1.0 - radius - 0.02),
        axis=2,
    )
    return {
        "policy": str(file.attrs.get("boundary_policy", "")),
        "contact_episode_count": int(np.sum(np.any(contact, axis=1))),
        "contact_frame_count": int(np.sum(contact)),
        "zero_contact": bool(not np.any(contact)),
        "near_boundary_episode_fraction_margin_0_02": float(np.mean(np.any(near, axis=1))),
        "note": "Near-boundary episodes are reported, not filtered; no trajectory is centered or clipped.",
    }


def _orientation_report(file: h5py.File) -> dict[str, Any]:
    angle = _reshape_frames(file, "object_angle").astype(np.float64)
    orientation = _reshape_frames(file, "object_orientation").astype(np.float64)
    angle_unit = str(file.attrs.get("object_angle_unit", ""))
    expected = _orientation_vectors(
        angle,
        angles_are_radians=(angle_unit == "radian"),
    )
    shape_ok = orientation.shape == expected.shape
    max_error = _max_abs(orientation, expected) if shape_ok else float("inf")
    norm_error = (
        float(np.max(np.abs(np.linalg.norm(orientation, axis=-1) - 1.0)))
        if orientation.ndim == 3 and orientation.shape[-1] == 2
        else float("inf")
    )
    return {
        "angle_unit": angle_unit,
        "shape": list(orientation.shape),
        "expected_shape": list(expected.shape),
        "shape_is_episode_time_xy": shape_ok,
        "max_abs_angle_vector_error": max_error,
        "max_abs_unit_norm_error": norm_error,
        "matches_object_angle": bool(
            angle_unit in {"degree", "radian"}
            and shape_ok
            and max_error <= 2.0e-6
            and norm_error <= 2.0e-6
        ),
    }


def _split_report(file: h5py.File) -> dict[str, Any]:
    split = _episode_values(file, "split_id").reshape(-1)
    trajectory = _episode_values(file, "trajectory_id").reshape(-1)
    conflicts = 0
    for value in np.unique(trajectory):
        if len(np.unique(split[trajectory == value])) != 1:
            conflicts += 1
    return {
        "counts": {str(value): int(np.sum(split == value)) for value in (0, 1, 2)},
        "trajectory_conflicts": conflicts,
        "disjoint": conflicts == 0,
    }


def _pair_report(file: h5py.File) -> dict[str, Any]:
    pair_id = _episode_values(file, "pair_id").reshape(-1)
    pair_role = _episode_values(file, "pair_role").reshape(-1)
    phase = _reshape_frames(file, "phase_id")
    action = _reshape_frames(file, "action")
    position = _reshape_frames(file, "object_position")
    split = _episode_values(file, "split_id").reshape(-1)
    lengths = _reshape_frames(file, "phase_lengths")[:, 0]
    failures: list[str] = []
    visible_history_identical = 0
    future_delta: list[float] = []
    valid_pairs = [int(value) for value in np.unique(pair_id) if value >= 0]
    for value in valid_pairs:
        members = np.flatnonzero(pair_id == value)
        if len(members) != 2 or set(pair_role[members].tolist()) != {0, 1}:
            failures.append(f"pair {value}: roles are not exactly 0/1")
            continue
        first, second = sorted(members.tolist(), key=lambda item: int(pair_role[item]))
        if not np.array_equal(action[first], action[second]):
            failures.append(f"pair {value}: action differs")
        if not np.array_equal(phase[first], phase[second]):
            failures.append(f"pair {value}: schedule differs")
        if split[first] != split[second]:
            failures.append(f"pair {value}: split differs")
        visible_history = phase[first] == 2
        if np.allclose(position[first, visible_history], position[second, visible_history]):
            visible_history_identical += 1
        reveal = int(sum(lengths[first, :4]))
        future_delta.append(float(abs(position[first, reveal, 1] - position[second, reveal, 1])))
    return {
        "pair_count": len(valid_pairs),
        "all_motion_action_schedule_invariants_pass": not failures,
        "failures": failures[:100],
        "future_y_delta_median": float(np.median(future_delta)) if future_delta else None,
        "visible_history_identical_pair_count": visible_history_identical,
        "semantics": (
            "matched source motion/action/schedule with renderer-based y intervention; "
            "not an exact matched-visible-history causal pair"
        ),
    }


def audit_dataset(path: Path, include_sha256: bool = True) -> dict[str, Any]:
    with h5py.File(path, "r") as file:
        lengths = _reshape_frames(file, "phase_lengths")[:, 0].astype(np.int16)
        split = _episode_values(file, "split_id").reshape(-1)
        motion_type = _episode_values(file, "motion_type").reshape(-1)
        motion_counts = {
            str(int(value)): int(np.sum(motion_type == value))
            for value in np.unique(motion_type)
        }
        motion_fractions = {
            key: count / max(len(motion_type), 1) for key, count in motion_counts.items()
        }
        schedule = _schedule_report(lengths, split)
        structure = _structure_report(file, lengths)
        weights = _weight_report(file, lengths)
        boundary = _boundary_report(file)
        orientation = _orientation_report(file)
        split_report = _split_report(file)
        pairs = _pair_report(file)
        attrs = {
            "dataset_version_v3": str(file.attrs.get("dataset_version", "")) == "primitive_v3",
            "completed": int(file.attrs.get("completed", 0)) == 1,
            "frame_loss_weight_policy": (
                str(file.attrs.get("frame_loss_weight_policy", "")) == FRAME_LOSS_WEIGHT_POLICY
            ),
            "boundary_policy": str(file.attrs.get("boundary_policy", "")) == BOUNDARY_POLICY,
            "first_occlusion_policy": (
                str(file.attrs.get("first_occlusion_policy", "")) == FIRST_OCCLUSION_POLICY
            ),
            "first_occlusion_not_randomized": int(file.attrs.get("first_occlusion_randomized", 1)) == 0,
            "object_orientation_policy": (
                str(file.attrs.get("object_orientation_policy", "")) == ORIENTATION_POLICY
            ),
        }
        acceptance = {
            **attrs,
            "exact_five_schedule_set": schedule["exact_schedule_set"],
            "schedule_balanced_in_each_split": schedule["per_split_balanced_within_one"],
            "first_occlusion_fixed": schedule["first_occlusion_fixed_onset_3_length_4"],
            "second_occlusion_fixed_h4": schedule["second_occlusion_fixed_length_4"],
            "structural_invariants": structure["all_pass"],
            "episode_weight_sum_20_2": weights["episode_sums_20_2"],
            "random_onset_weight_zero": weights["random_second_onset_weights_zero"],
            "reveal_weight_at_least_2_5": weights["final_reveal_weight_at_least_2_5"],
            "reveal_weight_ratio_at_most_1_25": weights["reveal_weight_mean_ratio"] <= 1.25,
            "no_boundary_contact": boundary["zero_contact"],
            "orientation_matches_angle": orientation["matches_object_angle"],
            "trajectory_splits_disjoint": split_report["disjoint"],
            "pair_invariants": pairs["all_motion_action_schedule_invariants_pass"],
            "motion_types_remain_diverse": (
                len(motion_counts) >= 2 and min(motion_fractions.values()) >= 0.25
            ),
        }
        report = {
            "path": str(path.resolve()),
            "sha256": _sha256(path) if include_sha256 else None,
            "dataset_variant": str(file.attrs.get("dataset_variant", "unknown")),
            "num_episodes": len(lengths),
            "motion_scale": float(file.attrs.get("motion_scale", float("nan"))),
            "motion_type_counts": motion_counts,
            "motion_type_fractions": motion_fractions,
            "schedule": schedule,
            "structure": structure,
            "frame_loss_weights": weights,
            "boundary": boundary,
            "orientation": orientation,
            "split": split_report,
            "matched_pairs": pairs,
            "distribution_and_future_correlation": _distribution_and_correlation_report(file),
            "shortcut_and_oracle_baselines": _baseline_report(file),
            "acceptance": acceptance,
            "all_required_acceptance_pass": bool(all(acceptance.values())),
        }
    return report


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    both_nan = np.isnan(left) & np.isnan(right)
    mismatch_nan = np.isnan(left) ^ np.isnan(right)
    if np.any(mismatch_nan):
        return float("inf")
    difference = np.abs(left - right)
    difference[both_nan] = 0.0
    return float(np.max(difference))


def audit_pair(original_path: Path, motion_path: Path) -> dict[str, Any]:
    with h5py.File(original_path, "r") as original, h5py.File(motion_path, "r") as motion:
        shared_keys = ("trajectory_id", "split_id", "pair_id", "pair_role", "phase_lengths")
        shared = {
            key: bool(np.array_equal(np.asarray(original[key][:]), np.asarray(motion[key][:])))
            for key in shared_keys
        }
        source_path = Path(str(original.attrs["source_dataset"]))
        if source_path.resolve() != Path(str(motion.attrs["source_dataset"])).resolve():
            raise AssertionError("V3 pair points to different source datasets")
        with h5py.File(source_path, "r") as source:
            trajectory = _episode_values(original, "trajectory_id").reshape(-1).astype(np.int64)
            source_position = _reshape_frames(source, "object_position")[trajectory]
            source_action = _reshape_frames(source, "action")[trajectory]
            source_velocity = _reshape_frames(source, "object_velocity")[trajectory]
            source_angle = _reshape_frames(source, "object_angle")[trajectory]
            source_angular_velocity = _reshape_frames(source, "object_angular_velocity")[trajectory]

        original_position = _reshape_frames(original, "object_position").astype(np.float64)
        motion_position = _reshape_frames(motion, "object_position").astype(np.float64)
        original_shift = _episode_values(original, "y_shift").reshape(-1).astype(np.float64)
        motion_shift = _episode_values(motion, "y_shift").reshape(-1).astype(np.float64)
        original_unshifted = original_position.copy()
        motion_unshifted = motion_position.copy()
        original_unshifted[:, :, 1] -= original_shift[:, None]
        motion_unshifted[:, :, 1] -= motion_shift[:, None]
        expected_motion_position = _scale_positions(source_position.astype(np.float64), MOTION_SCALE)

        expected_motion_angle = _scale_angles(source_angle, MOTION_SCALE)
        actual_motion_angle = _reshape_frames(motion, "object_angle")
        if np.nanmax(np.abs(source_angle)) <= 2.0 * math.pi + 0.1:
            angle_period = 2.0 * math.pi
        else:
            angle_period = 360.0
        angle_difference = (
            np.asarray(actual_motion_angle, dtype=np.float64)
            - np.asarray(expected_motion_angle, dtype=np.float64)
            + angle_period / 2.0
        ) % angle_period - angle_period / 2.0

        checks = {
            "shared_episode_schedule_split_pair_mapping": bool(all(shared.values())),
            "original_position_matches_source_after_pair_shift_removed": (
                _max_abs(original_unshifted, source_position) <= 1.0e-6
            ),
            "motion_position_is_anchor_preserving_1_5x_after_pair_shift_removed": (
                _max_abs(motion_unshifted, expected_motion_position) <= 2.0e-6
            ),
            "original_action_matches_source": (
                _max_abs(_reshape_frames(original, "action"), source_action) <= 1.0e-7
            ),
            "motion_action_is_1_5x": (
                _max_abs(_reshape_frames(motion, "action"), source_action * MOTION_SCALE) <= 2.0e-6
            ),
            "original_velocity_matches_source": (
                _max_abs(_reshape_frames(original, "object_velocity"), source_velocity) <= 1.0e-7
            ),
            "motion_velocity_is_1_5x": (
                _max_abs(
                    _reshape_frames(motion, "object_velocity"),
                    source_velocity * MOTION_SCALE,
                ) <= 2.0e-6
            ),
            "motion_angle_change_is_1_5x": float(np.max(np.abs(angle_difference))) <= 2.0e-5,
            "motion_angular_velocity_is_1_5x": (
                _max_abs(
                    _reshape_frames(motion, "object_angular_velocity"),
                    source_angular_velocity * MOTION_SCALE,
                ) <= 2.0e-6
            ),
        }
        original_step = np.linalg.norm(np.diff(original_unshifted, axis=1), axis=2)
        motion_step = np.linalg.norm(np.diff(motion_unshifted, axis=1), axis=2)
        positive = original_step > 1.0e-10
        displacement_ratio = float(np.median(motion_step[positive] / original_step[positive]))
        checks["median_step_displacement_increase_at_least_45_percent"] = displacement_ratio >= 1.45
        report = {
            "original": str(original_path.resolve()),
            "motion15x": str(motion_path.resolve()),
            "shared_fields": shared,
            "median_nonzero_step_displacement_ratio": displacement_ratio,
            "checks": checks,
            "all_required_acceptance_pass": bool(all(checks.values())),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--motion15x", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "original_motion": audit_dataset(args.original, include_sha256=not args.skip_sha256),
        "motion15x": audit_dataset(args.motion15x, include_sha256=not args.skip_sha256),
        "paired_comparison": audit_pair(args.original, args.motion15x),
    }
    for name, report in reports.items():
        path = args.output_dir / f"{name}.audit.json"
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps({"audit": name, "pass": report["all_required_acceptance_pass"]}))
        if args.enforce and not report["all_required_acceptance_pass"]:
            raise RuntimeError(f"primitive_v3 acceptance failed: {name}; inspect {path}")


if __name__ == "__main__":
    main()

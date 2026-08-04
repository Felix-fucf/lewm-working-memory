#!/usr/bin/env python3
"""Build the paired primitive_v3 five-reveal datasets.

This is deliberately a trajectory-derived dataset builder, not a physical
simulator.  Both outputs use the same primitive_v1 source episodes, split,
five-way schedule assignment, and matched-pair assignment.  The only motion
difference is a coherent 1.5x transform of displacement, velocity, action,
angle change, and angular velocity.  Every output frame is rendered from
state; source pixels are never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

try:
    from .build_occlusion_curriculum import (
        _angle_degrees,
        _copy_attrs,
        _create_dataset_like,
        _hash_uniform,
        _infer_position_columns,
        _selected_frame_values,
        _sha256,
        _split_assignments,
    )
    from .primitive_common import EPISODE_LENGTH, OBJECT_RADIUS
    from .primitive_render import TRAIN_CONTENTS, render_object
except ImportError:  # Direct execution on AutoDL.
    from build_occlusion_curriculum import (
        _angle_degrees,
        _copy_attrs,
        _create_dataset_like,
        _hash_uniform,
        _infer_position_columns,
        _selected_frame_values,
        _sha256,
        _split_assignments,
    )
    from primitive_common import EPISODE_LENGTH, OBJECT_RADIUS
    from primitive_render import TRAIN_CONTENTS, render_object


GENERATOR_SEED = 20260731
MOTION_SCALE = 1.5
PAIR_FRACTION = 0.20
FRAME_LOSS_WEIGHT_SUM = 20.2
FINAL_REVEAL_WEIGHT = 3.0
FRAME_LOSS_WEIGHT_POLICY = (
    "phase_relative_random_onset_zero_fixed_reveal_sum_normalized_v3_1"
)
BOUNDARY_POLICY = "preserve_anchor_reject_whole_episode_on_contact_v1"
FIRST_OCCLUSION_POLICY = "fixed_onset_3_length_4_v1"
ORIENTATION_POLICY = "cos_sin_from_object_angle_v1"
V3_SCHEDULES: tuple[tuple[int, int, int, int, int], ...] = tuple(
    (3, 4, middle_visible, 4, 11 - middle_visible)
    for middle_visible in range(4, 9)
)
VARIANTS = {
    "original_motion": 1.0,
    "motion15x": MOTION_SCALE,
}


@dataclass(frozen=True)
class SharedPlan:
    source_episode: int
    phase_lengths: tuple[int, int, int, int, int]
    split_id: int
    pair_id: int = -1
    pair_role: int = -1


def phase_vectors_v3(lengths: Sequence[int]) -> dict[str, np.ndarray]:
    schedule = tuple(int(value) for value in lengths)
    if schedule not in V3_SCHEDULES:
        raise ValueError(f"Not a frozen primitive_v3 schedule: {schedule}")
    phase_id = np.concatenate(
        [np.full(length, phase, dtype=np.int8) for phase, length in enumerate(schedule)]
    )
    visible = np.isin(phase_id, (0, 2, 4)).astype(np.uint8)
    reappearance = np.zeros(EPISODE_LENGTH, dtype=np.uint8)
    reappearance[np.flatnonzero((visible[1:] == 1) & (visible[:-1] == 0)) + 1] = 1
    final_reappearance = np.zeros(EPISODE_LENGTH, dtype=np.uint8)
    final_reappearance[sum(schedule[:4])] = 1
    return {
        "phase_id": phase_id,
        "object_visible": visible,
        "is_reappearance": reappearance,
        "is_final_reappearance": final_reappearance,
    }


def phase_relative_loss_weights_v3(lengths: Sequence[int]) -> np.ndarray:
    """Mask the random onset and keep reveal supervision schedule-invariant."""
    schedule = tuple(int(value) for value in lengths)
    phase_vectors_v3(schedule)
    parts: list[np.ndarray] = []
    for phase, length in enumerate(schedule):
        if phase == 0:
            values = np.ones(length, dtype=np.float64)
        elif phase in (1, 3):
            values = np.full(length, 0.1, dtype=np.float64)
        elif phase == 2:
            values = np.full(length, 0.1, dtype=np.float64)
            values[: min(length, 3)] = np.asarray([3.0, 2.0, 2.0])[:length]
        else:
            values = np.full(length, 2.0, dtype=np.float64)
            values[0] = 3.0
        parts.append(values)
    weights = np.concatenate(parts)
    second_onset = int(sum(schedule[:3]))
    final_reveal = int(sum(schedule[:4]))
    weights[second_onset] = 0.0
    # A single global rescale would dilute the reveal target whenever the final
    # visible phase is long (for reveal=15 it fell to 2.3488).  Reserve an
    # identical reveal weight for every schedule and normalize only the other
    # frames, preserving their phase-relative ratios.
    weights[final_reveal] = 0.0
    weights *= (FRAME_LOSS_WEIGHT_SUM - FINAL_REVEAL_WEIGHT) / float(weights.sum())
    weights[final_reveal] = FINAL_REVEAL_WEIGHT
    return weights.astype(np.float32)


def _episode_frames(file: h5py.File, key: str) -> np.ndarray:
    values = np.asarray(file[key][:])
    return values.reshape((len(file["ep_offset"]), EPISODE_LENGTH, *values.shape[1:]))


def _motion_types(file: h5py.File) -> np.ndarray:
    offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
    return np.asarray(file["motion_type"][offsets], dtype=np.int8).reshape(-1)


def _scale_positions(position: np.ndarray, scale: float) -> np.ndarray:
    anchor = position[:, :1]
    return anchor + float(scale) * (position - anchor)


def _angles_are_radians(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size and float(np.max(np.abs(finite))) <= 2.0 * math.pi + 0.1)


def _scale_angles(values: np.ndarray, scale: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    radians = source if _angles_are_radians(source) else np.radians(source)
    unwrapped = np.unwrap(radians, axis=1)
    transformed = unwrapped[:, :1] + float(scale) * (unwrapped - unwrapped[:, :1])
    if _angles_are_radians(source):
        if float(np.nanmin(source)) >= 0.0:
            transformed = np.mod(transformed, 2.0 * math.pi)
        else:
            transformed = (transformed + math.pi) % (2.0 * math.pi) - math.pi
        return transformed.astype(values.dtype, copy=False)
    degrees = np.degrees(transformed)
    if float(np.nanmin(source)) >= 0.0:
        degrees = np.mod(degrees, 360.0)
    else:
        degrees = (degrees + 180.0) % 360.0 - 180.0
    return degrees.astype(values.dtype, copy=False)


def _orientation_vectors(
    values: np.ndarray,
    *,
    angles_are_radians: bool | None = None,
) -> np.ndarray:
    """Convert scalar angles to the dataset's [cos(theta), sin(theta)] schema."""
    angles = np.asarray(values, dtype=np.float64)
    in_radians = _angles_are_radians(angles) if angles_are_radians is None else angles_are_radians
    radians = angles if in_radians else np.radians(angles)
    return np.stack((np.cos(radians), np.sin(radians)), axis=-1)


def _inside_boundary(position: np.ndarray, radius: float) -> np.ndarray:
    tolerance = 1.0e-7
    return np.all(
        (position > float(radius) + tolerance)
        & (position < 1.0 - float(radius) - tolerance),
        axis=(1, 2),
    )


def _feasible_pair_source(
    original: np.ndarray,
    amplified: np.ndarray,
    radius: float,
) -> bool:
    for position in (original, amplified):
        for schedule in V3_SCHEDULES:
            reveal = sum(schedule[:4])
            for target_y in (0.38, 0.62):
                shift = target_y - float(position[reveal, 1])
                shifted = position.copy()
                shifted[:, 1] += shift
                if not bool(_inside_boundary(shifted[None], radius)[0]):
                    return False
    return True


def _select_common_sources(
    source: h5py.File,
    seed: int,
    minimum: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    position = _episode_frames(source, "object_position").astype(np.float64)
    amplified = _scale_positions(position, MOTION_SCALE)
    original_ok = _inside_boundary(position, float(source.attrs.get("object_radius", OBJECT_RADIUS)))
    amplified_ok = _inside_boundary(amplified, float(source.attrs.get("object_radius", OBJECT_RADIUS)))
    common = np.flatnonzero(original_ok & amplified_ok)
    if len(common) < minimum:
        raise RuntimeError(
            f"Only {len(common)} source episodes survive the shared boundary policy; "
            f"minimum is {minimum}. Refusing to duplicate trajectories or lower 1.5x."
        )
    ordered = np.asarray(
        sorted(common.tolist(), key=lambda value: _hash_uniform(seed, value, 101)),
        dtype=np.int64,
    )
    usable_count = len(ordered) - len(ordered) % len(V3_SCHEDULES)
    if usable_count <= 0:
        raise RuntimeError("No complete five-schedule source block survives filtering")
    selected = ordered[:usable_count]
    report = {
        "source_episodes": int(len(position)),
        "original_boundary_eligible": int(original_ok.sum()),
        "motion15x_boundary_eligible": int(amplified_ok.sum()),
        "shared_boundary_eligible": int(len(common)),
        "selected_unique_sources": int(len(selected)),
        "dropped_for_five_way_balance": int(len(common) - len(selected)),
    }
    return selected, report


def _build_pair_mapping(
    selected: np.ndarray,
    motion_source: np.ndarray,
    pair_eligible: np.ndarray,
    seed: int,
    pair_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_source = selected.copy()
    pair_id = np.full(len(selected), -1, dtype=np.int64)
    pair_role = np.full(len(selected), -1, dtype=np.int8)
    output_motion = motion_source[output_source]
    next_pair = 0
    for motion in np.unique(output_motion):
        slots = np.flatnonzero(output_motion == motion)
        ordered_slots = sorted(
            slots.tolist(),
            key=lambda slot: _hash_uniform(seed, int(motion), int(output_source[slot]), 109),
        )
        pair_groups = int(round(pair_fraction * len(slots) / 2.0))
        eligible_slots = [slot for slot in ordered_slots if pair_eligible[output_source[slot]]]
        if len(eligible_slots) < pair_groups:
            raise RuntimeError(
                f"motion={motion} has {len(eligible_slots)} pair-eligible sources; "
                f"{pair_groups} are required"
            )
        bases = eligible_slots[:pair_groups]
        base_sources = {int(output_source[slot]) for slot in bases}
        replacements = [
            slot for slot in ordered_slots if int(output_source[slot]) not in base_sources
        ][:pair_groups]
        for base_slot, replacement_slot in zip(bases, replacements):
            output_source[replacement_slot] = output_source[base_slot]
            pair_id[base_slot] = next_pair
            pair_id[replacement_slot] = next_pair
            pair_role[base_slot] = 0
            pair_role[replacement_slot] = 1
            next_pair += 1
    return output_source, pair_id, pair_role


def _balanced_schedule_assignment(
    source_episode: np.ndarray,
    split_id: np.ndarray,
    seed: int,
) -> np.ndarray:
    result = np.empty((len(source_episode), 5), dtype=np.int16)
    for split in (0, 1, 2):
        split_indices = np.flatnonzero(split_id == split)
        groups: dict[int, list[int]] = {}
        for index in split_indices.tolist():
            groups.setdefault(int(source_episode[index]), []).append(index)
        pair_groups = [item for item in groups.items() if len(item[1]) == 2]
        single_groups = [item for item in groups.items() if len(item[1]) == 1]
        pair_groups.sort(key=lambda item: _hash_uniform(seed, split, item[0], 127))
        single_groups.sort(key=lambda item: _hash_uniform(seed, split, item[0], 131))
        count = len(split_indices)
        targets = np.full(5, count // 5, dtype=np.int64)
        for index in range(count % 5):
            targets[index] += 1
        remaining = targets.copy()
        assignment: dict[int, int] = {}
        for source_id, indices in pair_groups:
            feasible = np.flatnonzero(remaining >= len(indices))
            if not feasible.size:
                raise RuntimeError(f"Unable to balance pair group in split {split}")
            schedule_id = int(feasible[np.argmax(remaining[feasible])])
            assignment[source_id] = schedule_id
            remaining[schedule_id] -= len(indices)
        for source_id, indices in single_groups:
            feasible = np.flatnonzero(remaining > 0)
            if not feasible.size:
                raise RuntimeError(f"Schedule targets exhausted early in split {split}")
            schedule_id = int(feasible[np.argmax(remaining[feasible])])
            assignment[source_id] = schedule_id
            remaining[schedule_id] -= 1
        if np.any(remaining):
            raise RuntimeError(f"Schedule balancing did not close in split {split}: {remaining}")
        for index in split_indices.tolist():
            result[index] = V3_SCHEDULES[assignment[int(source_episode[index])]]
    return result


def build_shared_plan(
    source: h5py.File,
    seed: int,
    pair_fraction: float,
    minimum_eligible_episodes: int,
) -> tuple[list[SharedPlan], dict[str, Any]]:
    selected, selection_report = _select_common_sources(
        source, seed, minimum_eligible_episodes
    )
    position = _episode_frames(source, "object_position").astype(np.float64)
    amplified = _scale_positions(position, MOTION_SCALE)
    radius = float(source.attrs.get("object_radius", OBJECT_RADIUS))
    pair_eligible = np.asarray(
        [
            _feasible_pair_source(position[index], amplified[index], radius)
            for index in range(len(position))
        ],
        dtype=bool,
    )
    motion_source = _motion_types(source)
    output_source, pair_id, pair_role = _build_pair_mapping(
        selected, motion_source, pair_eligible, seed, pair_fraction
    )
    output_motion = motion_source[output_source]
    split_id = _split_assignments(output_source, output_motion, seed)
    phase_lengths = _balanced_schedule_assignment(output_source, split_id, seed)
    plans = [
        SharedPlan(
            source_episode=int(output_source[index]),
            phase_lengths=tuple(int(value) for value in phase_lengths[index]),
            split_id=int(split_id[index]),
            pair_id=int(pair_id[index]),
            pair_role=int(pair_role[index]),
        )
        for index in range(len(output_source))
    ]
    selection_report.update(
        output_episodes=len(plans),
        unique_output_trajectories=int(len(np.unique(output_source))),
        pair_groups=int(np.sum(pair_role == 0)),
        pair_eligible_sources=int(pair_eligible[selected].sum()),
    )
    return plans, selection_report


def _pair_y_shifts(position: np.ndarray, plans: Sequence[SharedPlan]) -> np.ndarray:
    shifts = np.zeros(len(plans), dtype=np.float32)
    for output_episode, plan in enumerate(plans):
        if plan.pair_role < 0:
            continue
        reveal = sum(plan.phase_lengths[:4])
        target_y = 0.38 if plan.pair_role == 0 else 0.62
        shifts[output_episode] = target_y - float(position[output_episode, reveal, 1])
    return shifts


def _set_frame_metadata(
    output: h5py.File,
    name: str,
    values: np.ndarray,
    dtype: Any,
) -> None:
    flat = np.repeat(np.asarray(values, dtype=dtype), EPISODE_LENGTH)
    if name in output:
        output[name][:] = flat.astype(output[name].dtype)
    else:
        dataset = output.create_dataset(
            name,
            shape=(len(flat),),
            dtype=dtype,
            compression="gzip",
            compression_opts=1,
        )
        dataset[:] = flat


def _write_variant(
    source_path: Path,
    output_path: Path,
    variant: str,
    scale: float,
    plans: Sequence[SharedPlan],
    seed: int,
    pair_fraction: float,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    with h5py.File(source_path, "r") as source:
        source_offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
        source_episode = np.asarray([plan.source_episode for plan in plans], dtype=np.int64)
        phase_lengths = np.asarray([plan.phase_lengths for plan in plans], dtype=np.int16)
        split_id = np.asarray([plan.split_id for plan in plans], dtype=np.int8)
        pair_id = np.asarray([plan.pair_id for plan in plans], dtype=np.int64)
        pair_role = np.asarray([plan.pair_role for plan in plans], dtype=np.int8)
        num_source_episodes = len(source_offsets)
        num_episodes = len(plans)
        source_total_frames = num_source_episodes * EPISODE_LENGTH
        total_frames = num_episodes * EPISODE_LENGTH

        source_position = _episode_frames(source, "object_position")
        output_position = _scale_positions(source_position[source_episode], scale).astype(
            source["object_position"].dtype
        )
        y_shift = _pair_y_shifts(output_position, plans)
        output_position[:, :, 1] += y_shift[:, None]
        radius = float(source.attrs.get("object_radius", OBJECT_RADIUS))
        if not bool(np.all(_inside_boundary(output_position, radius))):
            raise AssertionError(f"{variant} plan contains a boundary-contact episode")

        source_angle = _episode_frames(source, "object_angle")
        source_angles_are_radians = _angles_are_radians(source_angle)
        if "object_orientation" in source:
            source_orientation = _episode_frames(source, "object_orientation")
            expected_source_orientation = _orientation_vectors(
                source_angle,
                angles_are_radians=source_angles_are_radians,
            )
            if source_orientation.shape != expected_source_orientation.shape:
                raise ValueError(
                    "Source object_orientation must have shape "
                    f"{expected_source_orientation.shape}; got {source_orientation.shape}"
                )
            orientation_error = float(
                np.max(
                    np.abs(
                        source_orientation.astype(np.float64)
                        - expected_source_orientation
                    )
                )
            )
            if orientation_error > 2.0e-5:
                raise ValueError(
                    "Source object_orientation is not [cos(object_angle), "
                    f"sin(object_angle)]; max error={orientation_error:.6g}"
                )
        output_angle = _scale_angles(source_angle[source_episode], scale)
        output_orientation = _orientation_vectors(
            output_angle,
            angles_are_radians=source_angles_are_radians,
        )
        vectors = [phase_vectors_v3(schedule) for schedule in phase_lengths]
        phase_id = np.stack([value["phase_id"] for value in vectors])
        visible = np.stack([value["object_visible"] for value in vectors])
        reappearance = np.stack([value["is_reappearance"] for value in vectors])
        final_reappearance = np.stack(
            [value["is_final_reappearance"] for value in vectors]
        )

        with h5py.File(temp_path, "w", libver="latest") as output:
            _copy_attrs(source, output)
            output.attrs["source_num_episodes"] = int(num_source_episodes)
            output.attrs["num_episodes"] = int(num_episodes)
            output.attrs["dataset_version"] = "primitive_v3"
            output.attrs["dataset_variant"] = f"primitive_v3_h4_reveal5_{variant}"
            output.attrs["source_dataset"] = str(source_path.resolve())
            output.attrs["generator_seed"] = int(seed)
            output.attrs["trajectory_generation_semantics"] = (
                "derived_from_primitive_v1_state; not newly sampled physical trajectories"
            )
            output.attrs["motion_scale"] = float(scale)
            output.attrs["object_angle_unit"] = (
                "radian" if source_angles_are_radians else "degree"
            )
            output.attrs["object_orientation_policy"] = ORIENTATION_POLICY
            output.attrs["motion_transform"] = (
                "anchor_preserving_linear_displacement_velocity_action_angle_change_v1"
            )
            output.attrs["boundary_policy"] = BOUNDARY_POLICY
            output.attrs["first_occlusion_policy"] = FIRST_OCCLUSION_POLICY
            output.attrs["second_occlusion_policy"] = "fixed_length_4_v1"
            output.attrs["final_reveal_policy"] = "balanced_reveal_15_16_17_18_19_v1"
            output.attrs["frame_loss_weight_policy"] = FRAME_LOSS_WEIGHT_POLICY
            output.attrs["schedule_policy_json"] = json.dumps(
                [list(schedule) for schedule in V3_SCHEDULES]
            )
            output.attrs["pair_fraction"] = float(pair_fraction)
            output.attrs["first_occlusion_randomized"] = 0
            output.attrs["completed"] = 0

            for name, source_dataset in source.items():
                if name == "pixels":
                    continue
                if source_dataset.shape and source_dataset.shape[0] == source_total_frames:
                    target = _create_dataset_like(
                        output,
                        name,
                        source_dataset,
                        (total_frames, *source_dataset.shape[1:]),
                    )
                    target[:] = _selected_frame_values(source_dataset, source_episode)
                elif source_dataset.shape and source_dataset.shape[0] == num_source_episodes:
                    target = _create_dataset_like(
                        output,
                        name,
                        source_dataset,
                        (num_episodes, *source_dataset.shape[1:]),
                    )
                    target[:] = np.asarray(source_dataset[:])[source_episode]

            source_pixels = source["pixels"]
            pixels = _create_dataset_like(
                output,
                "pixels",
                source_pixels,
                (total_frames, *source_pixels.shape[1:]),
            )

            output["ep_offset"][:] = np.arange(num_episodes, dtype=np.int64) * EPISODE_LENGTH
            output["ep_len"][:] = EPISODE_LENGTH
            for name in ("episode_id", "ep_idx"):
                if name in output:
                    output[name][:] = np.repeat(
                        np.arange(num_episodes, dtype=output[name].dtype), EPISODE_LENGTH
                    )
            if "episode_start" in output:
                starts = np.zeros((num_episodes, EPISODE_LENGTH), dtype=output["episode_start"].dtype)
                starts[:, 0] = 1
                output["episode_start"][:] = starts.reshape(-1)
            for name in ("timestep", "step_idx"):
                if name in output:
                    output[name][:] = np.tile(
                        np.arange(EPISODE_LENGTH, dtype=output[name].dtype), num_episodes
                    )

            frame_weights = np.stack(
                [phase_relative_loss_weights_v3(schedule) for schedule in phase_lengths]
            )
            for name, values in (
                ("phase_id", phase_id),
                ("object_visible", visible),
                ("is_reappearance", reappearance),
                ("is_final_reappearance", final_reappearance),
                ("frame_loss_weight", frame_weights),
            ):
                if name in output:
                    output[name][:] = values.reshape(-1).astype(output[name].dtype)
                else:
                    output.create_dataset(name, data=values.reshape(-1))
            if "phase_lengths" in output:
                output["phase_lengths"][:] = np.repeat(
                    phase_lengths[:, None, :], EPISODE_LENGTH, axis=1
                ).reshape(-1, 5)
            else:
                output.create_dataset(
                    "phase_lengths",
                    data=np.repeat(phase_lengths[:, None, :], EPISODE_LENGTH, axis=1).reshape(-1, 5),
                )

            _set_frame_metadata(output, "trajectory_id", source_episode, np.int64)
            _set_frame_metadata(output, "source_episode", source_episode, np.int64)
            _set_frame_metadata(output, "renderer_seed", seed * 1_000_000 + source_episode, np.int64)
            _set_frame_metadata(output, "split_id", split_id, np.int8)
            _set_frame_metadata(output, "pair_id", pair_id, np.int64)
            _set_frame_metadata(output, "pair_role", pair_role, np.int8)
            _set_frame_metadata(output, "occlusion_start", phase_lengths[:, :3].sum(axis=1), np.int16)
            _set_frame_metadata(output, "occlusion_length", phase_lengths[:, 3], np.int16)
            _set_frame_metadata(output, "reveal_idx", phase_lengths[:, :4].sum(axis=1), np.int16)
            _set_frame_metadata(output, "y_shift", y_shift, np.float32)
            _set_frame_metadata(output, "motion_scale", np.full(num_episodes, scale), np.float32)

            output["object_position"][:] = output_position.reshape(-1, 2).astype(
                output["object_position"].dtype
            )
            if "object_velocity" in output:
                output["object_velocity"][:] = (
                    _episode_frames(source, "object_velocity")[source_episode] * scale
                ).reshape(output["object_velocity"].shape).astype(output["object_velocity"].dtype)
            if "action" in output:
                output["action"][:] = (
                    _episode_frames(source, "action")[source_episode] * scale
                ).reshape(output["action"].shape).astype(output["action"].dtype)
            output["object_angle"][:] = output_angle.reshape(output["object_angle"].shape).astype(
                output["object_angle"].dtype
            )
            if "object_orientation" in output:
                expected_shape = (total_frames, 2)
                if output["object_orientation"].shape != expected_shape:
                    raise ValueError(
                        "object_orientation must use the real primitive_v1 "
                        f"[cos(theta), sin(theta)] schema {expected_shape}; got "
                        f"{output['object_orientation'].shape}"
                    )
                output["object_orientation"][:] = output_orientation.reshape(
                    expected_shape
                ).astype(output["object_orientation"].dtype)
            if "object_angular_velocity" in output:
                output["object_angular_velocity"][:] = (
                    _episode_frames(source, "object_angular_velocity")[source_episode] * scale
                ).reshape(output["object_angular_velocity"].shape).astype(
                    output["object_angular_velocity"].dtype
                )
            for name in (
                "executed_acceleration",
                "translation_acceleration",
                "angular_acceleration",
                "object_acceleration",
                "object_angular_acceleration",
            ):
                if name in output:
                    output[name][:] = (np.asarray(output[name][:]) * scale).astype(output[name].dtype)

            for name in ("observation", "proprio"):
                if name not in output:
                    continue
                values = np.asarray(output[name][:])
                source_selected_position = _selected_frame_values(
                    source["object_position"], source_episode
                )
                columns = _infer_position_columns(values, source_selected_position)
                for axis, column in columns.items():
                    values[:, column] = output_position.reshape(-1, 2)[:, axis]
                output[name][:] = values
                output.attrs[f"{name}_position_columns_json"] = json.dumps(columns)

            image_size = int(source.attrs.get("image_size", source_pixels.shape[1]))
            content_id = np.asarray(source["content_id"][source_offsets]).reshape(-1)
            angles_degrees = _angle_degrees(output_angle.astype(np.float64))
            for episode, plan in enumerate(plans):
                content = TRAIN_CONTENTS[int(content_id[plan.source_episode])]
                rendered = np.empty((EPISODE_LENGTH, *source_pixels.shape[1:]), dtype=np.uint8)
                for timestep in range(EPISODE_LENGTH):
                    rendered[timestep] = np.asarray(
                        render_object(
                            content,
                            angle_deg=float(angles_degrees[episode, timestep]),
                            center_xy=tuple(float(value) for value in output_position[episode, timestep]),
                            object_radius=radius,
                            image_size=image_size,
                            visible=bool(visible[episode, timestep]),
                        ),
                        dtype=np.uint8,
                    )
                start = episode * EPISODE_LENGTH
                pixels[start : start + EPISODE_LENGTH] = rendered
                if (episode + 1) % 500 == 0 or episode + 1 == num_episodes:
                    output.flush()
                    print(f"[{variant}] rendered {episode + 1}/{num_episodes} episodes", flush=True)

            output.attrs["completed"] = 1
            output.attrs["completed_episodes"] = num_episodes
            output.flush()

    os.replace(temp_path, output_path)
    reveal = phase_lengths[:, :4].sum(axis=1)
    manifest = {
        "variant": variant,
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "num_episodes": len(plans),
        "unique_trajectories": int(len(np.unique(source_episode))),
        "motion_scale": float(scale),
        "first_occlusion_randomized": False,
        "schedule_counts": {
            str(schedule): int(np.sum(np.all(phase_lengths == schedule, axis=1)))
            for schedule in V3_SCHEDULES
        },
        "reveal_index_counts": {
            str(index): int(np.sum(reveal == index)) for index in range(15, 20)
        },
        "split_counts": {str(value): int(np.sum(split_id == value)) for value in (0, 1, 2)},
        "pair_groups": int(np.sum(pair_role == 0)),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    return manifest


def build_dataset_pair(
    source_path: Path,
    output_dir: Path,
    seed: int = GENERATOR_SEED,
    pair_fraction: float = PAIR_FRACTION,
    minimum_eligible_episodes: int = 16_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not 0.0 <= pair_fraction <= 0.5:
        raise ValueError("pair_fraction must be in [0, 0.5]")
    with h5py.File(source_path, "r") as source:
        if int(source.attrs["episode_length"]) != EPISODE_LENGTH:
            raise ValueError("primitive_v3 requires 22-frame primitive_v1 source episodes")
        plans, selection = build_shared_plan(
            source, seed, pair_fraction, minimum_eligible_episodes
        )
    manifests = {}
    for variant, scale in VARIANTS.items():
        output = output_dir / f"primitive_v3_h4_reveal5_{variant}.h5"
        manifests[variant] = _write_variant(
            source_path,
            output,
            variant,
            scale,
            plans,
            seed,
            pair_fraction,
            overwrite,
        )
    comparison = {
        "created_from": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "generator_seed": seed,
        "selection": selection,
        "variants": manifests,
        "first_occlusion_decision": {
            "randomized": False,
            "reason": (
                "The five-way second onset already tests absolute-time shortcuts; "
                "keeping the first onset fixed preserves a common memory burn-in and "
                "avoids adding pre-occlusion-history length as a confound."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "primitive_v3_pair_manifest.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=GENERATOR_SEED)
    parser.add_argument("--pair-fraction", type=float, default=PAIR_FRACTION)
    parser.add_argument("--minimum-eligible-episodes", type=int, default=16_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dataset_pair(
        args.source,
        args.output_dir,
        seed=args.seed,
        pair_fraction=args.pair_fraction,
        minimum_eligible_episodes=args.minimum_eligible_episodes,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

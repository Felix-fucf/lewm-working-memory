#!/usr/bin/env python3
"""Build renderer-based primitive_v2 occlusion curriculum datasets.

The primitive_v1 source has 22-frame episodes with phase lengths
``(3, 4, 3, 8, 4)``.  This builder keeps the underlying trajectory, actions,
and episode length fixed.  Dshort/Dmid keep the final reveal at frame 18;
Dfinal jitters it over frames 17/18/19 to break absolute-time shortcuts.  It
re-renders from state, never by editing pixels.

``dshort`` uses a two-frame second occlusion, ``dmid`` uses four frames, and
``dfinal`` mixes two/four/eight-frame occlusions at 50/30/20 percent.  Dfinal
also constructs renderer-based y-shift pairs.  Pair members share actions,
x-trajectories, timing, and occluded pixels, while visible y-history and the
future y-position differ.
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
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

try:
    from .primitive_common import EPISODE_LENGTH, OBJECT_RADIUS
    from .primitive_render import TRAIN_CONTENTS, render_object
except ImportError:  # Direct script execution on AutoDL.
    from primitive_common import EPISODE_LENGTH, OBJECT_RADIUS
    from primitive_render import TRAIN_CONTENTS, render_object


BASE_PHASE_LENGTHS = (3, 4, 3, 8, 4)
VARIANT_PHASE_LENGTHS: Mapping[str, tuple[int, int, int, int, int]] = {
    "dshort": (3, 4, 9, 2, 4),
    "dmid": (3, 4, 7, 4, 4),
    "dlong": BASE_PHASE_LENGTHS,
}
DFINAL_WEIGHTS = (("dshort", 0.50), ("dmid", 0.30), ("dlong", 0.20))
FINAL_REVEAL_INDEX = 18
SPLIT_FRACTIONS = (0.90, 0.05, 0.05)
FRAME_LOSS_WEIGHT_SUM = 20.2
FRAME_LOSS_WEIGHT_POLICY = "phase_relative_sum_normalized_v2"


@dataclass(frozen=True)
class EpisodePlan:
    source_episode: int
    phase_lengths: tuple[int, int, int, int, int]
    split_id: int
    pair_id: int = -1
    pair_role: int = -1
    y_shift: float = 0.0


def phase_vectors(
    lengths: Sequence[int],
) -> dict[str, np.ndarray]:
    lengths = tuple(int(value) for value in lengths)
    if len(lengths) != 5 or sum(lengths) != EPISODE_LENGTH:
        raise ValueError(f"Invalid 22-frame phase schedule: {lengths}")
    if lengths[0] != 3 or lengths[1] != 4:
        raise ValueError(f"Frozen phases changed unexpectedly: {lengths}")
    if not 17 <= sum(lengths[:4]) <= 19:
        raise ValueError(f"Final reveal must be in frames 17..19: {lengths}")

    phase_id = np.concatenate(
        [np.full(length, phase, dtype=np.int8) for phase, length in enumerate(lengths)]
    )
    visible = np.isin(phase_id, (0, 2, 4)).astype(np.uint8)
    reappearance = np.zeros(EPISODE_LENGTH, dtype=np.uint8)
    reappearance[np.flatnonzero((visible[1:] == 1) & (visible[:-1] == 0)) + 1] = 1
    final_reveal_index = int(sum(lengths[:4]))
    final_reappearance = np.zeros(EPISODE_LENGTH, dtype=np.uint8)
    final_reappearance[final_reveal_index] = 1
    return {
        "phase_id": phase_id,
        "object_visible": visible,
        "is_reappearance": reappearance,
        "is_final_reappearance": final_reappearance,
    }


def phase_relative_loss_weights(lengths: Sequence[int]) -> np.ndarray:
    """Create target-frame weights from the episode's actual phase schedule."""
    lengths = tuple(int(value) for value in lengths)
    phase_vectors(lengths)  # validate the schedule in one canonical place
    parts: list[np.ndarray] = []
    for phase, length in enumerate(lengths):
        if phase == 0:
            values = np.ones(length, dtype=np.float64)
        elif phase in (1, 3):
            values = np.full(length, 0.1, dtype=np.float64)
        elif phase == 2:
            values = np.full(length, 0.1, dtype=np.float64)
            values[: min(length, 3)] = np.asarray([3.0, 2.0, 2.0])[:length]
        else:
            values = np.full(length, 2.0, dtype=np.float64)
            if length:
                values[0] = 3.0
        parts.append(values)
    weights = np.concatenate(parts)
    weights *= FRAME_LOSS_WEIGHT_SUM / float(weights.sum())
    return weights.astype(np.float32)


def _hash_uniform(seed: int, *parts: int) -> float:
    payload = ":".join(str(int(value)) for value in (seed, *parts)).encode("ascii")
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return integer / float(2**64)


def _jitter_schedule(
    base: Sequence[int], seed: int, source_episode: int
) -> tuple[int, ...]:
    # Break the absolute-time shortcut in Dfinal while preserving the sampled
    # occlusion duration.  Moving one frame between the visible-history phase
    # and the final visible phase shifts both onset and reveal by {-1, 0, +1}.
    jitter = int(math.floor(3.0 * _hash_uniform(seed, source_episode, 19))) - 1
    return (base[0], base[1], base[2] + jitter, base[3], base[4] - jitter)


def _dfinal_schedule_map(
    source_episode: np.ndarray,
    seed: int,
) -> dict[int, tuple[int, ...]]:
    """Assign exactly 50/30/20 percent while keeping duplicate pairs together."""
    counts = Counter(int(value) for value in source_episode.tolist())
    total = len(source_episode)
    targets = {
        "dshort": int(round(DFINAL_WEIGHTS[0][1] * total)),
        "dmid": int(round(DFINAL_WEIGHTS[1][1] * total)),
    }
    targets["dlong"] = total - targets["dshort"] - targets["dmid"]
    remaining = dict(targets)
    ordered = sorted(
        counts,
        key=lambda group: (-counts[group], _hash_uniform(seed, group, 17)),
    )
    assignment: dict[int, tuple[int, ...]] = {}
    for group in ordered:
        weight = counts[group]
        feasible = [name for name in remaining if remaining[name] >= weight]
        if not feasible:
            raise RuntimeError("Unable to realize exact Dfinal curriculum proportions")
        selected = max(
            feasible,
            key=lambda name: (
                remaining[name] / max(targets[name], 1),
                _hash_uniform(seed, group, {"dshort": 1, "dmid": 2, "dlong": 3}[name]),
            ),
        )
        remaining[selected] -= weight
        assignment[group] = _jitter_schedule(
            VARIANT_PHASE_LENGTHS[selected], seed, group
        )
    if any(remaining.values()):
        raise RuntimeError(f"Dfinal schedule counts did not close: {remaining}")
    return assignment


def _split_assignments(
    source_episode: np.ndarray,
    motion_type: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Assign 90/5/5 splits while keeping duplicate trajectories together."""
    result = np.empty(len(source_episode), dtype=np.int8)
    for motion in np.unique(motion_type):
        output_indices = np.flatnonzero(motion_type == motion)
        groups = np.unique(source_episode[output_indices])
        ordered = sorted(groups.tolist(), key=lambda group: _hash_uniform(seed, motion, group))
        n_groups = len(ordered)
        n_train = int(round(SPLIT_FRACTIONS[0] * n_groups))
        n_val = int(round(SPLIT_FRACTIONS[1] * n_groups))
        train_groups = set(ordered[:n_train])
        val_groups = set(ordered[n_train : n_train + n_val])
        for output_index in output_indices:
            group = int(source_episode[output_index])
            if group in train_groups:
                result[output_index] = 0
            elif group in val_groups:
                result[output_index] = 1
            else:
                result[output_index] = 2
    return result


def _feasible_shift_interval(position: np.ndarray, radius: float) -> tuple[float, float]:
    y = np.asarray(position[:, 1], dtype=np.float64)
    return float(radius - y.min()), float(1.0 - radius - y.max())


def _build_output_sources(
    motion_type: np.ndarray,
    seed: int,
    pair_fraction: float,
    pair_eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace same-motion slots with duplicate sources to create pair groups."""
    source = np.arange(len(motion_type), dtype=np.int64)
    pair_id = np.full(len(source), -1, dtype=np.int64)
    pair_role = np.full(len(source), -1, dtype=np.int8)
    next_pair_id = 0

    for motion in np.unique(motion_type):
        indices = np.flatnonzero(motion_type == motion)
        ordered = np.asarray(
            sorted(indices.tolist(), key=lambda value: _hash_uniform(seed, motion, value, 31)),
            dtype=np.int64,
        )
        pair_groups = int(round(pair_fraction * len(indices) / 2.0))
        if pair_groups == 0:
            continue
        eligible = np.asarray(
            [value for value in ordered.tolist() if pair_eligible[value]],
            dtype=np.int64,
        )
        if pair_groups > len(eligible):
            raise ValueError(
                f"motion={motion} has {len(eligible)} eligible pair sources, "
                f"but {pair_groups} are required"
            )
        if 2 * pair_groups > len(ordered):
            raise ValueError("pair_fraction leaves no unpaired examples")

        bases = eligible[:pair_groups]
        base_set = set(bases.tolist())
        replacement_slots = np.asarray(
            [value for value in ordered.tolist() if value not in base_set][:pair_groups],
            dtype=np.int64,
        )
        for base, replacement in zip(bases.tolist(), replacement_slots.tolist()):
            source[replacement] = base
            pair_id[base] = next_pair_id
            pair_id[replacement] = next_pair_id
            pair_role[base] = 0
            pair_role[replacement] = 1
            next_pair_id += 1

    return source, pair_id, pair_role


def _episode_motion_type(file: h5py.File) -> np.ndarray:
    offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
    return np.asarray(file["motion_type"][offsets], dtype=np.int8).reshape(-1)


def build_episode_plan(
    file: h5py.File,
    variant: str,
    seed: int,
    pair_fraction: float,
) -> list[EpisodePlan]:
    offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
    num_episodes = len(offsets)
    motion_source = _episode_motion_type(file)
    source_position = np.asarray(file["object_position"][:], dtype=np.float64).reshape(
        num_episodes, EPISODE_LENGTH, 2
    )
    pair_eligible = np.zeros(num_episodes, dtype=bool)
    for episode in range(num_episodes):
        lower, upper = _feasible_shift_interval(source_position[episode], OBJECT_RADIUS)
        separations = []
        for reveal in (17, 18, 19):
            y = float(source_position[episode, reveal, 1])
            low_shift = float(np.clip(0.38 - y, lower, upper))
            high_shift = float(np.clip(0.62 - y, lower, upper))
            separations.append(high_shift - low_shift)
        pair_eligible[episode] = min(separations) >= 0.10

    if variant == "dfinal":
        source, pair_id, pair_role = _build_output_sources(
            motion_source, seed, pair_fraction, pair_eligible
        )
    else:
        source = np.arange(num_episodes, dtype=np.int64)
        pair_id = np.full(num_episodes, -1, dtype=np.int64)
        pair_role = np.full(num_episodes, -1, dtype=np.int8)

    output_motion = motion_source[source]
    split_id = _split_assignments(source, output_motion, seed)
    dfinal_schedule = (
        _dfinal_schedule_map(source, seed) if variant == "dfinal" else {}
    )

    final_positions = np.asarray(
        file["object_position"][offsets + FINAL_REVEAL_INDEX], dtype=np.float64
    )
    sigma_x = float(final_positions[:, 0].std())
    balance_amplitude = min(max(sigma_x, 0.12), 0.30)
    pair_amplitude = min(0.12, balance_amplitude)

    plans: list[EpisodePlan] = []
    for output_episode, source_episode in enumerate(source.tolist()):
        if variant == "dfinal":
            lengths = dfinal_schedule[source_episode]
        else:
            lengths = VARIANT_PHASE_LENGTHS[variant]

        shift = 0.0
        if variant == "dfinal":
            position = source_position[source_episode]
            lower, upper = _feasible_shift_interval(position, OBJECT_RADIUS)
            if pair_role[output_episode] >= 0:
                sign = -1.0 if pair_role[output_episode] == 0 else 1.0
                target_y = 0.5 + sign * pair_amplitude
            else:
                sign = -1.0 if _hash_uniform(seed, source_episode, 43) < 0.5 else 1.0
                target_y = 0.5 + sign * balance_amplitude
            reveal_index = int(sum(lengths[:4]))
            desired = target_y - float(position[reveal_index, 1])
            shift = float(np.clip(desired, lower, upper))

        plans.append(
            EpisodePlan(
                source_episode=source_episode,
                phase_lengths=tuple(int(value) for value in lengths),
                split_id=int(split_id[output_episode]),
                pair_id=int(pair_id[output_episode]),
                pair_role=int(pair_role[output_episode]),
                y_shift=shift,
            )
        )
    return plans


def _copy_attrs(source: h5py.File, output: h5py.File) -> None:
    for name, value in source.attrs.items():
        output.attrs[name] = value


def _create_dataset_like(
    output: h5py.File,
    name: str,
    source: h5py.Dataset,
    shape: tuple[int, ...],
) -> h5py.Dataset:
    if name == "pixels":
        return output.create_dataset(
            name,
            shape=shape,
            dtype=source.dtype,
            chunks=(1, *shape[1:]),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
    kwargs: dict[str, object] = {}
    if int(np.prod(shape, dtype=np.int64)) * source.dtype.itemsize > 1_000_000:
        kwargs.update(compression="gzip", compression_opts=1, shuffle=True, chunks=True)
    return output.create_dataset(name, shape=shape, dtype=source.dtype, **kwargs)


def _selected_frame_values(
    dataset: h5py.Dataset,
    source_episode: np.ndarray,
) -> np.ndarray:
    values = np.asarray(dataset[:])
    reshaped = values.reshape((len(values) // EPISODE_LENGTH, EPISODE_LENGTH, *values.shape[1:]))
    return reshaped[source_episode].reshape((-1, *values.shape[1:]))


def _infer_position_columns(
    values: np.ndarray,
    position: np.ndarray,
) -> dict[int, int]:
    if values.ndim != 2:
        return {}
    sample_count = min(len(values), 20_000)
    sample_indices = np.linspace(0, len(values) - 1, sample_count, dtype=np.int64)
    result: dict[int, int] = {}
    for axis in (0, 1):
        target = position[sample_indices, axis]
        for column in range(values.shape[1]):
            candidate = values[sample_indices, column]
            if np.allclose(candidate, target, rtol=1e-5, atol=1e-6, equal_nan=True):
                result[axis] = column
                break
    return result


def _angle_degrees(angle: np.ndarray) -> np.ndarray:
    finite = angle[np.isfinite(angle)]
    if finite.size and float(np.max(np.abs(finite))) <= 2.0 * math.pi + 0.1:
        return np.degrees(angle)
    return angle


def _render_episode(
    source_pixels: np.ndarray,
    content_id: int,
    position: np.ndarray,
    angle: np.ndarray,
    visible: np.ndarray,
    source_visible: np.ndarray,
    y_shift: float,
    image_size: int,
    object_radius: float,
) -> np.ndarray:
    pixels = np.asarray(source_pixels).copy()
    rerender_all = abs(float(y_shift)) > 1e-12
    content = TRAIN_CONTENTS[int(content_id)]
    angles = _angle_degrees(np.asarray(angle, dtype=np.float64))

    for timestep in range(EPISODE_LENGTH):
        changed_visibility = int(visible[timestep]) != int(source_visible[timestep])
        if not rerender_all and not changed_visibility:
            continue
        frame = render_object(
            content,
            angle_deg=float(angles[timestep]),
            center_xy=(float(position[timestep, 0]), float(position[timestep, 1])),
            object_radius=float(object_radius),
            image_size=int(image_size),
            visible=bool(visible[timestep]),
        )
        pixels[timestep] = np.asarray(frame, dtype=np.uint8)
    return pixels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_variant(
    source_path: Path,
    output_path: Path,
    variant: str,
    seed: int,
    pair_fraction: float,
    overwrite: bool,
) -> dict[str, object]:
    if variant not in {"dshort", "dmid", "dlong", "dfinal"}:
        raise ValueError(f"Unknown variant: {variant}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source:
        if int(source.attrs["episode_length"]) != EPISODE_LENGTH:
            raise ValueError("Only fixed 22-frame primitive_v1 datasets are supported")
        offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
        num_episodes = len(offsets)
        total_frames = num_episodes * EPISODE_LENGTH
        source_lengths = np.asarray(source["phase_lengths"][offsets])
        source_h = int(round(float(np.median(source_lengths[:, 3]))))
        if source_h != BASE_PHASE_LENGTHS[3]:
            raise ValueError(
                f"This frozen campaign expects source median H=8, found H={source_h}"
            )
        expected_short = max(2, int(round(source_h / 4)))
        expected_mid = max(expected_short + 1, int(round(source_h / 2)))
        if (
            VARIANT_PHASE_LENGTHS["dshort"][3] != expected_short
            or VARIANT_PHASE_LENGTHS["dmid"][3] != expected_mid
        ):
            raise AssertionError("Frozen Dshort/Dmid schedules disagree with H-derived rules")
        plans = build_episode_plan(source, variant, seed, pair_fraction)
        source_episode = np.asarray([plan.source_episode for plan in plans], dtype=np.int64)
        phase_lengths = np.asarray([plan.phase_lengths for plan in plans], dtype=np.int16)
        split_id = np.asarray([plan.split_id for plan in plans], dtype=np.int8)
        pair_id = np.asarray([plan.pair_id for plan in plans], dtype=np.int64)
        pair_role = np.asarray([plan.pair_role for plan in plans], dtype=np.int8)
        y_shift = np.asarray([plan.y_shift for plan in plans], dtype=np.float32)

        vectors = [phase_vectors(lengths) for lengths in phase_lengths]
        phase_id = np.stack([item["phase_id"] for item in vectors])
        visible = np.stack([item["object_visible"] for item in vectors])
        reappearance = np.stack([item["is_reappearance"] for item in vectors])
        final_reappearance = np.stack([item["is_final_reappearance"] for item in vectors])

        with h5py.File(temp_path, "w", libver="latest") as output:
            _copy_attrs(source, output)
            output.attrs["dataset_version"] = "primitive_v2"
            output.attrs["dataset_variant"] = f"primitive_v2_{variant}"
            output.attrs["source_dataset"] = str(source_path.resolve())
            output.attrs["generator_seed"] = int(seed)
            output.attrs["source_median_occlusion_length"] = source_h
            output.attrs["renderer_seed_semantics"] = (
                "deterministic rerender identifier generator_seed*1e6+source_episode; "
                "render_object itself makes no stochastic draws"
            )
            output.attrs["final_reveal_policy"] = (
                "fixed_18" if variant != "dfinal" else "deterministic_jitter_17_18_19"
            )
            output.attrs["pair_fraction"] = float(pair_fraction if variant == "dfinal" else 0.0)
            output.attrs["frame_loss_weight_policy"] = FRAME_LOSS_WEIGHT_POLICY
            output.attrs["schedule_policy_json"] = json.dumps(
                {name: list(lengths) for name, lengths in VARIANT_PHASE_LENGTHS.items()}
            )
            output.attrs["completed"] = 0

            # Copy all non-pixel data through the output-to-source episode map.
            for name, source_dataset in source.items():
                if name == "pixels":
                    continue
                if source_dataset.shape and source_dataset.shape[0] == total_frames:
                    shape = (total_frames, *source_dataset.shape[1:])
                    target = _create_dataset_like(output, name, source_dataset, shape)
                    target[:] = _selected_frame_values(source_dataset, source_episode)
                elif source_dataset.shape and source_dataset.shape[0] == num_episodes:
                    shape = (num_episodes, *source_dataset.shape[1:])
                    target = _create_dataset_like(output, name, source_dataset, shape)
                    target[:] = np.asarray(source_dataset[:])[source_episode]

            pixels_source = source["pixels"]
            pixels_output = _create_dataset_like(
                output,
                "pixels",
                pixels_source,
                (total_frames, *pixels_source.shape[1:]),
            )

            # Canonical episode/frame identity in output order.
            output["ep_offset"][:] = np.arange(num_episodes, dtype=np.int64) * EPISODE_LENGTH
            output["ep_len"][:] = EPISODE_LENGTH
            for name in ("episode_id", "ep_idx"):
                if name in output:
                    output[name][:] = np.repeat(np.arange(num_episodes, dtype=output[name].dtype), EPISODE_LENGTH)
            if "episode_start" in output:
                episode_start = np.zeros((num_episodes, EPISODE_LENGTH), dtype=output["episode_start"].dtype)
                episode_start[:, 0] = 1
                output["episode_start"][:] = episode_start.reshape(-1)
            for name in ("timestep", "step_idx"):
                if name in output:
                    output[name][:] = np.tile(np.arange(EPISODE_LENGTH, dtype=output[name].dtype), num_episodes)

            # Use phase-relative weights so reveal jitter does not change supervision strength.
            if "frame_loss_weight" in output:
                frame_weights = np.stack(
                    [phase_relative_loss_weights(lengths) for lengths in phase_lengths]
                )
                output["frame_loss_weight"][:] = frame_weights.reshape(-1).astype(
                    output["frame_loss_weight"].dtype
                )

            for name, values in (
                ("phase_id", phase_id),
                ("object_visible", visible),
                ("is_reappearance", reappearance),
                ("is_final_reappearance", final_reappearance),
            ):
                if name in output:
                    output[name][:] = values.reshape(-1).astype(output[name].dtype)
            if "phase_lengths" in output:
                output["phase_lengths"][:] = np.repeat(phase_lengths[:, None, :], EPISODE_LENGTH, axis=1).reshape(-1, 5)
            if "pair_id" in output:
                output["pair_id"][:] = np.repeat(pair_id, EPISODE_LENGTH)
            if "pair_role" in output:
                output["pair_role"][:] = np.repeat(pair_role, EPISODE_LENGTH)

            # Add canonical manifest fields as frame-level metadata.
            for name, dtype, values in (
                ("trajectory_id", np.int64, source_episode),
                ("renderer_seed", np.int64, seed * 1_000_000 + source_episode),
                ("split_id", np.int8, split_id),
                ("occlusion_start", np.int16, phase_lengths[:, :3].sum(axis=1)),
                ("occlusion_length", np.int16, phase_lengths[:, 3]),
                ("reveal_idx", np.int16, phase_lengths[:, :4].sum(axis=1)),
                ("y_shift", np.float32, y_shift),
            ):
                dataset = output.create_dataset(name, shape=(total_frames,), dtype=dtype, compression="gzip", compression_opts=1)
                dataset[:] = np.repeat(np.asarray(values, dtype=dtype), EPISODE_LENGTH)

            # Shift y in state-space, then update any exact position mirrors.
            position = np.asarray(output["object_position"][:], dtype=np.float32)
            position[:, 1] += np.repeat(y_shift, EPISODE_LENGTH)
            output["object_position"][:] = position
            for name in ("observation", "proprio"):
                if name not in output:
                    continue
                values = np.asarray(output[name][:])
                source_position = _selected_frame_values(source["object_position"], source_episode)
                columns = _infer_position_columns(values, source_position)
                if 1 in columns:
                    values[:, columns[1]] = position[:, 1]
                    output[name][:] = values
                output.attrs[f"{name}_position_columns_json"] = json.dumps(columns)

            image_size = int(source.attrs.get("image_size", pixels_source.shape[1]))
            object_radius = float(source.attrs.get("object_radius", OBJECT_RADIUS))
            source_visible_all = np.asarray(source["object_visible"][:]).reshape(num_episodes, EPISODE_LENGTH)
            source_content = np.asarray(source["content_id"][offsets]).reshape(-1)
            source_angle_all = np.asarray(source["object_angle"][:]).reshape(num_episodes, EPISODE_LENGTH)
            output_position = position.reshape(num_episodes, EPISODE_LENGTH, 2)

            for output_episode, plan in enumerate(plans):
                source_start = int(offsets[plan.source_episode])
                source_end = source_start + EPISODE_LENGTH
                source_pixels = np.asarray(pixels_source[source_start:source_end])
                rendered = _render_episode(
                    source_pixels=source_pixels,
                    content_id=int(source_content[plan.source_episode]),
                    position=output_position[output_episode],
                    angle=source_angle_all[plan.source_episode],
                    visible=visible[output_episode],
                    source_visible=source_visible_all[plan.source_episode],
                    y_shift=plan.y_shift,
                    image_size=image_size,
                    object_radius=object_radius,
                )
                out_start = output_episode * EPISODE_LENGTH
                pixels_output[out_start : out_start + EPISODE_LENGTH] = rendered
                if (output_episode + 1) % 500 == 0 or output_episode + 1 == num_episodes:
                    output.flush()
                    print(f"[{variant}] rendered {output_episode + 1}/{num_episodes} episodes", flush=True)

            output.attrs["completed"] = 1
            output.attrs["completed_episodes"] = num_episodes
            output.flush()

    os.replace(temp_path, output_path)
    manifest = {
        "variant": variant,
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "num_episodes": len(plans),
        "episode_length": EPISODE_LENGTH,
        "reveal_index_counts": {
            str(int(value)): int(np.sum(phase_lengths[:, :4].sum(axis=1) == value))
            for value in np.unique(phase_lengths[:, :4].sum(axis=1))
        },
        "schedule_counts": {
            str(tuple(schedule)): int(np.sum(np.all(phase_lengths == schedule, axis=1)))
            for schedule in np.unique(phase_lengths, axis=0)
        },
        "schedule_proportions": {
            str(tuple(schedule)): float(np.mean(np.all(phase_lengths == schedule, axis=1)))
            for schedule in np.unique(phase_lengths, axis=0)
        },
        "occlusion_length_counts": {
            str(int(length)): int(np.sum(phase_lengths[:, 3] == length))
            for length in np.unique(phase_lengths[:, 3])
        },
        "occlusion_length_proportions": {
            str(int(length)): float(np.mean(phase_lengths[:, 3] == length))
            for length in np.unique(phase_lengths[:, 3])
        },
        "split_counts": {str(value): int(np.sum(split_id == value)) for value in (0, 1, 2)},
        "paired_episodes": int(np.sum(pair_id >= 0)),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("dshort", "dmid", "dlong", "dfinal"),
        default=("dshort", "dmid", "dfinal"),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--pair-fraction", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not 0.0 <= args.pair_fraction <= 0.5:
        raise ValueError("pair_fraction must be in [0, 0.5]")
    manifests = []
    for variant in args.variants:
        output = args.output_dir / f"primitive_v2_{variant}.h5"
        manifests.append(
            build_variant(
                args.source,
                output,
                variant,
                args.seed,
                args.pair_fraction,
                args.overwrite,
            )
        )
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()

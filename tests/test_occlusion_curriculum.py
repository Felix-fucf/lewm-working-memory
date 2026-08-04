from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from schema_tools.audit_occlusion_dataset import audit_dataset
from schema_tools.build_occlusion_curriculum import (
    FRAME_LOSS_WEIGHT_POLICY,
    build_variant,
    phase_relative_loss_weights,
)
from schema_tools.primitive_common import (
    EPISODE_LENGTH,
    FRAME_LOSS_WEIGHTS,
    PHASE_LENGTHS,
    PHASE_VECTORS,
)
from schema_tools.primitive_render import TRAIN_CONTENTS, render_object


def _make_source(path: Path, episodes: int = 100, image_size: int = 32) -> None:
    rng = np.random.default_rng(12345)
    total = episodes * EPISODE_LENGTH
    offsets = np.arange(episodes, dtype=np.int64) * EPISODE_LENGTH
    position = np.empty((episodes, EPISODE_LENGTH, 2), dtype=np.float32)
    velocity = np.empty_like(position)
    angle = np.empty((episodes, EPISODE_LENGTH), dtype=np.float32)
    action = np.empty((episodes, EPISODE_LENGTH, 3), dtype=np.float32)
    pixels = np.empty((total, image_size, image_size, 3), dtype=np.uint8)

    for episode in range(episodes):
        time = np.arange(EPISODE_LENGTH, dtype=np.float32)
        base_x = 0.27 + 0.46 * episode / max(episodes - 1, 1)
        base_y = 0.49 + 0.02 * np.sin(episode)
        position[episode, :, 0] = np.clip(base_x + 0.015 * np.sin(time / 5), 0.2, 0.8)
        position[episode, :, 1] = np.clip(base_y + 0.006 * np.cos(time / 4), 0.2, 0.8)
        velocity[episode, 1:] = position[episode, 1:] - position[episode, :-1]
        velocity[episode, 0] = velocity[episode, 1]
        angle[episode] = (episode * 7 + time * 3) % 360
        action[episode] = rng.integers(-1, 2, size=(EPISODE_LENGTH, 3))
        for timestep in range(EPISODE_LENGTH):
            frame = render_object(
                TRAIN_CONTENTS[episode % len(TRAIN_CONTENTS)],
                angle_deg=float(angle[episode, timestep]),
                center_xy=tuple(position[episode, timestep]),
                object_radius=0.155,
                image_size=image_size,
                visible=bool(PHASE_VECTORS["object_visible"][timestep]),
            )
            pixels[episode * EPISODE_LENGTH + timestep] = np.asarray(frame)

    def frames(values: np.ndarray) -> np.ndarray:
        return np.asarray(values).reshape((total, *values.shape[2:]))

    with h5py.File(path, "w") as file:
        file.attrs["episode_length"] = EPISODE_LENGTH
        file.attrs["image_size"] = image_size
        file.attrs["object_radius"] = 0.155
        file.create_dataset("ep_offset", data=offsets)
        file.create_dataset("ep_len", data=np.full(episodes, EPISODE_LENGTH, dtype=np.int16))
        file.create_dataset("pixels", data=pixels)
        file.create_dataset("action", data=frames(action))
        file.create_dataset("object_position", data=frames(position))
        file.create_dataset("object_velocity", data=frames(velocity))
        file.create_dataset("object_angle", data=angle.reshape(-1))
        file.create_dataset("object_orientation", data=angle.reshape(-1))
        file.create_dataset("object_angular_velocity", data=np.full(total, 3.0, dtype=np.float32))
        file.create_dataset("phase_lengths", data=np.tile(PHASE_LENGTHS, (total, 1)))
        for name, values in PHASE_VECTORS.items():
            file.create_dataset(name, data=np.tile(values, episodes))
        file.create_dataset("valid_mask", data=np.ones(total, dtype=np.uint8))
        start = np.zeros((episodes, EPISODE_LENGTH), dtype=np.uint8)
        start[:, 0] = 1
        file.create_dataset("episode_start", data=start.reshape(-1))
        file.create_dataset("episode_id", data=np.repeat(np.arange(episodes), EPISODE_LENGTH))
        file.create_dataset("timestep", data=np.tile(np.arange(EPISODE_LENGTH), episodes))
        file.create_dataset("content_id", data=np.repeat(np.arange(episodes) % 8, EPISODE_LENGTH))
        file.create_dataset("shape_id", data=np.repeat(np.arange(episodes) % 8, EPISODE_LENGTH))
        file.create_dataset("color_id", data=np.repeat(np.arange(episodes) % 4, EPISODE_LENGTH))
        file.create_dataset("schema_id", data=np.zeros(total, dtype=np.int8))
        file.create_dataset("pair_id", data=np.full(total, -1, dtype=np.int64))
        file.create_dataset("pair_role", data=np.full(total, -1, dtype=np.int8))
        file.create_dataset("motion_type", data=np.repeat(np.arange(episodes) % 2, EPISODE_LENGTH))
        file.create_dataset("frame_loss_weight", data=np.tile(FRAME_LOSS_WEIGHTS, episodes))


def _episode_frames(file: h5py.File, key: str) -> np.ndarray:
    values = np.asarray(file[key][:])
    return values.reshape((len(file["ep_offset"]), EPISODE_LENGTH, *values.shape[1:]))


def test_short_mid_and_final_invariants(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    _make_source(source)
    outputs = {}
    for variant in ("dshort", "dmid", "dlong", "dfinal"):
        output = tmp_path / f"{variant}.h5"
        build_variant(source, output, variant, seed=20260728, pair_fraction=0.20, overwrite=False)
        outputs[variant] = output

    expected_lengths = {
        "dshort": {2},
        "dmid": {4},
        "dlong": {8},
        "dfinal": {2, 4, 8},
    }
    with h5py.File(source, "r") as original:
        source_actions = _episode_frames(original, "action")
        for variant, output in outputs.items():
            with h5py.File(output, "r") as generated:
                lengths = _episode_frames(generated, "phase_lengths")[:, 0]
                assert set(lengths[:, 3].tolist()) == expected_lengths[variant]
                reveal = lengths[:, :4].sum(axis=1)
                if variant == "dfinal":
                    assert set(reveal.tolist()) == {17, 18, 19}
                else:
                    assert np.all(reveal == 18)
                split = _episode_frames(generated, "split_id")[:, 0]
                trajectory = _episode_frames(generated, "trajectory_id")[:, 0]
                assert np.array_equal(
                    _episode_frames(generated, "action"), source_actions[trajectory]
                )
                assert generated.attrs["frame_loss_weight_policy"] == FRAME_LOSS_WEIGHT_POLICY
                generated_weights = _episode_frames(generated, "frame_loss_weight")
                expected_weights = np.stack(
                    [phase_relative_loss_weights(schedule) for schedule in lengths]
                )
                assert np.allclose(generated_weights, expected_weights)
                assert np.allclose(generated_weights.sum(axis=1), 20.2, atol=1e-5)
                reveal_weights = generated_weights[np.arange(len(reveal)), reveal]
                assert np.all(reveal_weights >= 2.5)
                # Prediction step reveal-1 targets frame reveal and must use its weight.
                target_aligned = generated_weights[:, 1:]
                assert np.array_equal(
                    target_aligned[np.arange(len(reveal)), reveal - 1],
                    reveal_weights,
                )
                for trajectory_id in np.unique(trajectory):
                    assert len(np.unique(split[trajectory == trajectory_id])) == 1

    report = audit_dataset(outputs["dfinal"], include_sha256=False)
    assert report["matched_pairs"]["pair_count"] == 10
    assert report["matched_pairs"]["all_invariants_pass"]
    assert report["matched_pairs"]["nuisance_auc"] <= 0.55
    assert report["all_required_acceptance_pass"]


def test_generation_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _make_source(source, episodes=20, image_size=24)
    build_variant(source, first, "dfinal", seed=77, pair_fraction=0.20, overwrite=False)
    build_variant(source, second, "dfinal", seed=77, pair_fraction=0.20, overwrite=False)
    with h5py.File(first, "r") as left, h5py.File(second, "r") as right:
        assert set(left.keys()) == set(right.keys())
        for key in left.keys():
            assert np.array_equal(left[key][:], right[key][:])

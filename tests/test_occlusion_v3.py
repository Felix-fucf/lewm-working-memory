from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from schema_tools.audit_occlusion_v3 import audit_dataset, audit_pair
from schema_tools.build_occlusion_v3 import (
    FRAME_LOSS_WEIGHT_POLICY,
    V3_SCHEDULES,
    build_dataset_pair,
    phase_relative_loss_weights_v3,
)
from schema_tools.primitive_common import EPISODE_LENGTH
from schema_tools.primitive_render import TRAIN_CONTENTS, render_object


def _make_source(path: Path, episodes: int = 200, image_size: int = 20) -> None:
    rng = np.random.default_rng(2031)
    total = episodes * EPISODE_LENGTH
    position = np.empty((episodes, EPISODE_LENGTH, 2), dtype=np.float32)
    velocity = np.empty_like(position)
    action = rng.normal(0.0, 0.01, size=(episodes, EPISODE_LENGTH, 3)).astype(np.float32)
    angle = np.empty((episodes, EPISODE_LENGTH), dtype=np.float32)
    pixels = np.zeros((total, image_size, image_size, 3), dtype=np.uint8)
    base_phase = np.asarray([3, 4, 3, 8, 4], dtype=np.int16)
    base_phase_id = np.repeat(np.arange(5, dtype=np.int8), base_phase)
    base_visible = np.isin(base_phase_id, (0, 2, 4)).astype(np.uint8)

    for episode in range(episodes):
        time = np.arange(EPISODE_LENGTH, dtype=np.float32)
        position[episode, :, 0] = 0.35 + 0.25 * (episode % 11) / 10 + 0.018 * np.sin(time / 4)
        position[episode, :, 1] = 0.42 + 0.16 * (episode % 7) / 6 + 0.012 * np.cos(time / 5)
        velocity[episode, 1:] = np.diff(position[episode], axis=0)
        velocity[episode, 0] = velocity[episode, 1]
        angle[episode] = (episode * 11 + time * 4) % 360
        content = TRAIN_CONTENTS[episode % len(TRAIN_CONTENTS)]
        for timestep in range(EPISODE_LENGTH):
            pixels[episode * EPISODE_LENGTH + timestep] = np.asarray(
                render_object(
                    content,
                    angle_deg=float(angle[episode, timestep]),
                    center_xy=tuple(position[episode, timestep]),
                    object_radius=0.155,
                    image_size=image_size,
                    visible=bool(base_visible[timestep]),
                )
            )

    def frames(values: np.ndarray) -> np.ndarray:
        return values.reshape((total, *values.shape[2:]))

    with h5py.File(path, "w") as file:
        file.attrs["episode_length"] = EPISODE_LENGTH
        file.attrs["image_size"] = image_size
        file.attrs["object_radius"] = 0.155
        file.create_dataset("ep_offset", data=np.arange(episodes) * EPISODE_LENGTH)
        file.create_dataset("ep_len", data=np.full(episodes, EPISODE_LENGTH, dtype=np.int16))
        file.create_dataset("pixels", data=pixels)
        file.create_dataset("action", data=frames(action))
        file.create_dataset("object_position", data=frames(position))
        file.create_dataset("object_velocity", data=frames(velocity))
        file.create_dataset("object_angle", data=angle.reshape(-1))
        angle_radians = np.radians(angle)
        orientation = np.stack(
            (np.cos(angle_radians), np.sin(angle_radians)),
            axis=-1,
        ).astype(np.float32)
        file.create_dataset("object_orientation", data=orientation.reshape(total, 2))
        file.create_dataset("object_angular_velocity", data=np.full(total, 4.0, dtype=np.float32))
        file.create_dataset("phase_lengths", data=np.tile(base_phase, (total, 1)))
        file.create_dataset("phase_id", data=np.tile(base_phase_id, episodes))
        file.create_dataset("object_visible", data=np.tile(base_visible, episodes))
        reappearance = np.zeros(EPISODE_LENGTH, dtype=np.uint8)
        reappearance[[7, 18]] = 1
        final = np.zeros(EPISODE_LENGTH, dtype=np.uint8)
        final[18] = 1
        file.create_dataset("is_reappearance", data=np.tile(reappearance, episodes))
        file.create_dataset("is_final_reappearance", data=np.tile(final, episodes))
        file.create_dataset("valid_mask", data=np.ones(total, dtype=np.uint8))
        starts = np.zeros((episodes, EPISODE_LENGTH), dtype=np.uint8)
        starts[:, 0] = 1
        file.create_dataset("episode_start", data=starts.reshape(-1))
        file.create_dataset("episode_id", data=np.repeat(np.arange(episodes), EPISODE_LENGTH))
        file.create_dataset("timestep", data=np.tile(np.arange(EPISODE_LENGTH), episodes))
        file.create_dataset("content_id", data=np.repeat(np.arange(episodes) % len(TRAIN_CONTENTS), EPISODE_LENGTH))
        file.create_dataset("shape_id", data=np.repeat(np.arange(episodes) % 8, EPISODE_LENGTH))
        file.create_dataset("color_id", data=np.repeat(np.arange(episodes) % 4, EPISODE_LENGTH))
        file.create_dataset("schema_id", data=np.zeros(total, dtype=np.int8))
        file.create_dataset("motion_type", data=np.repeat(np.arange(episodes) % 2, EPISODE_LENGTH))
        file.create_dataset("pair_id", data=np.full(total, -1, dtype=np.int64))
        file.create_dataset("pair_role", data=np.full(total, -1, dtype=np.int8))
        file.create_dataset("frame_loss_weight", data=np.ones(total, dtype=np.float32))


def test_five_schedule_weights_mask_only_random_second_onset() -> None:
    for schedule in V3_SCHEDULES:
        weights = phase_relative_loss_weights_v3(schedule)
        second_onset = sum(schedule[:3])
        reveal = sum(schedule[:4])
        assert len(weights) == EPISODE_LENGTH
        assert np.isclose(weights.sum(), 20.2, atol=1e-5)
        assert weights[second_onset] == 0.0
        assert weights[reveal] == 3.0
        # The first onset is fixed and therefore remains identifiable by time.
        assert weights[3] > 0.0


def test_v3_pair_generation_and_audit(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    output = tmp_path / "primitive_v3"
    _make_source(source)
    manifest = build_dataset_pair(
        source,
        output,
        seed=20260731,
        pair_fraction=0.20,
        minimum_eligible_episodes=0,
        overwrite=False,
    )
    original = output / "primitive_v3_h4_reveal5_original_motion.h5"
    motion = output / "primitive_v3_h4_reveal5_motion15x.h5"
    assert manifest["first_occlusion_decision"]["randomized"] is False
    original_report = audit_dataset(original, include_sha256=False)
    motion_report = audit_dataset(motion, include_sha256=False)
    pair_report = audit_pair(original, motion)
    assert original_report["all_required_acceptance_pass"], original_report["acceptance"]
    assert motion_report["all_required_acceptance_pass"], motion_report["acceptance"]
    assert pair_report["all_required_acceptance_pass"], pair_report["acceptance"]

    with h5py.File(original, "r") as left, h5py.File(motion, "r") as right:
        assert left.attrs["frame_loss_weight_policy"] == FRAME_LOSS_WEIGHT_POLICY
        assert int(left.attrs["first_occlusion_randomized"]) == 0
        assert np.array_equal(left["trajectory_id"][:], right["trajectory_id"][:])
        assert np.array_equal(left["phase_lengths"][:], right["phase_lengths"][:])
        assert left["object_orientation"].shape == (len(left["object_angle"]), 2)
        output_angle = np.radians(left["object_angle"][:])
        expected_orientation = np.stack(
            (np.cos(output_angle), np.sin(output_angle)), axis=-1
        )
        assert np.allclose(
            left["object_orientation"][:], expected_orientation, atol=2.0e-6
        )
        lengths = left["phase_lengths"][:].reshape(-1, EPISODE_LENGTH, 5)[:, 0]
        assert {tuple(row) for row in lengths} == set(V3_SCHEDULES)


def test_stop_gradient_changes_only_target_branch_gradient() -> None:
    context_encoder = torch.nn.Linear(3, 4)
    target_encoder = torch.nn.Linear(3, 4)
    predictor = torch.nn.Linear(4, 4)
    context_input = torch.randn(8, 3)
    target_input = torch.randn(8, 3)

    context = context_encoder(context_input)
    target = target_encoder(target_input)
    torch.nn.functional.mse_loss(predictor(context), target.detach()).backward()
    assert context_encoder.weight.grad is not None
    assert target_encoder.weight.grad is None

    context_encoder.zero_grad(set_to_none=True)
    target_encoder.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    context = context_encoder(context_input)
    target = target_encoder(target_input)
    torch.nn.functional.mse_loss(predictor(context), target).backward()
    assert context_encoder.weight.grad is not None
    assert target_encoder.weight.grad is not None

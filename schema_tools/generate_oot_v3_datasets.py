#!/usr/bin/env python3
"""
generate_oot_v3_datasets.py
===============================================================================

Occluded Object Transport V3：无可见平台、带噪声控制动力学。

核心语义
--------
画面中只有物体、两块遮挡板、背景和弱网格；不存在平台、机械手或其他可见
执行器。action 是抽象驱动器的二维归一化控制命令：

    action[t] = (u_x, u_y),  u_x,u_y in [-1,1]

它既不是物体真实位移，也不是事后测得的速度。环境通过带阻尼、执行增益误差、
控制偏置和时间相关过程噪声的动力学产生真实运动：

    executed_acceleration_t
        = acceleration_scale * actuator_gain * action_t
          + actuator_bias
          + process_noise_t

    velocity_(t+1)
        = damping * velocity_t + executed_acceleration_t

    position_(t+1)
        = position_t + velocity_(t+1)

因此通常：

    action[t] != executed_acceleration[t]
    action[t] != position[t+1] - position[t]

模型训练时默认只能读取 pixels 和 action。真实位置、速度、执行后加速度和过程
噪声只写入 HDF5 metadata，用于离线 probing 和数据自检。

时间语义
--------
frame t 展示状态 s_t；action[t] 是在看到 frame t 后发出的控制命令，用于产生
frame t+1。最后一帧没有下一帧，因此 action[T-1]=(0,0)。

Violation
---------
test_paired_normal 与 test_paired_violation 共享相同 action、执行器参数、过程
噪声、正常速度轨迹、遮挡时序和视觉参数。violation 只在第二次遮挡期间对物体
位置加入一次未被 action 解释的 y 方向跳变，不改变 x、再现时刻和遮挡长度。

中断安全
--------
每个 split 先写入 *.h5.tmp，完整生成并自检后再原子重命名为 *.h5。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image, ImageDraw


EPISODE_LENGTH = 28
ACTION_DIM = 2

PHASE_NAMES = {
    0: "visible_before_first_occluder",
    1: "occluded_by_first_occluder",
    2: "visible_between_occluders",
    3: "occluded_by_second_occluder",
    4: "visible_after_second_occluder",
}

SCHEMA_NAMES = {0: "hidden_actuator_controlled_transport"}
PAIR_ROLE_NAMES = {
    0: "unpaired_or_training",
    1: "paired_normal",
    2: "paired_violation",
}


@dataclass(frozen=True)
class ObjectContent:
    content_id: int
    shape_id: int
    color_id: int
    shape_name: str
    rgb: Tuple[int, int, int]


TRAIN_CONTENTS = (
    ObjectContent(0, 0, 0, "circle", (224, 62, 62)),
    ObjectContent(1, 1, 1, "square", (48, 152, 92)),
    ObjectContent(2, 2, 2, "diamond", (55, 108, 215)),
)

NEW_CONTENTS = (
    ObjectContent(3, 3, 3, "triangle", (238, 181, 48)),
    ObjectContent(4, 0, 4, "circle", (155, 82, 205)),
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    num_episodes: int
    contents: Tuple[ObjectContent, ...]
    long_second_occlusion: bool = False
    paired: bool = False
    violation: bool = False
    seed_offset: int = 0


@dataclass
class DynamicsParameters:
    actuator_gain: np.ndarray
    damping: np.ndarray
    actuator_bias: np.ndarray
    noise_beta: float
    noise_sigma: np.ndarray
    acceleration_scale: np.ndarray


@dataclass
class BaseEpisode:
    action: np.ndarray
    executed_acceleration: np.ndarray
    process_noise: np.ndarray
    object_position: np.ndarray
    object_velocity: np.ndarray
    actual_object_displacement: np.ndarray
    phase_lengths: np.ndarray
    phase_id: np.ndarray
    object_visible: np.ndarray
    occlusion_id: np.ndarray
    is_reappearance: np.ndarray
    is_final_reappearance: np.ndarray
    final_reveal_timestep: int
    occluders: Tuple[Tuple[float, float, float, float], ...]
    direction_id: int
    content: ObjectContent
    object_radius: float
    background_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    occluder_rgb: Tuple[int, int, int]
    dynamics: DynamicsParameters
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    goal_position: np.ndarray


# =============================================================================
# Phase schedule
# =============================================================================

def sample_phase_lengths(
    rng: np.random.Generator,
    long_second_occlusion: bool,
) -> np.ndarray:
    """随机五段长度，总和固定为 28。"""
    ranges = (
        ((4, 6), (3, 5), (2, 4), (9, 12), (4, 6))
        if long_second_occlusion
        else ((4, 7), (4, 7), (2, 5), (5, 9), (4, 7))
    )

    for _ in range(10_000):
        lengths = np.array(
            [rng.integers(low, high + 1) for low, high in ranges],
            dtype=np.int16,
        )
        if int(lengths.sum()) == EPISODE_LENGTH:
            return lengths

    raise RuntimeError("无法采样到总长为 28 的 phase schedule。")


def phase_ids_from_lengths(lengths: np.ndarray) -> np.ndarray:
    phase_id = np.empty(EPISODE_LENGTH, dtype=np.int8)
    cursor = 0
    for phase, length in enumerate(lengths.tolist()):
        phase_id[cursor : cursor + length] = phase
        cursor += length
    if cursor != EPISODE_LENGTH:
        raise RuntimeError("phase lengths 总和错误。")
    return phase_id


def make_reappearance_flags(
    visible: np.ndarray,
    phase_id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    reappearance = np.zeros(EPISODE_LENGTH, dtype=np.uint8)
    final_reappearance = np.zeros(EPISODE_LENGTH, dtype=np.uint8)

    for t in range(1, EPISODE_LENGTH):
        if visible[t - 1] == 0 and visible[t] == 1:
            reappearance[t] = 1
            if phase_id[t] == 4:
                final_reappearance[t] = 1

    return reappearance, final_reappearance


# =============================================================================
# Hidden actuator dynamics
# =============================================================================

def sample_dynamics_parameters(
    rng: np.random.Generator,
) -> DynamicsParameters:
    """
    正常噪声保持较小：足以使 action 不等于真实转移，但不会淹没可学习规律。
    """
    return DynamicsParameters(
        actuator_gain=rng.uniform(0.90, 1.10, size=2).astype(np.float32),
        damping=rng.uniform(0.885, 0.935, size=2).astype(np.float32),
        actuator_bias=rng.normal(
            0.0,
            [0.00035, 0.00025],
            size=2,
        ).astype(np.float32),
        noise_beta=float(rng.uniform(0.78, 0.90)),
        noise_sigma=rng.uniform(
            [0.00035, 0.00020],
            [0.00085, 0.00055],
        ).astype(np.float32),
        acceleration_scale=rng.uniform(
            [0.015, 0.012],
            [0.020, 0.017],
        ).astype(np.float32),
    )


def simulate_hidden_actuator(
    rng: np.random.Generator,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    goal_position: np.ndarray,
    dynamics: DynamicsParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    expert controller 依据隐藏状态产生 action；真实环境带噪执行 action。

    controller 能读真实状态只是为了生成离线轨迹。world model 训练时不能读取
    这些状态，只能看到 pixels 与 action。
    """
    position = np.zeros((EPISODE_LENGTH, 2), dtype=np.float32)
    velocity = np.zeros((EPISODE_LENGTH, 2), dtype=np.float32)
    action = np.zeros((EPISODE_LENGTH, 2), dtype=np.float32)
    executed_acceleration = np.zeros((EPISODE_LENGTH, 2), dtype=np.float32)
    process_noise = np.zeros((EPISODE_LENGTH, 2), dtype=np.float32)

    position[0] = initial_position
    velocity[0] = initial_velocity

    desired_vx = (
        goal_position[0] - initial_position[0]
    ) / (EPISODE_LENGTH - 1)
    target_lane_y = float(goal_position[1])

    policy_noise_state = np.zeros(2, dtype=np.float32)
    process_noise_state = np.zeros(2, dtype=np.float32)

    for t in range(EPISODE_LENGTH - 1):
        q = position[t]
        v = velocity[t]

        # controller 使用名义参数；真实执行器参数会略有偏差。
        nominal_damping = np.array([0.91, 0.91], dtype=np.float32)
        nominal_scale = np.array([0.0175, 0.0145], dtype=np.float32)

        feedforward_x = (
            (1.0 - nominal_damping[0]) * desired_vx / nominal_scale[0]
        )

        command_x = (
            feedforward_x
            + 9.0 * (desired_vx - v[0])
            + 0.08 * (goal_position[0] - q[0])
        )
        command_y = 2.2 * (target_lane_y - q[1]) - 5.5 * v[1]

        # policy 自身的平滑探索变化属于 action，模型可以观察到。
        policy_noise_state = (
            0.82 * policy_noise_state
            + rng.normal(0.0, [0.018, 0.025]).astype(np.float32)
        )

        command = np.array([command_x, command_y], dtype=np.float32)
        command = np.clip(command + policy_noise_state, -1.0, 1.0)
        action[t] = command

        # AR(1) 过程噪声属于环境内部，模型看不到。
        innovation = rng.normal(
            0.0,
            dynamics.noise_sigma,
            size=2,
        ).astype(np.float32)

        beta = dynamics.noise_beta
        process_noise_state = (
            beta * process_noise_state
            + math.sqrt(1.0 - beta * beta) * innovation
        )
        process_noise[t] = process_noise_state

        actual_acceleration = (
            dynamics.acceleration_scale
            * dynamics.actuator_gain
            * command
            + dynamics.actuator_bias
            + process_noise_state
        )
        executed_acceleration[t] = actual_acceleration

        velocity[t + 1] = dynamics.damping * v + actual_acceleration
        position[t + 1] = q + velocity[t + 1]

    action[-1] = 0.0
    executed_acceleration[-1] = 0.0
    process_noise[-1] = process_noise_state

    return action, executed_acceleration, process_noise, position, velocity


def compute_displacement(position: np.ndarray) -> np.ndarray:
    displacement = np.zeros_like(position, dtype=np.float32)
    displacement[:-1] = position[1:] - position[:-1]
    displacement[-1] = 0.0
    return displacement


# =============================================================================
# Occlusion geometry
# =============================================================================

def interval_for_segment(
    x: np.ndarray,
    radius: float,
    start: int,
    end: int,
    direction_id: int,
) -> Tuple[float, float]:
    """构造一块竖直遮挡板，使 start..end 恰好为遮挡帧。"""
    if start <= 0 or end >= EPISODE_LENGTH - 1:
        raise ValueError("遮挡段前后必须保留可见帧。")

    if direction_id == 0:  # 左 -> 右
        edge_a = 0.5 * ((x[start - 1] + radius) + (x[start] + radius))
        edge_b = 0.5 * ((x[end] - radius) + (x[end + 1] - radius))
    else:  # 右 -> 左
        edge_b = 0.5 * ((x[start - 1] - radius) + (x[start] - radius))
        edge_a = 0.5 * ((x[end] + radius) + (x[end + 1] + radius))

    left, right = float(min(edge_a, edge_b)), float(max(edge_a, edge_b))
    width = right - left

    if not 0.055 <= width <= 0.34:
        raise RuntimeError(f"遮挡板宽度不合理：{width:.4f}")
    if left < 0.02 or right > 0.98:
        raise RuntimeError("遮挡板越出画面。")

    return left, right


def visibility_from_geometry(
    position: np.ndarray,
    radius: float,
    occluders: Sequence[Tuple[float, float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    visible = np.ones(EPISODE_LENGTH, dtype=np.uint8)
    occlusion_id = np.zeros(EPISODE_LENGTH, dtype=np.int8)

    for t, (x, y) in enumerate(position):
        for index, (x0, x1, y0, y1) in enumerate(occluders, start=1):
            overlap_x = x + radius >= x0 and x - radius <= x1
            overlap_y = y + radius >= y0 and y - radius <= y1
            if overlap_x and overlap_y:
                visible[t] = 0
                occlusion_id[t] = index
                break

    return visible, occlusion_id


def build_occlusion_geometry(
    position: np.ndarray,
    radius: float,
    phase_lengths: np.ndarray,
    direction_id: int,
):
    phase_id = phase_ids_from_lengths(phase_lengths)
    p0, p1, p2, p3, _p4 = phase_lengths.tolist()

    first_start = p0
    first_end = p0 + p1 - 1
    second_start = p0 + p1 + p2
    second_end = second_start + p3 - 1

    first_left, first_right = interval_for_segment(
        position[:, 0], radius, first_start, first_end, direction_id
    )
    second_left, second_right = interval_for_segment(
        position[:, 0], radius, second_start, second_end, direction_id
    )

    # y 覆盖较大，确保 y-only violation 不改变再现时刻。
    occluders = (
        (first_left, first_right, 0.08, 0.92),
        (second_left, second_right, 0.08, 0.92),
    )

    visible, occlusion_id = visibility_from_geometry(
        position,
        radius,
        occluders,
    )

    expected_visible = np.isin(phase_id, [0, 2, 4]).astype(np.uint8)
    if not np.array_equal(visible, expected_visible):
        raise RuntimeError("几何 visibility 与 phase schedule 不一致。")

    expected_occlusion_id = np.zeros(EPISODE_LENGTH, dtype=np.int8)
    expected_occlusion_id[phase_id == 1] = 1
    expected_occlusion_id[phase_id == 3] = 2
    if not np.array_equal(occlusion_id, expected_occlusion_id):
        raise RuntimeError("occlusion_id 与 phase schedule 不一致。")

    reappearance, final_reappearance = make_reappearance_flags(
        visible,
        phase_id,
    )
    final_indices = np.flatnonzero(final_reappearance == 1)
    if len(final_indices) != 1:
        raise RuntimeError("最终再现必须恰好出现一次。")

    return (
        occluders,
        phase_id,
        visible,
        occlusion_id,
        reappearance,
        final_reappearance,
        int(final_indices[0]),
    )


# =============================================================================
# Rendering: no visible actuator
# =============================================================================

def normalized_to_pixel(value: float, image_size: int) -> int:
    return int(round(value * (image_size - 1)))


def draw_object(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    radius: int,
    content: ObjectContent,
) -> None:
    color = tuple(int(v) for v in content.rgb)
    outline = (35, 35, 35)

    if content.shape_name == "circle":
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=outline,
            width=max(1, radius // 4),
        )
    elif content.shape_name == "square":
        draw.rectangle(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=outline,
            width=max(1, radius // 4),
        )
    elif content.shape_name == "diamond":
        draw.polygon(
            [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)],
            fill=color,
            outline=outline,
        )
    elif content.shape_name == "triangle":
        draw.polygon(
            [(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)],
            fill=color,
            outline=outline,
        )
    else:
        raise ValueError(f"未知 shape：{content.shape_name}")


def render_frame(
    image_size: int,
    position: np.ndarray,
    visible: bool,
    radius: float,
    content: ObjectContent,
    occluders: Sequence[Tuple[float, float, float, float]],
    background_rgb: Tuple[int, int, int],
    grid_rgb: Tuple[int, int, int],
    occluder_rgb: Tuple[int, int, int],
) -> np.ndarray:
    """
    画面中没有平台或执行器。

    渲染顺序：背景 -> 弱网格 -> 物体 -> 遮挡板。
    """
    image = Image.new("RGB", (image_size, image_size), color=background_rgb)
    draw = ImageDraw.Draw(image)

    for fraction in (0.25, 0.50, 0.75):
        p = normalized_to_pixel(fraction, image_size)
        draw.line([(p, 0), (p, image_size - 1)], fill=grid_rgb, width=1)
        draw.line([(0, p), (image_size - 1, p)], fill=grid_rgb, width=1)

    # 对被标记为遮挡的帧直接不画物体。这样不会在遮挡板边缘泄漏
    # 少量像素，也保证 paired normal/violation 在最终再现前逐像素一致。
    if visible:
        x = normalized_to_pixel(float(position[0]), image_size)
        y = normalized_to_pixel(float(position[1]), image_size)
        radius_px = max(3, normalized_to_pixel(radius, image_size))
        draw_object(draw, x, y, radius_px, content)

    for x0, x1, y0, y1 in occluders:
        draw.rectangle(
            [
                normalized_to_pixel(x0, image_size),
                normalized_to_pixel(y0, image_size),
                normalized_to_pixel(x1, image_size),
                normalized_to_pixel(y1, image_size),
            ],
            fill=occluder_rgb,
            outline=(50, 50, 55),
            width=max(1, image_size // 128),
        )

    return np.asarray(image, dtype=np.uint8)


# =============================================================================
# Episode generation
# =============================================================================

def generate_base_episode(
    rng: np.random.Generator,
    split: SplitSpec,
) -> BaseEpisode:
    """Rejection sampling：只保留满足完整 OOT 几何结构的轨迹。"""
    for _attempt in range(3000):
        direction_id = int(rng.integers(0, 2))
        content = split.contents[int(rng.integers(0, len(split.contents)))]
        radius = float(rng.uniform(0.025, 0.034))
        lane_y = float(rng.uniform(0.38, 0.62))

        if direction_id == 0:
            initial_x = float(rng.uniform(0.045, 0.085))
            goal_x = float(rng.uniform(0.915, 0.955))
        else:
            initial_x = float(rng.uniform(0.915, 0.955))
            goal_x = float(rng.uniform(0.045, 0.085))

        initial_y = float(
            np.clip(lane_y + rng.normal(0.0, 0.020), 0.30, 0.70)
        )
        initial_position = np.array([initial_x, initial_y], dtype=np.float32)
        goal_position = np.array([goal_x, lane_y], dtype=np.float32)

        desired_vx = (goal_x - initial_x) / (EPISODE_LENGTH - 1)
        initial_velocity = np.array(
            [
                desired_vx * rng.uniform(0.75, 1.15),
                rng.normal(0.0, 0.0025),
            ],
            dtype=np.float32,
        )

        dynamics = sample_dynamics_parameters(rng)
        (
            action,
            executed_acceleration,
            process_noise,
            position,
            velocity,
        ) = simulate_hidden_actuator(
            rng,
            initial_position,
            initial_velocity,
            goal_position,
            dynamics,
        )

        dx = np.diff(position[:, 0])
        if direction_id == 0 and np.min(dx) <= 0.001:
            continue
        if direction_id == 1 and np.max(dx) >= -0.001:
            continue

        if np.min(position[:, 0]) < 0.015 or np.max(position[:, 0]) > 0.985:
            continue
        if np.min(position[:, 1]) < 0.20 or np.max(position[:, 1]) > 0.80:
            continue

        phase_lengths = sample_phase_lengths(
            rng,
            split.long_second_occlusion,
        )

        try:
            (
                occluders,
                phase_id,
                visible,
                occlusion_id,
                reappearance,
                final_reappearance,
                final_reveal_timestep,
            ) = build_occlusion_geometry(
                position,
                radius,
                phase_lengths,
                direction_id,
            )
        except RuntimeError:
            continue

        background_base = int(rng.integers(224, 239))
        background_rgb = (
            background_base,
            background_base,
            min(255, background_base + 3),
        )
        grid_rgb = tuple(min(255, value + 10) for value in background_rgb)

        occluder_base = int(rng.integers(108, 140))
        occluder_rgb = (
            occluder_base,
            occluder_base + 2,
            occluder_base + 7,
        )

        return BaseEpisode(
            action=action,
            executed_acceleration=executed_acceleration,
            process_noise=process_noise,
            object_position=position,
            object_velocity=velocity,
            actual_object_displacement=compute_displacement(position),
            phase_lengths=phase_lengths,
            phase_id=phase_id,
            object_visible=visible,
            occlusion_id=occlusion_id,
            is_reappearance=reappearance,
            is_final_reappearance=final_reappearance,
            final_reveal_timestep=final_reveal_timestep,
            occluders=occluders,
            direction_id=direction_id,
            content=content,
            object_radius=radius,
            background_rgb=background_rgb,
            grid_rgb=grid_rgb,
            occluder_rgb=occluder_rgb,
            dynamics=dynamics,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            goal_position=goal_position,
        )

    raise RuntimeError("连续 3000 次仍无法生成满足约束的 episode。")


def materialize_episode(
    base: BaseEpisode,
    image_size: int,
    pair_id: int,
    pair_role: int,
    violation: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    将 BaseEpisode 渲染为 normal 或 violation。

    violation 是位置 teleport：不改变 action、过程噪声或后续正常速度。
    """
    position = base.object_position.copy()
    violation_delta = np.zeros((EPISODE_LENGTH, 2), dtype=np.float32)
    violation_event_timestep = -1

    if violation:
        second_occlusion = np.flatnonzero(base.phase_id == 3)
        violation_event_timestep = int(
            second_occlusion[len(second_occlusion) // 2]
        )
        delta_y = float(
            rng.choice([-1.0, 1.0]) * rng.uniform(0.105, 0.145)
        )
        if not 0.24 <= position[-1, 1] + delta_y <= 0.76:
            delta_y = -delta_y

        position[violation_event_timestep:, 1] += delta_y
        violation_delta[violation_event_timestep:, 1] = delta_y

    visible, occlusion_id = visibility_from_geometry(
        position,
        base.object_radius,
        base.occluders,
    )
    if not np.array_equal(visible, base.object_visible):
        raise RuntimeError("violation 不应改变 visibility。")
    if not np.array_equal(occlusion_id, base.occlusion_id):
        raise RuntimeError("violation 不应改变 occlusion_id。")

    pixels = np.empty(
        (EPISODE_LENGTH, image_size, image_size, 3),
        dtype=np.uint8,
    )
    for t in range(EPISODE_LENGTH):
        pixels[t] = render_frame(
            image_size,
            position[t],
            bool(base.object_visible[t]),
            base.object_radius,
            base.content,
            base.occluders,
            base.background_rgb,
            base.grid_rgb,
            base.occluder_rgb,
        )

    actual_displacement = compute_displacement(position)

    def repeat_scalar(value, dtype):
        return np.full(EPISODE_LENGTH, value, dtype=dtype)

    def repeat_vec2(value):
        return np.repeat(
            np.asarray(value, dtype=np.float32)[None, :],
            EPISODE_LENGTH,
            axis=0,
        )

    observation = np.zeros((EPISODE_LENGTH, 10), dtype=np.float32)
    observation[:, 0:2] = position
    observation[:, 2:4] = base.object_velocity
    observation[:, 4:6] = base.action
    observation[:, 6] = base.object_visible
    observation[:, 7] = base.phase_id / 4.0
    observation[:, 8] = np.linspace(0.0, 1.0, EPISODE_LENGTH, dtype=np.float32)
    observation[:, 9] = float(violation)

    distance_to_target = np.linalg.norm(
        position - position[-1],
        axis=-1,
    ).astype(np.float32)

    reward = np.zeros(EPISODE_LENGTH, dtype=np.float32)
    reward[-1] = 1.0 if not violation else 0.0

    valid_mask = np.ones(EPISODE_LENGTH, dtype=np.uint8)
    episode_start = np.zeros(EPISODE_LENGTH, dtype=np.uint8)
    episode_start[0] = 1
    terminated = np.zeros(EPISODE_LENGTH, dtype=np.bool_)
    terminated[-1] = True

    return {
        "pixels": pixels,
        "action": base.action.copy(),
        "executed_acceleration": base.executed_acceleration.copy(),
        "process_noise": base.process_noise.copy(),
        "object_position": position,
        "object_velocity": base.object_velocity.copy(),
        "actual_object_displacement": actual_displacement,
        "violation_delta": violation_delta,
        "phase_lengths": np.repeat(
            base.phase_lengths[None, :],
            EPISODE_LENGTH,
            axis=0,
        ).astype(np.int16),
        "phase_id": base.phase_id.copy(),
        "object_visible": base.object_visible.copy(),
        "occlusion_id": base.occlusion_id.copy(),
        "is_reappearance": base.is_reappearance.copy(),
        "is_final_reappearance": base.is_final_reappearance.copy(),
        "final_reveal_timestep": repeat_scalar(
            base.final_reveal_timestep,
            np.int16,
        ),
        "content_id": repeat_scalar(base.content.content_id, np.int16),
        "shape_id": repeat_scalar(base.content.shape_id, np.int8),
        "color_id": repeat_scalar(base.content.color_id, np.int16),
        "direction_id": repeat_scalar(base.direction_id, np.int8),
        "schema_id": repeat_scalar(0, np.int8),
        "is_violation": repeat_scalar(int(violation), np.uint8),
        "pair_id": repeat_scalar(pair_id, np.int64),
        "pair_role": repeat_scalar(pair_role, np.int8),
        "violation_event_timestep": repeat_scalar(
            violation_event_timestep,
            np.int16,
        ),
        "actuator_gain": repeat_vec2(base.dynamics.actuator_gain),
        "dynamics_damping": repeat_vec2(base.dynamics.damping),
        "actuator_bias": repeat_vec2(base.dynamics.actuator_bias),
        "noise_beta": repeat_scalar(base.dynamics.noise_beta, np.float32),
        "noise_sigma": repeat_vec2(base.dynamics.noise_sigma),
        "acceleration_scale": repeat_vec2(base.dynamics.acceleration_scale),
        "initial_position": repeat_vec2(base.initial_position),
        "initial_velocity": repeat_vec2(base.initial_velocity),
        "goal_position": repeat_vec2(base.goal_position),
        "occluder_1": np.repeat(
            np.asarray(base.occluders[0], dtype=np.float32)[None, :],
            EPISODE_LENGTH,
            axis=0,
        ),
        "occluder_2": np.repeat(
            np.asarray(base.occluders[1], dtype=np.float32)[None, :],
            EPISODE_LENGTH,
            axis=0,
        ),
        "observation": observation,
        "pos_agent": position.copy(),
        "pos_target": repeat_vec2(base.goal_position),
        "proprio": position.copy(),
        "distance_to_target": distance_to_target,
        "reward": reward,
        "valid_mask": valid_mask,
        "episode_start": episode_start,
        "terminated": terminated,
        "truncated": np.zeros(EPISODE_LENGTH, dtype=np.bool_),
    }


# =============================================================================
# HDF5
# =============================================================================

VECTOR2_FIELDS = (
    "action",
    "executed_acceleration",
    "process_noise",
    "object_position",
    "object_velocity",
    "actual_object_displacement",
    "violation_delta",
    "actuator_gain",
    "dynamics_damping",
    "actuator_bias",
    "noise_sigma",
    "acceleration_scale",
    "initial_position",
    "initial_velocity",
    "goal_position",
    "pos_agent",
    "pos_target",
    "proprio",
)

SCALAR_FIELDS = {
    "phase_id": np.int8,
    "object_visible": np.uint8,
    "occlusion_id": np.int8,
    "is_reappearance": np.uint8,
    "is_final_reappearance": np.uint8,
    "final_reveal_timestep": np.int16,
    "content_id": np.int16,
    "shape_id": np.int8,
    "color_id": np.int16,
    "direction_id": np.int8,
    "schema_id": np.int8,
    "is_violation": np.uint8,
    "pair_id": np.int64,
    "pair_role": np.int8,
    "violation_event_timestep": np.int16,
    "noise_beta": np.float32,
    "distance_to_target": np.float32,
    "reward": np.float32,
    "valid_mask": np.uint8,
    "episode_start": np.uint8,
    "terminated": np.bool_,
    "truncated": np.bool_,
    "episode_id": np.int64,
    "ep_idx": np.int32,
    "step_idx": np.int16,
    "timestep": np.int16,
    "id": np.int64,
}


def create_dataset(
    file: h5py.File,
    name: str,
    total_frames: int,
    frame_shape: Tuple[int, ...],
    dtype,
    compression: Optional[str] = None,
) -> h5py.Dataset:
    chunks = (
        (1, *frame_shape)
        if name == "pixels"
        else (min(total_frames, 4096), *frame_shape)
    )
    return file.create_dataset(
        name,
        shape=(total_frames, *frame_shape),
        dtype=dtype,
        chunks=chunks,
        compression=compression,
        shuffle=compression is not None,
    )


def initialize_hdf5(
    file: h5py.File,
    split: SplitSpec,
    image_size: int,
    seed: int,
) -> Dict[str, h5py.Dataset]:
    total_frames = split.num_episodes * EPISODE_LENGTH

    file.attrs["dataset_name"] = "occluded_object_transport_v3"
    file.attrs["split_name"] = split.name
    file.attrs["num_episodes"] = split.num_episodes
    file.attrs["episode_length"] = EPISODE_LENGTH
    file.attrs["image_size"] = image_size
    file.attrs["seed"] = seed
    file.attrs["completed"] = 0
    file.attrs["visible_actuator"] = 0
    file.attrs["action_semantics"] = (
        "normalized hidden-actuator 2D acceleration/control command; "
        "not actual object displacement"
    )
    file.attrs["phase_names_json"] = json.dumps(PHASE_NAMES)
    file.attrs["schema_names_json"] = json.dumps(SCHEMA_NAMES)
    file.attrs["pair_role_names_json"] = json.dumps(PAIR_ROLE_NAMES)
    file.attrs["contents_json"] = json.dumps(
        [asdict(content) for content in split.contents]
    )
    file.attrs["long_second_occlusion"] = int(split.long_second_occlusion)
    file.attrs["paired"] = int(split.paired)
    file.attrs["violation"] = int(split.violation)

    datasets: Dict[str, h5py.Dataset] = {}
    datasets["pixels"] = create_dataset(
        file,
        "pixels",
        total_frames,
        (image_size, image_size, 3),
        np.uint8,
        compression="lzf",
    )

    for name in VECTOR2_FIELDS:
        datasets[name] = create_dataset(
            file,
            name,
            total_frames,
            (2,),
            np.float32,
        )

    datasets["phase_lengths"] = create_dataset(
        file,
        "phase_lengths",
        total_frames,
        (5,),
        np.int16,
    )
    datasets["occluder_1"] = create_dataset(
        file,
        "occluder_1",
        total_frames,
        (4,),
        np.float32,
    )
    datasets["occluder_2"] = create_dataset(
        file,
        "occluder_2",
        total_frames,
        (4,),
        np.float32,
    )
    datasets["observation"] = create_dataset(
        file,
        "observation",
        total_frames,
        (10,),
        np.float32,
    )

    for name, dtype in SCALAR_FIELDS.items():
        if name not in datasets:
            datasets[name] = create_dataset(
                file,
                name,
                total_frames,
                (),
                dtype,
            )

    datasets["ep_len"] = file.create_dataset(
        "ep_len",
        shape=(split.num_episodes,),
        dtype=np.int32,
    )
    datasets["ep_offset"] = file.create_dataset(
        "ep_offset",
        shape=(split.num_episodes,),
        dtype=np.int64,
    )
    datasets["ep_len"][:] = EPISODE_LENGTH
    datasets["ep_offset"][:] = (
        np.arange(split.num_episodes, dtype=np.int64) * EPISODE_LENGTH
    )

    return datasets


def write_episode(
    datasets: Dict[str, h5py.Dataset],
    episode: Dict[str, np.ndarray],
    episode_index: int,
) -> None:
    start = episode_index * EPISODE_LENGTH
    end = start + EPISODE_LENGTH
    frame_slice = slice(start, end)

    for name, value in episode.items():
        datasets[name][frame_slice] = value

    timesteps = np.arange(EPISODE_LENGTH, dtype=np.int16)
    datasets["episode_id"][frame_slice] = episode_index
    datasets["ep_idx"][frame_slice] = episode_index
    datasets["step_idx"][frame_slice] = timesteps
    datasets["timestep"][frame_slice] = timesteps
    datasets["id"][frame_slice] = np.arange(start, end, dtype=np.int64)


# =============================================================================
# Preview and validation
# =============================================================================

def save_contact_sheet(
    episode: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    frames: List[Image.Image] = []

    for t in range(EPISODE_LENGTH):
        frame = Image.fromarray(episode["pixels"][t])
        canvas = Image.new("RGB", (frame.width, frame.height + 28), (245, 245, 245))
        canvas.paste(frame, (0, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (4, 5),
            (
                f"t={t:02d} p={int(episode['phase_id'][t])} "
                f"v={int(episode['object_visible'][t])} "
                f"ux={episode['action'][t,0]:+.2f}"
            ),
            fill=(20, 20, 20),
        )
        frames.append(canvas)

    columns, rows = 7, 4
    cell_w, cell_h = frames[0].size
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (255, 255, 255))

    for index, frame in enumerate(frames):
        sheet.paste(
            frame,
            ((index % columns) * cell_w, (index // columns) * cell_h),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def validate_hdf5(
    path: Path,
    split: SplitSpec,
    image_size: int,
) -> Dict[str, object]:
    expected_frames = split.num_episodes * EPISODE_LENGTH

    with h5py.File(path, "r") as file:
        if int(file.attrs.get("completed", 0)) != 1:
            raise RuntimeError(f"{path} 尚未 completed。")
        if int(file.attrs.get("visible_actuator", 1)) != 0:
            raise RuntimeError("V3 不应包含可见执行器。")
        if file["pixels"].shape != (expected_frames, image_size, image_size, 3):
            raise RuntimeError("pixels shape 错误。")
        if file["action"].shape != (expected_frames, ACTION_DIM):
            raise RuntimeError("action shape 错误。")
        if len(np.unique(file["episode_id"][:])) != split.num_episodes:
            raise RuntimeError("episode_id 数量错误。")
        if not np.all(file["ep_len"][:] == EPISODE_LENGTH):
            raise RuntimeError("ep_len 不全为 28。")

        schedules = []
        reveal_times = []
        initial_x = []
        initial_vx = []
        action_equals_displacement = 0
        action_equals_acceleration = 0
        direction_counts = {"0": 0, "1": 0}

        for episode_index in range(split.num_episodes):
            start = episode_index * EPISODE_LENGTH
            end = start + EPISODE_LENGTH

            phase = file["phase_id"][start:end]
            visible = file["object_visible"][start:end]
            reappearance = file["is_reappearance"][start:end]
            final_flag = file["is_final_reappearance"][start:end]

            if set(np.unique(phase).tolist()) != {0, 1, 2, 3, 4}:
                raise RuntimeError(f"episode {episode_index} phase 不完整。")
            if int(reappearance.sum()) != 2:
                raise RuntimeError(f"episode {episode_index} 再现次数错误。")
            if int(final_flag.sum()) != 1:
                raise RuntimeError(f"episode {episode_index} 最终再现错误。")

            final_t = int(np.flatnonzero(final_flag == 1)[0])
            if EPISODE_LENGTH - final_t < 4:
                raise RuntimeError("最终可见帧不足 4。")
            if not np.array_equal(
                visible,
                np.isin(phase, [0, 2, 4]).astype(np.uint8),
            ):
                raise RuntimeError("visibility 与 phase 不一致。")

            schedules.append(tuple(file["phase_lengths"][start].tolist()))
            reveal_times.append(final_t)
            initial_x.append(float(file["object_position"][start, 0]))
            initial_vx.append(float(file["object_velocity"][start, 0]))

            action = file["action"][start : end - 1]
            displacement = file["actual_object_displacement"][start : end - 1]
            acceleration = file["executed_acceleration"][start : end - 1]

            if np.allclose(action, displacement, atol=1e-7, rtol=1e-7):
                action_equals_displacement += 1
            if np.allclose(action, acceleration, atol=1e-7, rtol=1e-7):
                action_equals_acceleration += 1

            direction_counts[str(int(file["direction_id"][start]))] += 1

        return {
            "path": str(path),
            "num_episodes": split.num_episodes,
            "num_frames": expected_frames,
            "pixels_shape": list(file["pixels"].shape),
            "action_shape": list(file["action"].shape),
            "unique_phase_schedules": len(set(schedules)),
            "unique_final_reveal_timesteps": sorted(set(reveal_times)),
            "initial_x_std": float(np.std(initial_x)),
            "initial_vx_std": float(np.std(initial_vx)),
            "direction_counts": direction_counts,
            "episodes_where_action_equals_displacement": action_equals_displacement,
            "episodes_where_action_equals_executed_acceleration": (
                action_equals_acceleration
            ),
        }


def validate_paired_files(
    normal_path: Path,
    violation_path: Path,
    num_episodes: int,
) -> Dict[str, object]:
    with h5py.File(normal_path, "r") as normal, h5py.File(
        violation_path,
        "r",
    ) as violation:
        checks = {
            "action_identical": np.array_equal(normal["action"][:], violation["action"][:]),
            "executed_acceleration_identical": np.array_equal(
                normal["executed_acceleration"][:],
                violation["executed_acceleration"][:],
            ),
            "process_noise_identical": np.array_equal(
                normal["process_noise"][:],
                violation["process_noise"][:],
            ),
            "velocity_identical": np.array_equal(
                normal["object_velocity"][:],
                violation["object_velocity"][:],
            ),
            "phase_timing_identical": np.array_equal(
                normal["phase_id"][:],
                violation["phase_id"][:],
            ),
            "visibility_identical": np.array_equal(
                normal["object_visible"][:],
                violation["object_visible"][:],
            ),
            "object_x_identical": np.array_equal(
                normal["object_position"][:, 0],
                violation["object_position"][:, 0],
            ),
        }

        pixels_before_reveal_identical = True
        y_shifts = []

        for episode_index in range(num_episodes):
            start = episode_index * EPISODE_LENGTH
            end = start + EPISODE_LENGTH
            final_t = int(normal["final_reveal_timestep"][start])

            if not np.array_equal(
                normal["pixels"][start : start + final_t],
                violation["pixels"][start : start + final_t],
            ):
                pixels_before_reveal_identical = False

            y_shifts.append(
                float(
                    violation["object_position"][end - 1, 1]
                    - normal["object_position"][end - 1, 1]
                )
            )

        checks["pixels_before_final_reveal_identical"] = (
            pixels_before_reveal_identical
        )

        if not all(checks.values()):
            raise RuntimeError(f"paired validation 失败：{checks}")

        return {
            **checks,
            "mean_abs_final_y_shift": float(np.mean(np.abs(y_shifts))),
            "min_abs_final_y_shift": float(np.min(np.abs(y_shifts))),
            "max_abs_final_y_shift": float(np.max(np.abs(y_shifts))),
        }


# =============================================================================
# Split generation and CLI
# =============================================================================

def generate_split(
    split: SplitSpec,
    output_dir: Path,
    image_size: int,
    base_seed: int,
    overwrite: bool,
    preview_episodes: int,
    progress_every: int,
    shared_bases: Optional[List[BaseEpisode]] = None,
):
    final_path = output_dir / f"{split.name}.h5"
    temporary_path = output_dir / f"{split.name}.h5.tmp"

    if final_path.exists():
        if not overwrite:
            raise FileExistsError(f"{final_path} 已存在；使用 --overwrite。")
        final_path.unlink()
    if temporary_path.exists():
        temporary_path.unlink()

    preview_dir = output_dir / "preview" / split.name
    if overwrite and preview_dir.exists():
        shutil.rmtree(preview_dir)

    rng = np.random.default_rng(base_seed + split.seed_offset)
    generated_bases: List[BaseEpisode] = []
    start_time = time.time()

    print()
    print("=" * 88)
    print(f"Generating split: {split.name}")
    print(f"episodes        : {split.num_episodes}")
    print("visible actuator: False")
    print(f"temporary file  : {temporary_path}")
    print("=" * 88)

    try:
        with h5py.File(temporary_path, "w") as file:
            datasets = initialize_hdf5(
                file,
                split,
                image_size,
                base_seed + split.seed_offset,
            )

            for episode_index in range(split.num_episodes):
                if shared_bases is None:
                    base = generate_base_episode(rng, split)
                    generated_bases.append(base)
                else:
                    base = shared_bases[episode_index]

                pair_id = episode_index if split.paired else -1
                pair_role = 2 if split.violation else (1 if split.paired else 0)

                episode = materialize_episode(
                    base,
                    image_size,
                    pair_id,
                    pair_role,
                    split.violation,
                    rng,
                )
                write_episode(datasets, episode, episode_index)

                if episode_index < preview_episodes:
                    prefix = "pair" if split.paired else "episode"
                    save_contact_sheet(
                        episode,
                        preview_dir / f"{prefix}_{episode_index:04d}.png",
                    )

                completed = episode_index + 1
                if (
                    completed == 1
                    or completed % progress_every == 0
                    or completed == split.num_episodes
                ):
                    elapsed = time.time() - start_time
                    rate = completed / max(elapsed, 1e-9)
                    eta = (
                        (split.num_episodes - completed)
                        / max(rate, 1e-9)
                        / 60.0
                    )
                    print(
                        f"[{split.name}] {completed:6d}/{split.num_episodes:6d} | "
                        f"{rate:7.2f} ep/s | ETA {eta:7.2f} min"
                    )
                    file.flush()

            file.attrs["completed"] = 1
            file.attrs["generation_seconds"] = time.time() - start_time
            file.flush()

        os.replace(temporary_path, final_path)
        summary = validate_hdf5(final_path, split, image_size)
        print(f"[OK] {split.name}: {final_path}")
        return summary, generated_bases

    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] 临时文件：{temporary_path}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate no-platform OOT V3 datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/world-model/datasets/oot_v3"),
    )
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--test-episodes", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-episodes", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_episodes = 64 if args.smoke else args.train_episodes
    test_episodes = 16 if args.smoke else args.test_episodes

    if args.image_size < 64:
        raise ValueError("--image-size 建议不小于 64。")
    if train_episodes <= 0 or test_episodes <= 0:
        raise ValueError("episode 数必须大于 0。")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Occluded Object Transport V3 — no visible actuator")
    print("=" * 88)
    print("output_dir      =", args.output_dir)
    print("episode_length  =", EPISODE_LENGTH)
    print("visible actuator=", False)

    split_specs = [
        SplitSpec("train", train_episodes, TRAIN_CONTENTS, seed_offset=0),
        SplitSpec("test_iid", test_episodes, TRAIN_CONTENTS, seed_offset=1000),
        SplitSpec(
            "test_new_content",
            test_episodes,
            NEW_CONTENTS,
            seed_offset=2000,
        ),
        SplitSpec(
            "test_long_occlusion",
            test_episodes,
            TRAIN_CONTENTS,
            long_second_occlusion=True,
            seed_offset=3000,
        ),
        SplitSpec(
            "test_paired_normal",
            test_episodes,
            TRAIN_CONTENTS,
            paired=True,
            seed_offset=4000,
        ),
        SplitSpec(
            "test_paired_violation",
            test_episodes,
            TRAIN_CONTENTS,
            paired=True,
            violation=True,
            seed_offset=5000,
        ),
    ]

    summaries = {}
    paired_bases: Optional[List[BaseEpisode]] = None

    for split in split_specs:
        if split.name == "test_paired_violation":
            summary, _ = generate_split(
                split,
                args.output_dir,
                args.image_size,
                args.seed,
                args.overwrite,
                args.preview_episodes,
                max(1, args.progress_every),
                shared_bases=paired_bases,
            )
        else:
            summary, bases = generate_split(
                split,
                args.output_dir,
                args.image_size,
                args.seed,
                args.overwrite,
                args.preview_episodes,
                max(1, args.progress_every),
            )
            if split.name == "test_paired_normal":
                paired_bases = bases

        summaries[split.name] = summary

    if paired_bases is None:
        raise RuntimeError("paired normal bases 未生成。")

    paired_validation = validate_paired_files(
        args.output_dir / "test_paired_normal.h5",
        args.output_dir / "test_paired_violation.h5",
        test_episodes,
    )

    manifest = {
        "dataset_name": "occluded_object_transport_v3",
        "episode_length": EPISODE_LENGTH,
        "image_size": args.image_size,
        "seed": args.seed,
        "visible_actuator": False,
        "action_semantics": (
            "normalized hidden-actuator 2D acceleration/control command; "
            "not actual object displacement"
        ),
        "phase_names": PHASE_NAMES,
        "schema_names": SCHEMA_NAMES,
        "splits": summaries,
        "paired_validation": paired_validation,
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("ALL OOT V3 SPLITS COMPLETED")
    print("manifest =", manifest_path)
    print("preview  =", args.output_dir / "preview")
    print("=" * 88)


if __name__ == "__main__":
    main()

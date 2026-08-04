#!/usr/bin/env python3
"""
primitive_v1 数据集的公共定义。

平动、旋转及后续组合运动必须共享：
1. 相同的 22 帧 phase 结构；
2. 相同的可见性节点；
3. 相同的逐帧 loss 权重；
4. 相同的三维 action 语义。
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Sequence structure
# ============================================================

PHASE_LENGTHS = (3, 4, 3, 8, 4)
EPISODE_LENGTH = sum(PHASE_LENGTHS)

PHASE_NAMES = {
    0: "visible_context",
    1: "short_occlusion",
    2: "first_reappearance",
    3: "long_occlusion",
    4: "final_reappearance",
}


# ============================================================
# Loss structure
# ============================================================

FRAME_LOSS_WEIGHTS = np.asarray(
    [
        # phase 0: visible context
        1.0, 1.0, 1.0,

        # phase 1: short occlusion
        0.1, 0.1, 0.1, 0.1,

        # phase 2: first reappearance
        3.0, 2.0, 2.0,

        # phase 3: long occlusion
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,

        # phase 4: final reappearance
        3.0, 2.0, 2.0, 2.0,
    ],
    dtype=np.float32,
)

assert EPISODE_LENGTH == 22
assert FRAME_LOSS_WEIGHTS.shape == (EPISODE_LENGTH,)
assert abs(float(FRAME_LOSS_WEIGHTS.sum()) - 20.2) < 1e-5


# ============================================================
# Motion/action definitions
# ============================================================

ACTION_DIM = 3

ACTION_NAMES = (
    "translation_acceleration_x",
    "translation_acceleration_y",
    "angular_acceleration",
)

MOTION_TYPE_TRANSLATION = 0
MOTION_TYPE_ROTATION = 1
MOTION_TYPE_COMBINED = 2

OBJECT_RADIUS = 0.155


def build_phase_vectors() -> dict[str, np.ndarray]:
    phase_id: list[int] = []
    visible: list[int] = []
    is_reappearance: list[int] = []
    is_final_reappearance: list[int] = []

    for current_phase, phase_length in enumerate(PHASE_LENGTHS):
        for local_timestep in range(phase_length):
            phase_id.append(current_phase)

            visible.append(
                1 if current_phase in (0, 2, 4) else 0
            )

            is_reappearance.append(
                1
                if (
                    current_phase in (2, 4)
                    and local_timestep == 0
                )
                else 0
            )

            is_final_reappearance.append(
                1
                if (
                    current_phase == 4
                    and local_timestep == 0
                )
                else 0
            )

    output = {
        "phase_id": np.asarray(
            phase_id,
            dtype=np.int8,
        ),
        "object_visible": np.asarray(
            visible,
            dtype=np.uint8,
        ),
        "is_reappearance": np.asarray(
            is_reappearance,
            dtype=np.uint8,
        ),
        "is_final_reappearance": np.asarray(
            is_final_reappearance,
            dtype=np.uint8,
        ),
    }

    for value in output.values():
        assert value.shape == (EPISODE_LENGTH,)

    return output


PHASE_VECTORS = build_phase_vectors()

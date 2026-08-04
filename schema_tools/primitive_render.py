#!/usr/bin/env python3
"""
primitive_v1 非对称物体渲染器。

本文件只负责：
1. 定义 train / new-content 的物体内容池；
2. 在固定背景上渲染较大的非对称物体；
3. 生成内容总览图与多角度旋转预览图。

它暂时不生成 HDF5 数据集。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


IMAGE_SIZE = 128
SUPERSAMPLE = 4

BACKGROUND_RGB = (238, 240, 243)
OUTLINE_RGB = (35, 39, 47)
MARKER_RGB = (250, 250, 250)


@dataclass(frozen=True)
class ShapeSpec:
    shape_id: int
    shape_name: str

    # 物体局部坐标，通常位于 [-1, 1] × [-1, 1]。
    polygon: tuple[tuple[float, float], ...]

    # 用于加强朝向辨识的局部标记点。
    marker: tuple[float, float]


@dataclass(frozen=True)
class ContentSpec:
    content_id: int
    split: str
    shape_id: int
    shape_name: str
    color_id: int
    color_name: str
    rgb: tuple[int, int, int]


# 所有形状都刻意避免中心对称和旋转对称。
SHAPES: tuple[ShapeSpec, ...] = (
    ShapeSpec(
        0,
        "arrow",
        (
            (-1.00, -0.34),
            (0.18, -0.34),
            (0.18, -0.76),
            (1.00, 0.00),
            (0.18, 0.76),
            (0.18, 0.34),
            (-1.00, 0.34),
        ),
        (0.48, 0.00),
    ),
    ShapeSpec(
        1,
        "L_block",
        (
            (-0.86, -0.86),
            (-0.22, -0.86),
            (-0.22, 0.20),
            (0.82, 0.20),
            (0.82, 0.84),
            (-0.86, 0.84),
        ),
        (-0.52, -0.52),
    ),
    ShapeSpec(
        2,
        "flag",
        (
            (-0.86, -0.88),
            (-0.52, -0.88),
            (-0.52, -0.63),
            (0.88, -0.63),
            (0.40, -0.04),
            (0.88, 0.48),
            (-0.52, 0.48),
            (-0.52, 0.88),
            (-0.86, 0.88),
        ),
        (0.29, -0.35),
    ),
    ShapeSpec(
        3,
        "hammer",
        (
            (-0.92, -0.76),
            (0.72, -0.76),
            (0.72, -0.28),
            (0.18, -0.28),
            (0.18, 0.88),
            (-0.30, 0.88),
            (-0.30, -0.28),
            (-0.92, -0.28),
        ),
        (0.43, -0.51),
    ),
    ShapeSpec(
        4,
        "hook",
        (
            (-0.84, -0.86),
            (-0.20, -0.86),
            (-0.20, 0.18),
            (0.76, 0.18),
            (0.76, -0.28),
            (0.18, -0.28),
            (0.18, -0.64),
            (0.92, -0.64),
            (0.92, 0.80),
            (-0.84, 0.80),
        ),
        (-0.51, -0.50),
    ),
    ShapeSpec(
        5,
        "dart_plane",
        (
            (-1.00, 0.08),
            (-0.22, -0.25),
            (-0.54, -0.86),
            (0.10, -0.50),
            (0.96, -0.08),
            (0.18, 0.20),
            (0.42, 0.82),
            (-0.30, 0.42),
        ),
        (0.43, -0.08),
    ),
    ShapeSpec(
        6,
        "bent_rod",
        (
            (-0.92, -0.70),
            (0.18, -0.70),
            (0.18, -0.22),
            (0.84, -0.22),
            (0.84, 0.78),
            (0.28, 0.78),
            (0.28, 0.30),
            (-0.92, 0.30),
        ),
        (-0.55, -0.43),
    ),
    ShapeSpec(
        7,
        "key",
        (
            (-0.96, -0.30),
            (0.24, -0.30),
            (0.24, -0.66),
            (0.88, -0.66),
            (0.88, 0.18),
            (0.46, 0.18),
            (0.46, 0.72),
            (0.02, 0.72),
            (0.02, 0.30),
            (-0.96, 0.30),
        ),
        (0.56, -0.39),
    ),

    # 以下四种仅用于 new-content 测试。
    ShapeSpec(
        8,
        "boot",
        (
            (-0.84, -0.88),
            (-0.20, -0.88),
            (-0.20, 0.16),
            (0.74, 0.16),
            (0.94, 0.72),
            (-0.84, 0.72),
        ),
        (-0.51, -0.55),
    ),
    ShapeSpec(
        9,
        "notched_wedge",
        (
            (-0.96, -0.68),
            (0.88, -0.68),
            (0.30, -0.08),
            (0.86, 0.72),
            (-0.26, 0.72),
            (-0.26, 0.18),
            (-0.96, 0.18),
        ),
        (0.46, -0.45),
    ),
    ShapeSpec(
        10,
        "boomerang",
        (
            (-0.94, -0.78),
            (-0.28, -0.78),
            (0.10, -0.12),
            (0.84, -0.52),
            (0.94, 0.04),
            (-0.02, 0.58),
            (-0.34, 0.88),
            (-0.64, 0.50),
            (-0.18, 0.02),
        ),
        (-0.57, -0.48),
    ),
    ShapeSpec(
        11,
        "fork",
        (
            (-0.84, -0.82),
            (-0.38, -0.82),
            (-0.38, -0.22),
            (-0.08, -0.22),
            (-0.08, -0.82),
            (0.34, -0.82),
            (0.34, -0.22),
            (0.72, -0.22),
            (0.72, 0.28),
            (0.18, 0.28),
            (0.18, 0.86),
            (-0.34, 0.86),
            (-0.34, 0.28),
            (-0.84, 0.28),
        ),
        (0.51, 0.02),
    ),
)


TRAIN_COLORS = (
    ("red", (220, 64, 67)),
    ("blue", (55, 112, 210)),
    ("green", (52, 157, 91)),
    ("orange", (234, 132, 43)),
    ("purple", (142, 84, 190)),
    ("cyan", (42, 165, 178)),
    ("magenta", (204, 70, 151)),
    ("gold", (224, 174, 48)),
)

NEW_COLORS = (
    ("coral", (224, 105, 82)),
    ("teal", (35, 143, 137)),
    ("indigo", (83, 85, 183)),
    ("lime", (132, 174, 55)),
)


TRAIN_CONTENTS: tuple[ContentSpec, ...] = tuple(
    ContentSpec(
        content_id=i,
        split="train",
        shape_id=i,
        shape_name=SHAPES[i].shape_name,
        color_id=i,
        color_name=TRAIN_COLORS[i][0],
        rgb=TRAIN_COLORS[i][1],
    )
    for i in range(8)
)

NEW_CONTENTS: tuple[ContentSpec, ...] = tuple(
    ContentSpec(
        content_id=8 + i,
        split="new_content",
        shape_id=8 + i,
        shape_name=SHAPES[8 + i].shape_name,
        color_id=8 + i,
        color_name=NEW_COLORS[i][0],
        rgb=NEW_COLORS[i][1],
    )
    for i in range(4)
)


def _rotate_point(
    point: tuple[float, float],
    angle_rad: float,
) -> tuple[float, float]:
    x, y = point
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)

    return (
        cosine * x - sine * y,
        sine * x + cosine * y,
    )


def render_object(
    content: ContentSpec,
    *,
    angle_deg: float,
    center_xy: tuple[float, float] = (0.5, 0.5),
    object_radius: float = 0.145,
    image_size: int = IMAGE_SIZE,
    visible: bool = True,
) -> Image.Image:
    """
    渲染单帧图像。

    center_xy 和 object_radius 都使用 [0,1] 归一化坐标。
    object_radius=0.145 对应约 37 像素的完整外接直径。
    """
    high_size = image_size * SUPERSAMPLE

    image = Image.new(
        "RGB",
        (high_size, high_size),
        BACKGROUND_RGB,
    )

    if not visible:
        resampling = getattr(Image, "Resampling", Image)
        return image.resize(
            (image_size, image_size),
            resampling.LANCZOS,
        )

    draw = ImageDraw.Draw(image)

    shape = SHAPES[content.shape_id]
    angle_rad = math.radians(angle_deg)

    center_x = center_xy[0] * high_size
    center_y = center_xy[1] * high_size
    radius_px = object_radius * high_size

    polygon_px = []
    for point in shape.polygon:
        rotated_x, rotated_y = _rotate_point(point, angle_rad)
        polygon_px.append(
            (
                center_x + rotated_x * radius_px,
                center_y + rotated_y * radius_px,
            )
        )

    outline_width = max(2, int(round(0.018 * high_size)))

    draw.polygon(
        polygon_px,
        fill=content.rgb,
        outline=OUTLINE_RGB,
        width=outline_width,
    )

    marker_x, marker_y = _rotate_point(shape.marker, angle_rad)
    marker_center_x = center_x + marker_x * radius_px
    marker_center_y = center_y + marker_y * radius_px
    marker_radius = 0.105 * radius_px

    marker_box = (
        marker_center_x - marker_radius,
        marker_center_y - marker_radius,
        marker_center_x + marker_radius,
        marker_center_y + marker_radius,
    )

    draw.ellipse(
        marker_box,
        fill=MARKER_RGB,
        outline=OUTLINE_RGB,
        width=max(1, outline_width // 2),
    )

    resampling = getattr(Image, "Resampling", Image)
    return image.resize(
        (image_size, image_size),
        resampling.LANCZOS,
    )


def _draw_label(
    canvas: Image.Image,
    xy: tuple[int, int],
    text: str,
) -> None:
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        xy,
        text,
        fill=(20, 23, 28),
        font=font,
    )


def make_content_gallery(out_path: Path) -> None:
    contents = TRAIN_CONTENTS + NEW_CONTENTS

    columns = 4
    rows = math.ceil(len(contents) / columns)

    cell_width = 160
    cell_height = 164

    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        (250, 250, 250),
    )

    for index, content in enumerate(contents):
        row = index // columns
        column = index % columns

        angle_deg = (index * 23.0) % 360.0

        frame = render_object(
            content,
            angle_deg=angle_deg,
        )

        x0 = column * cell_width + 16
        y0 = row * cell_height + 8

        canvas.paste(frame, (x0, y0))

        label = (
            f"{content.split} | id={content.content_id}\n"
            f"{content.shape_name} | {content.color_name}\n"
            f"angle={angle_deg:.0f} deg"
        )
        _draw_label(
            canvas,
            (column * cell_width + 8, row * cell_height + 137),
            label,
        )

    canvas.save(out_path)


def make_rotation_demo(out_path: Path) -> None:
    contents: Sequence[ContentSpec] = (
        TRAIN_CONTENTS[0],
        TRAIN_CONTENTS[1],
        TRAIN_CONTENTS[2],
        TRAIN_CONTENTS[5],
        NEW_CONTENTS[0],
        NEW_CONTENTS[2],
    )

    angles = (0.0, 45.0, 90.0, 135.0)

    cell_width = 150
    cell_height = 150

    canvas = Image.new(
        "RGB",
        (len(angles) * cell_width, len(contents) * cell_height),
        (250, 250, 250),
    )

    for row, content in enumerate(contents):
        for column, angle_deg in enumerate(angles):
            frame = render_object(
                content,
                angle_deg=angle_deg,
            )

            x0 = column * cell_width + 11
            y0 = row * cell_height + 6
            canvas.paste(frame, (x0, y0))

            _draw_label(
                canvas,
                (column * cell_width + 7, row * cell_height + 135),
                f"{content.shape_name} | {angle_deg:.0f} deg",
            )

    canvas.save(out_path)


def save_manifest(out_path: Path) -> None:
    payload = {
        "image_size": IMAGE_SIZE,
        "supersample": SUPERSAMPLE,
        "default_object_radius": 0.145,
        "train_contents": [
            asdict(content) for content in TRAIN_CONTENTS
        ],
        "new_contents": [
            asdict(content) for content in NEW_CONTENTS
        ],
    }

    out_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gallery_path = args.out_dir / "content_gallery.png"
    rotation_path = args.out_dir / "rotation_demo.png"
    manifest_path = args.out_dir / "content_manifest.json"

    make_content_gallery(gallery_path)
    make_rotation_demo(rotation_path)
    save_manifest(manifest_path)

    print("PASS: primitive renderer completed")
    print("content gallery :", gallery_path)
    print("rotation demo   :", rotation_path)
    print("manifest        :", manifest_path)


if __name__ == "__main__":
    main()

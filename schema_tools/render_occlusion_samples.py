#!/usr/bin/env python3
"""Render a compact contact sheet for manual occlusion-dataset inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def render_samples(dataset: Path, output: Path, rows: int = 4, size: int = 56) -> None:
    with h5py.File(dataset, "r") as file:
        offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
        schedules = np.asarray(file["phase_lengths"][offsets], dtype=np.int64)
        combos: dict[tuple[int, int], int] = {}
        for episode, lengths in enumerate(schedules):
            key = (int(lengths[3]), int(lengths[:4].sum()))
            combos.setdefault(key, episode)
        preferred = [(2, 17), (4, 18), (8, 19), (2, 19)]
        ordered_keys = [key for key in preferred if key in combos]
        ordered_keys.extend(key for key in sorted(combos) if key not in ordered_keys)
        selected = [combos[key] for key in ordered_keys[:rows]]
        frames = [
            np.asarray(file["pixels"][offsets[index]: offsets[index] + 22])
            for index in selected
        ]

    header = 22
    canvas = Image.new("RGB", (22 * size, len(selected) * (size + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (episode, episode_frames) in enumerate(zip(selected, frames)):
        lengths = schedules[episode]
        reveal = int(lengths[:4].sum())
        draw.text(
            (2, row * (size + header) + 3),
            f"episode={episode} occlusion={int(lengths[3])} reveal={reveal}",
            fill="black",
        )
        top = row * (size + header) + header
        for timestep, frame in enumerate(episode_frames):
            image = Image.fromarray(frame.astype(np.uint8)).resize((size, size))
            canvas.paste(image, (timestep * size, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--size", type=int, default=56)
    args = parser.parse_args()
    render_samples(args.dataset, args.output, rows=args.rows, size=args.size)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

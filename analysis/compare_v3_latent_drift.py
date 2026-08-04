#!/usr/bin/env python3
"""Compare deterministic validation anchor embeddings with linear CKA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"Anchor row mismatch: {left.shape} vs {right.shape}")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
    left_norm = np.linalg.norm(left.T @ left, ord="fro")
    right_norm = np.linalg.norm(right.T @ right, ord="fro")
    return float(cross / max(left_norm * right_norm, 1.0e-12))


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    with np.load(reference) as left, np.load(candidate) as right:
        left_episode = np.asarray(left["episode_id"])
        right_episode = np.asarray(right["episode_id"])
        if not np.array_equal(left_episode, right_episode):
            raise ValueError("Anchor episode ordering differs; CKA comparison is invalid")
        return {
            "reference": str(reference.resolve()),
            "candidate": str(candidate.resolve()),
            "anchor_rows": int(len(left_episode)),
            "target_linear_cka": linear_cka(left["target_emb"], right["target_emb"]),
            "prediction_linear_cka": linear_cka(left["pred_emb"], right["pred_emb"]),
            "interpretation": (
                "Representation-drift diagnostic only. Low target CKA plus lower latent error "
                "requires physical-probe checks before claiming better prediction."
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

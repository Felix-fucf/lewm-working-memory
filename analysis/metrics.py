"""Small, dependency-light metric helpers shared by audits and tests."""

from __future__ import annotations


def target_variance_from_sufficient_statistics(
    count: int,
    dimension: int,
    square_sum: float,
    summed_vector_square_sum: float,
) -> float:
    """Population variance averaged over samples and embedding dimensions."""
    if count <= 0 or dimension <= 0:
        raise ValueError("count and dimension must be positive")
    centered = float(square_sum) - float(summed_vector_square_sum) / float(count)
    return max(centered / float(count * dimension), 0.0)


def normalize_mse(raw_mse: float, target_variance: float) -> float:
    return float(raw_mse) / max(float(target_variance), 1.0e-12)

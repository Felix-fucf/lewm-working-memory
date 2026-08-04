from __future__ import annotations

import numpy as np

from analysis.metrics import (
    normalize_mse,
    target_variance_from_sufficient_statistics,
)


def _variance(values: np.ndarray) -> float:
    return target_variance_from_sufficient_statistics(
        len(values),
        values.shape[1],
        float(np.square(values).sum()),
        float(np.square(values.sum(axis=0)).sum()),
    )


def test_normalized_mse_matches_population_definition_and_is_scale_invariant() -> None:
    target = np.asarray([[1.0, 3.0], [3.0, 7.0], [5.0, 5.0]])
    prediction = target + np.asarray([[1.0, -1.0], [2.0, 0.0], [-1.0, 1.0]])
    raw = float(np.square(prediction - target).mean())
    expected_variance = float(np.square(target - target.mean(axis=0)).mean())
    assert np.isclose(_variance(target), expected_variance)
    expected_nmse = raw / expected_variance
    assert np.isclose(normalize_mse(raw, _variance(target)), expected_nmse)

    scale = 7.5
    scaled_raw = float(np.square(scale * prediction - scale * target).mean())
    assert np.isclose(
        normalize_mse(scaled_raw, _variance(scale * target)),
        expected_nmse,
    )

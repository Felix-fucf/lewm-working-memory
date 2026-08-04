from __future__ import annotations

import numpy as np

from analysis.compare_v3_latent_drift import linear_cka


def test_linear_cka_is_one_for_equal_and_orthogonally_rotated_features() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(128, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert np.isclose(linear_cka(features, features), 1.0)
    assert np.isclose(linear_cka(features, features @ q), 1.0)

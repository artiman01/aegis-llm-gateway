"""Mathematical invariant unit tests for Isotropic ZCA-Whitening Transformer."""

from __future__ import annotations

import numpy as np
import pytest

from infrastructure.cache.whitening_transformer import WhiteningTransformer


class TestWhiteningTransformer:
    """Rigorous mathematical tests for ZCA-whitening transformation."""

    def test_unfitted_transformer_returns_unit_normalized_vector(self) -> None:
        """Verify unfitted transformer falls back to unit L2 spherical projection."""
        transformer = WhiteningTransformer(dimension=4)
        assert not transformer.is_fitted

        raw_vec = [3.0, 4.0, 0.0, 0.0]
        whitened = transformer.whiten(raw_vec)

        assert len(whitened) == 4
        # Norm of [3, 4, 0, 0] is 5.0 -> [0.6, 0.8, 0, 0]
        assert pytest.approx(float(np.linalg.norm(whitened)), abs=1e-6) == 1.0
        assert pytest.approx(whitened[0], abs=1e-6) == 0.6
        assert pytest.approx(whitened[1], abs=1e-6) == 0.8

    def test_epsilon_positive_validation(self) -> None:
        """Verify non-positive epsilon values are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            WhiteningTransformer(dimension=4, epsilon=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            WhiteningTransformer(dimension=4, epsilon=-1e-5)

    def test_fit_validation_errors(self) -> None:
        """Verify dimension and sample size validations during fitting."""
        transformer = WhiteningTransformer(dimension=8)

        # Insufficient sample count (< 2)
        with pytest.raises(ValueError, match="requires at least 2 samples"):
            transformer.fit([[1.0] * 8])

        # Dimension mismatch
        with pytest.raises(ValueError, match=r"Expected embedding matrix of shape \(N, 8\)"):
            transformer.fit([[1.0, 2.0], [3.0, 4.0]])

    def test_whiten_dimension_mismatch_raises_error(self) -> None:
        """Verify vector dimension validation during whiten call."""
        transformer = WhiteningTransformer(dimension=4)
        with pytest.raises(ValueError, match=r"Expected vector of dimension 4"):
            transformer.whiten([1.0, 2.0, 3.0])

    def test_zca_mathematical_isotropization_invariants(self) -> None:
        """Verify empirical covariance of whitened data approximates identity matrix.

        Mathematical Invariants:
        1. Centering: E[Z] -> 0
        2. Covariance Diag: Cov(Z)_ii -> 1.0 (unit variance along all principal axes)
        3. Covariance Off-Diag: Cov(Z)_ij -> 0.0 for i != j (decorrelation / isotropy)
        4. Hyperspherical Output: ||whiten(x)||_2 == 1.0
        """
        dim = 8
        n_samples = 2000
        rng = np.random.default_rng(42)

        # Generate heavily anisotropic, correlated data with non-zero mean
        true_mean = np.array([2.5, -1.0, 0.5, 3.0, -2.0, 1.2, 0.8, -0.4])
        # Construct positive definite covariance with large eigenvalue spread
        a = rng.standard_normal((dim, dim))
        cov_matrix = a @ a.T + np.diag([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05])

        raw_data = rng.multivariate_normal(true_mean, cov_matrix, size=n_samples)
        sample_list: list[list[float]] = raw_data.tolist()

        transformer = WhiteningTransformer(dimension=dim, epsilon=1e-6)
        transformer.fit(sample_list)
        assert transformer.is_fitted

        # Compute pre-normalized whitened vectors (W * (x - mu)) directly to verify covariance
        # The whiten() method applies the final unit-norm projection.
        mean_vec = transformer._mean
        transform_mat = transformer._transform_matrix
        assert mean_vec is not None
        assert transform_mat is not None

        centered_raw = raw_data - mean_vec
        whitened_linear = centered_raw @ transform_mat.T

        # Verify linear whitened covariance is approximately identity
        empirical_cov = (whitened_linear.T @ whitened_linear) / n_samples
        identity = np.eye(dim)

        np.testing.assert_allclose(empirical_cov, identity, rtol=1e-1, atol=1e-1)

        # Verify whiten() produces exact unit-sphere vectors
        for i in range(5):
            whitened_unit = transformer.whiten(sample_list[i])
            norm = float(np.linalg.norm(whitened_unit))
            assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_numerical_stability_with_collinear_embeddings(self) -> None:
        """Verify Tikhonov regularization prevents zero-division on rank-deficient samples."""
        dim = 4
        # Two identical/collinear vectors
        samples = [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
        transformer = WhiteningTransformer(dimension=dim, epsilon=1e-4)
        # Must not raise LinAlgError
        transformer.fit(samples)
        assert transformer.is_fitted

        whitened = transformer.whiten([1.5, 1.5, 1.5, 1.5])
        assert len(whitened) == dim
        assert pytest.approx(float(np.linalg.norm(whitened)), abs=1e-5) == 1.0

    def test_dimension_384_baseline_calibration_is_fitted_out_of_box(self) -> None:
        """Dimension 384 transformer initializes with synthetic baseline fitted out-of-the-box."""
        transformer = WhiteningTransformer(dimension=384)
        assert transformer.is_fitted is True

        sample_vec = [0.1] * 384
        whitened = transformer.whiten(sample_vec)
        assert len(whitened) == 384
        assert pytest.approx(float(np.linalg.norm(whitened)), abs=1e-5) == 1.0

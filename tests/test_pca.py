import numpy as np
import pytest

from leadlag.pca import rolling_pca, subspace_regularized_pca


class TestSubspaceRegularizedPCA:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        S = np.cov(rng.standard_normal((100, 10)), rowvar=False)
        result = subspace_regularized_pca(S, 0.5, 3)
        assert result.shape == (10, 3)

    def test_orthogonality(self):
        rng = np.random.default_rng(0)
        S = np.cov(rng.standard_normal((100, 8)), rowvar=False)
        U = subspace_regularized_pca(S, 0.5, 3)
        gram = U.T @ U
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)

    def test_lambda_zero_equals_standard_pca(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5))
        S = np.cov(data, rowvar=False)

        U_reg = subspace_regularized_pca(S, 0.0, 2)

        eigenvalues, eigenvectors = np.linalg.eigh(S)
        idx = np.argsort(eigenvalues)[::-1][:2]
        U_std = eigenvectors[:, idx]

        # Eigenvectors can differ by sign, compare subspaces
        proj_reg = U_reg @ U_reg.T
        proj_std = U_std @ U_std.T
        np.testing.assert_allclose(proj_reg, proj_std, atol=1e-10)

    def test_lambda_one_uses_prev(self):
        rng = np.random.default_rng(0)
        S = np.cov(rng.standard_normal((100, 6)), rowvar=False)
        U_prev = np.linalg.qr(rng.standard_normal((6, 2)))[0][:, :2]

        U = subspace_regularized_pca(S, 1.0, 2, U_prev)
        proj = U @ U.T
        proj_prev = U_prev @ U_prev.T
        np.testing.assert_allclose(proj, proj_prev, atol=1e-10)

    def test_with_explicit_u_prev(self):
        rng = np.random.default_rng(1)
        S = np.cov(rng.standard_normal((100, 6)), rowvar=False)
        U_prev = np.linalg.qr(rng.standard_normal((6, 2)))[0][:, :2]
        result = subspace_regularized_pca(S, 0.5, 2, U_prev)
        assert result.shape == (6, 2)


class TestRollingPCA:
    def test_output_length(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((80, 6)))
        results = rolling_pca(df, window=60, lambda_=0.5, n_components=2)
        assert len(results) == 80 - 60  # 20

    def test_each_element_shape(self):
        import pandas as pd

        rng = np.random.default_rng(0)
        n_features = 8
        df = pd.DataFrame(rng.standard_normal((100, n_features)))
        results = rolling_pca(df, window=60, lambda_=0.5, n_components=3)
        for U in results:
            assert U.shape == (n_features, 3)

import numpy as np
import pandas as pd


def subspace_regularized_pca(
    S_sample: np.ndarray,
    lambda_: float,
    n_components: int,
    U_prev: np.ndarray | None = None,
) -> np.ndarray:
    n_features = S_sample.shape[0]

    if U_prev is None:
        rng = np.random.default_rng(42)
        U_prev = np.linalg.qr(rng.standard_normal((n_features, n_components)))[0][
            :, :n_components
        ]

    S_reg = lambda_ * (U_prev @ U_prev.T) + (1 - lambda_) * S_sample

    eigenvalues, eigenvectors = np.linalg.eigh(S_reg)
    # eigh returns ascending order, take last n_components (largest)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    return eigenvectors[:, idx]


def rolling_pca(
    joint_returns: pd.DataFrame,
    window: int,
    lambda_: float,
    n_components: int,
) -> list[np.ndarray]:
    results = []
    U_prev = None
    values = joint_returns.values

    for i in range(window, len(values)):
        chunk = values[i - window : i]
        S_sample = np.cov(chunk, rowvar=False)
        U = subspace_regularized_pca(S_sample, lambda_, n_components, U_prev)
        U_prev = U
        results.append(U)

    return results

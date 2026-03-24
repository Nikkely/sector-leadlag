import numpy as np
import pandas as pd


def build_signal(
    us_returns_today: pd.Series,
    eigenvectors: np.ndarray,
    us_dim: int,
    jp_dim: int,
) -> pd.Series:
    V_us = eigenvectors[:us_dim, :]  # (us_dim, n_components)
    V_jp = eigenvectors[us_dim : us_dim + jp_dim, :]  # (jp_dim, n_components)

    signal = V_jp @ V_us.T @ us_returns_today.values
    return pd.Series(signal, index=range(jp_dim))


def suggest(signal: pd.Series, jp_etf_map: dict[str, str], top_n: int) -> dict:
    etf_tickers = list(jp_etf_map.keys())
    etf_names = list(jp_etf_map.values())

    scores = pd.Series(signal.values, index=etf_tickers)
    sorted_scores = scores.sort_values(ascending=False)

    long = [
        {"etf": t, "sector": jp_etf_map[t], "score": round(float(s), 4)}
        for t, s in sorted_scores.head(top_n).items()
    ]
    short = [
        {"etf": t, "sector": jp_etf_map[t], "score": round(float(s), 4)}
        for t, s in sorted_scores.tail(top_n).items()
    ]

    return {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "long": long,
        "short": short,
    }

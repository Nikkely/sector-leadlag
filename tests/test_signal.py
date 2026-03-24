import numpy as np
import pandas as pd
import pytest

from leadlag.signal import build_signal, suggest


class TestBuildSignal:
    def test_output_length(self):
        us_dim, jp_dim, n_comp = 5, 8, 3
        rng = np.random.default_rng(0)
        eigvec = np.linalg.qr(rng.standard_normal((us_dim + jp_dim, n_comp)))[0][:, :n_comp]
        us_ret = pd.Series(rng.standard_normal(us_dim))
        signal = build_signal(us_ret, eigvec, us_dim, jp_dim)
        assert len(signal) == jp_dim

    def test_zero_returns_give_zero_signal(self):
        us_dim, jp_dim, n_comp = 4, 6, 2
        rng = np.random.default_rng(0)
        eigvec = np.linalg.qr(rng.standard_normal((us_dim + jp_dim, n_comp)))[0][:, :n_comp]
        us_ret = pd.Series(np.zeros(us_dim))
        signal = build_signal(us_ret, eigvec, us_dim, jp_dim)
        np.testing.assert_allclose(signal.values, 0.0, atol=1e-15)


class TestSuggest:
    def test_structure(self):
        signal = pd.Series([0.5, -0.3, 0.1, 0.8, -0.6])
        jp_map = {
            "1615.T": "Banks",
            "1616.T": "Financials",
            "1617.T": "Foods",
            "1618.T": "Energy",
            "1619.T": "Construction",
        }
        result = suggest(signal, jp_map, top_n=2)

        assert "date" in result
        assert "long" in result
        assert "short" in result
        assert len(result["long"]) == 2
        assert len(result["short"]) == 2

    def test_long_has_highest_scores(self):
        signal = pd.Series([0.5, -0.3, 0.1, 0.8, -0.6])
        jp_map = {
            "1615.T": "Banks",
            "1616.T": "Financials",
            "1617.T": "Foods",
            "1618.T": "Energy",
            "1619.T": "Construction",
        }
        result = suggest(signal, jp_map, top_n=2)
        long_scores = [e["score"] for e in result["long"]]
        short_scores = [e["score"] for e in result["short"]]
        assert all(l > s for l in long_scores for s in short_scores)

    def test_each_entry_has_required_keys(self):
        signal = pd.Series([0.1, 0.2, 0.3])
        jp_map = {"A": "SectorA", "B": "SectorB", "C": "SectorC"}
        result = suggest(signal, jp_map, top_n=1)
        for entry in result["long"] + result["short"]:
            assert "etf" in entry
            assert "sector" in entry
            assert "score" in entry

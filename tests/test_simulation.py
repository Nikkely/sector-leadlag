import numpy as np
import pandas as pd
import pytest

from leadlag.simulation import performance_metrics, run_simulation


class TestRunSimulation:
    def _make_data(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        rng = np.random.default_rng(42)
        jp_returns = pd.DataFrame(
            rng.standard_normal((10, 5)) * 0.01,
            index=dates,
        )
        signals = [pd.Series(rng.standard_normal(5)) for _ in range(10)]
        return signals, jp_returns, dates

    def test_output_columns(self):
        signals, jp_ret, dates = self._make_data()
        sim_df = run_simulation(signals, jp_ret, top_n=2, dates=dates)
        assert "daily_return" in sim_df.columns
        assert "cumulative_return" in sim_df.columns

    def test_cumulative_return_starts_near_one(self):
        signals, jp_ret, dates = self._make_data()
        sim_df = run_simulation(signals, jp_ret, top_n=2, dates=dates)
        assert abs(sim_df["cumulative_return"].iloc[0] - 1.0) < 0.1

    def test_empty_signals(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        jp_ret = pd.DataFrame(np.zeros((5, 3)), index=dates)
        sim_df = run_simulation([], jp_ret, top_n=1, dates=dates)
        assert sim_df.empty


class TestPerformanceMetrics:
    def test_keys(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        rng = np.random.default_rng(0)
        sim_df = pd.DataFrame(
            {
                "daily_return": rng.standard_normal(100) * 0.01,
                "cumulative_return": np.cumprod(1 + rng.standard_normal(100) * 0.01),
            },
            index=dates,
        )
        metrics = performance_metrics(sim_df)
        assert set(metrics.keys()) == {"cagr", "sharpe", "mdd", "win_rate"}

    def test_win_rate_range(self):
        dates = pd.bdate_range("2024-01-01", periods=50)
        rng = np.random.default_rng(1)
        daily = rng.standard_normal(50) * 0.01
        sim_df = pd.DataFrame(
            {
                "daily_return": daily,
                "cumulative_return": np.cumprod(1 + daily),
            },
            index=dates,
        )
        metrics = performance_metrics(sim_df)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_empty_df(self):
        sim_df = pd.DataFrame()
        metrics = performance_metrics(sim_df)
        assert metrics["cagr"] == 0

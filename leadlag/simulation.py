import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_simulation(
    signals_history: list[pd.Series],
    jp_returns: pd.DataFrame,
    top_n: int,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    daily_returns = []

    for i, signal in enumerate(signals_history):
        if i >= len(dates):
            break
        date = dates[i]
        if date not in jp_returns.index:
            continue

        sorted_idx = signal.argsort()[::-1]
        long_idx = sorted_idx[:top_n]

        day_return = jp_returns.iloc[jp_returns.index.get_loc(date)][long_idx].mean()
        daily_returns.append({"date": date, "daily_return": day_return})

    sim_df = pd.DataFrame(daily_returns)
    if sim_df.empty:
        return sim_df

    sim_df["cumulative_return"] = (1 + sim_df["daily_return"]).cumprod()
    sim_df = sim_df.set_index("date")
    return sim_df


def performance_metrics(sim_df: pd.DataFrame) -> dict:
    if sim_df.empty:
        return {"cagr": 0, "sharpe": 0, "mdd": 0, "win_rate": 0}

    daily = sim_df["daily_return"]
    n_days = len(daily)
    total_return = sim_df["cumulative_return"].iloc[-1]

    cagr = total_return ** (252 / n_days) - 1 if n_days > 0 else 0
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

    cum = sim_df["cumulative_return"]
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    mdd = drawdown.min()

    win_rate = (daily > 0).sum() / n_days if n_days > 0 else 0

    return {
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 4),
        "mdd": round(float(mdd), 4),
        "win_rate": round(float(win_rate), 4),
    }


def plot_simulation(sim_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sim_df.index, sim_df["cumulative_return"], linewidth=1.5)
    ax.set_title("Cumulative Return - Sector Lead-Lag Strategy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

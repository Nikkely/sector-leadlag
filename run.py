import argparse
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import yaml

from leadlag.fetch import align_dates, fetch_jp_returns, fetch_us_returns
from leadlag.pca import rolling_pca
from leadlag.signal import build_signal, suggest
from leadlag.simulation import (
    performance_metrics,
    plot_simulation,
    run_simulation,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_signal(config: dict) -> None:
    us_etfs = config["us_etfs"]
    jp_etfs = config["jp_etfs"]
    window = config["rolling_window"]
    lambda_ = config["lambda"]
    n_components = config["n_components"]
    top_n = config["top_n"]

    end = datetime.now()
    start = end - timedelta(days=int(window * 2.5))

    us_ret = fetch_us_returns(list(us_etfs.keys()), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    jp_ret = fetch_jp_returns(list(jp_etfs.keys()), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    us_aligned, jp_aligned = align_dates(us_ret, jp_ret)

    joint = pd.concat([us_aligned, jp_aligned], axis=1)
    joint = joint.dropna()

    eigenvectors_list = rolling_pca(joint, window, lambda_, n_components)

    if not eigenvectors_list:
        print("Not enough data for PCA computation.")
        return

    latest_eigvec = eigenvectors_list[-1]
    latest_us = us_aligned.iloc[-1]

    signal = build_signal(latest_us, latest_eigvec, len(us_etfs), len(jp_etfs))
    result = suggest(signal, jp_etfs, top_n)

    os.makedirs("output", exist_ok=True)
    filename = f"output/signal_{datetime.now().strftime('%Y%m%d')}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved to {filename}")


def cmd_simulate(config: dict, start: str, end: str, show_plot: bool) -> None:
    us_etfs = config["us_etfs"]
    jp_etfs = config["jp_etfs"]
    window = config["rolling_window"]
    lambda_ = config["lambda"]
    n_components = config["n_components"]
    top_n = config["top_n"]

    fetch_start = (pd.Timestamp(start) - timedelta(days=int(window * 2.5))).strftime("%Y-%m-%d")

    us_ret = fetch_us_returns(list(us_etfs.keys()), fetch_start, end)
    jp_ret = fetch_jp_returns(list(jp_etfs.keys()), fetch_start, end)

    us_aligned, jp_aligned = align_dates(us_ret, jp_ret)

    joint = pd.concat([us_aligned, jp_aligned], axis=1)
    joint = joint.dropna()

    eigenvectors_list = rolling_pca(joint, window, lambda_, n_components)

    if not eigenvectors_list:
        print("Not enough data for simulation.")
        return

    us_dim = len(us_etfs)
    jp_dim = len(jp_etfs)
    signal_dates = joint.index[window:]

    signals = []
    for i, eigvec in enumerate(eigenvectors_list):
        date = signal_dates[i]
        us_today = us_aligned.loc[date] if date in us_aligned.index else None
        if us_today is None:
            continue
        sig = build_signal(us_today, eigvec, us_dim, jp_dim)
        signals.append(sig)

    sim_dates = signal_dates[: len(signals)]
    sim_df = run_simulation(signals, jp_aligned, top_n, sim_dates)

    if sim_df.empty:
        print("No simulation results.")
        return

    os.makedirs("output", exist_ok=True)
    sim_df.to_csv("output/simulation_result.csv")
    plot_simulation(sim_df, "output/simulation_plot.png")

    metrics = performance_metrics(sim_df)
    print("=== Performance Metrics ===")
    print(f"  CAGR:      {metrics['cagr']:.2%}")
    print(f"  Sharpe:    {metrics['sharpe']:.4f}")
    print(f"  Max DD:    {metrics['mdd']:.2%}")
    print(f"  Win Rate:  {metrics['win_rate']:.2%}")
    print(f"\nSaved to output/simulation_result.csv, output/simulation_plot.png")

    if show_plot:
        import matplotlib.pyplot as plt

        img = plt.imread("output/simulation_plot.png")
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Sector Lead-Lag Analysis Tool")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("signal", help="Generate today's signal")

    sim_parser = subparsers.add_parser("simulate", help="Run backtest simulation")
    sim_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    sim_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    sim_parser.add_argument("--plot", action="store_true", help="Show plot")

    args = parser.parse_args()
    config = load_config()

    if args.command == "signal":
        cmd_signal(config)
    elif args.command == "simulate":
        cmd_simulate(config, args.start, args.end, args.plot)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

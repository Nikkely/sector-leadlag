import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _validate_download(
    data: pd.DataFrame, requested: list[str], label: str
) -> pd.DataFrame:
    """Drop tickers that returned no data and warn."""
    if data.empty:
        failed = requested
    else:
        close = data["Close"]
        if isinstance(close, pd.Series):
            # single ticker case
            close = close.to_frame(requested[0])
        missing = close.columns[close.isna().all()]
        failed = list(missing)
        data = data.drop(columns=failed, level=1, errors="ignore")

    if failed:
        logger.warning(
            "%s: skipping unavailable tickers: %s", label, ", ".join(failed)
        )

    if data.empty:
        raise ValueError(
            f"{label}: no valid data returned. Check tickers: {', '.join(requested)}"
        )

    return data


def fetch_us_returns(etf_list: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(etf_list, start=start, end=end, progress=False)
    data = _validate_download(data, etf_list, "US ETFs")
    close = data["Close"]
    returns = close.pct_change().dropna()
    return returns


def fetch_jp_returns(etf_list: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(etf_list, start=start, end=end, progress=False)
    data = _validate_download(data, etf_list, "JP ETFs")
    open_ = data["Open"]
    close = data["Close"]
    returns = (close - open_) / open_
    returns = returns.dropna()
    return returns


def align_dates(
    us_df: pd.DataFrame, jp_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 米国t日 → 日本t+1日: shift US index forward by 1 business day
    us_shifted = us_df.copy()
    us_shifted.index = us_shifted.index + pd.tseries.offsets.BDay(1)

    common = us_shifted.index.intersection(jp_df.index)
    return us_shifted.loc[common], jp_df.loc[common]

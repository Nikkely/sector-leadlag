import pandas as pd
import yfinance as yf


def fetch_us_returns(etf_list: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(etf_list, start=start, end=end, progress=False)
    close = data["Close"]
    returns = close.pct_change().dropna()
    return returns


def fetch_jp_returns(etf_list: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(etf_list, start=start, end=end, progress=False)
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

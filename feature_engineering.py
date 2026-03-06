import pandas as pd
import numpy as np

def add_intraday_pct(g: pd.DataFrame) -> pd.DataFrame:
    g["date"] = g.index.date
    g["day_open"] = g.groupby(g.index.date)["Open"].transform("first")
    g["intraday_pct"] = g["Close"] / g["day_open"] - 1
    return g


def add_basic_lags(g: pd.DataFrame, lags: int) -> pd.DataFrame:
    close = g["Close"]
    vol   = g["Volume"]

    returns = close.pct_change()
    hl_range = (g["High"] - g["Low"]) / close.replace(0, np.nan)

    vol_mean = vol.rolling(60, min_periods=10).mean()
    vol_std  = vol.rolling(60, min_periods=10).std()
    vol_z    = (vol - vol_mean) / vol_std

    for i in range(1, lags+1):
        g[f"ret_lag{i}"] = returns.shift(i)
        g[f"range_lag{i}"] = hl_range.shift(i)
        g[f"vol_lag{i}"] = vol.shift(i)
        g[f"volz_lag{i}"] = vol_z.shift(i)

    return g


def add_candle_structure(g: pd.DataFrame) -> pd.DataFrame:
    openp = g["Open"]
    close = g["Close"]
    high  = g["High"]
    low   = g["Low"]

    g["body"] = (close - openp) / openp
    g["upper_wick"] = (high - np.maximum(openp, close)) / close
    g["lower_wick"] = (np.minimum(openp, close) - low) / close
    return g


def add_return_bursts(g: pd.DataFrame) -> pd.DataFrame:
    close = g["Close"]
    g["burst3"] = close.pct_change(3)
    g["burst5"] = close.pct_change(5)
    return g


def add_rvol(g: pd.DataFrame) -> pd.DataFrame:
    vol = g["Volume"]
    g["rvol20"] = vol / vol.rolling(20).mean()
    return g


def build_all_features_per_symbol(g: pd.DataFrame, lags: int) -> pd.DataFrame:
    g = add_intraday_pct(g)
    g = add_basic_lags(g, lags)
    g = add_candle_structure(g)
    g = add_return_bursts(g)
    g = add_rvol(g)
    return g

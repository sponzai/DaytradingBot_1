# train_next1m_direction.py
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from joblib import dump


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame is indexed by a UTC-aware datetime index."""
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df.sort_index()

    # Try common timestamp columns
    for c in ["Datetime", "datetime", "Date", "date", "index"]:
        if c in df.columns:
            ts = pd.to_datetime(df[c], utc=True, errors="coerce")
            df = df.set_index(ts).drop(columns=[c]).sort_index()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df
    # Fallback: first column
    first = df.columns[0]
    ts = pd.to_datetime(df[first], utc=True, errors="coerce")
    df = df.set_index(ts).drop(columns=[first]).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def filter_rth(g: pd.DataFrame) -> pd.DataFrame:
    """Regular Trading Hours for US equities: 09:30–16:00 ET, weekdays."""
    if g.index.tz is None:
        g = g.tz_localize("UTC")
    ny = g.index.tz_convert("America/New_York")
    is_weekday = ny.dayofweek < 5
    hm = ny.hour * 60 + ny.minute
    rth_mask = (hm >= 9*60 + 30) & (hm <= 16*60)
    return g[is_weekday & rth_mask]


def build_lag_features_per_symbol(g: pd.DataFrame, lags: int) -> pd.DataFrame:
    """
    Build strictly past-dependent features to avoid leakage:
      - ret_lag{1..L}: close-to-close returns
      - range_lag{1..L}: (High-Low)/Close
      - vol_lag{1..L}: raw volume
      - volz_lag{1..L}: volume z-score (rolling 60)
    Target:
      - y: 1 if next minute close > current close else 0
      - fut_ret: next minute return
    """
    close = g["Close"].astype(float)
    high = g["High"].astype(float)
    low = g["Low"].astype(float)
    vol = g["Volume"].astype(float)

    # Base series
    ret1 = close.pct_change()
    hl_range = (high - low) / close.replace(0, np.nan)
    vol_mean = vol.rolling(60, min_periods=10).mean()
    vol_std = vol.rolling(60, min_periods=10).std()
    vol_z = (vol - vol_mean) / vol_std
    vol_z = vol_z.replace([np.inf, -np.inf], np.nan)

    # Lags (strictly past info: shift by >=1)
    for k in range(1, lags + 1):
        g[f"ret_lag{k}"] = ret1.shift(k)
        g[f"range_lag{k}"] = hl_range.shift(k)
        g[f"vol_lag{k}"] = vol.shift(k)
        g[f"volz_lag{k}"] = vol_z.shift(k)

    # Target: next minute direction
    g["fut_ret"] = close.shift(-1) / close - 1.0
    g["y"] = (g["fut_ret"] > 0).astype(int)

    # Drop rows that don't have full lag coverage or target
    req_cols = [f"ret_lag{k}" for k in range(1, lags + 1)] + \
               [f"range_lag{k}" for k in range(1, lags + 1)] + \
               [f"vol_lag{k}" for k in range(1, lags + 1)] + \
               [f"volz_lag{k}" for k in range(1, lags + 1)] + ["y", "fut_ret"]
    g = g.dropna(subset=req_cols)

    return g


def main():
    ap = argparse.ArgumentParser(description="Train a simple next-1m direction model using 3–6 lagged candles + volume.")
    ap.add_argument("--data", type=str, default="candles_1min_last7days.csv", help="Path to combined candles CSV")
    ap.add_argument("--lags", type=int, default=5, help="Number of past candles to use (3–6 recommended)")
    ap.add_argument("--rth", type=str, default="true", help="Use regular trading hours (true/false)")
    ap.add_argument("--test_days", type=float, default=1.0, help="Hold out last N days for testing")
    ap.add_argument("--outdir", type=str, default="ml_output_simple", help="Directory to save outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    df = ensure_datetime_index(df)

    # Basic sanity
    needed = ["Open", "High", "Low", "Close", "Volume", "symbol"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Apply RTH per symbol if requested
    if args.rth.lower() in ("true", "1", "yes", "y"):
        df = df.groupby("symbol", group_keys=False).apply(filter_rth)

    # Sort by time within each symbol
    df = df.sort_index()

    # Feature engineering (strictly per symbol)
    df_feat = df.groupby("symbol", group_keys=True).apply(
        lambda g: build_lag_features_per_symbol(g.copy(), lags=int(args.lags))
    )

    # Reset symbol to column (groupby added it to index level 0)
    if "symbol" not in df_feat.columns:
        df_feat = df_feat.reset_index(level=0).rename(columns={"level_0": "symbol"})

    # Time-based split: hold out last N days across all symbols
    cutoff = df_feat.index.max() - pd.Timedelta(days=float(args.test_days))
    train = df_feat.loc[df_feat.index <= cutoff].copy()
    test = df_feat.loc[df_feat.index > cutoff].copy()

    # Features
    lags = int(args.lags)
    feature_cols: List[str] = \
        [f"ret_lag{k}" for k in range(1, lags + 1)] + \
        [f"range_lag{k}" for k in range(1, lags + 1)] + \
        [f"vol_lag{k}" for k in range(1, lags + 1)] + \
        [f"volz_lag{k}" for k in range(1, lags + 1)]

    X_train = train[feature_cols]
    y_train = train["y"].astype(int)
    X_test = test[feature_cols]
    y_test = test["y"].astype(int)

    # Model: tree-based (no scaling needed), robust baseline
    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=100,  # slightly conservative for 1m noise
        l2_regularization=1e-3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_test, y_pred)

    # Simple trading proxy: long when pred=1, flat otherwise, realized next-minute return
    test = test.copy()
    test["pred"] = y_pred
    test["proba"] = proba
    long_mask = test["pred"] == 1
    hit_rate = (test.loc[long_mask, "y"] == 1).mean() if long_mask.any() else np.nan
    mean_fut_ret = test.loc[long_mask, "fut_ret"].mean() if long_mask.any() else np.nan
    sum_fut_ret = test.loc[long_mask, "fut_ret"].sum() if long_mask.any() else np.nan

    # Save artifacts
    metrics = f"""
Lags used: {lags}
RTH: {args.rth}
Test window (days): {args.test_days}
Samples — train: {len(train):,} | test: {len(test):,} | tickers in test: {test['symbol'].nunique():,}

Classification:
  Accuracy:  {acc:.4f}
  Precision: {prec:.4f}
  Recall:    {rec:.4f}
  F1:        {f1:.4f}
  ROC-AUC:   {auc:.4f}

Confusion matrix [ [TN, FP], [FN, TP] ]:
{cm}

Trading proxy (long-only when pred=1):
  Signals (pred=1) count: {int(long_mask.sum()):,}
  Hit rate on signals:    {hit_rate:.4f}
  Mean fut_ret on longs:  {mean_fut_ret:.6f}
  Sum fut_ret on longs:   {sum_fut_ret:.6f}
"""
    print(metrics)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(metrics)

    test_out = test[["symbol"] + feature_cols + ["y", "pred", "proba", "fut_ret"]].copy()
    test_out.to_csv(os.path.join(args.outdir, "test_predictions.csv"), index=True)
    dump(model, os.path.join(args.outdir, "model.joblib"))

    print(f"\nSaved: {os.path.join(args.outdir, 'metrics.txt')}")
    print(f"Saved: {os.path.join(args.outdir, 'test_predictions.csv')}")
    print(f"Saved: {os.path.join(args.outdir, 'model.joblib')}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    main()
``

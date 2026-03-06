# download_candles.py

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from ticker_list import tickers

# Last 7 days (Yahoo Finance limit for 1-minute data)
end_time = datetime.now()
start_time = end_time - timedelta(days=7)

all_data = []

for symbol in tickers:
    print(f"Downloading {symbol}...")
    df = yf.download(
        symbol,
        interval="1m",
        start=start_time,
        end=end_time,
        progress=False
    )

    if df.empty:
        print(f"No data for {symbol}, skipping.")
        continue

    df["symbol"] = symbol
    all_data.append(df)

# Combine and save
if all_data:
    final_df = pd.concat(all_data)
    final_df.to_csv("candles_1min_last7days.csv")
    print("Saved: candles_1min_last7days.csv")
else:
    print("No data downloaded.")

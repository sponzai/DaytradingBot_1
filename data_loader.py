import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Loads and ensures the dataframe has a proper UTC datetime index.
    """
    df = pd.read_csv(path)

    # Try to identify a datetime column
    for col in ["Datetime", "datetime", "Date", "date", "timestamp", df.columns[0]]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], utc=True)
                df = df.set_index(col)
                break
            except:
                continue

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("No valid datetime column found.")

    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    return df

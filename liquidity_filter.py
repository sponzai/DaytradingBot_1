import pandas as pd

def filter_liquidity(df: pd.DataFrame,
                     min_median_volume: int = 50000,
                     min_price: float = 2.0) -> pd.DataFrame:
    """
    Removes illiquid symbols based on median volume & price.
    """

    good_symbols = []

    for sym, g in df.groupby("symbol"):
        if g["Volume"].median() >= min_median_volume and g["Close"].median() >= min_price:
            good_symbols.append(sym)

    df = df[df["symbol"].isin(good_symbols)]
    return df
``

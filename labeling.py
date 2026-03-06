import pandas as pd

def add_label_next_minute(g: pd.DataFrame) -> pd.DataFrame:
    g["fut_ret"] = g["Close"].shift(-1) / g["Close"] - 1
    g["y"] = (g["fut_ret"] > 0).astype(int)
    return g

import pandas as pd
from data_loader import load_data
from liquidity_filter import filter_liquidity
from feature_engineering import build_all_features_per_symbol
from labeling import add_label_next_minute
from model_training import train_model, evaluate_model
from walkforward import walkforward_cv

DATA_PATH = "candles_1min_last7days.csv"
LAGS = 5

print("Loading data...")
df = load_data(DATA_PATH)

print("Filtering liquidity...")
df = filter_liquidity(df)

print("Building features...")
df = df.groupby("symbol", group_keys=False).apply(
    lambda g: build_all_features_per_symbol(g, LAGS)
)

print("Adding labels...")
df = df.groupby("symbol", group_keys=False).apply(add_label_next_minute)

df = df.dropna()

# Time split
cutoff = df.index.max() - pd.Timedelta(days=1)
train = df[df.index <= cutoff]
test  = df[df.index > cutoff]

# Feature cols
feature_cols = [c for c in df.columns if c not in ["y", "fut_ret", "symbol", "date"]]

print("Training model...")
model = train_model(train[feature_cols], train["y"])

print("Evaluating model...")
metrics = evaluate_model(model, test[feature_cols], test["y"])

print(metrics)

print("Running walkforward CV...")
wf_scores = walkforward_cv(df, feature_cols)

print("Walkforward hit rate:", sum(wf_scores) / len(wf_scores))

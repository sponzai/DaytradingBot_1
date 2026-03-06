import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

def walkforward_cv(df, feature_cols, horizon_days=1):
    unique_days = sorted(df.index.date)

    results = []

    for i in range(len(unique_days) - horizon_days):
        train_days = unique_days[:i+1]
        test_days  = unique_days[i+1:i+1+horizon_days]

        train = df[df.index.date.isin(train_days)]
        test  = df[df.index.date.isin(test_days)]

        model = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=200
        )

        model.fit(train[feature_cols], train["y"])

        proba = model.predict_proba(test[feature_cols])[:, 1]
        y_pred = (proba >= 0.55).astype(int)
        y_true = test["y"]

        hit_rate = (y_pred == y_true).mean()

        results.append(hit_rate)

    return results

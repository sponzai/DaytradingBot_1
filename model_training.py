import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_default_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1e-3,
        random_state=42
    )


def train_model(X_train, y_train):
    model = get_default_model()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, proba_threshold=0.55):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= proba_threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "y_pred": y_pred,
        "proba": proba
    }

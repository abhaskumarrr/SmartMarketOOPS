#!/usr/bin/env python3
"""train_classifier.py
Train a LightGBM classifier on the generated dataset to predict trading actions.
"""
from __future__ import annotations
import argparse
import pathlib
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    X = df.drop(columns=["label"])
    # Map labels from (-1, 0, 1) to (0, 1, 2) for LightGBM multiclass
    y = df["label"].astype(int) + 1
    return X, y


def train_lightgbm(X: pd.DataFrame, y: pd.Series):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time-series order preserved
    )

    # Encode object columns numerically to avoid categorical feature mismatch at inference
    object_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    if object_cols:
        for col in object_cols:
            combined = pd.concat([X_train[col], X_val[col]], axis=0)
            mapping = {k: i for i, k in enumerate(combined.unique())}
            X_train[col] = X_train[col].map(mapping).astype("int32")
            X_val[col] = X_val[col].map(mapping).astype("int32")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    params = dict(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        metric=["multi_logloss"],
        verbose=-1,
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)],
    )

    preds = model.predict(X_val)
    y_pred = preds.argmax(axis=1)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print("Validation Accuracy:", acc)
    print("Validation F1:", f1)
    print(classification_report(y_val, y_pred))

    return model


def cli():
    parser = argparse.ArgumentParser(description="Train LightGBM trading classifier")
    parser.add_argument("--dataset", required=True, help="Path to Parquet dataset")
    parser.add_argument("--model_out", default="../../ml/models/trading_model.pkl")
    args = parser.parse_args()

    X, y = load_dataset(args.dataset)
    model = train_lightgbm(X, y)

    out_path = pathlib.Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"✅ Model saved to {out_path}")

if __name__ == "__main__":
    cli() 
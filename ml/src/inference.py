#!/usr/bin/env python3
"""inference.py
Load trained model and output trading action for the latest feature vector.
"""
from __future__ import annotations
import pathlib
import joblib
import numpy as np
import pandas as pd
from typing import Literal

Action = Literal["long", "short", "hold"]

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "trading_model.pkl"


def load_model(model_path: pathlib.Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_action(features: pd.DataFrame, model=None, threshold: float = 0.55) -> Action:
    """Return action based on model probabilities.

    Args:
        features: DataFrame with a single row of features (same columns as training).
        model: pre-loaded model.
        threshold: probability edge threshold over 0.5.
    """
    if model is None:
        model = load_model()

    # Encode object columns to numeric codes to match training
    obj_cols = [c for c in features.columns if features[c].dtype == "object"]
    if obj_cols:
        for col in obj_cols:
            features[col] = features[col].astype("category").cat.codes.astype("int32")

    proba = model.predict(features)  # shape (1,3)
    long_p, hold_p, short_p = proba[0][2], proba[0][1], proba[0][0]

    if long_p > max(hold_p, short_p) and long_p > threshold:
        return "long"
    if short_p > max(hold_p, long_p) and short_p > threshold:
        return "short"
    return "hold"


if __name__ == "__main__":
    import argparse, json
    from data.dataset_builder import fetch_ohlcv, build_features

    parser = argparse.ArgumentParser(description="Run live prediction on the latest candle")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--exchange", default="binance")
    args = parser.parse_args()

    raw = fetch_ohlcv(args.symbol, args.timeframe, limit=200, exchange_name=args.exchange)
    feat = build_features(raw).tail(1)
    action = predict_action(feat)
    print(json.dumps({"action": action})) 
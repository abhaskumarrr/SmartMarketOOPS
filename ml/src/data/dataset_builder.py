#!/usr/bin/env python3
"""dataset_builder.py
Builds supervised learning datasets for the trading model from live (or historical) exchange data.

Usage (example):
    python dataset_builder.py --symbol BTC/USDT --timeframe 1h --limit 5000 --horizon 4 \
        --exchange binance --output ../../data/datasets/btcusdt_1h.parquet

The script will:
1. Fetch OHLCV via CCXT.
2. Run build_features() to generate advanced feature set.
3. Create forward-return labels (+1, 0, -1) based on horizon & thresholds.
4. Persist the feature/label dataframe in Parquet format.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import ccxt
import numpy as np
import pandas as pd

# Local imports
from features.build_features import build_features

LABEL_THRESH = 0.004  # 0.4% move threshold for classification

def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 5000,
    exchange_name: str = "binance",
) -> pd.DataFrame:
    """Download OHLCV data via CCXT and return as DataFrame indexed by timestamp."""
    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    exchange.load_markets()

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def label_forward_return(df: pd.DataFrame, horizon: int = 4, thresh: float = LABEL_THRESH) -> pd.Series:
    """Return discrete labels: +1(long) / -1(short) / 0(hold) based on forward return."""
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[fwd_ret > thresh] = 1
    labels[fwd_ret < -thresh] = -1
    return labels

def build_dataset(
    symbol: str,
    timeframe: str,
    limit: int,
    horizon: int,
    exchange_name: str = "binance",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Full pipeline: fetch data → features → labels."""
    raw_df = fetch_ohlcv(symbol, timeframe, limit, exchange_name)

    features = build_features(raw_df)
    labels = label_forward_return(raw_df, horizon)

    # Align lengths
    aligned = features.join(labels.rename("label"), how="inner").dropna()
    X = aligned.drop(columns=["label"])
    y = aligned["label"].astype(int)
    return X, y

def cli():
    parser = argparse.ArgumentParser(description="Build ML dataset for trading model")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=4, help="Label horizon in candles")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--output", required=True, help="Output Parquet file path")

    args = parser.parse_args()

    X, y = build_dataset(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        horizon=args.horizon,
        exchange_name=args.exchange,
    )

    df_out = X.copy()
    df_out["label"] = y
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path)
    print(f"✅ Dataset saved: {output_path} — rows: {len(df_out)}, features: {X.shape[1]}")

if __name__ == "__main__":
    cli() 
"""
Model Training Module

This module provides functions for training Smart Money Concepts models.
"""

import os
import logging
import json
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from ..data.data_loader import MarketDataLoader
from ..models.model_registry import ModelRegistry
from ..utils.metrics import calculate_metrics
from ..models.base_model import ModelFactory

# Configure logging
logger = logging.getLogger(__name__)

def train_model(
    symbol: str,
    model_type: str,
    data_path: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    sequence_length: int = 60,
    forecast_horizon: int = 1,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    class_weights_mode: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a model for the specified symbol.
    
    Args:
        symbol: Trading symbol (e.g., "BTC-USDT")
        model_type: Type of model to train (e.g., "smc_transformer")
        data_path: Path to load data from
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
        sequence_length: Number of time steps in input sequence
        forecast_horizon: Number of time steps to predict
        batch_size: Batch size for training
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        class_weights_mode: Mode for class weighting (e.g., "balanced", None)
        **kwargs: Additional model-specific parameters
        
    Returns:
        Dictionary containing model info and metrics
    """
    logger.info(f"Training {model_type} model for {symbol}")
    
    # Load data (raw DataFrame and feature engineering)
    loader = MarketDataLoader(timeframe='1h', symbols=[symbol.replace('USD', '/USDT')])
    try:
        df = loader.load_from_csv(symbol=symbol.replace('USD', '/USDT'), file_path=data_path)
    except Exception as e:
        logger.warning(f"Could not load from CSV: {e}. Fetching from exchange instead.")
        df_dict = loader.fetch_historical_data(start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), end_date=datetime.now().strftime('%Y-%m-%d'))
        df = df_dict.get(symbol.replace('USD', '/USDT'))
    if df is None:
        raise ValueError(f"No data found for symbol {symbol}")
    df_proc = loader.preprocess_for_smc(df)
    X, y = loader.create_features(df_proc)

    # Create trainer
    from .trainer import ModelTrainer
    input_dim = X.shape[1]
    trainer = ModelTrainer(
        model_type=model_type,
        input_shape=(sequence_length, input_dim),
        output_units=forecast_horizon,
        batch_size=batch_size,
        epochs=num_epochs,
        patience=early_stopping_patience,
        learning_rate=learning_rate,
        model_dir="models",
        log_dir="logs",
        experiment_name=None,
        random_state=42
    )
    # Preprocess and split data using ModelTrainer
    # Reshape X to (samples, sequence_length, features) for LSTM/GRU
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    X_train, X_val, y_train, y_val = trainer.preprocess_data(X_seq, y_seq, scaling_method='standard', target_scaling=False)
    # Create DataLoaders
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # For test set, use the remaining data
    X_test = X_seq[-len(X_val):]
    y_test = y_seq[-len(y_val):]
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainer.model = ModelFactory.create_model(
        model_type=model_type,
        input_dim=input_dim,
        output_dim=forecast_horizon,
        seq_len=sequence_length,
        forecast_horizon=forecast_horizon,
        **kwargs
    )
    # Train the model
    trainer.fit(train_loader, val_loader, class_weights_mode=class_weights_mode)
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    
    # Collect fitted preprocessors if available
    preprocessor = None
    if hasattr(trainer, 'feature_scaler') and hasattr(trainer, 'target_scaler'):
        if getattr(trainer, 'feature_scaler', None) is not None and getattr(trainer, 'target_scaler', None) is not None:
            preprocessor = {
                'feature_scaler': trainer.feature_scaler,
                'target_scaler': trainer.target_scaler
            }
        elif getattr(trainer, 'feature_scaler', None) is not None:
            preprocessor = trainer.feature_scaler
        elif getattr(trainer, 'target_scaler', None) is not None:
            preprocessor = trainer.target_scaler
    elif hasattr(trainer, 'feature_scaler') and getattr(trainer, 'feature_scaler', None) is not None:
        preprocessor = trainer.feature_scaler
    elif hasattr(trainer, 'target_scaler') and getattr(trainer, 'target_scaler', None) is not None:
        preprocessor = trainer.target_scaler

    # Save the model and preprocessor
    version = ModelRegistry().save_model(
        model=trainer.model,
        symbol=symbol,
        metrics=test_metrics,
        metadata={
            "sequence_length": sequence_length,
            "forecast_horizon": forecast_horizon,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            **kwargs
        },
        preprocessor=preprocessor
    )
    return {"version": version, "metrics": test_metrics} 
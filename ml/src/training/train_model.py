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

from ..data.unified_data_processor import UnifiedDataProcessor
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
    data_processor = UnifiedDataProcessor(
        timeframe='1h', 
        symbols=[symbol.replace('USD', '/USDT')],
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon
    )

    if data_path:
        df = data_processor.load_from_csv(symbol=symbol.replace('USD', '/USDT'), file_path=data_path)
    else:
        df_dict = data_processor.fetch_historical_data(start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), end_date=datetime.now().strftime('%Y-%m-%d'))
        df = df_dict.get(symbol.replace('USD', '/USDT'))

    if df is None:
        raise ValueError(f"No data found for symbol {symbol}")

    processed_data = data_processor.fit_transform(
        df,
        target_column='close',
        train_split=train_ratio
    )

    X_train, y_train = processed_data['X_train'], processed_data['y_train']
    X_val, y_val = processed_data['X_val'], processed_data['y_val']
    X_test, y_test = processed_data['X_test'], processed_data['y_test']

    # Create DataLoaders
    train_dataset = data_processor.create_pytorch_datasets(processed_data)[0]
    val_dataset = data_processor.create_pytorch_datasets(processed_data)[1]
    test_dataset = data_processor.create_pytorch_datasets(processed_data)[2]

    train_loader, val_loader, test_loader = data_processor.create_pytorch_dataloaders(
        (train_dataset, val_dataset, test_dataset),
        batch_size=batch_size
    )

    # Create trainer
    from .trainer import ModelTrainer
    input_dim = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
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
    preprocessor = data_processor.scalers # UnifiedDataProcessor stores scalers in .scalers

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
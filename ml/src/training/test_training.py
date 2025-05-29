"""
Test script for training modules

This script tests the functionality of the training modules with mock data.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import LSTMModel, GRUModel, ModelFactory
from src.training import (
    Trainer, 
    TimeSeriesDataset, 
    prepare_time_series_data,
    evaluate_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_time_series_data(n_samples=1000, n_features=10):
    """
    Create mock time series data for testing
    
    Args:
        n_samples: Number of time points
        n_features: Number of features
        
    Returns:
        DataFrame with time series data
    """
    # Create timestamps
    start_date = datetime(2022, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Create feature data with some patterns
    data = {}
    
    # Create a price series with trend and cycles
    t = np.linspace(0, 4*np.pi, n_samples)
    price = 1000 + 10 * t + 50 * np.sin(t) + 20 * np.sin(0.5*t) + 10 * np.random.randn(n_samples)
    
    # Add OHLCV data
    data['timestamp'] = timestamps
    data['open'] = price
    data['high'] = price * (1 + 0.01 * np.random.rand(n_samples))
    data['low'] = price * (1 - 0.01 * np.random.rand(n_samples))
    data['close'] = price * (1 + 0.005 * (np.random.rand(n_samples) - 0.5))
    data['volume'] = 1000 + 500 * np.random.rand(n_samples) + 300 * np.sin(0.5*t)
    
    # Add some additional features
    for i in range(n_features - 5):  # -5 because we already added 5 features (OHLCV)
        if i % 3 == 0:
            # Trend feature
            data[f'feature_{i}'] = 0.5 * t + 0.1 * np.random.randn(n_samples)
        elif i % 3 == 1:
            # Cyclical feature
            data[f'feature_{i}'] = np.sin(0.1 * i * t) + 0.05 * np.random.randn(n_samples)
        else:
            # Random feature
            data[f'feature_{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


def test_time_series_dataset():
    """Test TimeSeriesDataset class"""
    logger.info("Testing TimeSeriesDataset")
    
    # Create mock data
    n_samples = 500
    n_features = 5
    seq_len = 50
    forecast_horizon = 5
    
    # Create feature data
    X = np.random.randn(n_samples, n_features)
    y = np.sin(np.linspace(0, 10, n_samples))  # Simple sine wave target
    
    # Create dataset
    dataset = TimeSeriesDataset(X, y, seq_len, forecast_horizon)
    
    # Test dataset size
    expected_size = max(0, n_samples - seq_len - forecast_horizon + 1)
    assert len(dataset) == expected_size, f"Expected {expected_size} sequences, got {len(dataset)}"
    
    # Test getting an item
    sequence, target = dataset[0]
    assert sequence.shape == (seq_len, n_features), f"Expected sequence shape {(seq_len, n_features)}, got {sequence.shape}"
    assert target.shape == (forecast_horizon, 1), f"Expected target shape {(forecast_horizon, 1)}, got {target.shape}"
    
    logger.info("TimeSeriesDataset tests passed")


def test_data_preparation():
    """Test data preparation functions"""
    logger.info("Testing data preparation functions")
    
    # Create mock data
    data = create_mock_time_series_data(n_samples=500, n_features=10)
    
    # Test prepare_time_series_data
    data_dict = prepare_time_series_data(
        data=data,
        target_column='close',
        seq_len=50,
        forecast_horizon=5,
        batch_size=16,
        val_ratio=0.2,
        test_ratio=0.1,
        num_workers=0  # Use 0 workers for testing
    )
    
    # Check keys
    expected_keys = [
        'train_loader', 'val_loader', 'test_loader',
        'feature_columns', 'target_column', 'seq_len', 'forecast_horizon',
        'input_dim', 'output_dim', 'train_size', 'val_size', 'test_size'
    ]
    for key in expected_keys:
        assert key in data_dict, f"Missing key: {key}"
    
    # Check data loaders
    batch = next(iter(data_dict['train_loader']))
    X_batch, y_batch = batch
    
    assert X_batch.shape[0] == 16, f"Expected batch size 16, got {X_batch.shape[0]}"
    assert X_batch.shape[1] == 50, f"Expected sequence length 50, got {X_batch.shape[1]}"
    assert X_batch.shape[2] == data_dict['input_dim'], f"Expected {data_dict['input_dim']} features, got {X_batch.shape[2]}"
    
    assert y_batch.shape[0] == 16, f"Expected batch size 16, got {y_batch.shape[0]}"
    assert y_batch.shape[1] == 5, f"Expected forecast horizon 5, got {y_batch.shape[1]}"
    
    logger.info("Data preparation tests passed")


def test_trainer():
    """Test Trainer class with a small model"""
    logger.info("Testing Trainer class")
    
    # Create a small model for testing
    model = LSTMModel(
        input_dim=5,
        output_dim=1,
        seq_len=10,
        forecast_horizon=2,
        hidden_dim=8,
        num_layers=1
    )
    
    # Create small datasets
    X_train = np.random.randn(100, 10, 5)  # 100 sequences of length 10 with 5 features
    y_train = np.random.randn(100, 2, 1)   # 100 targets of length 2 with 1 feature
    
    X_val = np.random.randn(20, 10, 5)
    y_val = np.random.randn(20, 2, 1)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create trainer
    experiment_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("logs", "tensorboard", experiment_name)
    checkpoints_dir = os.path.join("models", "checkpoints", experiment_name)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        log_dir=log_dir,
        checkpoints_dir=checkpoints_dir,
        experiment_name=experiment_name
    )
    
    # Train for a few epochs
    train_losses, val_losses = trainer.train(num_epochs=2, early_stopping_patience=5)
    
    # Check training results
    assert len(train_losses) == 2, f"Expected 2 training losses, got {len(train_losses)}"
    assert len(val_losses) == 2, f"Expected 2 validation losses, got {len(val_losses)}"
    
    # Check that checkpoint files were created
    assert os.path.exists(os.path.join(checkpoints_dir, "best.pt")), "Best checkpoint file not created"
    
    logger.info("Trainer tests passed")


def test_model_factory():
    """Test ModelFactory"""
    logger.info("Testing ModelFactory")
    
    # Create a model from the factory
    model = ModelFactory.create_model(
        model_type="lstm",
        input_dim=5,
        output_dim=1,
        seq_len=10,
        forecast_horizon=2,
        hidden_dim=8,
        num_layers=1
    )
    
    # Test if it's an LSTM model
    assert isinstance(model, LSTMModel), f"Expected LSTMModel, got {type(model)}"
    
    # Create a GRU model
    model = ModelFactory.create_model(
        model_type="gru",
        input_dim=5,
        output_dim=1,
        seq_len=10,
        forecast_horizon=2,
        hidden_dim=8,
        num_layers=1
    )
    
    # Test if it's a GRU model
    assert isinstance(model, GRUModel), f"Expected GRUModel, got {type(model)}"
    
    logger.info("ModelFactory tests passed")


def test_end_to_end():
    """Test end-to-end training and evaluation"""
    logger.info("Testing end-to-end training and evaluation")
    
    # Create mock data
    data = create_mock_time_series_data(n_samples=300, n_features=10)
    
    # Prepare datasets
    data_dict = prepare_time_series_data(
        data=data,
        target_column='close',
        seq_len=30,
        forecast_horizon=5,
        batch_size=16,
        val_ratio=0.2,
        test_ratio=0.1,
        num_workers=0  # Use 0 workers for testing
    )
    
    # Create a model
    model = ModelFactory.create_model(
        model_type="lstm",
        input_dim=data_dict['input_dim'],
        output_dim=data_dict['output_dim'],
        seq_len=data_dict['seq_len'],
        forecast_horizon=data_dict['forecast_horizon'],
        hidden_dim=16,
        num_layers=1
    )
    
    # Create trainer
    experiment_name = f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("logs", "tensorboard", experiment_name)
    checkpoints_dir = os.path.join("models", "checkpoints", experiment_name)
    
    trainer = Trainer(
        model=model,
        train_dataloader=data_dict['train_loader'],
        val_dataloader=data_dict['val_loader'],
        log_dir=log_dir,
        checkpoints_dir=checkpoints_dir,
        experiment_name=experiment_name
    )
    
    # Train for a few epochs
    train_losses, val_losses = trainer.train(num_epochs=2, early_stopping_patience=5)
    
    # Evaluate model
    results = evaluate_model(model, data_dict['test_loader'])
    metrics = results['metrics']
    
    logger.info(f"Test metrics: RMSE={metrics['rmse']:.4f}, Directional Accuracy={metrics['directional_accuracy']:.4f}")
    
    logger.info("End-to-end training and evaluation tests passed")


if __name__ == "__main__":
    logger.info("Running training module tests")
    
    # Test individual components
    test_time_series_dataset()
    test_data_preparation()
    test_model_factory()
    test_trainer()
    
    # Test end-to-end
    test_end_to_end()
    
    logger.info("All tests passed!") 
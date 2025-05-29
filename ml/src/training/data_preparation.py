"""
Time Series Data Preparation for Model Training

This module provides utilities for preparing time series data for model training,
including creating sliding window datasets, train-validation-test splits, and DataLoaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union, Dict
from sklearn.model_selection import TimeSeriesSplit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with sliding windows"""
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
        forecast_horizon: int,
        step: int = 1,
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            targets: Target data array of shape (n_samples,) or (n_samples, target_dim)
            seq_len: Length of input sequence
            forecast_horizon: Number of steps to forecast
            step: Step size for sliding window
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.step = step
        
        # Calculate valid start indices
        self.valid_indices = []
        
        for i in range(0, len(data) - seq_len - forecast_horizon + 1, step):
            if i + seq_len + forecast_horizon <= len(data):
                self.valid_indices.append(i)
        
        logger.info(f"Created TimeSeriesDataset with {len(self.valid_indices)} sequences")
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (sequence, target)
        """
        # Get start index for this sequence
        start_idx = self.valid_indices[idx]
        
        # Get sequence
        sequence = self.data[start_idx:start_idx + self.seq_len]
        
        # Get target sequence
        target = self.targets[start_idx + self.seq_len:start_idx + self.seq_len + self.forecast_horizon]
        
        # Reshape target if it's 1D
        if len(target.shape) == 1:
            target = target.unsqueeze(-1)
        
        return sequence, target


def prepare_time_series_data(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    seq_len: int = 60,
    forecast_horizon: int = 5,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    step: int = 1,
    shuffle_train: bool = True,
    num_workers: int = 4,
) -> Dict[str, Union[DataLoader, np.ndarray]]:
    """
    Prepare time series data for model training.
    
    Args:
        data: Input DataFrame containing time series data
        target_column: Name of the target column
        feature_columns: List of feature column names (if None, use all columns except target)
        seq_len: Length of input sequence
        forecast_horizon: Number of steps to forecast
        batch_size: Batch size for DataLoaders
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        step: Step size for sliding window
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for DataLoaders
        
    Returns:
        Dictionary containing DataLoaders and data info
    """
    logger.info(f"Preparing time series data with {len(data)} samples")
    
    # Ensure data is sorted by time
    if 'timestamp' in data.columns or 'date' in data.columns:
        time_col = 'timestamp' if 'timestamp' in data.columns else 'date'
        data = data.sort_values(by=time_col)
    
    # Select feature columns if not provided
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column and 'timestamp' not in col and 'date' not in col]
    
    logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Extract features and target
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Split into train, validation, and test sets
    # For time series data, we need to split sequentially
    test_size = int(len(data) * test_ratio)
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size - test_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len, forecast_horizon, step)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len, forecast_horizon, step)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len, forecast_horizon, step)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Return everything in a dictionary
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_columns': feature_columns,
        'target_column': target_column,
        'seq_len': seq_len,
        'forecast_horizon': forecast_horizon,
        'input_dim': len(feature_columns),
        'output_dim': 1 if len(y.shape) == 1 else y.shape[1],
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
    }


def prepare_multi_target_data(
    data: pd.DataFrame,
    target_columns: List[str],
    feature_columns: Optional[List[str]] = None,
    seq_len: int = 60,
    forecast_horizon: int = 5,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    step: int = 1,
    shuffle_train: bool = True,
    num_workers: int = 4,
) -> Dict[str, Union[DataLoader, np.ndarray]]:
    """
    Prepare multi-target time series data for model training.
    
    Args:
        data: Input DataFrame containing time series data
        target_columns: List of target column names
        feature_columns: List of feature column names (if None, use all columns except targets)
        seq_len: Length of input sequence
        forecast_horizon: Number of steps to forecast
        batch_size: Batch size for DataLoaders
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        step: Step size for sliding window
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for DataLoaders
        
    Returns:
        Dictionary containing DataLoaders and data info
    """
    logger.info(f"Preparing multi-target time series data with {len(data)} samples")
    
    # Ensure data is sorted by time
    if 'timestamp' in data.columns or 'date' in data.columns:
        time_col = 'timestamp' if 'timestamp' in data.columns else 'date'
        data = data.sort_values(by=time_col)
    
    # Select feature columns if not provided
    if feature_columns is None:
        feature_columns = [col for col in data.columns 
                          if col not in target_columns 
                          and 'timestamp' not in col 
                          and 'date' not in col]
    
    logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
    logger.info(f"Predicting {len(target_columns)} targets: {target_columns}")
    
    # Extract features and targets
    X = data[feature_columns].values
    y = data[target_columns].values
    
    # Split into train, validation, and test sets
    # For time series data, we need to split sequentially
    test_size = int(len(data) * test_ratio)
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size - test_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len, forecast_horizon, step)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len, forecast_horizon, step)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len, forecast_horizon, step)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Return everything in a dictionary
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'feature_columns': feature_columns,
        'target_columns': target_columns,
        'seq_len': seq_len,
        'forecast_horizon': forecast_horizon,
        'input_dim': len(feature_columns),
        'output_dim': len(target_columns),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
    }


def create_walk_forward_cv_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 30,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward cross-validation splits for time series data.
    
    Args:
        data: Input DataFrame containing time series data
        n_splits: Number of splits
        test_size: Size of test set
        gap: Gap between train and test sets
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    splits = []
    
    for train_idx, test_idx in tscv.split(data):
        splits.append((train_idx, test_idx))
    
    return splits 
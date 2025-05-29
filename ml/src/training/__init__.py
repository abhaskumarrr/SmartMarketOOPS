"""
Training Package

This package provides utilities for training and evaluating time series forecasting models
with TensorBoard integration and Apple Silicon optimization.
"""

from .trainer import ModelTrainer as Trainer
from .data_preparation import (
    TimeSeriesDataset, 
    prepare_time_series_data, 
    prepare_multi_target_data,
    create_walk_forward_cv_splits
)

__all__ = [
    'Trainer',
    'TimeSeriesDataset',
    'prepare_time_series_data',
    'prepare_multi_target_data',
    'create_walk_forward_cv_splits',
] 
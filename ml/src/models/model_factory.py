from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

from .base_model import BaseModel
from .lstm_model import LSTMModel
from .gru_model import GRUModel
from .transformer_model import TransformerModel
from .cnn_lstm_model import CNNLSTMModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating different model architectures.
    Allows easy switching between model types for experimentation.
    """

    @staticmethod
    def create_model(
        model_type: str,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        forecast_horizon: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: Optional[str] = None,
        **kwargs: Any
    ) -> BaseModel:
        """
        Create a model instance of the specified type.

        Args:
            model_type: Type of model to create ('lstm', 'gru', 'transformer', 'cnn_lstm')
            input_dim: Number of input features
            output_dim: Number of output features
            seq_len: Length of input sequences
            forecast_horizon: Number of steps to forecast
            hidden_dim: Size of hidden layers
            num_layers: Number of layers (for RNNs and Transformer)
            dropout: Dropout rate
            device: Device to use for computation
            **kwargs: Additional model-specific parameters

        Returns:
            Model instance
        """
        print(f"ModelFactory receiving model_type: {model_type}")
        model_type = model_type.lower()

        if model_type == 'lstm':
            return LSTMModel(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                **kwargs
            )
        elif model_type == 'gru':
            return GRUModel(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                **kwargs
            )
        elif model_type == 'transformer':
            return TransformerModel(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                forecast_horizon=forecast_horizon,
                d_model=hidden_dim,
                nhead=kwargs.get('nhead', 4),
                num_layers=num_layers,
                dropout=dropout,
                device=device,
                **kwargs
            )
        elif model_type == 'cnn_lstm':
             # Explicitly map parameters from create_model and kwargs to CNNLSTMModel constructor arguments
             cnn_input_size = input_dim # Map input_dim to input_size
             # Get cnn_channels from kwargs first, with a default if not found
             cnn_cnn_channels = kwargs.get('cnn_channels', 64)
             cnn_lstm_hidden = hidden_dim # Map hidden_dim to lstm_hidden
             cnn_lstm_layers = num_layers # Map num_layers to lstm_layers
             cnn_dropout = dropout # Map dropout
             cnn_num_classes = output_dim # Map output_dim to num_classes
             print(f"Attempting to instantiate: {CNNLSTMModel}") # Added print statement to inspect the class object
             return CNNLSTMModel(
                input_size=cnn_input_size,
                cnn_channels=cnn_cnn_channels,
                lstm_hidden=cnn_lstm_hidden,
                lstm_layers=cnn_lstm_layers,
                dropout=cnn_dropout,
                num_classes=cnn_num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}") 
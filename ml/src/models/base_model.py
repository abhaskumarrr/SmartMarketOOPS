"""
Base Model for all model implementations
Provides common functionality and interface
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
# from .model_factory import ModelFactory # Removed circular import

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all models.
    Enforces a consistent interface for all models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the base model.
        
        Args:
            model_path: Path to a saved model file (if loading a pre-trained model)
        """
        self.model = None
    
    @abstractmethod
    def _build_model(self):
        """
        Build the model architecture.
        To be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def fit_model(
        self,
        train_loader,
        val_loader,
        num_epochs=100,
        lr=0.001,
        early_stopping_patience=10,
        checkpoint_dir=None
    ):
        """
        Train the model.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, model_path: str) -> None:
        """
        Save the model.
        
        Args:
            model_path: Path to save the model
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, model_path: str, **kwargs):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model
        """
        pass


class DirectionalLoss(nn.Module):
    """
    Custom loss function that penalizes incorrect direction predictions more heavily.
    
    This is especially important for trading models where predicting the direction
    correctly (up/down) is often more important than the exact magnitude.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 2.0):
        """
        Initialize the directional loss function.
        
        Args:
            alpha: Weight for the MSE component (0-1)
            beta: Multiplier for incorrect direction predictions
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the directional loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss value
        """
        # Calculate MSE component
        mse_loss = self.mse(y_pred, y_true)
        
        # Calculate directional component
        # 1 if direction is the same, 0 if different
        direction_pred = (y_pred[:, 1:] - y_pred[:, :-1]) > 0
        direction_true = (y_true[:, 1:] - y_true[:, :-1]) > 0
        
        # Calculate directional accuracy
        direction_match = (direction_pred == direction_true).float()
        direction_loss = 1.0 - direction_match.mean()
        
        # Combined loss with higher penalty for incorrect directions
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * self.beta * direction_loss
        
        return combined_loss


# ModelFactory class was moved to model_factory.py 
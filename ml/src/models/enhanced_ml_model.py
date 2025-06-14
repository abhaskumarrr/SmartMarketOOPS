"""
Enhanced ML Model Base Class
Serves as a foundation for all ML models in the SmartMarketOOPS system
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedMLModel(nn.Module, ABC):
    """
    Base class for all ML models in the SmartMarketOOPS system.
    Implements common functionality, persistence, and evaluation metrics.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_name: str = "enhanced_ml_model",
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        # Model architecture parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        
        # Determine device
        self.device = self._determine_device(device)
        
        # Training history
        self.training_history = {
            "train_losses": [],
            "val_losses": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "learning_rates": [],
            "epochs": 0,
            "best_val_loss": float("inf"),
            "best_val_accuracy": 0.0,
            "training_time": 0,
            "last_trained": None
        }
        
        # Performance metrics
        self.metrics = {}
        
        # Version tracking
        self.version = kwargs.get("version", "1.0.0")
        self.creation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized {self.model_name} (v{self.version}) on {self.device}")
    
    def _determine_device(self, device: Optional[str]) -> torch.device:
        """Determine the appropriate device for the model"""
        if device is not None:
            return torch.device(device)
        
        if torch.cuda.is_available():
            selected_device = torch.device("cuda:0")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Check for Apple M1/M2 GPU support
            selected_device = torch.device("mps")
            logger.info("Using Apple MPS (Metal Performance Shaders)")
        else:
            selected_device = torch.device("cpu")
            logger.info("Using CPU")
        
        return selected_device
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model (must be implemented by subclasses)"""
        pass
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        lr: float = 0.001,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model with early stopping and checkpointing
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of training epochs
            lr: Learning rate
            early_stopping_patience: Epochs to wait before early stopping
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Dictionary with training history
        """
        start_time = time.time()
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Set up loss function based on output dimension
        if self.output_dim == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for X, y in train_loader:
                X = X.to(self.device).float()
                
                if self.output_dim == 1:
                    y = y.to(self.device).float()
                else:
                    y = y.to(self.device).long()
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self(X)
                
                if self.output_dim == 1:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y)
                    # Calculate accuracy for binary classification
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    loss = criterion(outputs, y)
                    # Calculate accuracy for multi-class classification
                    _, predicted = torch.max(outputs.data, 1)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Accumulate metrics
                train_loss += loss.item() * X.size(0)
                
                if self.output_dim == 1:
                    train_correct += (predicted == y).sum().item()
                else:
                    train_correct += (predicted == y).sum().item()
                
                train_total += y.size(0)
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / train_total
            epoch_train_accuracy = train_correct / train_total
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(self.device).float()
                    
                    if self.output_dim == 1:
                        y = y.to(self.device).float()
                    else:
                        y = y.to(self.device).long()
                    
                    outputs = self(X)
                    
                    if self.output_dim == 1:
                        outputs = outputs.squeeze(-1)
                        loss = criterion(outputs, y)
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                    else:
                        loss = criterion(outputs, y)
                        _, predicted = torch.max(outputs.data, 1)
                    
                    val_loss += loss.item() * X.size(0)
                    
                    if self.output_dim == 1:
                        val_correct += (predicted == y).sum().item()
                    else:
                        val_correct += (predicted == y).sum().item()
                    
                    val_total += y.size(0)
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / val_total
            epoch_val_accuracy = val_correct / val_total
            
            # Update learning rate
            scheduler.step(epoch_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Train Acc: {epoch_train_accuracy:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc: {epoch_val_accuracy:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Update training history
            self.training_history["train_losses"].append(epoch_train_loss)
            self.training_history["val_losses"].append(epoch_val_loss)
            self.training_history["train_accuracies"].append(epoch_train_accuracy)
            self.training_history["val_accuracies"].append(epoch_val_accuracy)
            self.training_history["learning_rates"].append(current_lr)
            self.training_history["epochs"] = epoch + 1
            
            # Check for improvement
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                self.training_history["best_val_loss"] = best_val_loss
                self.training_history["best_val_accuracy"] = epoch_val_accuracy
                patience_counter = 0
                
                # Save checkpoint
                if checkpoint_dir is not None:
                    self.save(os.path.join(checkpoint_dir, f"{self.model_name}_best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Calculate training time
        training_time = time.time() - start_time
        self.training_history["training_time"] = training_time
        self.training_history["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.training_history
    
    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input data tensor or numpy array
            
        Returns:
            Numpy array of predictions
        """
        self.eval()
        
        # Convert numpy array to tensor if needed
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Ensure input is on the correct device
        X = X.to(self.device).float()
        
        with torch.no_grad():
            outputs = self(X)
            
            if self.output_dim == 1:
                outputs = outputs.squeeze(-1)
                # Apply sigmoid for binary classification
                predictions = torch.sigmoid(outputs).cpu().numpy()
            else:
                # Apply softmax for multi-class classification
                predictions = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return predictions
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Set up loss function based on output dimension
        if self.output_dim == 1:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device).float()
                
                if self.output_dim == 1:
                    y = y.to(self.device).float()
                else:
                    y = y.to(self.device).long()
                
                outputs = self(X)
                
                if self.output_dim == 1:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y)
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).float()
                else:
                    loss = criterion(outputs, y)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(outputs.data, 1)
                
                test_loss += loss.item() * X.size(0)
                
                # Collect predictions and targets
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Convert to numpy arrays
        y_pred = np.array(all_predictions)
        y_true = np.array(all_targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred)
        metrics["test_loss"] = test_loss / len(test_loader.dataset)
        
        # Store metrics
        self.metrics.update(metrics)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate accuracy
        metrics["accuracy"] = np.mean(y_pred == y_true)
        
        if self.output_dim == 1:
            # Binary classification metrics
            try:
                from sklearn.metrics import (
                    precision_score, recall_score, f1_score, 
                    roc_auc_score, average_precision_score
                )
                
                metrics["precision"] = precision_score(y_true, y_pred)
                metrics["recall"] = recall_score(y_true, y_pred)
                metrics["f1_score"] = f1_score(y_true, y_pred)
                
                # Try to calculate ROC AUC and PR AUC
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
                    metrics["pr_auc"] = average_precision_score(y_true, y_pred)
                except:
                    logger.warning("Could not calculate ROC AUC or PR AUC")
            
            except ImportError:
                logger.warning("scikit-learn not available for detailed metrics calculation")
        
        else:
            # Multi-class classification metrics
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                metrics["precision_macro"] = precision_score(y_true, y_pred, average='macro')
                metrics["recall_macro"] = recall_score(y_true, y_pred, average='macro')
                metrics["f1_score_macro"] = f1_score(y_true, y_pred, average='macro')
                
                metrics["precision_weighted"] = precision_score(y_true, y_pred, average='weighted')
                metrics["recall_weighted"] = recall_score(y_true, y_pred, average='weighted')
                metrics["f1_score_weighted"] = f1_score(y_true, y_pred, average='weighted')
            
            except ImportError:
                logger.warning("scikit-learn not available for detailed metrics calculation")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save model parameters and metadata
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        state_dict = {
            "model_state": self.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "model_name": self.model_name,
            "version": self.version,
            "creation_timestamp": self.creation_timestamp,
            "training_history": self.training_history,
            "metrics": self.metrics
        }
        
        torch.save(state_dict, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata separately for easy inspection
        metadata_path = filepath.replace(".pt", "_metadata.json")
        with open(metadata_path, "w") as f:
            # Convert any non-serializable objects
            metadata = {
                "model_name": self.model_name,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "version": self.version,
                "creation_timestamp": self.creation_timestamp,
                "training_history": self.training_history,
                "metrics": self.metrics
            }
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "EnhancedMLModel":
        """
        Load model from file
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        state_dict = torch.load(filepath, map_location="cpu")
        
        # Extract model parameters
        input_dim = state_dict["input_dim"]
        output_dim = state_dict["output_dim"]
        model_name = state_dict["model_name"]
        version = state_dict.get("version", "1.0.0")
        
        # Create new model instance
        model = cls(
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=model_name,
            device=device,
            version=version
        )
        
        # Load model state
        model.load_state_dict(state_dict["model_state"])
        
        # Load training history and metrics if available
        if "training_history" in state_dict:
            model.training_history = state_dict["training_history"]
        
        if "metrics" in state_dict:
            model.metrics = state_dict["metrics"]
        
        if "creation_timestamp" in state_dict:
            model.creation_timestamp = state_dict["creation_timestamp"]
        
        model.eval()
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            Dictionary of model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "version": self.version,
            "creation_timestamp": self.creation_timestamp,
            "training_history": self.training_history,
            "metrics": self.metrics
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get parameter count statistics
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params
        } 
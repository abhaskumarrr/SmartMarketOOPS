"""
Comprehensive ML Training Pipeline for Smart Money Concepts trading models
"""

import os
import logging
import numpy as np
import pandas as pd
import json
import time
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Type
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

# Import models
from ..models.base_model import BaseModel
from ..models.lstm_model import LSTMModel
from ..models.gru_model import GRUModel
from ..models.transformer_model import TransformerModel
from ..models.cnn_lstm_model import CNNLSTMModel

# Setup logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    PyTorch-based model trainer for Smart Money Concepts trading models.
    Handles data preprocessing, model initialization, training, evaluation, saving, and experiment tracking.
    """
    def __init__(
        self,
        model_type: str,
        input_shape: Tuple[int, int],
        output_units: int = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10, 
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        model_dir: str = 'models',
        log_dir: str = 'logs',
        experiment_name: Optional[str] = None,
        random_state: int = 42,
        **model_params
    ):
        self.model_type = model_type.lower()
        self.input_shape = input_shape
        self.output_units = output_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.random_state = random_state
        self.model_params = model_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model().to(self.device)
    
    def _create_model(self) -> BaseModel:
        common_params = {
            'input_dim': self.input_shape[1],
            'output_dim': self.output_units,
            'seq_len': self.input_shape[0],
            'forecast_horizon': 1,
            'device': self.device,
            **self.model_params
        }
        if self.model_type == 'lstm':
            return LSTMModel(**common_params)
        elif self.model_type == 'gru':
            return GRUModel(**common_params)
        elif self.model_type == 'transformer':
            return TransformerModel(**common_params)
        elif self.model_type == 'cnn_lstm':
            return CNNLSTMModel(**common_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def fit(self, train_loader, val_loader, class_weights_mode: Optional[str] = None):
        logger.info(f"Training {self.model_type} model on device {self.device}")
        
        calculated_class_weights = None
        if class_weights_mode == "balanced" and hasattr(train_loader.dataset, 'tensors'):
            y_train_tensor = train_loader.dataset.tensors[1]
            if y_train_tensor is not None:
                y_train_numpy = y_train_tensor.cpu().numpy()
                if self.output_units == 1: # Binary classification
                    # Ensure y_train_numpy is 1D for compute_class_weight
                    y_train_numpy = y_train_numpy.flatten()
                    unique_classes = np.unique(y_train_numpy)
                    if len(unique_classes) == 2: # Expecting 0 and 1
                        weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train_numpy)
                        # BCEWithLogitsLoss expects pos_weight for the positive class (class 1)
                        # Order of weights from compute_class_weight corresponds to np.unique(classes)
                        # If unique_classes is [0, 1], then weights[1] is for class 1.
                        calculated_class_weights = torch.tensor(weights[1], dtype=torch.float32) 
                        logger.info(f"Calculated pos_weight for BCEWithLogitsLoss: {calculated_class_weights.item()}")
                    else:
                        logger.warning(f"Cannot compute balanced class weights for binary classification. Expected 2 unique classes, found {len(unique_classes)}. Proceeding without class weights.")
                else: # Multi-class classification
                    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_numpy), y=y_train_numpy)
                    calculated_class_weights = torch.tensor(weights, dtype=torch.float32)
                    logger.info(f"Calculated class weights for CrossEntropyLoss: {calculated_class_weights}")
            else:
                logger.warning("Could not extract labels from train_loader to compute class weights.")
        elif class_weights_mode is not None and class_weights_mode != "balanced":
            logger.warning(f"Unsupported class_weights_mode: {class_weights_mode}. Proceeding without class weights.")

        history = self.model.fit_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.epochs,
            lr=self.learning_rate,
            early_stopping_patience=self.patience,
            checkpoint_dir=self.model_dir,
            class_weights=calculated_class_weights
        )
        return history

    def evaluate(self, test_loader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device).float()
                y = y.cpu().numpy()
                outputs = self.model(X)
                if self.output_units == 1:
                    preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                    preds = (preds > 0.5).astype(int)
                else:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_true.extend(y.flatten())
                y_pred.extend(preds)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1_score': f1_score(y_true, y_pred, average='macro')
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save(self, model_path: str):
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def preprocess_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scaling_method: str = 'standard',
        target_scaling: bool = False,
        test_size: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data for training.
        
        Args:
            X: Input features
            y: Target values
            scaling_method: Scaling method ('standard', 'minmax', 'none')
            target_scaling: Whether to scale targets
            test_size: Size of test split (if None, use validation_split)
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # Use validation split as test size if not specified
        if test_size is None:
            test_size = self.validation_split
            
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Apply feature scaling
        if scaling_method.lower() != 'none':
            # Scale each feature independently across samples and time
            orig_shape = X_train.shape
            
            # Reshape to 2D for scaling: (samples*time, features)
            X_train_2d = X_train.reshape(-1, X_train.shape[2])
            X_val_2d = X_val.reshape(-1, X_val.shape[2])
            
            if scaling_method.lower() == 'standard':
                self.feature_scaler = StandardScaler()
            elif scaling_method.lower() == 'minmax':
                self.feature_scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {scaling_method}")
            
            # Fit scaler on training data only
            X_train_2d = self.feature_scaler.fit_transform(X_train_2d)
            X_val_2d = self.feature_scaler.transform(X_val_2d)
            
            # Reshape back to original shape
            X_train = X_train_2d.reshape(orig_shape)
            X_val = X_val_2d.reshape(X_val.shape)
            
            logger.info(f"Applied {scaling_method} scaling to features")
        
        # Apply target scaling for regression problems
        if target_scaling and self.output_units == 1 and len(y_train.shape) == 2:
            if scaling_method.lower() == 'standard':
                self.target_scaler = StandardScaler()
            elif scaling_method.lower() == 'minmax':
                self.target_scaler = MinMaxScaler()
                
            # Reshape if needed
            y_train_2d = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
            y_val_2d = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
            
            # Fit scaler on training data only
            y_train = self.target_scaler.fit_transform(y_train_2d)
            y_val = self.target_scaler.transform(y_val_2d)
            
            logger.info(f"Applied {scaling_method} scaling to targets")
        
        return X_train, X_val, y_train, y_val
    
    def create_callbacks(self) -> List[torch.nn.Module]:
        """
        Create callbacks for training.
        
        Returns:
            List of PyTorch callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=self.patience // 2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoints
        checkpoint_path = os.path.join(
            self.model_dir, 
            f"{self.model_type}_" + "{epoch:02d}_{val_loss:.4f}.pt"
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_weights_only=True,
            save_weights_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # TensorBoard logging
        tensorboard = TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            profile_batch=0
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_logger = CSVLogger(
            os.path.join(self.log_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        
        # Custom epoch end callback for saving the best epoch
        def on_epoch_end(epoch, logs):
            # Save current epoch metrics
            self.metrics[epoch] = {k: float(v) for k, v in logs.items()}
            
            # Update best epoch if val_loss improved
            if epoch > 0 and 'val_loss' in logs:
                best_val_loss = min([m.get('val_loss', float('inf')) for e, m in self.metrics.items() if e < epoch])
                if logs['val_loss'] < best_val_loss:
                    self.best_epoch = epoch
        
        epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        callbacks.append(epoch_end_callback)
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training data
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            optimizer_cls: Optimizer class to use (default: Adam)
            optimizer_kwargs: Arguments for optimizer initialization
            loss_fn: Loss function to use (default: DirectionalLoss)
            lr_scheduler_cls: Optional learning rate scheduler class
            lr_scheduler_kwargs: Arguments for scheduler initialization
            log_dir: Directory for TensorBoard logs
            checkpoints_dir: Directory for model checkpoints
            experiment_name: Name for the experiment
            mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Set up optimizer with default parameters if not provided
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.001, "weight_decay": 1e-5}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)
        
        # Set up loss function
        self.loss_fn = loss_fn if loss_fn is not None else DirectionalLoss(alpha=0.7, beta=2.0)
        
        # Set up learning rate scheduler if provided
        self.lr_scheduler = None
        if lr_scheduler_cls is not None:
            if lr_scheduler_kwargs is None:
                lr_scheduler_kwargs = {}
            self.lr_scheduler = lr_scheduler_cls(self.optimizer)
        
        # Set up experiment name and directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"{model.__class__.__name__}_{timestamp}"
        
        # Set up TensorBoard writer
        self.log_dir = log_dir or os.path.join("logs", "tensorboard", self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Set up checkpoints directory
        self.checkpoints_dir = checkpoints_dir or os.path.join("models", "checkpoints", self.experiment_name)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Set up mixed precision training
        self.mixed_precision = mixed_precision
        if torch.cuda.is_available() and mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            if mixed_precision:
                logger.warning("Mixed precision training is enabled but not supported on this device")
                self.mixed_precision = False
        
        # Initialize training metrics
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.training_time = 0
        
        # Log model architecture and hyperparameters
        self._log_model_info()
        
        logger.info(f"Trainer initialized with {model.__class__.__name__} on {self.device}")
        logger.info(f"TensorBoard logs will be saved to {self.log_dir}")
        logger.info(f"Checkpoints will be saved to {self.checkpoints_dir}")
    
    def _log_model_info(self):
        """Log model architecture and hyperparameters to TensorBoard"""
        # Create a text summary of model architecture
        model_summary = str(self.model)
        
        # Log model architecture as text
        self.writer.add_text("Model/Architecture", model_summary, 0)
        
        # Log hyperparameters
        hparams = {
            "model_type": self.model.__class__.__name__,
            "input_dim": self.model.input_dim,
            "output_dim": self.model.output_dim,
            "seq_len": self.model.seq_len,
            "forecast_horizon": self.model.forecast_horizon,
            "optimizer": self.optimizer.__class__.__name__,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "batch_size": self.train_dataloader.batch_size,
            "mixed_precision": self.mixed_precision,
        }
        
        # Add more detailed model hyperparameters if available
        if hasattr(self.model, "hidden_dim"):
            hparams["hidden_dim"] = self.model.hidden_dim
        if hasattr(self.model, "num_layers"):
            hparams["num_layers"] = self.model.num_layers
        if hasattr(self.model, "dropout"):
            hparams["dropout"] = self.model.dropout
        
        # Convert all values to strings for TensorBoard
        hparams = {k: str(v) for k, v in hparams.items()}
        
        # Log hyperparameters as text
        self.writer.add_text("Hyperparameters", json.dumps(hparams, indent=2), 0)
        
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_dataloader):
            # Move data to the appropriate device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(X_batch)
                    loss = self.loss_fn(outputs, y_batch)
                    
                # Backward and optimize with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass and optimization
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Log batch loss (every 10 batches)
            if batch_idx % 10 == 0:
                logger.info(f"Epoch: {self.current_epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.6f}")
                self.writer.add_scalar('Loss/train_batch', loss.item(), 
                                     self.current_epoch * len(self.train_dataloader) + batch_idx)
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        epoch_time = time.time() - start_time
        self.training_time += epoch_time
        
        # Log epoch metrics
        self.writer.add_scalar('Loss/train', avg_loss, self.current_epoch)
        self.writer.add_scalar('Time/epoch', epoch_time, self.current_epoch)
        self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
        # Update learning rate if scheduler exists
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Store loss history
        self.train_losses.append(avg_loss)
        
        logger.info(f"Epoch {self.current_epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.6f}")
        return avg_loss
    
    def validate(self) -> float:
        """
        Validate the model on the validation dataset.
        
        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            logger.warning("No validation data provided, skipping validation")
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_dataloader:
                # Move data to the appropriate device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                
                # Update metrics
                total_loss += loss.item()
                batch_count += 1
                
        # Calculate average loss
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        # Log validation metrics
        self.writer.add_scalar('Loss/validation', avg_loss, self.current_epoch)
        
        # Calculate and log directional accuracy
        if self.val_dataloader is not None:
            direction_accuracy = self._calculate_directional_accuracy()
            self.writer.add_scalar('Metrics/directional_accuracy', direction_accuracy, self.current_epoch)
            logger.info(f"Validation - Directional Accuracy: {direction_accuracy:.4f}")
        
        # Store validation loss history
        self.val_losses.append(avg_loss)
        
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss
    
    def _calculate_directional_accuracy(self) -> float:
        """
        Calculate directional accuracy on validation set.
        
        Returns:
            Directional accuracy value (0-1)
        """
        self.model.eval()
        direction_correct = 0
        total_directions = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_dataloader:
                # Move data to the appropriate device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Calculate direction (up/down) for predictions and targets
                pred_direction = (outputs[:, 1:] - outputs[:, :-1]) > 0
                true_direction = (y_batch[:, 1:] - y_batch[:, :-1]) > 0
                
                # Compare directions
                direction_correct += torch.sum(pred_direction == true_direction).item()
                total_directions += pred_direction.numel()
        
        # Calculate directional accuracy
        accuracy = direction_correct / total_directions if total_directions > 0 else 0
        return accuracy
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save a checkpoint of the current model state.
        
        Args:
            is_best: Whether this checkpoint is the best so far
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{self.current_epoch}.pt")
        
        # Create checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'seq_len': self.model.seq_len,
                'forecast_horizon': self.model.forecast_horizon,
                'model_type': self.model.__class__.__name__
            }
        }
        
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # If this is the best model so far, also save as best.pt
        if is_best:
            best_path = os.path.join(self.checkpoints_dir, "best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint and restore the training state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            return
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore learning rate scheduler if available
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        # Update best validation loss if available
        if checkpoint['val_loss'] is not None:
            self.best_val_loss = min(self.best_val_loss, checkpoint['val_loss'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {self.current_epoch})")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10, save_frequency: int = 5) -> Tuple[List[float], List[float]]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs with no improvement before stopping early
            save_frequency: Save checkpoints every n epochs
            
        Returns:
            Training and validation loss history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        start_time = time.time()
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Check if this is the best model so far
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                is_best = True
                logger.info(f"New best validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter} epochs (best: {self.best_val_loss:.6f})")
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Calculate total training time
        total_time = time.time() - start_time
        
        # Log final metrics
        self.writer.add_text("Training/Summary", 
                           f"Training completed in {total_time:.2f}s ({num_epochs} epochs)\n"
                           f"Best validation loss: {self.best_val_loss:.6f} (epoch {best_epoch+1})", 0)
        
        logger.info(f"Training completed in {total_time:.2f}s ({num_epochs} epochs)")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f} (epoch {best_epoch+1})")
        
        # Close the TensorBoard writer
        self.writer.close()
        
        return self.train_losses, self.val_losses 
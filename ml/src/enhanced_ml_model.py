"""
Enhanced ML Model for Trading Predictions
Production-grade ML model with the following features:
- Feature engineering and preprocessing with technical indicators
- Neural network architecture with LSTM and attention mechanism
- Training pipeline with validation, early stopping, and learning rate scheduling
- Prediction logic with confidence scoring and risk assessment
- Model persistence and loading with state management
- Performance monitoring and evaluation metrics
- Position sizing based on confidence and volatility
"""

import logging
import numpy as np
import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import base model
from ml.src.models.base_model import BaseModel, DirectionalLoss

logger = logging.getLogger(__name__)

class EnhancedNeuralNetwork(nn.Module):
    """Neural network architecture for the Enhanced ML Model"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize the neural network
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(EnhancedNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers for sequential data
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output layers
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Price prediction
        )
        
        # Direction prediction (up/down)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary classification (up/down)
            nn.Softmax(dim=1)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Tuple of (price_prediction, direction_prediction, confidence)
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract features
        features = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)
        for i in range(seq_len):
            features[:, i, :] = self.feature_extractor(x[:, i, :])
        
        # Process with LSTM
        lstm_out, _ = self.lstm(features)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Generate outputs
        price_pred = self.regressor(context_vector)
        direction_pred = self.classifier(context_vector)
        confidence = self.confidence_estimator(context_vector)
        
        return price_pred, direction_pred, confidence


class EnhancedMLModel(BaseModel):
    """Enhanced ML Model for trading predictions"""
    
    def __init__(self, 
                 input_dim: int = 20, 
                 hidden_dim: int = 128, 
                 num_layers: int = 2, 
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 model_path: Optional[str] = None):
        """
        Initialize the enhanced ML model
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            model_path: Path to load a pre-trained model
        """
        super().__init__(model_path)
        self.model_name = "EnhancedMLModel"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.confidence_threshold = 0.65
        self.is_trained = False
        
        # Initialize preprocessing components
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Build or load model
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._build_model()
        
        logger.info(f"Initialized {self.model_name} with {input_dim} features, {hidden_dim} hidden units, {num_layers} layers")
    
    def _build_model(self):
        """Build the model architecture"""
        self.model = EnhancedNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize loss functions
        self.price_criterion = nn.MSELoss()
        self.direction_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.BCELoss()
        self.directional_loss = DirectionalLoss(alpha=0.7, beta=1.5)
        
        logger.info(f"Built {self.model_name} architecture")
    
    def _preprocess_features(self, features: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Preprocess features for prediction or training
        
        Args:
            features: Raw input features
            is_training: Whether this is for training (to fit the scaler)
            
        Returns:
            Preprocessed features
        """
        # Check if features need reshaping
        if len(features.shape) == 2:
            # Add sequence dimension if missing
            features = np.expand_dims(features, axis=0)
        
        # Extract shape information
        batch_size, seq_len, feat_dim = features.shape
        
        # Reshape for scaling
        reshaped_features = features.reshape(-1, feat_dim)
        
        # Apply scaling
        if is_training:
            scaled_features = self.feature_scaler.fit_transform(reshaped_features)
        elif self.is_trained:
            scaled_features = self.feature_scaler.transform(reshaped_features)
        else:
            # If not trained and not training, just standardize
            scaled_features = (reshaped_features - np.mean(reshaped_features, axis=0)) / (np.std(reshaped_features, axis=0) + 1e-8)
        
        # Reshape back to original dimensions
        scaled_features = scaled_features.reshape(batch_size, seq_len, feat_dim)
        
        # Extract technical indicators
        enhanced_features = self._extract_technical_indicators(scaled_features)
        
        return enhanced_features
        
    def _extract_technical_indicators(self, features: np.ndarray) -> np.ndarray:
        """
        Extract technical indicators from price data
        
        Args:
            features: Price data features
            
        Returns:
            Enhanced features with technical indicators
        """
        # This is a simplified version - in production, you'd implement more indicators
        batch_size, seq_len, feat_dim = features.shape
        
        # Assuming the first feature is the closing price
        close_prices = features[:, :, 0].reshape(batch_size, seq_len)
        
        # Initialize arrays for indicators
        ma_short = np.zeros((batch_size, seq_len, 1))
        ma_long = np.zeros((batch_size, seq_len, 1))
        momentum = np.zeros((batch_size, seq_len, 1))
        volatility = np.zeros((batch_size, seq_len, 1))
        
        # Calculate indicators for each batch
        for b in range(batch_size):
            # Moving averages
            for i in range(seq_len):
                if i >= 5:
                    ma_short[b, i, 0] = np.mean(close_prices[b, i-5:i])
                else:
                    ma_short[b, i, 0] = close_prices[b, i]
                    
                if i >= 20:
                    ma_long[b, i, 0] = np.mean(close_prices[b, i-20:i])
                else:
                    ma_long[b, i, 0] = close_prices[b, i]
            
            # Momentum (price change over last 10 periods)
            for i in range(seq_len):
                if i >= 10:
                    momentum[b, i, 0] = (close_prices[b, i] / close_prices[b, i-10]) - 1
            
            # Volatility (rolling standard deviation)
            for i in range(seq_len):
                if i >= 10:
                    volatility[b, i, 0] = np.std(close_prices[b, i-10:i])
        
        # Concatenate with original features
        enhanced_features = np.concatenate([
            features,
            ma_short,
            ma_long,
            momentum,
            volatility
        ], axis=2)
        
        return enhanced_features
    
    def _estimate_volatility(self, features: np.ndarray) -> float:
        """
        Estimate price volatility from features
        
        Args:
            features: Input features
            
        Returns:
            Estimated volatility
        """
        # Extract closing prices (assuming first feature is close price)
        if len(features.shape) == 3:
            close_prices = features[0, :, 0]  # First batch, all time steps, first feature
        else:
            close_prices = features[:, 0]  # All time steps, first feature
        
        # Calculate returns
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(returns)
        
        return volatility
    
    def _calculate_stop_loss(self, price: float, action: str, volatility: float) -> float:
        """
        Calculate stop loss level based on price, action and volatility
        
        Args:
            price: Current price
            action: Trading action ('buy', 'sell', 'hold')
            volatility: Price volatility
            
        Returns:
            Stop loss price level
        """
        # Adjust stop loss distance based on volatility
        # Higher volatility = wider stop loss
        stop_distance = max(0.01, volatility * 2)  # Minimum 1% stop
        
        if action == 'buy':
            return price * (1 - stop_distance)
        elif action == 'sell':
            return price * (1 + stop_distance)
        else:
            return None
    
    def _calculate_take_profit(self, price: float, action: str, volatility: float) -> float:
        """
        Calculate take profit level based on price, action and volatility
        
        Args:
            price: Current price
            action: Trading action ('buy', 'sell', 'hold')
            volatility: Price volatility
            
        Returns:
            Take profit price level
        """
        # Take profit should be at least 1.5x the stop loss distance
        take_profit_distance = max(0.02, volatility * 3)
        
        if action == 'buy':
            return price * (1 + take_profit_distance)
        elif action == 'sell':
            return price * (1 - take_profit_distance)
        else:
            return None
    
    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        """
        Calculate position size based on confidence and volatility
        
        Args:
            confidence: Model confidence (0-1)
            volatility: Price volatility
            
        Returns:
            Position size as a fraction of available capital (0-1)
        """
        # Base position size on confidence
        base_size = confidence * 0.5  # Max 50% of capital
        
        # Adjust for volatility - reduce size for higher volatility
        volatility_factor = 1.0 / (1.0 + 10 * volatility)
        
        # Calculate final position size
        position_size = base_size * volatility_factor
        
        # Ensure position size is within reasonable limits
        position_size = max(0.01, min(position_size, 0.5))
        
        return position_size
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Make trading predictions
        
        Args:
            features: Input features for prediction
            
        Returns:
            Dictionary with prediction results including:
            - action: 'buy', 'sell', or 'hold'
            - confidence: Confidence score (0-1)
            - price_target: Predicted price
            - stop_loss: Suggested stop loss level
            - take_profit: Suggested take profit level
            - position_size: Suggested position size based on confidence and risk
        """
        if not self.is_trained:
            logger.warning("Model is not trained yet. Returning default prediction.")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'price_target': None,
                'stop_loss': None,
                'take_profit': None,
                'position_size': 0.0
            }
        
        try:
            # Preprocess features
            processed_features = self._preprocess_features(features)
            
            # Convert to PyTorch tensor
            features_tensor = torch.tensor(processed_features, dtype=torch.float32)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                price_pred, direction_pred, confidence = self.model(features_tensor)
                
                # Extract values
                price_value = price_pred.item()
                direction_values = direction_pred.squeeze().numpy()
                confidence_value = confidence.item()
                
                # Inverse transform price prediction if needed
                if hasattr(self, 'target_scaler') and self.target_scaler is not None:
                    price_value = self.target_scaler.inverse_transform([[price_value]])[0][0]
                
                # Determine action based on direction prediction
                buy_prob = direction_values[1] if len(direction_values.shape) == 1 else direction_values[0][1]
                sell_prob = direction_values[0] if len(direction_values.shape) == 1 else direction_values[0][0]
                
                if buy_prob > 0.5 + (self.confidence_threshold - 0.5) and confidence_value > self.confidence_threshold:
                    action = 'buy'
                elif sell_prob > 0.5 + (self.confidence_threshold - 0.5) and confidence_value > self.confidence_threshold:
                    action = 'sell'
                else:
                    action = 'hold'
                
                # Calculate risk parameters
                volatility = self._estimate_volatility(features)
                stop_loss = self._calculate_stop_loss(price_value, action, volatility)
                take_profit = self._calculate_take_profit(price_value, action, volatility)
                position_size = self._calculate_position_size(confidence_value, volatility)
                
                # Create prediction dictionary
                prediction = {
                    'action': action,
                    'confidence': float(confidence_value),
                    'price_target': float(price_value),
                    'stop_loss': float(stop_loss) if stop_loss is not None else None,
                    'take_profit': float(take_profit) if take_profit is not None else None,
                    'position_size': float(position_size),
                    'direction_probabilities': {
                        'buy': float(buy_prob),
                        'sell': float(sell_prob)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.debug(f"Prediction: {prediction}")
                return prediction
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return safe default prediction on error
            return {
                'action': 'hold',
                'confidence': 0.0,
                'price_target': None,
                'stop_loss': None,
                'take_profit': None,
                'position_size': 0.0,
                'error': str(e)
            }
    
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
        Train the model with data loaders
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            lr: Learning rate
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Training {self.model_name} with {num_epochs} epochs...")
        
        # Set model to training mode
        self.model.train()
        
        # Initialize optimizer with learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Initialize learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Initialize early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                price_pred, direction_pred, confidence = self.model(data)
                
                # Calculate losses
                price_loss = self.price_criterion(price_pred, targets[:, 0:1])
                
                # Direction labels (1 for up, 0 for down)
                direction_labels = (targets[:, 1] > 0).long()
                direction_loss = self.direction_criterion(direction_pred, direction_labels)
                
                # Confidence should match accuracy
                with torch.no_grad():
                    direction_accuracy = (torch.argmax(direction_pred, dim=1) == direction_labels).float()
                confidence_loss = self.confidence_criterion(confidence, direction_accuracy.unsqueeze(1))
                
                # Combined loss
                loss = price_loss + direction_loss + confidence_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Record loss
                train_losses.append(loss.item())
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Calculate average training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_losses = []
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for data, targets in val_loader:
                    # Forward pass
                    price_pred, direction_pred, confidence = self.model(data)
                    
                    # Calculate losses
                    price_loss = self.price_criterion(price_pred, targets[:, 0:1])
                    
                    # Direction labels
                    direction_labels = (targets[:, 1] > 0).long()
                    direction_loss = self.direction_criterion(direction_pred, direction_labels)
                    
                    # Confidence should match accuracy
                    direction_accuracy = (torch.argmax(direction_pred, dim=1) == direction_labels).float()
                    confidence_loss = self.confidence_criterion(confidence, direction_accuracy.unsqueeze(1))
                    
                    # Combined loss
                    loss = price_loss + direction_loss + confidence_loss
                    
                    # Record loss and predictions
                    val_losses.append(loss.item())
                    all_preds.append(price_pred.cpu().numpy())
                    all_targets.append(targets[:, 0:1].cpu().numpy())
            
            # Calculate average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            # Calculate validation metrics
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            
            val_metrics = {
                'mse': mean_squared_error(all_targets, all_preds),
                'mae': mean_absolute_error(all_targets, all_preds),
                'r2': r2_score(all_targets, all_preds)
            }
            history['val_metrics'].append(val_metrics)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MSE: {val_metrics['mse']:.4f}")
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"{self.model_name}_best.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'val_metrics': val_metrics
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Mark as trained
        self.is_trained = True
        
        logger.info(f"Training completed with best validation loss: {best_val_loss:.4f}")
        return history
    
    def train(self, data: np.ndarray, labels: np.ndarray, val_split: float = 0.2, 
              batch_size: int = 32, num_epochs: int = 100, 
              early_stopping_patience: int = 10, checkpoint_dir: str = None) -> Dict:
        """
        Train the model with numpy arrays
        
        Args:
            data: Training data
            labels: Training labels
            val_split: Validation split ratio
            batch_size: Batch size
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Training {self.model_name}...")
        
        try:
            # Validate inputs
            if data.shape[0] != labels.shape[0]:
                raise ValueError(f"Data and labels must have the same number of samples. Got {data.shape[0]} and {labels.shape[0]}")
            
            # Preprocess data
            processed_data = self._preprocess_features(data, is_training=True)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                processed_data, labels, test_size=val_split, random_state=42
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Train the model
            history = self.fit_model(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                lr=self.learning_rate,
                early_stopping_patience=early_stopping_patience,
                checkpoint_dir=checkpoint_dir
            )
            
            # Save feature scaler
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                scaler_path = os.path.join(checkpoint_dir, f"{self.model_name}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.feature_scaler, f)
                logger.info(f"Saved feature scaler to {scaler_path}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            logger.warning("Model is not trained yet. Evaluation may not be meaningful.")
        
        try:
            # Preprocess features
            processed_features = self._preprocess_features(X)
            
            # Convert to PyTorch tensor
            features_tensor = torch.tensor(processed_features, dtype=torch.float32)
            labels_tensor = torch.tensor(y, dtype=torch.float32)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make predictions
            with torch.no_grad():
                price_pred, direction_pred, confidence = self.model(features_tensor)
                
                # Extract values
                price_values = price_pred.cpu().numpy()
                direction_values = direction_pred.cpu().numpy()
                confidence_values = confidence.cpu().numpy()
                
                # Calculate metrics
                price_mse = mean_squared_error(y[:, 0], price_values)
                price_mae = mean_absolute_error(y[:, 0], price_values)
                price_r2 = r2_score(y[:, 0], price_values)
                
                # Direction accuracy
                direction_labels = (y[:, 1] > 0).astype(int)
                direction_preds = np.argmax(direction_values, axis=1)
                direction_accuracy = np.mean(direction_preds == direction_labels)
                
                # Confidence calibration
                confidence_error = np.mean(np.abs(confidence_values - (direction_preds == direction_labels).reshape(-1, 1)))
                
                metrics = {
                    'mse': float(price_mse),
                    'mae': float(price_mae),
                    'r2': float(price_r2),
                    'direction_accuracy': float(direction_accuracy),
                    'confidence_error': float(confidence_error)
                }
                
                logger.info(f"Evaluation metrics: {metrics}")
                return metrics
                
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return {
                'error': str(e),
                'mse': float('nan'),
                'mae': float('nan'),
                'r2': float('nan'),
                'direction_accuracy': float('nan'),
                'confidence_error': float('nan')
            }
    
    def save(self, model_path: str) -> None:
        """
        Save the model
        
        Args:
            model_path: Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model state
            model_state = {
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'confidence_threshold': self.confidence_threshold,
                'is_trained': self.is_trained,
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save model
            torch.save(model_state, model_path)
            
            # Save feature scaler separately
            scaler_path = model_path.replace('.pt', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Feature scaler saved to {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, model_path: str, **kwargs):
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model
        """
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model state
            model_state = torch.load(model_path)
            
            # Create model instance with saved parameters
            model = cls(
                input_dim=model_state.get('input_dim', kwargs.get('input_dim', 20)),
                hidden_dim=model_state.get('hidden_dim', kwargs.get('hidden_dim', 128)),
                num_layers=model_state.get('num_layers', kwargs.get('num_layers', 2)),
                dropout=model_state.get('dropout', kwargs.get('dropout', 0.2)),
                learning_rate=model_state.get('learning_rate', kwargs.get('learning_rate', 0.001)),
                batch_size=model_state.get('batch_size', kwargs.get('batch_size', 64)),
                model_path=None  # Don't load model again
            )
            
            # Load model state dict
            model.model.load_state_dict(model_state['model_state_dict'])
            
            # Set model attributes
            model.confidence_threshold = model_state.get('confidence_threshold', 0.65)
            model.is_trained = model_state.get('is_trained', True)
            model.model_name = model_state.get('model_name', 'EnhancedMLModel')
            
            # Load feature scaler if available
            scaler_path = model_path.replace('.pt', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    model.feature_scaler = pickle.load(f)
                logger.info(f"Feature scaler loaded from {scaler_path}")
            
            # Set model to evaluation mode
            model.model.eval()
            
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'name': self.model_name,
            'is_trained': self.is_trained,
            'confidence_threshold': self.confidence_threshold,
            'type': 'enhanced_ml',
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            },
            'training': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        }

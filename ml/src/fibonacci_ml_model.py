"""
Fibonacci ML Model for Trading Predictions
Production-grade Fibonacci ML trading model with the following features:
- Fibonacci retracement level calculation
- Support and resistance identification
- Trend analysis integration
- Price action pattern recognition
- Risk management rules
- Position sizing logic
- Swing high/low detection
- Market structure analysis
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
from ml.src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class FibonacciNetwork(nn.Module):
    """Neural network architecture for the Fibonacci ML Model"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize the neural network
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(FibonacciNetwork, self).__init__()
        
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
        
        # LSTM for sequential data
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fibonacci level detection
        self.fib_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 5)  # 5 Fibonacci levels (0.236, 0.382, 0.5, 0.618, 0.786)
        )
        
        # Support/resistance detection
        self.sr_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Support and resistance strength
        )
        
        # Trend analysis
        self.trend_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # Uptrend, downtrend, sideways
            nn.Softmax(dim=1)
        )
        
        # Price prediction
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Price prediction
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Tuple of (fib_levels, support_resistance, trend, price_prediction, confidence)
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract features
        features = torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device)
        for i in range(seq_len):
            features[:, i, :] = self.feature_extractor(x[:, i, :])
        
        # Process with LSTM
        lstm_out, _ = self.lstm(features)
        
        # Use the last output for predictions
        last_output = lstm_out[:, -1, :]
        
        # Generate outputs
        fib_levels = self.fib_detector(last_output)
        support_resistance = self.sr_detector(last_output)
        trend = self.trend_analyzer(last_output)
        price_pred = self.price_predictor(last_output)
        confidence = self.confidence_estimator(last_output)
        
        return fib_levels, support_resistance, trend, price_pred, confidence


class FibonacciMLModel(BaseModel):
    """Fibonacci ML Model for trading predictions"""
    
    def __init__(self, 
                 input_dim: int = 20, 
                 hidden_dim: int = 128, 
                 num_layers: int = 2, 
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 model_path: Optional[str] = None):
        """
        Initialize the Fibonacci ML model
        
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
        self.model_name = "FibonacciMLModel"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_trained = False
        
        # Fibonacci levels
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
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
        self.model = FibonacciNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Define loss functions
        self.fib_criterion = nn.CrossEntropyLoss()
        self.sr_criterion = nn.MSELoss()
        self.trend_criterion = nn.CrossEntropyLoss()
        self.price_criterion = nn.MSELoss()
        self.confidence_criterion = nn.BCELoss()
        
        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        logger.info(f"Built {self.model_name} architecture")
    
    def _preprocess_features(self, features: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Preprocess features for prediction or training
        
        Args:
            features: Raw input features
            is_training: Whether this is for training (to fit the scaler)
            
        Returns:
            Preprocessed features with Fibonacci analysis
        """
        # Check if features need reshaping
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        batch_size, seq_len, feat_dim = features.shape
        
        # Reshape for scaling
        reshaped_features = features.reshape(-1, feat_dim)
        
        # Apply scaling
        if is_training:
            scaled_features = self.feature_scaler.fit_transform(reshaped_features)
        elif self.is_trained:
            scaled_features = self.feature_scaler.transform(reshaped_features)
        else:
            scaled_features = (reshaped_features - np.mean(reshaped_features, axis=0)) / (np.std(reshaped_features, axis=0) + 1e-8)
        
        # Reshape back to original dimensions
        scaled_features = scaled_features.reshape(batch_size, seq_len, feat_dim)
        
        # Extract Fibonacci-specific features
        enhanced_features = self._extract_fibonacci_features(scaled_features)
        
        return enhanced_features
    
    def _extract_fibonacci_features(self, features: np.ndarray) -> np.ndarray:
        """
        Extract Fibonacci-specific features from price data
        
        Args:
            features: Price data features
            
        Returns:
            Enhanced features with Fibonacci analysis
        """
        batch_size, seq_len, feat_dim = features.shape
        
        # Assuming the first 4 features are OHLC prices
        if feat_dim >= 4:
            high_prices = features[:, :, 1]  # High prices
            low_prices = features[:, :, 2]   # Low prices
            close_prices = features[:, :, 3] # Close prices
        else:
            # If only close prices available
            close_prices = features[:, :, 0]
            high_prices = close_prices
            low_prices = close_prices
        
        # Initialize arrays for Fibonacci features
        fib_retracements = np.zeros((batch_size, seq_len, 5))  # 5 Fibonacci levels
        swing_highs = np.zeros((batch_size, seq_len, 1))
        swing_lows = np.zeros((batch_size, seq_len, 1))
        support_resistance = np.zeros((batch_size, seq_len, 2))
        trend_strength = np.zeros((batch_size, seq_len, 1))
        
        for b in range(batch_size):
            # Calculate swing highs and lows
            for i in range(seq_len):
                # Look back window for swing detection
                lookback = min(10, i)
                if lookback > 2:
                    # Swing high detection
                    if i >= 2 and i < seq_len - 2:
                        if (high_prices[b, i] > high_prices[b, i-1] and 
                            high_prices[b, i] > high_prices[b, i+1]):
                            swing_highs[b, i, 0] = 1.0
                    
                    # Swing low detection
                    if i >= 2 and i < seq_len - 2:
                        if (low_prices[b, i] < low_prices[b, i-1] and 
                            low_prices[b, i] < low_prices[b, i+1]):
                            swing_lows[b, i, 0] = 1.0
                
                # Calculate Fibonacci retracements
                if i >= 20:  # Need enough data for meaningful retracements
                    recent_high = np.max(high_prices[b, i-20:i])
                    recent_low = np.min(low_prices[b, i-20:i])
                    current_price = close_prices[b, i]
                    
                    if recent_high != recent_low:
                        # Calculate retracement levels
                        price_range = recent_high - recent_low
                        for j, level in enumerate(self.fibonacci_levels):
                            fib_level_price = recent_high - (price_range * level)
                            # Distance from current price to Fibonacci level (normalized)
                            fib_retracements[b, i, j] = abs(current_price - fib_level_price) / price_range
                
                # Support/Resistance strength
                if i >= 10:
                    # Simple support/resistance based on price levels
                    price_window = close_prices[b, i-10:i]
                    current_price = close_prices[b, i]
                    
                    # Support strength (how many times price bounced from lower levels)
                    support_count = np.sum(price_window <= current_price * 0.99)
                    support_resistance[b, i, 0] = support_count / 10.0
                    
                    # Resistance strength (how many times price was rejected from higher levels)
                    resistance_count = np.sum(price_window >= current_price * 1.01)
                    support_resistance[b, i, 1] = resistance_count / 10.0
                
                # Trend strength calculation
                if i >= 20:
                    # Simple trend strength based on price momentum
                    price_change = close_prices[b, i] - close_prices[b, i-20]
                    price_volatility = np.std(close_prices[b, i-20:i])
                    if price_volatility > 0:
                        trend_strength[b, i, 0] = price_change / price_volatility
        
        # Concatenate with original features
        enhanced_features = np.concatenate([
            features,
            fib_retracements,
            swing_highs,
            swing_lows,
            support_resistance,
            trend_strength
        ], axis=2)
        
        return enhanced_features
    
    def _calculate_fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            high: Swing high price
            low: Swing low price
            
        Returns:
            Dictionary of Fibonacci levels
        """
        price_range = high - low
        levels = {}
        
        for level in self.fibonacci_levels:
            levels[f"fib_{level}"] = high - (price_range * level)
        
        return levels
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Make trading predictions based on Fibonacci analysis
        
        Args:
            features: Input features for prediction
            
        Returns:
            Dictionary with prediction results including Fibonacci analysis
        """
        if not self.is_trained:
            logger.warning("Model not trained yet, predictions may be unreliable")
        
        try:
            # Preprocess features
            processed_features = self._preprocess_features(features)
            
            # Convert to tensor
            features_tensor = torch.tensor(processed_features, dtype=torch.float32)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                fib_levels, sr_levels, trend, price_pred, confidence = self.model(features_tensor)
            
            # Process predictions
            fib_probs = torch.softmax(fib_levels, dim=1).squeeze().numpy()
            sr_values = torch.sigmoid(sr_levels).squeeze().numpy()
            trend_probs = trend.squeeze().numpy()
            price_value = price_pred.item()
            confidence_value = confidence.item()
            
            # Determine most likely Fibonacci level
            best_fib_idx = np.argmax(fib_probs)
            best_fib_level = self.fibonacci_levels[best_fib_idx]
            
            # Determine trend direction
            trend_labels = ['downtrend', 'sideways', 'uptrend']
            trend_idx = np.argmax(trend_probs)
            trend_direction = trend_labels[trend_idx]
            
            # Generate trading action based on Fibonacci analysis and trend
            current_price = features[-1, -1, 0] if len(features.shape) == 3 else features[-1, 0]
            
            # Calculate Fibonacci levels for current swing
            if len(features.shape) == 3:
                recent_prices = features[0, :, 0]
            else:
                recent_prices = features[:, 0]
            
            recent_high = np.max(recent_prices[-20:])
            recent_low = np.min(recent_prices[-20:])
            fib_levels_dict = self._calculate_fibonacci_levels(recent_high, recent_low)
            
            # Trading logic based on Fibonacci levels and trend
            action = 'hold'
            if trend_direction == 'uptrend' and confidence_value > 0.6:
                # Look for buying opportunities at Fibonacci support levels
                if best_fib_level in [0.382, 0.5, 0.618]:  # Key retracement levels
                    action = 'buy'
            elif trend_direction == 'downtrend' and confidence_value > 0.6:
                # Look for selling opportunities at Fibonacci resistance levels
                if best_fib_level in [0.382, 0.5, 0.618]:
                    action = 'sell'
            
            # Calculate risk parameters based on Fibonacci levels
            if action == 'buy':
                stop_loss = fib_levels_dict.get('fib_0.786', current_price * 0.98)
                take_profit = fib_levels_dict.get('fib_0.236', current_price * 1.03)
            elif action == 'sell':
                stop_loss = fib_levels_dict.get('fib_0.236', current_price * 1.02)
                take_profit = fib_levels_dict.get('fib_0.786', current_price * 0.97)
            else:
                stop_loss = None
                take_profit = None
            
            # Position sizing based on confidence and support/resistance strength
            sr_strength = np.mean(sr_values)
            position_size = confidence_value * sr_strength * 0.3  # Max 30% of capital
            position_size = max(0.01, min(position_size, 0.3))
            
            prediction = {
                'action': action,
                'confidence': float(confidence_value),
                'fibonacci_level': float(best_fib_level),
                'fibonacci_probability': float(fib_probs[best_fib_idx]),
                'trend_direction': trend_direction,
                'trend_probabilities': {
                    'downtrend': float(trend_probs[0]),
                    'sideways': float(trend_probs[1]),
                    'uptrend': float(trend_probs[2])
                },
                'support_strength': float(sr_values[0]) if len(sr_values) > 1 else float(sr_values),
                'resistance_strength': float(sr_values[1]) if len(sr_values) > 1 else float(sr_values),
                'price_target': float(price_value),
                'stop_loss': float(stop_loss) if stop_loss is not None else None,
                'take_profit': float(take_profit) if take_profit is not None else None,
                'position_size': float(position_size),
                'fibonacci_levels': {f"fib_{level}": float(price) for level, price in fib_levels_dict.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Fibonacci prediction: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error during Fibonacci prediction: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'fibonacci_level': 0.5,
                'price_target': None,
                'stop_loss': None,
                'take_profit': None,
                'position_size': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
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
                fib_levels, sr_levels, trend, price_pred, confidence = self.model(data)
                
                # Calculate losses
                # Fibonacci level loss (assuming targets contain Fibonacci level labels)
                if targets.shape[1] > 5:
                    fib_labels = targets[:, 5:10].long()  # Fibonacci level labels
                    fib_loss = self.fib_criterion(fib_levels, torch.argmax(fib_labels, dim=1))
                else:
                    fib_loss = torch.tensor(0.0)
                
                # Support/Resistance loss
                if targets.shape[1] > 3:
                    sr_targets = targets[:, 3:5]  # Support/Resistance targets
                    sr_loss = self.sr_criterion(sr_levels, sr_targets)
                else:
                    sr_loss = torch.tensor(0.0)
                
                # Trend loss
                if targets.shape[1] > 2:
                    trend_labels = targets[:, 2].long()  # Trend labels
                    trend_loss = self.trend_criterion(trend, trend_labels)
                else:
                    trend_loss = torch.tensor(0.0)
                
                # Price prediction loss
                price_loss = self.price_criterion(price_pred, targets[:, 0:1])
                
                # Confidence loss (should match prediction accuracy)
                with torch.no_grad():
                    # Simple accuracy based on price prediction
                    price_accuracy = (torch.abs(price_pred - targets[:, 0:1]) < 0.01).float()
                confidence_loss = self.confidence_criterion(confidence, price_accuracy)
                
                # Combined loss
                loss = price_loss + 0.5 * fib_loss + 0.3 * sr_loss + 0.3 * trend_loss + 0.2 * confidence_loss
                
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
                    fib_levels, sr_levels, trend, price_pred, confidence = self.model(data)
                    
                    # Calculate validation loss (simplified)
                    loss = self.price_criterion(price_pred, targets[:, 0:1])
                    
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
        Train the Fibonacci model with numpy arrays
        
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
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make predictions
            with torch.no_grad():
                fib_levels, sr_levels, trend, price_pred, confidence = self.model(features_tensor)
                
                # Extract values
                price_values = price_pred.cpu().numpy().flatten()
                fib_probs = torch.softmax(fib_levels, dim=1).cpu().numpy()
                trend_probs = trend.cpu().numpy()
                confidence_values = confidence.cpu().numpy().flatten()
                
                # Calculate metrics
                price_mse = mean_squared_error(y[:, 0], price_values)
                price_mae = mean_absolute_error(y[:, 0], price_values)
                price_r2 = r2_score(y[:, 0], price_values)
                
                # Fibonacci level accuracy (if available in labels)
                fib_accuracy = 0.0
                if y.shape[1] > 5:
                    fib_labels = np.argmax(y[:, 5:10], axis=1)
                    fib_preds = np.argmax(fib_probs, axis=1)
                    fib_accuracy = np.mean(fib_preds == fib_labels)
                
                # Trend accuracy (if available in labels)
                trend_accuracy = 0.0
                if y.shape[1] > 2:
                    trend_labels = y[:, 2].astype(int)
                    trend_preds = np.argmax(trend_probs, axis=1)
                    trend_accuracy = np.mean(trend_preds == trend_labels)
                
                # Confidence calibration
                confidence_error = np.mean(np.abs(confidence_values - 0.5))  # Simplified
                
                metrics = {
                    'mse': float(price_mse),
                    'mae': float(price_mae),
                    'r2': float(price_r2),
                    'fibonacci_accuracy': float(fib_accuracy),
                    'trend_accuracy': float(trend_accuracy),
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
                'fibonacci_accuracy': float('nan'),
                'trend_accuracy': float('nan'),
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
                'fibonacci_levels': self.fibonacci_levels,
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
            model.fibonacci_levels = model_state.get('fibonacci_levels', [0.236, 0.382, 0.5, 0.618, 0.786])
            model.is_trained = model_state.get('is_trained', True)
            model.model_name = model_state.get('model_name', 'FibonacciMLModel')
            
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
            'fibonacci_levels': self.fibonacci_levels,
            'type': 'fibonacci_ml',
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            },
            'training': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            },
            'features': [
                'fibonacci_retracement_levels',
                'swing_high_low_detection',
                'support_resistance_identification',
                'trend_analysis',
                'price_action_patterns'
            ]
        }

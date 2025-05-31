#!/usr/bin/env python3
"""
Meta-Learning Ensemble System for Enhanced SmartMarketOOPS
Implements adaptive ensemble methods that learn to adapt to market regimes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from collections import deque
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Advanced market regime detection using clustering and statistical analysis"""
    
    def __init__(self, n_regimes: int = 7, lookback_window: int = 100):
        """Initialize market regime detector"""
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.is_fitted = False
        
        # Regime characteristics
        self.regime_names = [
            'trending_bullish', 'trending_bearish', 'volatile_high', 'volatile_low',
            'ranging_tight', 'ranging_wide', 'breakout'
        ]
        
        logger.info(f"Market Regime Detector initialized with {n_regimes} regimes")
    
    def extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection"""
        features = []
        
        # Price-based features
        returns = data['close'].pct_change().fillna(0)
        features.extend([
            returns.mean(),  # Average return
            returns.std(),   # Volatility
            returns.skew(),  # Skewness
            returns.kurt(),  # Kurtosis
        ])
        
        # Trend features
        if len(data) >= 20:
            sma_20 = data['close'].rolling(20).mean()
            trend_strength = (data['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
            features.append(trend_strength)
        else:
            features.append(0.0)
        
        # Volatility features
        high_low_ratio = (data['high'] / data['low']).mean() - 1
        features.append(high_low_ratio)
        
        # Volume features
        if 'volume' in data.columns:
            volume_trend = data['volume'].pct_change().mean()
            features.append(volume_trend)
        else:
            features.append(0.0)
        
        # Range features
        price_range = (data['high'] - data['low']) / data['close']
        features.extend([
            price_range.mean(),
            price_range.std()
        ])
        
        # Momentum features
        if len(data) >= 10:
            momentum = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]
            features.append(momentum)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def fit(self, historical_data: Dict[str, pd.DataFrame]):
        """Fit regime detector on historical data"""
        all_features = []
        
        for symbol, data in historical_data.items():
            # Extract features for overlapping windows
            for i in range(self.lookback_window, len(data), 10):  # Every 10 periods
                window_data = data.iloc[i-self.lookback_window:i]
                features = self.extract_regime_features(window_data)
                all_features.append(features)
        
        if len(all_features) == 0:
            logger.warning("No features extracted for regime detection")
            return
        
        # Fit scaler and clustering
        feature_matrix = np.array(all_features)
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        self.kmeans.fit(feature_matrix)
        self.is_fitted = True
        
        logger.info(f"Regime detector fitted on {len(all_features)} feature vectors")
    
    def detect_regime(self, data: pd.DataFrame) -> Tuple[int, str, float]:
        """Detect current market regime"""
        if not self.is_fitted:
            return 0, 'unknown', 0.0
        
        # Extract features
        features = self.extract_regime_features(data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict regime
        regime_id = self.kmeans.predict(features_scaled)[0]
        regime_name = self.regime_names[regime_id] if regime_id < len(self.regime_names) else f'regime_{regime_id}'
        
        # Calculate confidence (distance to cluster center)
        distances = self.kmeans.transform(features_scaled)[0]
        confidence = 1.0 / (1.0 + distances[regime_id])  # Inverse distance
        
        return regime_id, regime_name, confidence


class MetaLearningNetwork(nn.Module):
    """Neural network for meta-learning ensemble weights"""
    
    def __init__(self, input_dim: int, n_models: int, hidden_dim: int = 128):
        """Initialize meta-learning network"""
        super(MetaLearningNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.n_models = n_models
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_models),
            nn.Softmax(dim=1)
        )
        
        # Performance predictor
        self.performance_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + n_models, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Encode context
        context_features = self.context_encoder(context)
        
        # Generate ensemble weights
        weights = self.weight_generator(context_features)
        
        # Predict performance
        combined_features = torch.cat([context_features, weights], dim=1)
        performance = self.performance_predictor(combined_features)
        
        return weights, performance


class AdaptiveEnsemble:
    """Adaptive ensemble system with meta-learning"""
    
    def __init__(self, model_names: List[str], context_dim: int = 20):
        """Initialize adaptive ensemble"""
        self.model_names = model_names
        self.n_models = len(model_names)
        self.context_dim = context_dim
        
        # Meta-learning network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_network = MetaLearningNetwork(context_dim, self.n_models).to(self.device)
        self.optimizer = optim.Adam(self.meta_network.parameters(), lr=0.001)
        
        # Performance tracking
        self.model_performance_history = {name: deque(maxlen=1000) for name in model_names}
        self.ensemble_performance_history = deque(maxlen=1000)
        
        # Regime-specific weights
        self.regime_weights = {}
        
        logger.info(f"Adaptive Ensemble initialized with {self.n_models} models")
    
    def extract_context_features(self, market_data: pd.DataFrame, 
                                regime_info: Tuple[int, str, float],
                                model_predictions: Dict[str, float],
                                model_confidences: Dict[str, float]) -> np.ndarray:
        """Extract context features for meta-learning"""
        features = []
        
        # Market features
        if len(market_data) > 0:
            returns = market_data['close'].pct_change().fillna(0)
            features.extend([
                returns.iloc[-1] if len(returns) > 0 else 0.0,  # Latest return
                returns.std() if len(returns) > 1 else 0.0,     # Volatility
                market_data['volume'].iloc[-1] / market_data['volume'].mean() if 'volume' in market_data.columns else 1.0
            ])
        else:
            features.extend([0.0, 0.0, 1.0])
        
        # Regime features
        regime_id, regime_name, regime_confidence = regime_info
        features.extend([
            regime_id / 10.0,  # Normalized regime ID
            regime_confidence
        ])
        
        # Model prediction features
        predictions = list(model_predictions.values())
        confidences = list(model_confidences.values())
        
        features.extend([
            np.mean(predictions),  # Average prediction
            np.std(predictions),   # Prediction disagreement
            np.mean(confidences),  # Average confidence
            np.std(confidences),   # Confidence disagreement
        ])
        
        # Model-specific features
        for i, model_name in enumerate(self.model_names):
            pred = model_predictions.get(model_name, 0.5)
            conf = model_confidences.get(model_name, 0.0)
            
            # Recent performance
            recent_perf = np.mean(list(self.model_performance_history[model_name])[-10:]) if self.model_performance_history[model_name] else 0.5
            
            features.extend([pred, conf, recent_perf])
        
        # Pad or truncate to context_dim
        features = features[:self.context_dim]
        while len(features) < self.context_dim:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_adaptive_weights(self, context_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Get adaptive ensemble weights using meta-learning"""
        context_tensor = torch.FloatTensor(context_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            weights, predicted_performance = self.meta_network(context_tensor)
            weights = weights.cpu().numpy().flatten()
            predicted_performance = predicted_performance.cpu().item()
        
        return weights, predicted_performance
    
    def combine_predictions(self, model_predictions: Dict[str, float],
                          model_confidences: Dict[str, float],
                          weights: np.ndarray) -> Dict[str, Any]:
        """Combine model predictions using adaptive weights"""
        
        # Ensure we have predictions for all models
        predictions = []
        confidences = []
        
        for i, model_name in enumerate(self.model_names):
            pred = model_predictions.get(model_name, 0.5)
            conf = model_confidences.get(model_name, 0.0)
            predictions.append(pred)
            confidences.append(conf)
        
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Weighted ensemble prediction
        ensemble_prediction = np.sum(weights * predictions)
        
        # Weighted ensemble confidence
        ensemble_confidence = np.sum(weights * confidences)
        
        # Calculate prediction uncertainty
        prediction_variance = np.sum(weights * (predictions - ensemble_prediction) ** 2)
        uncertainty = np.sqrt(prediction_variance)
        
        # Quality score based on agreement and confidence
        agreement_score = 1.0 - (np.std(predictions) / (np.mean(predictions) + 1e-8))
        confidence_score = ensemble_confidence
        quality_score = (agreement_score * 0.6) + (confidence_score * 0.4)
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': ensemble_confidence,
            'quality_score': quality_score,
            'uncertainty': uncertainty,
            'weights': weights.tolist(),
            'individual_predictions': {
                model_name: pred for model_name, pred in zip(self.model_names, predictions)
            },
            'individual_confidences': {
                model_name: conf for model_name, conf in zip(self.model_names, confidences)
            }
        }
    
    def update_performance(self, model_predictions: Dict[str, float],
                          actual_outcome: float, ensemble_prediction: float):
        """Update model and ensemble performance tracking"""
        
        # Update individual model performance
        for model_name, prediction in model_predictions.items():
            # Calculate accuracy (binary classification)
            pred_class = 1 if prediction > 0.5 else 0
            actual_class = 1 if actual_outcome > 0.5 else 0
            accuracy = 1.0 if pred_class == actual_class else 0.0
            
            self.model_performance_history[model_name].append(accuracy)
        
        # Update ensemble performance
        ensemble_pred_class = 1 if ensemble_prediction > 0.5 else 0
        actual_class = 1 if actual_outcome > 0.5 else 0
        ensemble_accuracy = 1.0 if ensemble_pred_class == actual_class else 0.0
        
        self.ensemble_performance_history.append(ensemble_accuracy)
    
    def train_meta_network(self, training_data: List[Dict[str, Any]], epochs: int = 100):
        """Train meta-learning network"""
        logger.info(f"Training meta-learning network on {len(training_data)} samples")
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for data_point in training_data:
                context = torch.FloatTensor(data_point['context']).unsqueeze(0).to(self.device)
                target_weights = torch.FloatTensor(data_point['optimal_weights']).unsqueeze(0).to(self.device)
                target_performance = torch.FloatTensor([data_point['actual_performance']]).unsqueeze(0).to(self.device)
                
                # Forward pass
                pred_weights, pred_performance = self.meta_network(context)
                
                # Calculate loss
                weight_loss = F.mse_loss(pred_weights, target_weights)
                performance_loss = F.mse_loss(pred_performance, target_performance)
                total_loss_sample = weight_loss + performance_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_sample.backward()
                self.optimizer.step()
                
                total_loss += total_loss_sample.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(training_data)
                logger.info(f"Meta-learning epoch {epoch}: Average loss = {avg_loss:.6f}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models and ensemble"""
        summary = {
            'ensemble_performance': {
                'recent_accuracy': np.mean(list(self.ensemble_performance_history)[-100:]) if self.ensemble_performance_history else 0.0,
                'total_predictions': len(self.ensemble_performance_history)
            },
            'model_performance': {}
        }
        
        for model_name in self.model_names:
            history = self.model_performance_history[model_name]
            summary['model_performance'][model_name] = {
                'recent_accuracy': np.mean(list(history)[-100:]) if history else 0.0,
                'total_predictions': len(history)
            }
        
        return summary
    
    def save(self, filepath: str):
        """Save meta-learning ensemble"""
        state = {
            'meta_network_state_dict': self.meta_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_names': self.model_names,
            'context_dim': self.context_dim,
            'performance_history': {
                'models': dict(self.model_performance_history),
                'ensemble': list(self.ensemble_performance_history)
            }
        }
        
        torch.save(state, filepath)
        logger.info(f"Meta-learning ensemble saved to {filepath}")
    
    def load(self, filepath: str):
        """Load meta-learning ensemble"""
        state = torch.load(filepath, map_location=self.device)
        
        self.meta_network.load_state_dict(state['meta_network_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Restore performance history
        if 'performance_history' in state:
            for model_name, history in state['performance_history']['models'].items():
                self.model_performance_history[model_name] = deque(history, maxlen=1000)
            
            self.ensemble_performance_history = deque(state['performance_history']['ensemble'], maxlen=1000)
        
        logger.info(f"Meta-learning ensemble loaded from {filepath}")


class MetaLearningEnsembleSystem:
    """Complete meta-learning ensemble system"""
    
    def __init__(self, symbols: List[str] = None):
        """Initialize meta-learning ensemble system"""
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        
        # Model names for ensemble
        self.model_names = [
            'enhanced_transformer',
            'cnn_lstm',
            'technical_indicators',
            'smc_analyzer',
            'reinforcement_learning'
        ]
        
        # Components
        self.regime_detector = MarketRegimeDetector()
        self.adaptive_ensembles = {}
        
        # Initialize ensemble for each symbol
        for symbol in self.symbols:
            self.adaptive_ensembles[symbol] = AdaptiveEnsemble(self.model_names)
        
        logger.info(f"Meta-Learning Ensemble System initialized for {len(self.symbols)} symbols")
    
    def fit_regime_detector(self, historical_data: Dict[str, pd.DataFrame]):
        """Fit regime detector on historical data"""
        self.regime_detector.fit(historical_data)
    
    def get_enhanced_prediction(self, symbol: str, market_data: pd.DataFrame,
                              model_predictions: Dict[str, float],
                              model_confidences: Dict[str, float]) -> Dict[str, Any]:
        """Get enhanced prediction using meta-learning ensemble"""
        
        if symbol not in self.adaptive_ensembles:
            logger.warning(f"No ensemble found for {symbol}")
            return {
                'prediction': np.mean(list(model_predictions.values())),
                'confidence': np.mean(list(model_confidences.values())),
                'enhanced': False
            }
        
        # Detect market regime
        regime_info = self.regime_detector.detect_regime(market_data)
        
        # Extract context features
        ensemble = self.adaptive_ensembles[symbol]
        context_features = ensemble.extract_context_features(
            market_data, regime_info, model_predictions, model_confidences
        )
        
        # Get adaptive weights
        weights, predicted_performance = ensemble.get_adaptive_weights(context_features)
        
        # Combine predictions
        ensemble_result = ensemble.combine_predictions(
            model_predictions, model_confidences, weights
        )
        
        # Add regime information
        ensemble_result.update({
            'regime_id': regime_info[0],
            'regime_name': regime_info[1],
            'regime_confidence': regime_info[2],
            'predicted_performance': predicted_performance,
            'enhanced': True,
            'meta_learning': True
        })
        
        return ensemble_result
    
    def update_performance(self, symbol: str, model_predictions: Dict[str, float],
                          ensemble_prediction: float, actual_outcome: float):
        """Update performance for meta-learning"""
        if symbol in self.adaptive_ensembles:
            self.adaptive_ensembles[symbol].update_performance(
                model_predictions, actual_outcome, ensemble_prediction
            )
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get system-wide performance summary"""
        performance = {}
        
        for symbol, ensemble in self.adaptive_ensembles.items():
            performance[symbol] = ensemble.get_performance_summary()
        
        return performance
    
    def save_system(self, directory: str):
        """Save complete meta-learning system"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save regime detector
        regime_path = os.path.join(directory, 'regime_detector.pkl')
        with open(regime_path, 'wb') as f:
            pickle.dump(self.regime_detector, f)
        
        # Save ensembles
        for symbol, ensemble in self.adaptive_ensembles.items():
            ensemble_path = os.path.join(directory, f'meta_ensemble_{symbol}.pth')
            ensemble.save(ensemble_path)
        
        logger.info(f"Meta-learning ensemble system saved to {directory}")
    
    def load_system(self, directory: str):
        """Load complete meta-learning system"""
        import os
        
        # Load regime detector
        regime_path = os.path.join(directory, 'regime_detector.pkl')
        if os.path.exists(regime_path):
            with open(regime_path, 'rb') as f:
                self.regime_detector = pickle.load(f)
        
        # Load ensembles
        for symbol in self.symbols:
            ensemble_path = os.path.join(directory, f'meta_ensemble_{symbol}.pth')
            if os.path.exists(ensemble_path):
                if symbol not in self.adaptive_ensembles:
                    self.adaptive_ensembles[symbol] = AdaptiveEnsemble(self.model_names)
                self.adaptive_ensembles[symbol].load(ensemble_path)
        
        logger.info(f"Meta-learning ensemble system loaded from {directory}")


def main():
    """Test meta-learning ensemble system"""
    # Create sample data
    symbols = ['BTCUSDT', 'ETHUSDT']
    historical_data = {}
    
    for symbol in symbols:
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
        prices = 45000 + np.cumsum(np.random.randn(1000) * 100)
        
        historical_data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'open': prices + np.random.randn(1000) * 50,
            'volume': np.random.lognormal(15, 1, 1000)
        })
    
    # Initialize system
    meta_system = MetaLearningEnsembleSystem(symbols)
    meta_system.fit_regime_detector(historical_data)
    
    # Test prediction
    test_data = historical_data['BTCUSDT'].tail(100)
    model_predictions = {
        'enhanced_transformer': 0.75,
        'cnn_lstm': 0.68,
        'technical_indicators': 0.72,
        'smc_analyzer': 0.71,
        'reinforcement_learning': 0.69
    }
    model_confidences = {
        'enhanced_transformer': 0.85,
        'cnn_lstm': 0.78,
        'technical_indicators': 0.65,
        'smc_analyzer': 0.82,
        'reinforcement_learning': 0.73
    }
    
    result = meta_system.get_enhanced_prediction(
        'BTCUSDT', test_data, model_predictions, model_confidences
    )
    
    print(f"Meta-learning ensemble result: {result}")


if __name__ == "__main__":
    main()

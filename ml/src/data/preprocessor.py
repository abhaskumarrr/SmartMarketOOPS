"""
Enhanced Data Preprocessing Module

This module extends the basic preprocessing functionality from data_loader.py,
adding more advanced features like standard scaling, additional technical indicators,
and other preprocessing techniques to improve model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from .data_loader import MarketDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPreprocessor:
    """Extended preprocessing functionality for crypto market data"""
    
    def __init__(self, data_loader: Optional[MarketDataLoader] = None):
        """
        Initialize the enhanced preprocessor
        
        Args:
            data_loader: Crypto data loader instance (creates new one if None)
        """
        self.data_loader = data_loader or MarketDataLoader()
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators and features beyond the basic ones
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added advanced features
        """
        # Start with the basic technical indicators
        data = self.data_loader.add_technical_indicators(df.copy())
        
        # Add more advanced features
        
        # Stochastic Oscillator
        high_14 = data['high'].rolling(window=14).max()
        low_14 = data['low'].rolling(window=14).min()
        
        # %K
        data['stoch_k'] = 100 * (data['close'] - low_14) / (high_14 - low_14)
        # %D (3-day moving average of %K)
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        tr1 = abs(data['high'] - data['low'])
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        data['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['atr'] = data['true_range'].rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        data['obv'] = 0.0  # Initialize with zeros
        
        # Use integer indexing with iloc instead of loc for datetime index
        # Set first value
        data.iloc[0, data.columns.get_loc('obv')] = data.iloc[0]['volume']
        
        # Calculate OBV for the rest of the data
        for i in range(1, len(data)):
            if data.iloc[i]['close'] > data.iloc[i-1]['close']:
                data.iloc[i, data.columns.get_loc('obv')] = data.iloc[i-1]['obv'] + data.iloc[i]['volume']
            elif data.iloc[i]['close'] < data.iloc[i-1]['close']:
                data.iloc[i, data.columns.get_loc('obv')] = data.iloc[i-1]['obv'] - data.iloc[i]['volume']
            else:
                data.iloc[i, data.columns.get_loc('obv')] = data.iloc[i-1]['obv']
        
        # Ichimoku Cloud indicators
        # Conversion Line (Tenkan-sen)
        data['tenkan_sen'] = (data['high'].rolling(window=9).max() + 
                              data['low'].rolling(window=9).min()) / 2
        # Base Line (Kijun-sen)
        data['kijun_sen'] = (data['high'].rolling(window=26).max() + 
                             data['low'].rolling(window=26).min()) / 2
        # Leading Span A (Senkou Span A)
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        # Leading Span B (Senkou Span B)
        data['senkou_span_b'] = ((data['high'].rolling(window=52).max() + 
                                 data['low'].rolling(window=52).min()) / 2).shift(26)
        # Lagging Span (Chikou Span)
        data['chikou_span'] = data['close'].shift(-26)
        
        # Volatility indicators - Normalized by price to make them more comparable
        # Standard deviation of close price (normalized by price)
        data['norm_std_dev'] = data['close'].rolling(window=20).std() / data['close']
        
        # Price acceleration (2nd derivative)
        data['price_accel'] = data['close'].diff().diff()
        
        # Log returns instead of percentage changes
        data['log_return_1d'] = np.log(data['close'] / data['close'].shift(1))
        data['log_return_3d'] = np.log(data['close'] / data['close'].shift(3))
        data['log_return_7d'] = np.log(data['close'] / data['close'].shift(7))
        
        # Price relative to moving averages
        data['close_rel_ma7'] = data['close'] / data['ma7']
        data['close_rel_ma30'] = data['close'] / data['ma30']
        
        # Recent range (high/low spread) relative to price
        data['recent_range_rel'] = (data['high'].rolling(window=7).max() - 
                                   data['low'].rolling(window=7).min()) / data['close']
        
        # Volume relative to moving average
        data['vol_rel_ma7'] = data['volume'] / data['volume_ma7']
        
        return data
    
    def standard_scale_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
        """
        Normalize data using standard scaling (zero mean, unit variance)
        
        Args:
            df: DataFrame with feature data
            
        Returns:
            Tuple containing:
                - Normalized data as numpy array
                - StandardScaler fitted on the data
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.values)
        return scaled_data, scaler
    
    def robust_scale_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, RobustScaler]:
        """
        Normalize data using robust scaling (less sensitive to outliers)
        
        Args:
            df: DataFrame with feature data
            
        Returns:
            Tuple containing:
                - Normalized data as numpy array
                - RobustScaler fitted on the data
        """
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df.values)
        return scaled_data, scaler
    
    def preprocess_with_standard_scaling(
        self, 
        df: pd.DataFrame, 
        add_features: bool = True,
        add_advanced_features: bool = True,
        target_column: str = 'close',
        sequence_length: int = 48,
        train_split: float = 0.8,
        forecast_horizon: int = 1
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray, Any]]:
        """
        Preprocess data using standard scaling
        
        Args:
            df: DataFrame with OHLCV data
            add_features: Whether to add basic technical indicators
            add_advanced_features: Whether to add advanced features
            target_column: Column to use as prediction target
            sequence_length: Number of time steps for each sample
            train_split: Proportion of data to use for training
            forecast_horizon: Number of steps ahead to predict
            
        Returns:
            Dict containing preprocessed data and metadata
        """
        # Make a copy to avoid modifying the original DataFrame
        data = df.copy()
        
        # Verify required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Add technical indicators as features if requested
        if add_features:
            if add_advanced_features:
                data = self.add_advanced_features(data)
            else:
                data = self.data_loader.add_technical_indicators(data)
        
        # Drop rows with NaN values (e.g., from technical indicators)
        data.dropna(inplace=True)
        
        # Select feature columns - exclude the timestamp if it's in columns
        feature_columns = [col for col in data.columns if col != 'timestamp']
        
        # Apply standard scaling
        scaled_data, scaler = self.standard_scale_data(data[feature_columns])
        
        # Create sequences for time-series prediction
        X, y = self.data_loader.create_sequences(
            data=scaled_data,
            target_idx=feature_columns.index(target_column),
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        # Split into training and validation sets
        split_idx = int(len(X) * train_split)
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train, 
            'y_train': y_train,
            'X_val': X_val, 
            'y_val': y_val,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'scaler': scaler
        }
    
    def preprocess_with_robust_scaling(
        self, 
        df: pd.DataFrame, 
        add_features: bool = True,
        add_advanced_features: bool = True,
        target_column: str = 'close',
        sequence_length: int = 48,
        train_split: float = 0.8,
        forecast_horizon: int = 1
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray, Any]]:
        """
        Preprocess data using robust scaling (less sensitive to outliers)
        
        Args:
            df: DataFrame with OHLCV data
            add_features: Whether to add basic technical indicators
            add_advanced_features: Whether to add advanced features
            target_column: Column to use as prediction target
            sequence_length: Number of time steps for each sample
            train_split: Proportion of data to use for training
            forecast_horizon: Number of steps ahead to predict
            
        Returns:
            Dict containing preprocessed data and metadata
        """
        # Make a copy to avoid modifying the original DataFrame
        data = df.copy()
        
        # Verify required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Add technical indicators as features if requested
        if add_features:
            if add_advanced_features:
                data = self.add_advanced_features(data)
            else:
                data = self.data_loader.add_technical_indicators(data)
        
        # Drop rows with NaN values (e.g., from technical indicators)
        data.dropna(inplace=True)
        
        # Select feature columns - exclude the timestamp if it's in columns
        feature_columns = [col for col in data.columns if col != 'timestamp']
        
        # Apply robust scaling
        scaled_data, scaler = self.robust_scale_data(data[feature_columns])
        
        # Create sequences for time-series prediction
        X, y = self.data_loader.create_sequences(
            data=scaled_data,
            target_idx=feature_columns.index(target_column),
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        # Split into training and validation sets
        split_idx = int(len(X) * train_split)
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train, 
            'y_train': y_train,
            'X_val': X_val, 
            'y_val': y_val,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'scaler': scaler
        }
    
    def denormalize_with_scaler(self, data: np.ndarray, scaler: Union[StandardScaler, RobustScaler], feature_idx: int) -> np.ndarray:
        """
        Denormalize data that was scaled with StandardScaler or RobustScaler
        
        Args:
            data: Normalized data
            scaler: The scaler used for normalization
            feature_idx: Index of the feature to denormalize
            
        Returns:
            Denormalized data
        """
        # Create a dummy array with zeros except for the target feature
        dummy = np.zeros((data.shape[0], scaler.n_features_in_))
        dummy[:, feature_idx] = data
        
        # Apply inverse transform and extract only the target feature
        return scaler.inverse_transform(dummy)[:, feature_idx]
    
    def cross_validation_split(
        self, 
        data: np.ndarray, 
        targets: np.ndarray, 
        n_splits: int = 5
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create time-series cross-validation splits
        
        Args:
            data: Input data
            targets: Target values
            n_splits: Number of cross-validation splits
            
        Returns:
            List of dictionaries containing train/val split for each fold
        """
        if n_splits < 2:
            raise ValueError("Number of splits should be at least 2")
        
        total_samples = len(data)
        splits = []
        
        # For time series, we do forward-chaining cross-validation
        # Each split has more training data than the previous one
        for i in range(n_splits):
            # Calculate split point ensuring a minimum validation size
            val_size = total_samples // (n_splits + 1)
            split_idx = total_samples - (i + 1) * val_size
            
            if split_idx <= 10:  # Ensure minimum training data
                continue
                
            X_train, X_val = data[:split_idx], data[split_idx:split_idx + val_size]
            y_train, y_val = targets[:split_idx], targets[split_idx:split_idx + val_size]
            
            splits.append({
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'split_idx': split_idx
            })
        
        return splits


def preprocess_with_enhanced_features(
    symbol: str = "BTCUSD",
    interval: str = "1h",
    days_back: int = 30,
    use_cache: bool = True,
    scaling_method: str = "standard",
    add_advanced_features: bool = True,
    **kwargs
) -> Dict:
    """
    Convenience function to load and preprocess data with enhanced features
    
    Args:
        symbol: Trading symbol
        interval: Timeframe
        days_back: Number of days to look back
        use_cache: Whether to use cached data
        scaling_method: Scaling method ('minmax', 'standard', 'robust')
        add_advanced_features: Whether to add advanced features
        **kwargs: Additional arguments for preprocessing
        
    Returns:
        Preprocessed data dictionary
    """
    loader = MarketDataLoader()
    preprocessor = EnhancedPreprocessor(loader)
    
    # Load data
    df = loader.get_data(symbol, interval, days_back, use_cache)
    
    # Apply preprocessing based on selected scaling method
    if scaling_method.lower() == 'minmax':
        return loader.preprocess_data(df, **kwargs)
    elif scaling_method.lower() == 'standard':
        return preprocessor.preprocess_with_standard_scaling(
            df, 
            add_advanced_features=add_advanced_features,
            **kwargs
        )
    elif scaling_method.lower() == 'robust':
        return preprocessor.preprocess_with_robust_scaling(
            df,
            add_advanced_features=add_advanced_features,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}") 
"""
Enhanced Data Preprocessing Pipeline for Transformer Models
Optimized for financial time series data and multi-modal inputs
Implements Subtask 24.2: Data Preprocessing for Transformer Input
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class TransformerDataset(Dataset):
    """
    Custom Dataset for Transformer models with financial time series data
    Supports variable sequence lengths and multi-modal features
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        sequence_lengths: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize the dataset

        Args:
            sequences: Input sequences [num_samples, max_seq_len, num_features]
            targets: Target values [num_samples, num_targets]
            sequence_lengths: Actual lengths of each sequence
            feature_names: Names of input features
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.sequence_lengths = torch.LongTensor(sequence_lengths) if sequence_lengths is not None else None
        self.feature_names = feature_names or []

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sequence_lengths is not None:
            return self.sequences[idx], self.targets[idx], self.sequence_lengths[idx]
        return self.sequences[idx], self.targets[idx]


class TransformerPreprocessor:
    """
    Enhanced preprocessing pipeline for Transformer models
    Optimized for financial time series with attention mechanisms
    """

    def __init__(
        self,
        sequence_length: int = 100,
        forecast_horizon: int = 1,
        scaling_method: str = 'standard',
        feature_engineering: bool = True,
        multi_timeframe: bool = True,
        attention_features: bool = True
    ):
        """
        Initialize the preprocessor

        Args:
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to predict ahead
            scaling_method: 'standard', 'robust', or 'minmax'
            feature_engineering: Whether to add engineered features
            multi_timeframe: Whether to include multi-timeframe features
            attention_features: Whether to add attention-specific features
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaling_method = scaling_method
        self.feature_engineering = feature_engineering
        self.multi_timeframe = multi_timeframe
        self.attention_features = attention_features

        # Scalers for different feature groups
        self.price_scaler = None
        self.volume_scaler = None
        self.technical_scaler = None
        self.feature_names = []

        logger.info(f"TransformerPreprocessor initialized: seq_len={sequence_length}, "
                   f"forecast_horizon={forecast_horizon}, scaling={scaling_method}")

    def _create_scaler(self) -> Union[StandardScaler, RobustScaler]:
        """Create appropriate scaler based on method"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")

    def _add_attention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features specifically designed for attention mechanisms
        These help the model focus on important market aspects
        """
        if not self.attention_features:
            return df

        # Price momentum features (for price attention)
        df['price_momentum_short'] = df['close'].pct_change(5)
        df['price_momentum_medium'] = df['close'].pct_change(20)
        df['price_momentum_long'] = df['close'].pct_change(50)

        # Volume attention features
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)

        # Volatility attention features
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volatility_short'] = df['close'].rolling(10).std() / df['close']
        df['volatility_long'] = df['close'].rolling(50).std() / df['close']

        # Market structure features for attention
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

        return df

    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-timeframe features for better temporal understanding
        """
        if not self.multi_timeframe:
            return df

        # Different timeframe moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ma_{period}_ratio'] = df['close'] / df[f'ma_{period}']

        # Multi-timeframe RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Bollinger Bands for different periods
        for period in [10, 20, 50]:
            rolling_mean = df['close'].rolling(period).mean()
            rolling_std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = rolling_mean + (rolling_std * 2)
            df[f'bb_lower_{period}'] = rolling_mean - (rolling_std * 2)
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

        return df

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators and engineered features
        """
        if not self.feature_engineering:
            return df

        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']

        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['price_volume'] = df['close'] * df['volume']

        # Volatility features
        df['volatility'] = df['returns'].rolling(20).std()
        df['atr'] = self._calculate_atr(df)

        # Trend features
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()

    def _separate_feature_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Separate features into groups for specialized scaling
        """
        # Exclude timestamp and datetime columns
        numeric_cols = [col for col in df.columns if col not in ['timestamp'] and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        # Price-related features
        price_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['price', 'close', 'open', 'high', 'low', 'ma_', 'ema', 'bb_'])]

        # Volume-related features
        volume_cols = [col for col in numeric_cols if any(x in col.lower() for x in ['volume'])]

        # Technical indicators and ratios
        technical_cols = [col for col in numeric_cols if col not in price_cols + volume_cols]

        return df[price_cols], df[volume_cols], df[technical_cols]

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str = 'close',
        train_split: float = 0.8
    ) -> Dict[str, Any]:
        """
        Fit the preprocessor and transform the data

        Args:
            df: Input DataFrame with OHLCV data
            target_column: Column to use as prediction target
            train_split: Proportion of data for training

        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Starting Transformer preprocessing pipeline")

        # Make a copy to avoid modifying original data
        data = df.copy()

        # Add engineered features
        if self.feature_engineering:
            data = self._add_engineered_features(data)

        # Add multi-timeframe features
        if self.multi_timeframe:
            data = self._add_multi_timeframe_features(data)

        # Add attention-specific features
        if self.attention_features:
            data = self._add_attention_features(data)

        # Remove NaN values
        data = data.dropna()

        # Separate features into groups
        price_features, volume_features, technical_features = self._separate_feature_groups(data)

        # Fit scalers on training data
        train_size = int(len(data) * train_split)

        # Scale price features
        if not price_features.empty:
            self.price_scaler = self._create_scaler()
            price_scaled = self.price_scaler.fit_transform(price_features.iloc[:train_size])
            price_scaled_full = self.price_scaler.transform(price_features)
        else:
            price_scaled_full = np.array([]).reshape(len(data), 0)

        # Scale volume features
        if not volume_features.empty:
            self.volume_scaler = self._create_scaler()
            volume_scaled = self.volume_scaler.fit_transform(volume_features.iloc[:train_size])
            volume_scaled_full = self.volume_scaler.transform(volume_features)
        else:
            volume_scaled_full = np.array([]).reshape(len(data), 0)

        # Scale technical features
        if not technical_features.empty:
            self.technical_scaler = self._create_scaler()
            technical_scaled = self.technical_scaler.fit_transform(technical_features.iloc[:train_size])
            technical_scaled_full = self.technical_scaler.transform(technical_features)
        else:
            technical_scaled_full = np.array([]).reshape(len(data), 0)

        # Combine all scaled features
        scaled_features = np.hstack([price_scaled_full, volume_scaled_full, technical_scaled_full])

        # Store feature names
        self.feature_names = list(price_features.columns) + list(volume_features.columns) + list(technical_features.columns)

        # Create sequences
        sequences, targets = self._create_sequences(scaled_features, data[target_column].values)

        # Split into train/validation
        train_size_seq = int(len(sequences) * train_split)

        result = {
            'X_train': sequences[:train_size_seq],
            'y_train': targets[:train_size_seq],
            'X_val': sequences[train_size_seq:],
            'y_val': targets[train_size_seq:],
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'sequence_length': self.sequence_length,
            'scalers': {
                'price': self.price_scaler,
                'volume': self.volume_scaler,
                'technical': self.technical_scaler
            }
        }

        logger.info(f"Preprocessing complete: {len(self.feature_names)} features, "
                   f"{len(sequences)} sequences, train/val split: {train_size_seq}/{len(sequences) - train_size_seq}")

        return result

    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction

        Args:
            features: Scaled feature array
            targets: Target values

        Returns:
            Tuple of (sequences, targets)
        """
        X, y = [], []

        for i in range(self.sequence_length, len(features) - self.forecast_horizon + 1):
            # Input sequence
            X.append(features[i - self.sequence_length:i])

            # Target (future value)
            if self.forecast_horizon == 1:
                y.append(targets[i])
            else:
                y.append(targets[i:i + self.forecast_horizon])

        return np.array(X), np.array(y)

    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training and validation

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data

        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = TransformerDataset(X_train, y_train, feature_names=self.feature_names)
        val_dataset = TransformerDataset(X_val, y_val, feature_names=self.feature_names)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

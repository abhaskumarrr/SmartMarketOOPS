import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Union
import logging
import ccxt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import ta # Technical Analysis library
import yaml

logger = logging.getLogger(__name__)

# Define a small threshold for determining 'Neutral' movement (must match loader.py)
LOG_RETURN_THRESHOLD = 0.0005

class UnifiedDataProcessor:
    """
    Unified data processor for cryptocurrency market data.
    Handles data fetching, loading, comprehensive feature engineering,
    scaling, splitting, and preparation for various ML model architectures.
    """

    def __init__(self,
                 data_dir: str = os.path.join('data', 'raw'),
                 processed_dir: str = os.path.join('data', 'processed'),
                 timeframe: str = '1h',
                 symbols: List[str] = None,
                 sequence_length: int = 60,
                 forecast_horizon: int = 1,
                 scaling_method: str = 'standard', # 'standard', 'robust', 'minmax'
                 feature_engineering_config: Dict[str, bool] = None):
        
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.timeframe = timeframe
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaling_method = scaling_method
        self.feature_engineering_config = feature_engineering_config or {
            'smc': True,
            'technical_indicators': True,
            'lagged_features': True,
            'multi_timeframe': True,
            'attention_features': True
        }

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        self.exchange_handlers = {
            'binance': ccxt.binance,
            'kucoin': ccxt.kucoin,
            'coinbase': ccxt.coinbase,
        }
        
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def fetch_historical_data(self,
                             exchange: str = 'binance',
                             start_date: str = None,
                             end_date: str = None,
                             limit: int = 1000) -> Dict[str, pd.DataFrame]:
        # ... (existing fetch_historical_data logic from data_loader.py) ...
        """
        Fetch historical OHLCV data from exchange.
        
        Args:
            exchange: Exchange to fetch data from
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            limit: Maximum number of candles per request
            
        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """
        if exchange not in self.exchange_handlers:
            raise ValueError(f"Unsupported exchange: {exchange}. Available: {list(self.exchange_handlers.keys())}")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Convert dates to timestamps
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        until = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        logger.info(f"Fetching data from {exchange} for {self.symbols} from {start_date} to {end_date}")
        
        # Initialize exchange
        try:
            exchange_class = self.exchange_handlers[exchange]()
            exchange_class.load_markets()
        except Exception as e:
            logger.error(f"Failed to initialize {exchange}: {e}")
            raise
        
        data = {}
        for symbol in self.symbols:
            try:
                logger.info(f"Fetching {symbol} data...")
                ohlcv = []
                current_since = since
                
                while current_since < until:
                    batch = exchange_class.fetch_ohlcv(symbol, self.timeframe, current_since, limit)
                    if not batch:
                        break
                    
                    ohlcv.extend(batch)
                    # Update timestamp for next iteration
                    current_since = batch[-1][0] + 1
                    
                if ohlcv:
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    data[symbol] = df
                    
                    # Save raw data
                    filename = f"{symbol.replace('/', '_')}_{self.timeframe}_{start_date}_{end_date}.csv"
                    df.to_csv(os.path.join(self.data_dir, filename))
                    logger.info(f"Saved {len(df)} records for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
                
        return data

    def load_from_csv(self,
                     symbol: str = None,
                     file_path: str = None,
                     start_date: str = None,
                     end_date: str = None) -> pd.DataFrame:
        # ... (existing load_from_csv logic from data_loader.py) ...
        """
        Load OHLCV data from CSV file.
        
        Args:
            symbol: Symbol to load (used to construct filename if file_path not provided)
            file_path: Path to CSV file
            start_date: Filter data starting from this date
            end_date: Filter data ending at this date
            
        Returns:
            DataFrame with OHLCV data
        """
        if file_path is None and symbol is None:
            raise ValueError("Either file_path or symbol must be provided")
            
        if file_path is None:
            # Find latest file for this symbol
            symbol_str = symbol.replace('/', '_')
            files = [f for f in os.listdir(self.data_dir) if f.startswith(symbol_str)]
            if not files:
                raise FileNotFoundError(f"No files found for {symbol}")
            
            files.sort(reverse=True)  # Most recent first
            file_path = os.path.join(self.data_dir, files[0])
        
        # Load data
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Filter by date if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        return df

    def _create_scaler(self) -> Union[StandardScaler, RobustScaler, MinMaxScaler]:
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features...")
        df_copy = df.copy()

        # Basic price features
        df_copy['returns'] = df_copy['close'].pct_change()
        df_copy['log_returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        df_copy['price_change'] = df_copy['close'] - df_copy['open']
        df_copy['price_range'] = df_copy['high'] - df_copy['low']

        # Volume features
        df_copy['volume_change'] = df_copy['volume'].pct_change()
        df_copy['price_volume'] = df_copy['close'] * df_copy['volume']

        # Volatility features
        df_copy['volatility'] = df_copy['returns'].rolling(20).std()
        df_copy['atr'] = self._calculate_atr(df_copy)

        # Trend features
        df_copy['ema_12'] = df_copy['close'].ewm(span=12).mean()
        df_copy['ema_26'] = df_copy['close'].ewm(span=26).mean()
        df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
        df_copy['macd_signal'] = df_copy['macd'].ewm(span=9).mean()

        # SMC features (from MarketDataLoader.preprocess_for_smc)
        if self.feature_engineering_config.get('smc', False):
            window_size = 20 # Default window size for SMC
            df_copy['swing_high'] = df_copy['high'].rolling(window=window_size, center=True).apply(
                lambda x: 1 if x.iloc[window_size//2] == max(x) else 0, raw=False
            )
            df_copy['swing_low'] = df_copy['low'].rolling(window=window_size, center=True).apply(
                lambda x: 1 if x.iloc[window_size//2] == min(x) else 0, raw=False
            )
            df_copy['bullish_fvg'] = (
                (df_copy['low'].shift(1) > df_copy['high'].shift(-1)) &
                (df_copy['close'].shift(1) < df_copy['open'].shift(1)) &
                (df_copy['close'].shift(-1) > df_copy['open'].shift(-1))
            ).astype(int)
            df_copy['bearish_fvg'] = (
                (df_copy['high'].shift(1) < df_copy['low'].shift(-1)) &
                (df_copy['close'].shift(1) > df_copy['open'].shift(1)) &
                (df_copy['close'].shift(-1) < df_copy['open'].shift(-1))
            ).astype(int)
            df_copy['price_change_smc'] = df_copy['close'].pct_change(5)
            df_copy['bullish_ob'] = (
                (df_copy['close'] < df_copy['open']) &
                (df_copy['price_change_smc'].shift(-1) > 0.02)
            ).astype(int)
            df_copy['bearish_ob'] = (
                (df_copy['close'] > df_copy['open']) &
                (df_copy['price_change_smc'].shift(-1) < -0.02)
            ).astype(int)
            df_copy['buy_liquidity'] = df_copy['swing_high'].rolling(window=10).max().fillna(0)
            df_copy['sell_liquidity'] = df_copy['swing_low'].rolling(window=10).max().fillna(0)

        # Technical indicators (from TradingDataset and preprocessing.py)
        if self.feature_engineering_config.get('technical_indicators', False):
            df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=14)
            for ma_period in [20, 50, 200]:
                df_copy[f'sma_{ma_period}'] = ta.trend.sma_indicator(df_copy['close'], window=ma_period)
            df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
            df_copy['obv'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
            df_copy['stoch_k'] = ta.momentum.stoch(df_copy['high'], df_copy['low'], df_copy['close'])
            df_copy['stoch_d'] = ta.momentum.stoch_signal(df_copy['high'], df_copy['low'], df_copy['close'])

        # Lagged features (from MarketDataLoader.create_features)
        if self.feature_engineering_config.get('lagged_features', False):
            for col in ['close', 'volume', 'rsi', 'macd']: # Assuming rsi and macd are already engineered
                for lag in range(1, 6):
                    df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)

        # Multi-timeframe features (from TransformerPreprocessor)
        if self.feature_engineering_config.get('multi_timeframe', False):
            for period in [5, 10, 20, 50, 100, 200]:
                df_copy[f'ma_{period}'] = df_copy['close'].rolling(period).mean()
                df_copy[f'ma_{period}_ratio'] = df_copy['close'] / df_copy[f'ma_{period}']
            for period in [7, 14, 21]:
                delta = df_copy['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df_copy[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            for period in [10, 20, 50]:
                rolling_mean = df_copy['close'].rolling(period).mean()
                rolling_std = df_copy['close'].rolling(period).std()
                df_copy[f'bb_upper_{period}'] = rolling_mean + (rolling_std * 2)
                df_copy[f'bb_lower_{period}'] = rolling_mean - (rolling_std * 2)
                df_copy[f'bb_position_{period}'] = (df_copy['close'] - df_copy[f'bb_lower_{period}']) / (df_copy[f'bb_upper_{period}'] - df_copy[f'bb_lower_{period}'])

        # Attention features (from TransformerPreprocessor)
        if self.feature_engineering_config.get('attention_features', False):
            df_copy['price_momentum_short'] = df_copy['close'].pct_change(5)
            df_copy['price_momentum_medium'] = df_copy['close'].pct_change(20)
            df_copy['price_momentum_long'] = df_copy['close'].pct_change(50)
            df_copy['volume_ma_ratio'] = df_copy['volume'] / df_copy['volume'].rolling(20).mean()
            df_copy['volume_spike'] = (df_copy['volume'] > df_copy['volume'].rolling(20).mean() * 2).astype(int)
            df_copy['price_range_att'] = (df_copy['high'] - df_copy['low']) / df_copy['close']
            df_copy['volatility_short'] = df_copy['close'].rolling(10).std() / df_copy['close']
            df_copy['volatility_long'] = df_copy['close'].rolling(50).std() / df_copy['close']
            df_copy['higher_high'] = (df_copy['high'] > df_copy['high'].shift(1)).astype(int)
            df_copy['lower_low'] = (df_copy['low'] < df_copy['low'].shift(1)).astype(int)

        return df_copy.dropna() # Drop NaNs created by feature engineering

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()

    def _get_feature_columns(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'next_log_close', 'target_log_return', 'target_direction']
        
        # Filter out non-numeric columns and excluded columns
        feature_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64', 'float32', 'int32'] and col not in exclude_cols]
        return feature_cols

    def fit_transform(self, df: pd.DataFrame, target_column: str = 'close', train_split: float = 0.8) -> Dict[str, Any]:
        logger.info("Starting data processing pipeline (fit_transform)...")
        data = df.copy()

        data = self._engineer_features(data)
        
        # Define target
        data['log_close'] = np.log(data[target_column])
        data['next_log_close'] = data['log_close'].shift(-self.forecast_horizon)
        data['target_log_return'] = data['next_log_close'] - data['log_close']
        data['target_direction'] = data['target_log_return'].apply(
            lambda x: 0 if x < -LOG_RETURN_THRESHOLD else (2 if x > LOG_RETURN_THRESHOLD else 1)
        )
        
        data = data.dropna() # Drop NaNs after target creation

        self.feature_names = self._get_feature_columns(data, exclude_cols=['timestamp', 'next_log_close', 'target_log_return', 'target_direction'])
        logger.info(f"Identified {len(self.feature_names)} features: {self.feature_names}")

        features_df = data[self.feature_names]
        targets_series = data['target_direction']

        # Fit scalers on training data
        train_size = int(len(features_df) * train_split)
        train_features = features_df.iloc[:train_size]

        scaler = self._create_scaler()
        self.scalers['features'] = scaler.fit(train_features)
        
        scaled_features = self.scalers['features'].transform(features_df)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(scaled_features, targets_series.values)

        train_end_idx = int(len(X_seq) * train_split)
        val_end_idx = train_end_idx + int(len(X_seq) * (1 - train_split) / 2) # Simple split for now

        X_train, y_train = X_seq[:train_end_idx], y_seq[:train_end_idx]
        X_val, y_val = X_seq[train_end_idx:val_end_idx], y_seq[train_end_idx:val_end_idx]
        X_test, y_test = X_seq[val_end_idx:], y_seq[val_end_idx:]

        logger.info(f"Data processing complete. Train: {len(X_train)} samples, Val: {len(X_val)} samples, Test: {len(X_test)} samples.")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'scalers': self.scalers
        }

    def transform_inference_data(self, df: pd.DataFrame) -> np.ndarray:
        logger.info("Transforming inference data...")
        data = df.copy()
        data = self._engineer_features(data)
        
        # Ensure feature columns match
        current_features = self._get_feature_columns(data, exclude_cols=['timestamp'])
        if set(current_features) != set(self.feature_names):
            logger.warning("Inference data features do not exactly match training features. Attempting to reindex.")
            data = data.reindex(columns=self.feature_names, fill_value=0) # Fill missing with 0, might need better strategy
        else:
            data = data[self.feature_names] # Ensure order

        if 'features' not in self.scalers:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        scaled_features = self.scalers['features'].transform(data)

        if len(scaled_features) < self.sequence_length:
            raise ValueError(f"Not enough data to create a sequence of length {self.sequence_length}. "
                             f"Provided: {len(scaled_features)} data points.")

        inference_sequence = scaled_features[-self.sequence_length:]
        return inference_sequence.reshape(1, self.sequence_length, len(self.feature_names))

    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i : i + self.sequence_length])
            y.append(targets[i + self.sequence_length + self.forecast_horizon - 1])
        return np.array(X), np.array(y)

    def create_pytorch_datasets(self, processed_data: Dict[str, np.ndarray]) -> Tuple[Dataset, Dataset, Dataset]:
        X_train, y_train = processed_data['X_train'], processed_data['y_train']
        X_val, y_val = processed_data['X_val'], processed_data['y_val']
        X_test, y_test = processed_data['X_test'], processed_data['y_test']

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        
        return train_dataset, val_dataset, test_dataset

    def create_pytorch_dataloaders(self, datasets: Tuple[Dataset, Dataset, Dataset], batch_size: int = 32, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, val_dataset, test_dataset = datasets
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

class RealTimeDataNormalizer:
    """
    Normalizes raw WebSocket data from Delta Exchange into a standard format for downstream processing.
    Handles trades, order book, and ticker data. Robust to missing/erroneous fields.
    """
    def __init__(self):
        pass

    def normalize(self, raw_data):
        # Determine message type
        msg_type = raw_data.get('type')
        payload = raw_data.get('data') or raw_data.get('payload') or raw_data
        result = []

        if msg_type == 'trade' or (isinstance(payload, dict) and 'trades' in payload):
            trades = payload.get('trades', []) if isinstance(payload, dict) else []
            for trade in trades:
                result.append({
                    'timestamp': int(trade.get('timestamp', trade.get('time', 0))),
                    'price': float(trade.get('price', 0)),
                    'volume': float(trade.get('size', trade.get('volume', 0))),
                    'symbol': trade.get('symbol', payload.get('symbol', '')),
                    'type': 'trade',
                    'side': trade.get('side', None),
                    'id': trade.get('id', None)
                })
        elif msg_type == 'orderbook' or (isinstance(payload, dict) and ('bids' in payload or 'asks' in payload)):
            # Order book snapshot or update
            result.append({
                'timestamp': int(payload.get('timestamp', payload.get('time', 0))),
                'symbol': payload.get('symbol', ''),
                'type': 'orderbook',
                'bids': payload.get('bids', []),
                'asks': payload.get('asks', [])
            })
        elif msg_type == 'ticker' or (isinstance(payload, dict) and 'mark_price' in payload):
            result.append({
                'timestamp': int(payload.get('timestamp', payload.get('time', 0))),
                'symbol': payload.get('symbol', ''),
                'type': 'ticker',
                'price': float(payload.get('mark_price', payload.get('price', 0))),
                'volume': float(payload.get('volume', 0))
            })
        else:
            # Unknown or unhandled type, pass through for logging/debug
            result.append({'raw': raw_data, 'type': 'unknown'})
        return result

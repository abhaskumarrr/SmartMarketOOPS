"""
Data Loader for Crypto Market Data

This module provides functionality to load historical cryptocurrency data 
from Delta Exchange API or CSV files, and prepare it for ML model training.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import ccxt
import torch
from torch.utils.data import TensorDataset, DataLoader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataLoader:
    """
    Data loader for cryptocurrency market data with support for Smart Money Concepts (SMC) features.
    """
    
    def __init__(self, 
                 data_dir: str = os.path.join('data', 'raw'),
                 processed_dir: str = os.path.join('data', 'processed'),
                 timeframe: str = '1h',
                 symbols: List[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory for raw data
            processed_dir: Directory for processed data
            timeframe: Timeframe for data ('1m', '5m', '15m', '1h', '4h', '1d')
            symbols: List of cryptocurrency symbols to load
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.timeframe = timeframe
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Available exchange handlers
        self.exchange_handlers = {
            'binance': ccxt.binance,
            'kucoin': ccxt.kucoin,
            'coinbase': ccxt.coinbase,
            # 'ftx': ccxt.ftx,  # FTX is no longer supported in ccxt
        }
    
    def fetch_historical_data(self, 
                             exchange: str = 'binance',
                             start_date: str = None,
                             end_date: str = None,
                             limit: int = 1000) -> Dict[str, pd.DataFrame]:
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
    
    def preprocess_for_smc(self, df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
        """
        Add Smart Money Concepts features to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Window size for identifying swings and order blocks
            
        Returns:
            DataFrame with SMC features
        """
        # Copy the DataFrame to avoid modifying the original
        df_smc = df.copy()
        
        # 1. Identify swing highs and lows
        df_smc['swing_high'] = df_smc['high'].rolling(window=window_size, center=True).apply(
            lambda x: 1 if x.iloc[window_size//2] == max(x) else 0, raw=False
        )
        df_smc['swing_low'] = df_smc['low'].rolling(window=window_size, center=True).apply(
            lambda x: 1 if x.iloc[window_size//2] == min(x) else 0, raw=False
        )
        
        # 2. Calculate Fair Value Gaps (imbalances)
        df_smc['bullish_fvg'] = (
            (df_smc['low'].shift(1) > df_smc['high'].shift(-1)) & 
            (df_smc['close'].shift(1) < df_smc['open'].shift(1)) &  # Bearish candle before gap
            (df_smc['close'].shift(-1) > df_smc['open'].shift(-1))  # Bullish candle after gap
        ).astype(int)
        
        df_smc['bearish_fvg'] = (
            (df_smc['high'].shift(1) < df_smc['low'].shift(-1)) & 
            (df_smc['close'].shift(1) > df_smc['open'].shift(1)) &  # Bullish candle before gap
            (df_smc['close'].shift(-1) < df_smc['open'].shift(-1))  # Bearish candle after gap
        ).astype(int)
        
        # 3. Identify potential order blocks
        # Bullish order block: Last bearish candle before a bullish move
        # Bearish order block: Last bullish candle before a bearish move
        df_smc['price_change'] = df_smc['close'].pct_change(5)  # 5-period change
        df_smc['bullish_ob'] = (
            (df_smc['close'] < df_smc['open']) &  # Bearish candle
            (df_smc['price_change'].shift(-1) > 0.02)  # Followed by strong bullish move
        ).astype(int)
        
        df_smc['bearish_ob'] = (
            (df_smc['close'] > df_smc['open']) &  # Bullish candle
            (df_smc['price_change'].shift(-1) < -0.02)  # Followed by strong bearish move
        ).astype(int)
        
        # 4. Calculate liquidity levels (previous swing highs/lows)
        df_smc['buy_liquidity'] = df_smc['swing_high'].rolling(window=10).max().fillna(0)
        df_smc['sell_liquidity'] = df_smc['swing_low'].rolling(window=10).max().fillna(0)
        
        # 5. Calculate technical indicators
        # RSI
        delta = df_smc['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_smc['rsi'] = 100 - (100 / (1 + rs))
        
        # Add moving averages
        for ma_period in [20, 50, 200]:
            df_smc[f'sma_{ma_period}'] = df_smc['close'].rolling(window=ma_period).mean()
            
        # MACD
        df_smc['ema_12'] = df_smc['close'].ewm(span=12, adjust=False).mean()
        df_smc['ema_26'] = df_smc['close'].ewm(span=26, adjust=False).mean()
        df_smc['macd'] = df_smc['ema_12'] - df_smc['ema_26']
        df_smc['macd_signal'] = df_smc['macd'].ewm(span=9, adjust=False).mean()
        df_smc['macd_hist'] = df_smc['macd'] - df_smc['macd_signal']
        
        # Fill NaN values
        df_smc = df_smc.fillna(0)
        
        return df_smc
    
    def create_features(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create feature matrix and target variable for machine learning.
        
        Args:
            df: DataFrame with OHLCV and SMC features
            target_column: Column to use as target
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Define the forecast horizon (e.g., price after n periods)
        horizon = 24 if self.timeframe == '1h' else 5
        
        # Create target variable (future price change %)
        df['target'] = df[target_column].pct_change(periods=horizon).shift(-horizon)
        
        # Binary target for classification (1 for price increase, 0 for decrease)
        df['target_binary'] = (df['target'] > 0).astype(int)
        
        # Drop rows with NaN target
        df_ml = df.dropna(subset=['target', 'target_binary'])
        
        # Select features
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'swing_high', 'swing_low', 
            'bullish_fvg', 'bearish_fvg',
            'bullish_ob', 'bearish_ob',
            'buy_liquidity', 'sell_liquidity',
            'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        # Add SMA columns
        feature_columns.extend([f'sma_{period}' for period in [20, 50, 200]])
        
        # Create lag features (last n values)
        for col in ['close', 'volume', 'rsi', 'macd']:
            for lag in range(1, 6):
                df_ml.loc[:, f'{col}_lag_{lag}'] = df_ml[col].shift(lag)
                feature_columns.append(f'{col}_lag_{lag}')
        
        # Drop rows with NaN features
        df_ml = df_ml.dropna()
        
        # Return features and target
        X = df_ml[feature_columns]
        y = df_ml['target_binary']  # Use binary target for classification
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  train_size: float = 0.7, 
                  val_size: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation and test sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Time series split (no random shuffling)
        train_end = int(len(X) * train_size)
        val_end = train_end + int(len(X) * val_size)
        
        X_train = X.iloc[:train_end].values
        X_val = X.iloc[train_end:val_end].values
        X_test = X.iloc[val_end:].values
        
        y_train = y.iloc[:train_end].values
        y_val = y.iloc[train_end:val_end].values
        y_test = y.iloc[val_end:].values
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_sequence_data(self, X: pd.DataFrame, y: pd.Series, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM/GRU models.
        
        Args:
            X: Feature matrix
            y: Target variable
            sequence_length: Number of timesteps in each sequence
            
        Returns:
            Tuple of (X_seq, y_seq) where X_seq has shape (samples, sequence_length, features)
        """
        X_array = X.values
        y_array = y.values
        
        X_seq = []
        y_seq = []
        
        for i in range(len(X_array) - sequence_length):
            X_seq.append(X_array[i:i+sequence_length])
            y_seq.append(y_array[i+sequence_length])
            
        return np.array(X_seq), np.array(y_seq)

    def create_image_data(self, df: pd.DataFrame, window_size: int = 60, 
                         features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create image-like data for CNN models by stacking multiple features into channels.
        
        Args:
            df: DataFrame with features
            window_size: Size of the window (height and width of the image)
            features: List of features to use as channels
            
        Returns:
            Tuple of (X_images, y_labels) where X_images has shape (samples, height, width, channels)
        """
        if features is None:
            features = ['close', 'volume', 'rsi', 'macd']
            
        # Normalize each feature
        df_norm = df.copy()
        for feature in features:
            df_norm[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
            
        X_images = []
        y_labels = []
        
        # Get future returns (target)
        future_returns = df['close'].pct_change(5).shift(-5)
        
        for i in range(len(df_norm) - window_size - 5):  # -5 for future returns
            # Create multi-channel image
            image = np.zeros((window_size, window_size, len(features)))
            
            for j, feature in enumerate(features):
                # Convert time series to 2D image-like array
                feature_series = df_norm[feature].iloc[i:i+window_size].values
                
                # Method 1: Simple reshaping into a square
                feature_matrix = np.reshape(feature_series, (int(window_size**0.5), -1))
                
                # Resize to window_size x window_size
                from skimage.transform import resize
                feature_matrix = resize(feature_matrix, (window_size, window_size), 
                                      anti_aliasing=True, mode='reflect')
                
                image[:, :, j] = feature_matrix
                
            X_images.append(image)
            
            # Binary target: 1 if price increases, 0 if it decreases
            y_labels.append(1 if future_returns.iloc[i+window_size] > 0 else 0)
            
        return np.array(X_images), np.array(y_labels)


def load_data(
    symbol: str = "BTCUSD",
    interval: str = "1h",
    days_back: int = 30,
    use_cache: bool = True,
    preprocess: bool = True,
    batch_size: int = 32,
    sequence_length: int = 60,
    **preprocess_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads and preprocesses data, returning train/val/test DataLoaders for PyTorch models.
    """
    loader = MarketDataLoader(timeframe=interval, symbols=[symbol.replace('USD', '/USDT')])
    if use_cache:
        try:
            df = loader.load_from_csv(symbol=symbol.replace('USD', '/USDT'))
        except Exception as e:
            logger.warning(f"Could not load from CSV: {e}. Fetching from exchange instead.")
            df_dict = loader.fetch_historical_data(start_date=(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'), end_date=datetime.now().strftime('%Y-%m-%d'))
            df = df_dict.get(symbol.replace('USD', '/USDT'))
    else:
        df_dict = loader.fetch_historical_data(start_date=(datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'), end_date=datetime.now().strftime('%Y-%m-%d'))
        df = df_dict.get(symbol.replace('USD', '/USDT'))
    if df is None:
        raise ValueError(f"No data found for symbol {symbol}")
    # Only pass supported kwargs to preprocess_for_smc
    supported_keys = ['window_size']
    filtered_kwargs = {k: v for k, v in preprocess_kwargs.items() if k in supported_keys}
    df_proc = loader.preprocess_for_smc(df, **filtered_kwargs)
    X, y = loader.create_features(df_proc)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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


if __name__ == "__main__":
    # Example usage
    loader = MarketDataLoader(timeframe='1h', symbols=['BTC/USDT'])
    
    try:
        # Fetch data (uncomment to fetch new data)
        # data = loader.fetch_historical_data(start_date='2023-01-01', end_date='2023-03-01')
        
        # Or load existing data
        df = loader.load_from_csv(symbol='BTC/USDT')
        
        # Preprocess for SMC features
        df_smc = loader.preprocess_for_smc(df)
        
        # Create features and target
        X, y = loader.create_features(df_smc)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
        
        # Create sequence data for LSTM/GRU
        X_seq, y_seq = loader.prepare_sequence_data(X, y)
        
        # Create image data for CNN
        X_img, y_img = loader.create_image_data(df_smc)
        
        # Print shapes
        print(f"Regular data shapes: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"Sequence data shapes: X_seq {X_seq.shape}, y_seq {y_seq.shape}")
        print(f"Image data shapes: X_img {X_img.shape}, y_img {y_img.shape}")
        
    except Exception as e:
        logger.error(f"Error: {e}") 
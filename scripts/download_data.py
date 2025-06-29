import ccxt
import pandas as pd
import os
from datetime import datetime

def download_data(symbol, timeframe, since, limit, output_folder):
    """
    Downloads historical OHLCV data from Binance and saves it to a CSV file.
    """
    exchange = ccxt.binance()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    since_timestamp = exchange.parse8601(since)
    
    print(f"Downloading {symbol} {timeframe} data from {since}...")
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since_timestamp, limit)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    filepath = os.path.join(output_folder, filename)
    
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

if __name__ == '__main__':
    # Configuration
    symbol = 'BTC/USDT'
    timeframes = ['15m', '1h', '4h', '1d', '1w', '1M']
    since = '2023-01-01T00:00:00Z'  # Start date
    limit = 1000  # Number of candles to fetch
    output_folder = 'data/raw'
    
    for timeframe in timeframes:
        download_data(symbol, timeframe, since, limit, output_folder)

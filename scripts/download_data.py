import os
import pandas as pd
import ccxt

def download_data(symbol, timeframe, limit, output_dir):
    """Downloads historical market data from Binance and saves it to a CSV file."""
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_path = os.path.join(output_dir, f'{symbol.replace("/", "-")}-{timeframe}.csv')
            df.to_csv(file_path, index=False)
            print(f"Data downloaded and saved to {file_path}")
        else:
            print(f"No data fetched for {symbol} from Binance.")
    except Exception as e:
        print(f"Error fetching data from Binance: {e}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download historical market data.')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='The trading symbol to download data for.')
    parser.add_argument('--timeframe', type=str, default='1h', help='The timeframe to download data for.')
    parser.add_argument('--limit', type=int, default=1000, help='The number of candles to download.')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='The directory to save the downloaded data to.')
    args = parser.parse_args()

    download_data(args.symbol, args.timeframe, args.limit, args.output_dir)

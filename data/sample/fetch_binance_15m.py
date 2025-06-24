import requests
import pandas as pd
import time
import os

BINANCE_URL = 'https://api.binance.com/api/v3/klines'
SYMBOL = 'BTCUSDT'
INTERVAL = '15m'
LIMIT = 1000  # Max per request

# Set your desired start and end timestamps (in milliseconds)
# Example: 2019-01-01 to now
START_DATE = '2019-01-01 00:00:00'
END_DATE = None  # None = fetch up to latest

OUTPUT_CSV = 'BTCUSD_15m_large.csv'

# Convert date string to milliseconds since epoch
def date_to_millis(date_str):
    return int(pd.Timestamp(date_str).timestamp() * 1000)

def fetch_binance_klines(symbol, interval, start_time, end_time=None, limit=1000):
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'limit': limit
    }
    if end_time:
        params['endTime'] = end_time
    resp = requests.get(BINANCE_URL, params=params)
    resp.raise_for_status()
    return resp.json()

def main():
    start_ts = date_to_millis(START_DATE)
    end_ts = date_to_millis(END_DATE) if END_DATE else int(time.time() * 1000)
    all_data = []
    last_ts = start_ts
    print(f"Fetching data from {pd.to_datetime(start_ts, unit='ms')} to {pd.to_datetime(end_ts, unit='ms')}")
    while last_ts < end_ts:
        data = fetch_binance_klines(SYMBOL, INTERVAL, last_ts, end_ts, LIMIT)
        if not data:
            break
        all_data.extend(data)
        last_ts = data[-1][0] + 1  # Next candle
        print(f"Fetched up to {pd.to_datetime(last_ts, unit='ms')}, total rows: {len(all_data)}")
        time.sleep(0.5)  # Avoid rate limits
        if len(data) < LIMIT:
            break
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    # Merge with existing if present
    if os.path.exists(OUTPUT_CSV):
        old = pd.read_csv(OUTPUT_CSV)
        old['timestamp'] = pd.to_datetime(old['timestamp'])
        df = pd.concat([old, df]).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")

if __name__ == '__main__':
    main() 
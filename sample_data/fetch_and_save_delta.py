import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# Initialize Delta Exchange
exchange = ccxt.delta()
symbol = 'BTC/USDT'

# Helper to convert OHLCV to DataFrame
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
def ohlcv_to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Fetch 370 days (~1 year) of data
now = int(time.time() * 1000)
since_days_ago = 370
since_ts = int((datetime.utcnow() - timedelta(days=since_days_ago)).timestamp() * 1000)

# Fetch and save OHLCV for different timeframes
timeframes = {'1d': 'BTCUSD_1d.csv', '4h': 'BTCUSD_4h.csv', '15m': 'BTCUSD_15m.csv'}
for tf, fname in timeframes.items():
    print(f'Fetching {tf} data...')
    all_ohlcv = []
    since = since_ts
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        if len(ohlcv) < 1000:
            break
        since = ohlcv[-1][0] + 1  # next batch
        time.sleep(exchange.rateLimit / 1000)
    df = ohlcv_to_df(all_ohlcv)
    df = df[df['timestamp'] >= pd.to_datetime(datetime.utcnow() - timedelta(days=since_days_ago))]
    df.to_csv(f'sample_data/{fname}', index=False)
    print(f'Saved {fname} ({len(df)} rows)')

# Fetch and save orderbook (top 50 bids/asks)
print('Fetching orderbook...')
orderbook = exchange.fetch_order_book(symbol, limit=50)
orderbook_df = pd.DataFrame({
    'bid_price': [b[0] for b in orderbook['bids']],
    'bid_amount': [b[1] for b in orderbook['bids']],
    'ask_price': [a[0] for a in orderbook['asks']],
    'ask_amount': [a[1] for a in orderbook['asks']],
})
orderbook_df.to_csv('sample_data/orderbook.csv', index=False)
print('Saved orderbook.csv')

# Fetch and save recent trades (last 1000)
print('Fetching trades...')
trades = exchange.fetch_trades(symbol, limit=1000)
trades_df = pd.DataFrame(trades)
trades_df.to_csv('sample_data/trades.csv', index=False)
print('Saved trades.csv')

print('All data fetched and saved to sample_data/.') 
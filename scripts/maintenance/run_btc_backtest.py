import pandas as pd
from ml.backend.src.strategy.backtest import Backtester

def always_buy(data):
    return 'buy'

df = pd.read_csv('sample_data/BTCUSD_1d.csv')
bt = Backtester(df, always_buy, stop_loss_pct=0.02, take_profit_pct=0.04, max_risk=0.1, order_type='standard')
result = bt.run()
print('BTC 1D Backtest Results:')
for k, v in result.items():
    print(f'{k}: {v}') 
import pandas as pd
from typing import List, Dict

class SMCDector:
    def __init__(self, ohlcv: pd.DataFrame):
        self.ohlcv = ohlcv

    def detect_order_blocks(self, lookback=20) -> List[Dict]:
        # Simple logic: find large bullish/bearish candles at swing highs/lows
        order_blocks = []
        for i in range(lookback, len(self.ohlcv) - lookback):
            candle = self.ohlcv.iloc[i]
            prev_high = self.ohlcv['high'].iloc[i-lookback:i].max()
            prev_low = self.ohlcv['low'].iloc[i-lookback:i].min()
            if candle['close'] > candle['open'] and candle['low'] <= prev_low:
                order_blocks.append({'type': 'bullish', 'price': candle['low'], 'timestamp': candle['timestamp']})
            elif candle['close'] < candle['open'] and candle['high'] >= prev_high:
                order_blocks.append({'type': 'bearish', 'price': candle['high'], 'timestamp': candle['timestamp']})
        return order_blocks

    def detect_fvg(self, min_gap=0.002) -> List[Dict]:
        # FVG: gap between previous high and next low (bullish), or previous low and next high (bearish)
        fvg_list = []
        for i in range(1, len(self.ohlcv) - 1):
            prev_high = self.ohlcv['high'].iloc[i-1]
            next_low = self.ohlcv['low'].iloc[i+1]
            if next_low > prev_high and (next_low - prev_high) / prev_high > min_gap:
                fvg_list.append({'type': 'bullish', 'price': (prev_high + next_low) / 2, 'timestamp': self.ohlcv['timestamp'].iloc[i]})
            prev_low = self.ohlcv['low'].iloc[i-1]
            next_high = self.ohlcv['high'].iloc[i+1]
            if prev_low > next_high and (prev_low - next_high) / prev_low > min_gap:
                fvg_list.append({'type': 'bearish', 'price': (prev_low + next_high) / 2, 'timestamp': self.ohlcv['timestamp'].iloc[i]})
        return fvg_list

    def detect_liquidity_zones(self, window=20) -> List[Dict]:
        # Highs/lows with many touches = liquidity zones
        liquidity_zones = []
        highs = self.ohlcv['high'].rolling(window=window).apply(lambda x: (x == x.max()).sum(), raw=True)
        lows = self.ohlcv['low'].rolling(window=window).apply(lambda x: (x == x.min()).sum(), raw=True)
        for i in range(window, len(self.ohlcv)):
            if highs.iloc[i] >= 3:
                liquidity_zones.append({'type': 'high', 'price': self.ohlcv['high'].iloc[i], 'timestamp': self.ohlcv['timestamp'].iloc[i]})
            if lows.iloc[i] >= 3:
                liquidity_zones.append({'type': 'low', 'price': self.ohlcv['low'].iloc[i], 'timestamp': self.ohlcv['timestamp'].iloc[i]})
        return liquidity_zones

    def detect_all(self) -> Dict[str, List[Dict]]:
        return {
            'order_blocks': self.detect_order_blocks(),
            'fvg': self.detect_fvg(),
            'liquidity_zones': self.detect_liquidity_zones()
        }

# Example usage:
# ohlcv = pd.read_csv('BTCUSD_15m.csv')
# smc = SMCDector(ohlcv)
# result = smc.detect_all()
# print(result) 
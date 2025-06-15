import pandas as pd
from typing import List, Dict

class CandlestickPatternDetector:
    def __init__(self, ohlcv: pd.DataFrame):
        self.ohlcv = ohlcv

    def detect_engulfing(self) -> List[Dict]:
        patterns = []
        for i in range(1, len(self.ohlcv)):
            prev = self.ohlcv.iloc[i-1]
            curr = self.ohlcv.iloc[i]
            # Bullish engulfing
            if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']:
                patterns.append({'type': 'bullish_engulfing', 'price': curr['close'], 'timestamp': curr['timestamp']})
            # Bearish engulfing
            if prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']:
                patterns.append({'type': 'bearish_engulfing', 'price': curr['close'], 'timestamp': curr['timestamp']})
        return patterns

    def detect_pin_bar(self, threshold=0.66) -> List[Dict]:
        patterns = []
        for i in range(len(self.ohlcv)):
            candle = self.ohlcv.iloc[i]
            body = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']
            total = candle['high'] - candle['low']
            if total == 0:
                continue
            # Bullish pin bar
            if lower_wick / total > threshold and body / total < (1 - threshold):
                patterns.append({'type': 'bullish_pin_bar', 'price': candle['close'], 'timestamp': candle['timestamp']})
            # Bearish pin bar
            if upper_wick / total > threshold and body / total < (1 - threshold):
                patterns.append({'type': 'bearish_pin_bar', 'price': candle['close'], 'timestamp': candle['timestamp']})
        return patterns

    def detect_inside_bar(self) -> List[Dict]:
        patterns = []
        for i in range(1, len(self.ohlcv)):
            prev = self.ohlcv.iloc[i-1]
            curr = self.ohlcv.iloc[i]
            if curr['high'] < prev['high'] and curr['low'] > prev['low']:
                patterns.append({'type': 'inside_bar', 'price': curr['close'], 'timestamp': curr['timestamp']})
        return patterns

    def detect_all(self) -> List[Dict]:
        return self.detect_engulfing() + self.detect_pin_bar() + self.detect_inside_bar()

# Example usage:
# ohlcv = pd.read_csv('BTCUSD_15m.csv')
# detector = CandlestickPatternDetector(ohlcv)
# patterns = detector.detect_all()
# print(patterns) 
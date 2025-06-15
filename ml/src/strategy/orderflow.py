import pandas as pd
from typing import Dict, Any

class OrderFlowAnalyzer:
    def __init__(self, orderbook: pd.DataFrame, trades: pd.DataFrame):
        self.orderbook = orderbook
        self.trades = trades

    def detect_imbalance(self, threshold=1.5) -> bool:
        # Compare bid/ask volume
        bid_vol = self.orderbook['bid_amount'].sum()
        ask_vol = self.orderbook['ask_amount'].sum()
        if ask_vol == 0:
            return False
        return (bid_vol / ask_vol) > threshold

    def detect_large_trades(self, min_size=10000) -> bool:
        return (self.trades['size'] > min_size).any() if 'size' in self.trades.columns else False

    def detect_spoofing(self, window=10, ratio=3) -> bool:
        # Simple spoofing: large orders appear/disappear quickly
        spoofing = False
        for i in range(window, len(self.orderbook)):
            prev = self.orderbook.iloc[i-window:i]
            curr = self.orderbook.iloc[i]
            if (prev['bid_amount'].max() > ratio * curr['bid_amount']) or (prev['ask_amount'].max() > ratio * curr['ask_amount']):
                spoofing = True
                break
        return spoofing

    def analyze(self) -> Dict[str, Any]:
        return {
            'imbalance': self.detect_imbalance(),
            'large_trades': self.detect_large_trades(),
            'spoofing': self.detect_spoofing()
        }

# Example usage:
# orderbook = pd.read_csv('orderbook.csv')
# trades = pd.read_csv('trades.csv')
# analyzer = OrderFlowAnalyzer(orderbook, trades)
# result = analyzer.analyze()
# print(result) 
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator

class MacroBiasAnalyzer:
    def __init__(self, ohlcv_1d: pd.DataFrame, ohlcv_4h: pd.DataFrame):
        self.ohlcv_1d = ohlcv_1d
        self.ohlcv_4h = ohlcv_4h

    def calculate_moving_averages(self, df, windows=[50, 200]):
        for window in windows:
            df[f'sma_{window}'] = SMAIndicator(close=df['close'], window=window).sma_indicator()
            df[f'ema_{window}'] = EMAIndicator(close=df['close'], window=window).ema_indicator()
        return df

    def detect_trend(self, df, short_window=50, long_window=200):
        if df[f'sma_{short_window}'].iloc[-1] > df[f'sma_{long_window}'].iloc[-1]:
            return 'bullish'
        elif df[f'sma_{short_window}'].iloc[-1] < df[f'sma_{long_window}'].iloc[-1]:
            return 'bearish'
        else:
            return 'neutral'

    def find_swing_highs_lows(self, df, lookback=20):
        highs = df['high'].rolling(window=lookback, min_periods=1).max()
        lows = df['low'].rolling(window=lookback, min_periods=1).min()
        return highs, lows

    def analyze(self):
        # Analyze 1D timeframe
        df_1d = self.calculate_moving_averages(self.ohlcv_1d.copy())
        bias_1d = self.detect_trend(df_1d)
        highs_1d, lows_1d = self.find_swing_highs_lows(df_1d)
        # Analyze 4H timeframe
        df_4h = self.calculate_moving_averages(self.ohlcv_4h.copy())
        bias_4h = self.detect_trend(df_4h)
        highs_4h, lows_4h = self.find_swing_highs_lows(df_4h)
        # Combine signals
        if bias_1d == bias_4h:
            macro_bias = bias_1d
        else:
            macro_bias = 'neutral'
        return {
            'macro_bias': macro_bias,
            '1d_bias': bias_1d,
            '4h_bias': bias_4h,
            '1d_highs': highs_1d,
            '1d_lows': lows_1d,
            '4h_highs': highs_4h,
            '4h_lows': lows_4h
        }

# Example usage:
# ohlcv_1d = pd.read_csv('BTCUSD_1d.csv')
# ohlcv_4h = pd.read_csv('BTCUSD_4h.csv')
# analyzer = MacroBiasAnalyzer(ohlcv_1d, ohlcv_4h)
# result = analyzer.analyze()
# print(result) 
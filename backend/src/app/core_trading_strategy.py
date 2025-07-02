import pandas as pd
import numpy as np
import pandas_ta as ta

# Assuming data_fetcher is a module that can get HTF data
# from . import data_fetcher 

class CoreTradingStrategy:
    """
    Implements the high-probability trading strategy based on the guide.txt specification.
    This strategy focuses on trend-following entries at HTF candle opens within
    extreme zones, confirmed by micro-structure analysis.
    """
    def __init__(self, risk_per_trade=0.01, atr_multiplier_sl=1.5, atr_multiplier_tp=2.5):
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        print("✅ CoreTradingStrategy initialized with ADX/ATR logic.")

    def detect_regime(self, daily_data: pd.DataFrame, adx_threshold=25) -> str:
        """
        Analyzes the daily chart using ADX to determine market regime.
        """
        if daily_data is None or len(daily_data) < 50:
            return "range" # Not enough data
            
        adx = daily_data.ta.adx(length=20)
        if adx is None or adx.empty:
            return "range"
        
        adx_value = adx.iloc[-1]['ADX_20']
        dmp = adx.iloc[-1]['DMP_20']
        dmn = adx.iloc[-1]['DMN_20']

        if adx_value > adx_threshold:
            return "trend_up" if dmp > dmn else "trend_down"
        return "range"

    def get_extreme_zone_details(self, daily_data: pd.DataFrame) -> dict:
        """
        Determines if the current price is in an extreme zone and its percentile rank.
        """
        if len(daily_data) < 20:
            return {'zone': 'middle', 'percentile': 0.5}

        rolling_window = daily_data['close'].rolling(window=20)
        last_close = daily_data['close'].iloc[-1]
        
        # Calculate percentile rank of the last close within the window
        # Using a lambda to handle potential NaNs in the window
        percentile = rolling_window.apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if not x.empty else 0.5, raw=False).iloc[-1]

        if pd.isna(percentile):
             return {'zone': 'middle', 'percentile': 0.5}

        zone = 'middle'
        if percentile >= 0.95:
            zone = 'top'
        elif percentile <= 0.05:
            zone = 'bottom'
            
        return {'zone': zone, 'percentile': percentile}

    def identify_smc_zones(self, h1_data: pd.DataFrame) -> list:
        """
        Detects Order Blocks and Fair Value Gaps (FVG) on the 1H chart.
        """
        zones = []
        if h1_data is None or len(h1_data) < 3:
            return zones

        # Simplified FVG detection
        for i in range(2, len(h1_data)):
            # Bullish FVG: Low of candle i is higher than High of candle i-2
            if h1_data.iloc[i-1]['low'] > h1_data.iloc[i-2]['high']:
                zones.append({'type': 'fvg_bullish', 'top': h1_data.iloc[i-1]['low'], 'bottom': h1_data.iloc[i-2]['high']})
            # Bearish FVG: High of candle i is lower than Low of candle i-2
            if h1_data.iloc[i-1]['high'] < h1_data.iloc[i-2]['low']:
                zones.append({'type': 'fvg_bearish', 'top': h1_data.iloc[i-2]['low'], 'bottom': h1_data.iloc[i-1]['high']})
        
        # Simplified Order Block detection
        # Bullish OB: Last down candle before an up move
        if h1_data.iloc[-2]['close'] < h1_data.iloc[-2]['open'] and h1_data.iloc[-1]['close'] > h1_data.iloc[-1]['open']:
             zones.append({'type': 'order_block_bullish', 'price_level': h1_data.iloc[-2]['low']})
        
        return zones
    
    def verify_micro_structure(self, m15_data: pd.DataFrame, bias: str) -> bool:
        """
        Confirms a 15m "break of structure" (BoS).
        Looks for a recent swing high/low being broken.
        """
        if m15_data is None or len(m15_data) < 10:
            return False

        recent_data = m15_data.tail(10)
        
        if bias == 'bullish':
            swing_high = recent_data['high'].iloc[:-1].max()
            return recent_data.iloc[-1]['close'] > swing_high
        if bias == 'bearish':
            swing_low = recent_data['low'].iloc[:-1].min()
            return recent_data.iloc[-1]['close'] < swing_low
        return False

    def compute_atr(self, m15_data: pd.DataFrame, period=14) -> float:
        """
        Calculates the 14-period ATR on the 15m timeframe.
        """
        if m15_data is None or len(m15_data) < period:
            return 0.0
        atr = m15_data.ta.atr(length=period)
        return atr.iloc[-1]

    def generate_signal(self, data: dict) -> dict:
        """
        The master function to generate a trading signal.
        """
        daily_data = pd.DataFrame(data.get('daily', []))
        h4_data = pd.DataFrame(data.get('h4', []))
        m15_data = pd.DataFrame(data.get('m15', []))

        if daily_data.empty or h4_data.empty or m15_data.empty:
            return {"signal": "hold", "reason": "Missing or empty dataframes"}

        # 1. Determine Market Regime
        regime = self.detect_regime(daily_data)
        if 'trend' not in regime:
            return {"signal": "hold", "confidence": 0, "reason": "Ranging market"}
        
        bias = "bullish" if regime == "trend_up" else "bearish"

        # 2. Check for HTF Extreme Zone
        zone_details = self.get_extreme_zone_details(h4_data)
        extreme_location = zone_details['zone']
        
        # 3. Verify Micro-structure confirmation
        micro_structure_confirmed = self.verify_micro_structure(m15_data, bias)
        
        # 4. Calculate confidence
        confidence = 0
        if bias == 'bullish' and extreme_location == 'bottom':
            confidence = (1 - zone_details['percentile']) * 100
        elif bias == 'bearish' and extreme_location == 'top':
            confidence = zone_details['percentile'] * 100
        
        if not micro_structure_confirmed:
            confidence *= 0.5 # Reduce confidence if no micro-structure confirmation

        # 5. Generate Signal
        if bias == 'bullish' and extreme_location == 'bottom' and micro_structure_confirmed:
            current_price = m15_data['close'].iloc[-1]
            atr = self.compute_atr(m15_data)
            stop_loss = current_price - (atr * self.atr_multiplier_sl)
            target_price = current_price + (atr * self.atr_multiplier_tp)
            return {
                "signal": "buy",
                "confidence": round(confidence, 2),
                "stop_loss": round(stop_loss, 2),
                "target_price": round(target_price, 2),
                "reason": "Bullish trend, price at HTF demand, and 15m bullish structure confirmed."
            }
        elif bias == 'bearish' and extreme_location == 'top' and micro_structure_confirmed:
            current_price = m15_data['close'].iloc[-1]
            atr = self.compute_atr(m15_data)
            stop_loss = current_price + (atr * self.atr_multiplier_sl)
            target_price = current_price - (atr * self.atr_multiplier_tp)
            return {
                "signal": "sell",
                "confidence": round(confidence, 2),
                "stop_loss": round(stop_loss, 2),
                "target_price": round(target_price, 2),
                "reason": "Bearish trend, price at HTF supply, and 15m bearish structure confirmed."
            }
        else:
            # Provide a detailed reason for holding
            hold_reason = f"Conditions not met. Bias: {bias}, Zone: {extreme_location}, Micro-structure: {micro_structure_confirmed}"
            return {
                "signal": "hold",
                "confidence": round(confidence, 2),
                "reason": hold_reason
            }

# Example Usage (for testing)
if __name__ == '__main__':
    def create_dummy_data(rows):
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=rows, freq='H'))
        return pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(42000, 42100, size=rows),
            'high': np.random.uniform(42100, 42200, size=rows),
            'low': np.random.uniform(41900, 42000, size=rows),
            'close': np.random.uniform(42000, 42100, size=rows),
            'volume': np.random.uniform(100, 200, size=rows)
        })

    mock_data = {
        'daily': create_dummy_data(100),
        'h4': create_dummy_data(100),
        'm15': create_dummy_data(100)
    }
    
    strategy = CoreTradingStrategy()
    signal = strategy.generate_signal(mock_data)
    
    import json
    print(json.dumps(signal, indent=2)) 
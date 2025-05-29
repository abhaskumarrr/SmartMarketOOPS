from typing import Dict, Any

class CRTConfluence:
    def __init__(self, macro_bias, smc_signals, candlestick_signals, orderflow_signals):
        self.macro_bias = macro_bias
        self.smc_signals = smc_signals
        self.candlestick_signals = candlestick_signals
        self.orderflow_signals = orderflow_signals

    def compute_confluence(self) -> Dict[str, Any]:
        score = 0
        factors = []
        # Macro bias
        if self.macro_bias['macro_bias'] == 'bullish':
            score += 1
            factors.append('macro_bullish')
        elif self.macro_bias['macro_bias'] == 'bearish':
            score -= 1
            factors.append('macro_bearish')
        # SMC signals
        if self.smc_signals['order_blocks']:
            score += 1
            factors.append('order_block')
        if self.smc_signals['fvg']:
            score += 1
            factors.append('fvg')
        if self.smc_signals['liquidity_zones']:
            score += 1
            factors.append('liquidity_zone')
        # Candlestick
        if self.candlestick_signals:
            score += 1
            factors.append('candlestick')
        # Orderflow
        if self.orderflow_signals.get('imbalance'):
            score += 1
            factors.append('orderflow_imbalance')
        # Action
        action = 'buy' if score >= 3 else 'sell' if score <= -3 else 'wait'
        return {
            'score': score,
            'factors': factors,
            'action': action
        }

# Example usage:
# crt = CRTConfluence(macro_bias, smc_signals, candlestick_signals, orderflow_signals)
# result = crt.compute_confluence()
# print(result) 
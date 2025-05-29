from typing import Dict

class RiskManager:
    def __init__(self, account_balance, max_risk=0.01, max_drawdown=0.2):
        self.account_balance = account_balance
        self.max_risk = max_risk
        self.max_drawdown = max_drawdown
        self.equity_peak = account_balance
        self.equity_trough = account_balance

    def position_size(self, stop_loss_pct, kelly_fraction=0.5) -> float:
        # Fixed fractional or Kelly
        risk_amount = self.account_balance * self.max_risk * kelly_fraction
        return risk_amount / stop_loss_pct if stop_loss_pct > 0 else 0

    def update_equity(self, new_balance):
        self.account_balance = new_balance
        if new_balance > self.equity_peak:
            self.equity_peak = new_balance
        if new_balance < self.equity_trough:
            self.equity_trough = new_balance

    def check_drawdown(self) -> bool:
        drawdown = (self.equity_peak - self.account_balance) / self.equity_peak
        return drawdown > self.max_drawdown

    def recommend(self, stop_loss_pct, take_profit_pct) -> Dict:
        size = self.position_size(stop_loss_pct)
        stop = stop_loss_pct
        take_profit = take_profit_pct
        risk_status = 'halt' if self.check_drawdown() else 'ok'
        return {
            'size': size,
            'stop_loss': stop,
            'take_profit': take_profit,
            'risk_status': risk_status
        }

# Example usage:
# rm = RiskManager(account_balance=10000)
# rec = rm.recommend(stop_loss_pct=0.02, take_profit_pct=0.04)
# print(rec) 
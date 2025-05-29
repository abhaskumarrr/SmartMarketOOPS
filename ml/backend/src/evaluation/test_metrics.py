import numpy as np
import pytest
from evaluation.metrics import sharpe_ratio, sortino_ratio, max_drawdown
import pandas as pd
from strategy.backtest import Backtester, grid_search_optimizer
from strategy.risk_management import RiskManager

def test_sharpe_ratio():
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    expected = np.mean(returns) / (np.std(returns) + 1e-8)
    assert np.isclose(sharpe_ratio(returns), expected)

def test_sortino_ratio():
    returns = np.array([0.01, 0.02, -0.03, 0.04, -0.05])
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    expected = np.mean(returns) / (downside_deviation + 1e-8)
    assert np.isclose(sortino_ratio(returns), expected)

def test_max_drawdown():
    returns = np.array([0.1, -0.2, 0.05, -0.1, 0.2])
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    expected = np.min(drawdown)
    assert np.isclose(max_drawdown(returns), expected)

def test_backtester_risk_management():
    # Create synthetic price data
    data = pd.DataFrame({'close': [100, 102, 98, 104, 96, 110, 90, 120, 80, 130, 70]})
    # Simple strategy: always buy
    def always_buy(df):
        return 'buy'
    # Test with tight stop-loss and take-profit
    bt = Backtester(data, always_buy, stop_loss_pct=0.02, take_profit_pct=0.04, max_risk=0.1)
    results = bt.run()
    # There should be at least one trade
    assert len(results['trades']) > 0
    # Check that stop-loss or take-profit is hit (PnL should not always be the same)
    assert any(abs(t) > 0 for t in results['trades'])
    # Position sizing should be less than or equal to max_risk * balance
    # (We can't check directly, but can check that final_balance is not negative)
    assert results['final_balance'] > 0

def test_grid_search_optimizer():
    # Synthetic data
    data = pd.DataFrame({'close': [100, 102, 98, 104, 96, 110, 90, 120, 80, 130, 70]})
    def always_buy(df):
        return 'buy'
    stop_loss_range = [0.01, 0.02]
    take_profit_range = [0.03, 0.04]
    max_risk_range = [0.05, 0.1]
    result = grid_search_optimizer(
        data,
        always_buy,
        stop_loss_range,
        take_profit_range,
        max_risk_range,
        objective_metric='sharpe',
    )
    assert 'best_params' in result
    assert 'best_result' in result
    assert result['best_result']['sharpe'] == max(r['result']['sharpe'] for r in result['all_results'])

def test_trailing_stop():
    data = pd.DataFrame({'close': [100, 105, 110, 108, 112, 115, 113, 117, 120, 118, 125]})
    def always_buy(df):
        return 'buy'
    bt = Backtester(data, always_buy, stop_loss_pct=0.03, take_profit_pct=0.1, max_risk=0.1, order_type='trailing_stop')
    results = bt.run()
    # Should exit on trailing stop or take-profit
    assert len(results['trades']) > 0
    assert results['final_balance'] > 0

def test_oco_order():
    data = pd.DataFrame({'close': [100, 98, 97, 99, 101, 103, 105, 104, 102, 100, 98]})
    def always_buy(df):
        return 'buy'
    bt = Backtester(data, always_buy, stop_loss_pct=0.02, take_profit_pct=0.04, max_risk=0.1, order_type='oco')
    results = bt.run()
    # Should exit on stop-loss or take-profit, but not both
    assert len(results['trades']) > 0
    assert results['final_balance'] > 0

def test_custom_execution_logic():
    data = pd.DataFrame({'close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120]})
    def always_buy(df):
        return 'buy'
    def custom_exit_next_bar(data, entry_idx, entry_price, stop_loss, take_profit, size, order_type):
        # Always exit at the next bar (or last bar)
        exit_idx = min(entry_idx + 1, len(data) - 1)
        return data['close'].iloc[exit_idx]
    bt = Backtester(data, always_buy, stop_loss_pct=0.02, take_profit_pct=0.04, max_risk=0.1, order_type='standard', execution_logic=custom_exit_next_bar)
    results = bt.run()
    # Should have trades, and each trade should be the difference between consecutive closes
    assert len(results['trades']) > 0
    balance = 10000
    for i, pnl in enumerate(results['trades']):
        if i + 2 < len(data):
            entry_price = data['close'].iloc[i+1]
            exit_price = data['close'].iloc[i+2]
            risk_manager = RiskManager(balance, max_risk=bt.max_risk)
            size = risk_manager.position_size(bt.stop_loss_pct, kelly_fraction=0.5)
            expected_pnl = (exit_price - entry_price) / entry_price * size
            # Use a tolerance of 1e-4 due to floating point arithmetic in financial calculations.
            assert abs(pnl - expected_pnl) < 1e-4, f"Trade {i}: Expected {expected_pnl}, got {pnl}"
            balance += pnl
 
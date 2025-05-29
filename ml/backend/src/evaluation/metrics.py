import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)

def max_drawdown(returns):
    returns = np.asarray(returns)
    if len(returns) == 0 or np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
        return np.nan
    # Prevent -100% returns (which would wipe out the portfolio)
    returns = np.clip(returns, -0.999, None)
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (cumulative - peak) / peak
        drawdown[peak == 0] = 0
    return float(np.min(drawdown))

def sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    return np.mean(excess_returns) / (downside_deviation + 1e-8)

def classification_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    } 
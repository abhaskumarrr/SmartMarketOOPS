import json
import importlib
import pandas as pd
import numpy as np
import os
import sys

# Add the project root's 'ml' directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
ml_root = os.path.abspath(os.path.join(project_root, '..', 'ml'))
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

# A simple, standardized backtest function to run on each strategy
def run_simple_backtest(strategy_logic, data):
    """
    A simplified backtesting engine.
    - strategy_logic: A function that takes a data row and returns 'buy', 'sell', or 'hold'.
    - data: A pandas DataFrame with OHLCV data.
    """
    positions = []
    trades = []
    capital = 10000
    position_size = 0.1 # BTC
    
    for i in range(1, len(data)):
        row = data.iloc[i]
        try:
            # Pass historical data up to current point
            signal = strategy_logic(data.iloc[:i]) 
        except Exception:
            signal = 'hold' # If strategy fails, just hold.
        
        # Simple position management
        if signal == 'buy' and not positions:
            entry_price = row['close']
            positions.append({'entry_price': entry_price, 'type': 'long'})
            trades.append({'entry': entry_price, 'exit': np.nan, 'pnl': np.nan})
        elif signal == 'sell' and positions:
            exit_price = row['close']
            entry_price = positions.pop(0)['entry_price']
            pnl = (exit_price - entry_price) * position_size
            trades[-1].update({'exit': exit_price, 'pnl': pnl})
            capital += pnl
            
    return trades, capital

def calculate_metrics(trades):
    if not trades:
        return { "win_rate": 0, "profit_factor": 0, "max_drawdown": 0, "sharpe_ratio": 0, "trade_count": 0 }

    pnls = [t['pnl'] for t in trades if not np.isnan(t['pnl'])]
    if not pnls:
        return { "win_rate": 0, "profit_factor": 0, "max_drawdown": 0, "sharpe_ratio": 0, "trade_count": 0 }

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    win_rate = len(wins) / len(pnls) if pnls else 0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Simplified Drawdown
    equity_curve = np.cumsum([10000] + pnls)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Simplified Sharpe
    returns = pd.Series(pnls).pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0 # Annualized
    
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "trade_count": len(pnls)
    }

def get_strategy_logic_from_module(module):
    """
    Tries to find a runnable strategy function within a module.
    This part is highly heuristic due to the inconsistent codebase.
    """
    # Look for a function that is likely to be the strategy entry point
    if hasattr(module, 'run_backtest'):
        # This is the ideal case. We assume it returns a function that generates signals.
        # This is a placeholder for a more complex integration if needed.
        # For now, we will prefer a direct signal generation function.
        pass

    if hasattr(module, 'generate_signals'):
        return module.generate_signals
    
    # A common pattern might be a class that can be instantiated and called
    for attr_name in dir(module):
        if 'Strategy' in attr_name or 'System' in attr_name:
             # Check if it's a class
            potential_class = getattr(module, attr_name)
            if isinstance(potential_class, type):
                try:
                    strategy_instance = potential_class()
                    if hasattr(strategy_instance, 'predict'):
                        return strategy_instance.predict
                    if hasattr(strategy_instance, 'generate_signals'):
                        return strategy_instance.generate_signals
                except Exception:
                    continue # Cannot instantiate without params, skip

    # Fallback to a dummy function if no clear entry point is found
    return lambda data: 'hold'


def benchmark_all_strategies(manifest_file, data_file, results_file):
    with open(manifest_file, 'r') as f:
        strategies = json.load(f)
        
    data = pd.read_csv(data_file)
    results = []

    print(f"🚀 Starting benchmark of {len(strategies)} strategies...")

    for i, s in enumerate(strategies):
        print(f"[{i+1}/{len(strategies)}] Testing: {s['name']}...")
        try:
            # Dynamically import the module
            module_path = s['module']
            # The python path is set to 'ml', so we import from 'src...'
            if module_path.startswith('src.'):
                 module = importlib.import_module(module_path)
            else:
                 # Fallback for unexpected module path structures
                 fixed_module_path = 'src.' + module_path.split('src.')[-1]
                 module = importlib.import_module(fixed_module_path)

            
            # Heuristically find the strategy logic function
            strategy_logic = get_strategy_logic_from_module(module)
            
            if strategy_logic.__code__.co_code == (lambda data: 'hold').__code__.co_code:
                 print(f"  ⚠️ Could not find a clear strategy function in {s['name']}. Skipping.")
                 metrics = {}
            else:
                # Run the simplified backtest
                trades, final_capital = run_simple_backtest(strategy_logic, data)
                
                # Calculate performance metrics
                metrics = calculate_metrics(trades)

            result_entry = {
                "strategy_name": s['name'],
                **metrics
            }
            results.append(result_entry)
            print(f"  ✅ Complete. Trades: {metrics.get('trade_count', 0)}, P/F: {metrics.get('profit_factor', 0):.2f}")

        except Exception as e:
            print(f"  ❌ FAILED to test {s['name']}. Reason: {e}")
            results.append({
                "strategy_name": s['name'], "error": str(e)
            })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    print(f"\n🎉 Benchmark complete! Results saved to {results_file}")


if __name__ == "__main__":
    manifest = 'strategy_manifest.json'
    data_path = 'data/sample/BTC-USDT-1h.csv'
    results_path = 'data/strategy_benchmark.csv'
    benchmark_all_strategies(manifest, data_path, results_path) 
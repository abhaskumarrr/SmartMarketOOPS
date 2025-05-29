import pandas as pd
import matplotlib.pyplot as plt
from .macro_bias import MacroBiasAnalyzer
from .smc_detection import SMCDector
from .candlestick_patterns import CandlestickPatternDetector
from .crt_confluence import CRTConfluence
from .orderflow import OrderFlowAnalyzer
from .risk_management import RiskManager
from .backtest import Backtester
import torch
from ml.backend.src.models.cnn_lstm import CNNLSTMModel
import numpy as np
import mplfinance as mpf
import ta
import os
import json

def main(symbol: str = 'BTCUSD'):
    # Load sample data (replace with your data source)
    ohlcv_1d = pd.read_csv('sample_data/BTCUSD_1d.csv')
    ohlcv_4h = pd.read_csv('sample_data/BTCUSD_4h.csv')
    ohlcv_15m = pd.read_csv('sample_data/BTCUSD_15m.csv')
    orderbook = pd.read_csv('sample_data/orderbook.csv')
    trades = pd.read_csv('sample_data/trades.csv')

    # Macro Bias
    macro = MacroBiasAnalyzer(ohlcv_1d, ohlcv_4h)
    macro_result = macro.analyze()

    # SMC Detection
    smc = SMCDector(ohlcv_15m)
    smc_result = smc.detect_all()

    # Candlestick Patterns
    candle = CandlestickPatternDetector(ohlcv_15m)
    candle_result = candle.detect_all()

    # Order Flow
    orderflow = OrderFlowAnalyzer(orderbook, trades)
    orderflow_result = orderflow.analyze()

    # CRT Confluence
    crt = CRTConfluence(macro_result, smc_result, candle_result, orderflow_result)
    crt_result = crt.compute_confluence()

    # Risk Management
    risk = RiskManager(account_balance=10000)
    risk_result = risk.recommend(stop_loss_pct=0.02, take_profit_pct=0.04)

    # Load data
    large_csv = 'sample_data/BTCUSD_15m_large.csv'
    default_csv = 'sample_data/BTCUSD_15m.csv'
    if os.path.exists(large_csv):
        print(f"[INFO] Using large dataset: {large_csv}")
        ohlcv_15m = pd.read_csv(large_csv)
    else:
        print(f"[INFO] Using default dataset: {default_csv}")
        ohlcv_15m = pd.read_csv(default_csv)
    # Use the same feature engineering as in TradingDataset
    ohlcv_15m['sma_50'] = ta.trend.sma_indicator(ohlcv_15m['close'], window=50)
    ohlcv_15m['sma_200'] = ta.trend.sma_indicator(ohlcv_15m['close'], window=200)
    ohlcv_15m['ema_20'] = ta.trend.ema_indicator(ohlcv_15m['close'], window=20)
    ohlcv_15m['rsi_14'] = ta.momentum.rsi(ohlcv_15m['close'], window=14)
    ohlcv_15m['macd'] = ta.trend.macd(ohlcv_15m['close'])
    ohlcv_15m['macd_signal'] = ta.trend.macd_signal(ohlcv_15m['close'])
    ohlcv_15m['obv'] = ta.volume.on_balance_volume(ohlcv_15m['close'], ohlcv_15m['volume'])
    ohlcv_15m['stoch_k'] = ta.momentum.stoch(ohlcv_15m['high'], ohlcv_15m['low'], ohlcv_15m['close'])
    ohlcv_15m['stoch_d'] = ta.momentum.stoch_signal(ohlcv_15m['high'], ohlcv_15m['low'], ohlcv_15m['close'])
    ohlcv_15m['log_close'] = np.log(ohlcv_15m['close'])
    ohlcv_15m['log_return_1'] = ohlcv_15m['log_close'].diff(1)
    ohlcv_15m['log_return_2'] = ohlcv_15m['log_close'].diff(2)
    ohlcv_15m['log_return_3'] = ohlcv_15m['log_close'].diff(3)
    ohlcv_15m['next_log_close'] = ohlcv_15m['log_close'].shift(-1)
    ohlcv_15m['target_class'] = (ohlcv_15m['next_log_close'] > ohlcv_15m['log_close']).astype(int)
    feature_cols = [c for c in ohlcv_15m.columns if c not in ['timestamp', 'next_log_close', 'target_log_return', 'target_class']]
    # Diagnostics: missing values before imputation
    print("[DEBUG] Missing values before imputation:")
    print(ohlcv_15m[feature_cols + ['target_class']].isnull().sum())
    # Forward-fill, then back-fill, then drop remaining NaNs
    ohlcv_15m[feature_cols] = ohlcv_15m[feature_cols].fillna(method='ffill').fillna(method='bfill')
    ohlcv_15m['target_class'] = ohlcv_15m['target_class'].fillna(method='ffill').fillna(method='bfill')
    # Diagnostics: missing values after imputation
    print("[DEBUG] Missing values after imputation:")
    print(ohlcv_15m[feature_cols + ['target_class']].isnull().sum())
    before_drop = len(ohlcv_15m)
    ohlcv_15m = ohlcv_15m.dropna(subset=feature_cols + ['target_class']).reset_index(drop=True)
    after_drop = len(ohlcv_15m)
    print(f"[DEBUG] Feature engineering: {before_drop} rows before NaN drop, {after_drop} after, {before_drop - after_drop} dropped.")
    # Normalize using training min/max
    norm_params_file = 'ml/backend/src/data/norm_params.npy'
    if os.path.exists(norm_params_file):
        norm = np.load(norm_params_file, allow_pickle=True).item()
        min_vals = norm['min']
        max_vals = norm['max']
        ohlcv_15m[feature_cols] = (ohlcv_15m[feature_cols] - min_vals) / (max_vals - min_vals + 1e-8)
    # Model input size
    input_size = len(feature_cols)
    model_path = 'models/cnnlstm_trained.pt'
    # Load model checkpoint (weights + hyperparams)
    checkpoint = torch.load(model_path, map_location='cpu')
    best_params = checkpoint.get('hyperparams', {'cnn_channels': 32, 'lstm_hidden': 64, 'lstm_layers': 2, 'dropout': 0.2})
    print(f"[INFO] Loaded model hyperparameters from checkpoint: {best_params}")
    model = CNNLSTMModel(input_size=input_size, cnn_channels=best_params['cnn_channels'], lstm_hidden=best_params['lstm_hidden'], lstm_layers=best_params['lstm_layers'], dropout=best_params['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load normalization params
    norm_params_file = 'ml/backend/src/data/norm_params.npy'
    if os.path.exists(norm_params_file):
        norm = np.load(norm_params_file, allow_pickle=True).item()
        min_vals = norm['min']
        max_vals = norm['max']
    else:
        min_vals = None
        max_vals = None

    seq_len = 32
    def model_strategy(df):
        if not hasattr(model_strategy, 'call_count'):
            model_strategy.call_count = 0
            model_strategy.wait_count = 0
            model_strategy.buy_count = 0
        model_strategy.call_count += 1
        if len(df) < seq_len:
            print(f"[DEBUG] len(df)={len(df)} < seq_len={seq_len}: returning 'wait'")
            model_strategy.wait_count += 1
            return 'wait'
        window = df[feature_cols].values[-seq_len:]
        if np.isnan(window).any():
            print(f"[DEBUG] NaN detected in model input window, skipping prediction.")
            model_strategy.wait_count += 1
            return 'wait'
        # Normalize
        if min_vals is not None and max_vals is not None:
            window = (window - min_vals) / (max_vals - min_vals + 1e-8)
        window = np.expand_dims(window, axis=0)  # (1, seq_len, input_size)
        window_tensor = torch.tensor(window, dtype=torch.float32)
        with torch.no_grad():
            logits = model(window_tensor).item()
            prob = 1 / (1 + np.exp(-logits))  # Sigmoid
        action = 'buy' if prob > 0.5 else 'wait'
        if action == 'buy':
            model_strategy.buy_count += 1
        else:
            model_strategy.wait_count += 1
        print(f"[DEBUG] logits={logits}, prob={prob:.4f}, action={action}")
        return action

    def sma_crossover_strategy(df):
        if len(df) < 50:
            print(f"[DEBUG][SMA] len(df)={len(df)} < 50: returning 'wait'")
            return 'wait'
        sma10 = df['close'].rolling(window=10).mean().iloc[-1]
        sma50 = df['close'].rolling(window=50).mean().iloc[-1]
        action = 'buy' if sma10 > sma50 else 'wait'
        print(f"[DEBUG][SMA] Action: {action} (sma10={sma10}, sma50={sma50})")
        return action

    def always_buy_strategy(df):
        # Always returns 'buy' for every bar after the warmup period
        return 'buy'

    backtester = Backtester(ohlcv_15m, model_strategy)
    backtest_result = backtester.run()

    # Run baseline strategy
    print("\n[INFO] Running baseline always-buy strategy...")
    baseline_backtester = Backtester(ohlcv_15m, always_buy_strategy)
    baseline_results = baseline_backtester.run()
    with open('runs/cnnlstm/backtest_baseline.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    print("[INFO] Baseline strategy results saved to runs/cnnlstm/backtest_baseline.json")

    # Compare model vs. baseline
    try:
        with open('runs/cnnlstm/backtest_BTCUSD.json') as f:
            model_results = json.load(f)
        with open('runs/cnnlstm/backtest_baseline.json') as f:
            baseline_results = json.load(f)
        with open('runs/cnnlstm/backtest_comparison.md', 'w') as f:
            f.write("# Backtest Comparison: Model vs. Baseline\n\n")
            f.write(f"**Model**: {model_results['metrics']}\n\n")
            f.write(f"**Baseline**: {baseline_results['metrics']}\n\n")
            f.write("---\n")
            for k in model_results['metrics']:
                f.write(f"- {k}: Model={model_results['metrics'][k]}, Baseline={baseline_results['metrics'].get(k, 'N/A')}\n")
        print("[INFO] Comparison report saved to runs/cnnlstm/backtest_comparison.md")
    except Exception as e:
        print(f"[WARNING] Could not generate comparison report: {e}")

    # Debug: log class distribution
    print(f"[DEBUG] model_strategy called {model_strategy.call_count} times: {model_strategy.buy_count} 'buy', {model_strategy.wait_count} 'wait' (classification, threshold=0.5)")
    print(f"[DEBUG] Class distribution in data: {ohlcv_15m['target_class'].value_counts().to_dict()}")

    # Generate candlestick chart with SMA 50 overlay
    ohlcv_1d_mpf = ohlcv_1d.copy()
    ohlcv_1d_mpf = ohlcv_1d_mpf.rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    ohlcv_1d_mpf['Date'] = pd.to_datetime(ohlcv_1d_mpf['Date'])
    ohlcv_1d_mpf.set_index('Date', inplace=True)
    mpf.plot(ohlcv_1d_mpf, type='candle', style='charles', title=f'{symbol} 1D Candlestick', ylabel='Price', volume=True, mav=(50), savefig='candlestick_chart.png')

    # Optional: Plot extracted feature (e.g., 50-period SMA on 1D)
    plt.figure(figsize=(12,6))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['close'], label='Close Price')
    plt.plot(macro.calculate_moving_averages(ohlcv_1d.copy())['sma_50'], label='SMA 50')
    plt.title(f'{symbol} 1D Close Price and 50-period SMA')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_verification_chart.png')
    plt.close()

    # --- Feature Engineering ---
    features = {}
    # SMA 50, SMA 200
    ohlcv_1d['sma_50'] = ta.trend.sma_indicator(ohlcv_1d['close'], window=50)
    ohlcv_1d['sma_200'] = ta.trend.sma_indicator(ohlcv_1d['close'], window=200)
    # EMA 20
    ohlcv_1d['ema_20'] = ta.trend.ema_indicator(ohlcv_1d['close'], window=20)
    # RSI
    ohlcv_1d['rsi'] = ta.momentum.rsi(ohlcv_1d['close'], window=14)
    # MACD
    ohlcv_1d['macd'] = ta.trend.macd(ohlcv_1d['close'])
    ohlcv_1d['macd_signal'] = ta.trend.macd_signal(ohlcv_1d['close'])
    # OBV
    ohlcv_1d['obv'] = ta.volume.on_balance_volume(ohlcv_1d['close'], ohlcv_1d['volume'])
    # Stochastic Oscillator
    ohlcv_1d['stoch_k'] = ta.momentum.stoch(ohlcv_1d['high'], ohlcv_1d['low'], ohlcv_1d['close'])
    ohlcv_1d['stoch_d'] = ta.momentum.stoch_signal(ohlcv_1d['high'], ohlcv_1d['low'], ohlcv_1d['close'])
    features['sma_50'] = ohlcv_1d['sma_50']
    features['sma_200'] = ohlcv_1d['sma_200']
    features['ema_20'] = ohlcv_1d['ema_20']
    features['rsi'] = ohlcv_1d['rsi']
    features['macd'] = ohlcv_1d['macd']
    features['macd_signal'] = ohlcv_1d['macd_signal']
    features['obv'] = ohlcv_1d['obv']
    features['stoch_k'] = ohlcv_1d['stoch_k']
    features['stoch_d'] = ohlcv_1d['stoch_d']

    # --- Plot all features ---
    # SMA/EMA
    plt.figure(figsize=(12,6))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['close'], label='Close')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['sma_50'], label='SMA 50')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['sma_200'], label='SMA 200')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['ema_20'], label='EMA 20')
    plt.title(f'{symbol} 1D Close, SMA, EMA')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_sma_ema.png')
    plt.close()
    # RSI
    plt.figure(figsize=(12,4))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['rsi'], label='RSI')
    plt.axhline(70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(30, color='g', linestyle='--', alpha=0.5)
    plt.title(f'{symbol} 1D RSI')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_rsi.png')
    plt.close()
    # MACD
    plt.figure(figsize=(12,4))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['macd'], label='MACD')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['macd_signal'], label='MACD Signal')
    plt.title(f'{symbol} 1D MACD')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_macd.png')
    plt.close()
    # OBV
    plt.figure(figsize=(12,4))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['obv'], label='OBV')
    plt.title(f'{symbol} 1D OBV')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_obv.png')
    plt.close()
    # Stochastic Oscillator
    plt.figure(figsize=(12,4))
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['stoch_k'], label='%K')
    plt.plot(ohlcv_1d['timestamp'], ohlcv_1d['stoch_d'], label='%D')
    plt.axhline(80, color='r', linestyle='--', alpha=0.5)
    plt.axhline(20, color='g', linestyle='--', alpha=0.5)
    plt.title(f'{symbol} 1D Stochastic Oscillator')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_stoch.png')
    plt.close()

    return {
        'symbol': symbol,
        'macro_bias': macro_result,
        'smc': smc_result,
        'candlestick_patterns': candle_result,
        'orderflow': orderflow_result,
        'confluence': crt_result,
        'risk': risk_result,
        'backtest': backtest_result,
        'feature_chart': 'feature_verification_chart.png',
        'candlestick_chart': 'candlestick_chart.png',
        'features': features,
        'feature_charts': [
            'feature_sma_ema.png',
            'feature_rsi.png',
            'feature_macd.png',
            'feature_obv.png',
            'feature_stoch.png',
        ],
    }

if __name__ == "__main__":
    result = main('BTCUSD')
    for k, v in result.items():
        print(f"{k}: {v}")
    print("Feature chart saved as feature_verification_chart.png") 
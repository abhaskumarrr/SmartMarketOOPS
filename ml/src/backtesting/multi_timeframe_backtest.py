import pandas as pd
import numpy as np
import torch
import os
import sys
from collections import Counter

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the model definition and feature builder
from ml.src.training.train_hybrid_model import HybridModel
from ml.src.features.build_features import build_features

class MultiTimeframeBacktester:
    def __init__(self, models_dir, data_dir, timeframes):
        self.models = self._load_models(models_dir, timeframes)
        self.data = self._load_data(data_dir, timeframes)
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.trade_history = []

    def _load_models(self, models_dir, timeframes):
        models = {}
        model_path = os.path.join(models_dir, 'hybrid_model.pth')
        if os.path.exists(model_path):
            # Input_dim is 10 based on the features in build_features.py
            model = HybridModel(input_dim=10)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            # We use the same model for all timeframes in this backtest
            for tf in timeframes:
                models[tf] = model
            print(f"Loaded hybrid model for all timeframes")
        else:
            print(f"Warning: Hybrid model not found at {model_path}")
        return models

    def _load_data(self, data_dir, timeframes):
        data = {}
        for tf in timeframes:
            file_path = os.path.join(data_dir, f"BTC_USDT_{tf}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['price_change'] = df['close'].pct_change()
                df['volume_change'] = df['volume'].pct_change()
                df = df.dropna()
                data[tf] = df.set_index('timestamp')
        return data

    def get_decision(self, timestamp):
        """Gets a trading decision based on a majority vote from all models."""
        predictions = []
        for timeframe, model in self.models.items():
            df = self.data[timeframe]
            try:
                # Find the most recent data point for the given timestamp
                data_point = df.loc[:timestamp].iloc[-1]
                
                # Use the same features as in the training script
                features_df = build_features(df.loc[:timestamp])
                latest_features = features_df.iloc[-1]
                model_features = latest_features.drop(['regime', 'bias']).values.astype(np.float32)
                
                features_tensor = torch.FloatTensor(np.array(model_features).reshape(1, -1))
                
                with torch.no_grad():
                    output = model(features_tensor)
                    decision = torch.argmax(output, dim=1).item()
                    predictions.append(decision)
            except IndexError:
                # Not enough data for this timeframe yet
                continue
        
        if not predictions:
            return 0 # Hold if no models have an opinion

        # Majority vote
        vote_counts = Counter(predictions)
        most_common = vote_counts.most_common(1)
        return most_common[0][0] if most_common else 0

    def run(self):
        """Runs the backtest."""
        # We'll iterate through the 1-hour timeframe as our main clock
        main_timeframe = '1h'
        if main_timeframe not in self.data:
            print(f"Error: Main timeframe data ('{main_timeframe}') not found.")
            return

        for index, row in self.data[main_timeframe].iterrows():
            decision = self.get_decision(index)
            current_price = row['close']

            # Execute trade based on decision
            if self.position == 0: # If no open position
                if decision == 1: # Buy
                    self.position = 1
                    self.entry_price = current_price
                elif decision == 2: # Sell
                    self.position = -1
                    self.entry_price = current_price
            elif self.position == 1: # If long
                if decision == 2: # Close long and open short
                    pnl = current_price - self.entry_price
                    self.balance += pnl
                    self.trade_history.append(pnl)
                    self.position = -1
                    self.entry_price = current_price
            elif self.position == -1: # If short
                if decision == 1: # Close short and open long
                    pnl = self.entry_price - current_price
                    self.balance += pnl
                    self.trade_history.append(pnl)
                    self.position = 1
                    self.entry_price = current_price
        
        self.print_report()

    def print_report(self):
        """Prints a summary of the backtest results."""
        print("\n--- Multi-Timeframe Backtest Report ---")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance:   ${self.balance:.2f}")
        
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        print(f"Total Return:    {total_return:.2f}%")
        
        num_trades = len(self.trade_history)
        if num_trades > 0:
            wins = sum(1 for pnl in self.trade_history if pnl > 0)
            losses = num_trades - wins
            win_rate = (wins / num_trades) * 100
            avg_win = sum(pnl for pnl in self.trade_history if pnl > 0) / wins if wins > 0 else 0
            avg_loss = sum(pnl for pnl in self.trade_history if pnl < 0) / losses if losses > 0 else 0
            
            print(f"Total Trades:    {num_trades}")
            print(f"Win Rate:        {win_rate:.2f}%")
            print(f"Average Win:     ${avg_win:.2f}")
            print(f"Average Loss:    ${avg_loss:.2f}")
        else:
            print("No trades were executed.")
        print("-----------------------------------------")


if __name__ == '__main__':
    models_dir = os.path.join(os.path.dirname(__file__), '../../models')
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data/raw')
    timeframes = ['15m', '1h', '4h', '1d', '1w', '1M']
    
    backtester = MultiTimeframeBacktester(models_dir, data_dir, timeframes)
    backtester.run()

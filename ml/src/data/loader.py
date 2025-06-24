import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import os
import numpy as np
import ta

# Define a small threshold for determining 'Neutral' movement
LOG_RETURN_THRESHOLD = 0.0005 # Example threshold, may need tuning

class TradingDataset(Dataset):
    def __init__(self, csv_file, schema_file, seq_len=32, transform=None, norm_params_file=None, save_norm_params=False):
        self.data = pd.read_csv(csv_file)
        with open(schema_file, 'r') as f:
            self.schema = yaml.safe_load(f)
        self.transform = transform
        self.seq_len = seq_len
        self._validate_schema()
        # --- Add technical indicators as features ---
        if 'sma_50' not in self.data.columns:
            self.data['sma_50'] = ta.trend.sma_indicator(self.data['close'], window=50)
        if 'sma_200' not in self.data.columns:
            self.data['sma_200'] = ta.trend.sma_indicator(self.data['close'], window=200)
        if 'ema_20' not in self.data.columns:
            self.data['ema_20'] = ta.trend.ema_indicator(self.data['close'], window=20)
        if 'rsi_14' not in self.data.columns:
            self.data['rsi_14'] = ta.momentum.rsi(self.data['close'], window=14)
        if 'macd' not in self.data.columns:
            self.data['macd'] = ta.trend.macd(self.data['close'])
        if 'macd_signal' not in self.data.columns:
            self.data['macd_signal'] = ta.trend.macd_signal(self.data['close'])
        if 'obv' not in self.data.columns:
            self.data['obv'] = ta.volume.on_balance_volume(self.data['close'], self.data['volume'])
        if 'stoch_k' not in self.data.columns:
            self.data['stoch_k'] = ta.momentum.stoch(self.data['high'], self.data['low'], self.data['close'])
        if 'stoch_d' not in self.data.columns:
            self.data['stoch_d'] = ta.momentum.stoch_signal(self.data['high'], self.data['low'], self.data['close'])
        # --- Add lagged returns ---
        self.data['log_close'] = np.log(self.data['close'])
        self.data['log_return_1'] = self.data['log_close'].diff(1)
        self.data['log_return_2'] = self.data['log_close'].diff(2)
        self.data['log_return_3'] = self.data['log_close'].diff(3)
        # --- Target: next log-return ---
        self.data['next_log_close'] = self.data['log_close'].shift(-1)
        self.data['target_log_return'] = self.data['next_log_close'] - self.data['log_close']
        # --- Convert target to categorical direction ---
        self.data['target_direction'] = self.data['target_log_return'].apply(lambda x: 0 if x < -LOG_RETURN_THRESHOLD else (2 if x > LOG_RETURN_THRESHOLD else 1)) # 0: Down, 1: Neutral, 2: Up
        # --- End feature engineering ---
        # Print target class distribution
        print("\nTarget Class Distribution:")
        print(self.data['target_direction'].value_counts())
        print("---")
        self.feature_cols = [c for c in self.data.columns if c not in ['timestamp', 'next_log_close', 'target_log_return', 'target_direction']]
        print(f"TradingDataset Feature Columns: {self.feature_cols}")
        # Drop rows with NaN in any feature or target
        self.data = self.data.dropna(subset=self.feature_cols + ['target_direction']).reset_index(drop=True)
        # Normalization
        if norm_params_file and os.path.exists(norm_params_file):
            norm = np.load(norm_params_file, allow_pickle=True).item()
            self.min = norm['min']
            self.max = norm['max']
        else:
            self.min = self.data[self.feature_cols].min().values
            self.max = self.data[self.feature_cols].max().values
            if save_norm_params and norm_params_file:
                np.save(norm_params_file, {'min': self.min, 'max': self.max})
        self.close_idx = self.data.columns.get_loc('close') - (1 if 'timestamp' in self.data.columns else 0)

    def _validate_schema(self):
        required_cols = [k for k in self.schema.keys() if k != 'features']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start = max(0, idx - self.seq_len + 1)
        window = self.data.iloc[start:idx+1][self.feature_cols].values.astype(float)
        if window.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - window.shape[0], window.shape[1]))
            window = np.vstack([pad, window])
        window = (window - self.min) / (self.max - self.min + 1e-8)
        if self.transform:
            window = self.transform(window)
        target_direction = self.data.iloc[idx]['target_direction'] # Get categorical target
        return torch.tensor(window, dtype=torch.float32), torch.tensor(target_direction, dtype=torch.long) # Return as LongTensor for classification

def get_dataloader(csv_file, schema_file, batch_size=64, shuffle=True, seq_len=32, transform=None, norm_params_file=None, save_norm_params=False, num_workers=0):
    dataset = TradingDataset(csv_file, schema_file, seq_len=seq_len, transform=transform, norm_params_file=norm_params_file, save_norm_params=save_norm_params)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) 
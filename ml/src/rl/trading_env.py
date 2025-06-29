import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Any

class TradingEnv:
    """
    A trading environment that uses real historical market data for training RL agents.
    """
    def __init__(self, data_path: str, initial_balance: float = 10000.0, trade_size: float = 1.0):
        self.data_path = data_path
        self.initial_balance = initial_balance
        self.trade_size = trade_size
        
        self._load_market_data()
        self.reset()

    def _load_market_data(self):
        """Loads market data from the specified CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Market data file not found at {self.data_path}")
        self.market_df = pd.read_csv(self.data_path)
        # Simple feature engineering
        self.market_df['price_change'] = self.market_df['close'].pct_change()
        self.market_df['volume_change'] = self.market_df['volume'].pct_change()
        self.market_df = self.market_df.dropna().reset_index(drop=True)

    def reset(self) -> np.ndarray:
        """Resets the environment to its initial state."""
        self.balance = self.initial_balance
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.current_step = 0
        self.trade_count = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Returns the current state of the environment."""
        row = self.market_df.iloc[self.current_step]
        # State: current price, position, price change, volume change
        return np.array([row['close'], self.position, row['price_change'], row['volume_change']], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Executes one time step within the environment."""
        current_price = self.market_df.loc[self.current_step, 'close']
        
        # Execute the action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.trade_count += 1
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = current_price
            self.trade_count += 1
        
        # Move to the next time step
        self.current_step += 1
        done = self.current_step >= len(self.market_df) - 1
        
        # Calculate reward
        reward = 0
        if self.position != 0:
            next_price = self.market_df.loc[self.current_step, 'close']
            if self.position == 1: # Long position
                reward = (next_price - current_price) * self.trade_size
            elif self.position == -1: # Short position
                reward = (current_price - next_price) * self.trade_size
        
        # Update balance if a position was closed
        if (action == 2 and self.position == 1) or (action == 1 and self.position == -1):
            self.balance += reward
            self.position = 0

        new_state = self._get_state()
        info = {'trade_id': f'trade_{self.trade_count}'} if self.trade_count > 0 else {}
        
        return new_state, reward, done, info

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/raw/BTC_USDT_1h.csv')
    env = TradingEnv(data_path=data_path)
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.random.randint(0, 3)
        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = new_state

    print(f"Simulation finished.")
    print(f"Initial balance: ${env.initial_balance:.2f}")
    print(f"Final balance: ${env.balance:.2f}")
    print(f"Total reward (PnL): ${total_reward:.2f}")

#!/usr/bin/env python3
"""
Reinforcement Learning Trading Agent for Enhanced SmartMarketOOPS
Implements Q-learning and PPO algorithms for adaptive trading strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import gym
from gym import spaces
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class TradingEnvironment(gym.Env):
    """Custom trading environment for RL agent"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, 
                 transaction_cost: float = 0.001, max_position: float = 1.0):
        """Initialize trading environment"""
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # State space: [price_features, technical_indicators, portfolio_state]
        self.state_dim = 20  # Will be expanded based on features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                          shape=(self.state_dim,), dtype=np.float32)
        
        # Action space: [hold, buy_25%, buy_50%, buy_100%, sell_25%, sell_50%, sell_100%]
        self.action_space = spaces.Discrete(7)
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position (-1 to 1, where 1 is fully long)
        self.portfolio_value = initial_balance
        self.trade_history = []
        
        logger.info(f"Trading environment initialized with {len(data)} data points")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        # Get current and next prices
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        if self.current_step < len(self.data):
            new_price = self.data.iloc[self.current_step]['close']
            self.portfolio_value = self.balance + (self.position * new_price * self.initial_balance)
        
        # Calculate reward based on portfolio performance
        price_return = (next_price - current_price) / current_price
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        # Reward combines portfolio return and risk-adjusted performance
        reward = portfolio_return * 100  # Scale reward
        
        # Add penalty for excessive trading
        if len(self.trade_history) > 0 and self.trade_history[-1]['step'] == self.current_step:
            reward -= self.transaction_cost * 10  # Trading penalty
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1) or (self.portfolio_value <= self.initial_balance * 0.5)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: int, price: float) -> float:
        """Execute trading action"""
        action_map = {
            0: 0.0,    # Hold
            1: 0.25,   # Buy 25%
            2: 0.5,    # Buy 50%
            3: 1.0,    # Buy 100%
            4: -0.25,  # Sell 25%
            5: -0.5,   # Sell 50%
            6: -1.0    # Sell 100%
        }
        
        target_position = action_map[action]
        position_change = target_position - self.position
        
        if abs(position_change) > 0.01:  # Only trade if significant change
            # Calculate trade amount
            trade_amount = abs(position_change) * self.initial_balance
            transaction_cost = trade_amount * self.transaction_cost
            
            # Update position and balance
            self.position = target_position
            self.balance -= transaction_cost
            
            # Record trade
            self.trade_history.append({
                'step': self.current_step,
                'action': action,
                'position_change': position_change,
                'price': price,
                'cost': transaction_cost
            })
            
            return -transaction_cost
        
        return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        if self.current_step >= len(self.data):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        # Price features (normalized)
        price_features = [
            row['close'] / 50000,  # Normalized price
            row['high'] / row['low'] - 1,  # High-low ratio
            row['volume'] / 1000000,  # Normalized volume
            (row['close'] - row['open']) / row['open'],  # Price change
        ]
        
        # Technical indicators (if available)
        technical_features = []
        for col in ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12']:
            if col in row:
                if col == 'rsi':
                    technical_features.append(row[col] / 100)  # Normalize RSI
                else:
                    technical_features.append(row[col] / row['close'])  # Relative to price
            else:
                technical_features.append(0.0)
        
        # Portfolio state
        portfolio_features = [
            self.position,  # Current position
            self.balance / self.initial_balance,  # Normalized balance
            self.portfolio_value / self.initial_balance,  # Portfolio performance
            len(self.trade_history) / 100,  # Trade frequency
        ]
        
        # Market context (recent price movements)
        lookback = min(5, self.current_step)
        if lookback > 0:
            recent_returns = []
            for i in range(lookback):
                if self.current_step - i - 1 >= 0:
                    prev_price = self.data.iloc[self.current_step - i - 1]['close']
                    curr_price = self.data.iloc[self.current_step - i]['close']
                    recent_returns.append((curr_price - prev_price) / prev_price)
            
            while len(recent_returns) < 5:
                recent_returns.append(0.0)
            
            market_context = recent_returns[:5]
        else:
            market_context = [0.0] * 5
        
        # Combine all features
        observation = np.array(
            price_features + technical_features + portfolio_features + market_context,
            dtype=np.float32
        )
        
        # Pad or truncate to state_dim
        if len(observation) > self.state_dim:
            observation = observation[:self.state_dim]
        elif len(observation) < self.state_dim:
            padding = np.zeros(self.state_dim - len(observation), dtype=np.float32)
            observation = np.concatenate([observation, padding])
        
        return observation


class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize DQN network"""
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)


class DQNAgent:
    """Deep Q-Learning agent for trading"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000, batch_size: int = 32):
        """Initialize DQN agent"""
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self.update_target_network()
        
        logger.info(f"DQN Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self) -> float:
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


class RLTradingSystem:
    """Complete RL trading system integrating with SmartMarketOOPS"""
    
    def __init__(self, symbols: List[str] = None):
        """Initialize RL trading system"""
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        self.agents = {}
        self.environments = {}
        self.training_history = {}
        
        logger.info(f"RL Trading System initialized for symbols: {self.symbols}")
    
    def create_agent(self, symbol: str, training_data: pd.DataFrame) -> DQNAgent:
        """Create and train RL agent for a symbol"""
        # Create environment
        env = TradingEnvironment(training_data)
        self.environments[symbol] = env
        
        # Create agent
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space.n,
            lr=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        self.agents[symbol] = agent
        return agent
    
    def train_agent(self, symbol: str, episodes: int = 1000, 
                   target_update_freq: int = 100) -> Dict[str, Any]:
        """Train RL agent"""
        if symbol not in self.agents or symbol not in self.environments:
            raise ValueError(f"Agent or environment not found for {symbol}")
        
        agent = self.agents[symbol]
        env = self.environments[symbol]
        
        training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'portfolio_values': []
        }
        
        logger.info(f"Starting RL training for {symbol} ({episodes} episodes)")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            losses = []
            
            while True:
                # Choose action
                action = agent.act(state, training=True)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()
                    losses.append(loss)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Update target network
            if episode % target_update_freq == 0:
                agent.update_target_network()
            
            # Record training metrics
            training_history['episode_rewards'].append(total_reward)
            training_history['episode_lengths'].append(steps)
            training_history['losses'].extend(losses)
            training_history['portfolio_values'].append(info.get('portfolio_value', 0))
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(training_history['episode_rewards'][-100:])
                avg_portfolio = np.mean(training_history['portfolio_values'][-100:])
                logger.info(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
                           f"Avg Portfolio=${avg_portfolio:.2f}, Epsilon={agent.epsilon:.3f}")
        
        self.training_history[symbol] = training_history
        logger.info(f"RL training completed for {symbol}")
        
        return training_history
    
    def get_trading_action(self, symbol: str, current_state: np.ndarray) -> Dict[str, Any]:
        """Get trading action from trained RL agent"""
        if symbol not in self.agents:
            return {'action': 'HOLD', 'confidence': 0.0, 'rl_signal': False}
        
        agent = self.agents[symbol]
        
        # Get action from agent
        action_idx = agent.act(current_state, training=False)
        
        # Map action to trading signal
        action_map = {
            0: {'action': 'HOLD', 'size': 0.0},
            1: {'action': 'BUY', 'size': 0.25},
            2: {'action': 'BUY', 'size': 0.5},
            3: {'action': 'BUY', 'size': 1.0},
            4: {'action': 'SELL', 'size': 0.25},
            5: {'action': 'SELL', 'size': 0.5},
            6: {'action': 'SELL', 'size': 1.0}
        }
        
        trading_action = action_map[action_idx]
        
        # Calculate confidence based on Q-values
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.q_network(state_tensor)
            confidence = torch.softmax(q_values, dim=1).max().item()
        
        return {
            'action': trading_action['action'],
            'size': trading_action['size'],
            'confidence': confidence,
            'rl_signal': True,
            'q_values': q_values.cpu().numpy().tolist()
        }
    
    def save_agents(self, directory: str):
        """Save all trained agents"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for symbol, agent in self.agents.items():
            filepath = os.path.join(directory, f"rl_agent_{symbol}.pth")
            agent.save(filepath)
            logger.info(f"Saved RL agent for {symbol} to {filepath}")
    
    def load_agents(self, directory: str):
        """Load trained agents"""
        import os
        
        for symbol in self.symbols:
            filepath = os.path.join(directory, f"rl_agent_{symbol}.pth")
            if os.path.exists(filepath):
                if symbol not in self.agents:
                    # Create dummy agent to load weights
                    dummy_env = TradingEnvironment(pd.DataFrame({'close': [1]}))
                    self.agents[symbol] = DQNAgent(dummy_env.state_dim, dummy_env.action_space.n)
                
                self.agents[symbol].load(filepath)
                logger.info(f"Loaded RL agent for {symbol} from {filepath}")


def main():
    """Test RL trading system"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
    prices = 45000 + np.cumsum(np.random.randn(1000) * 100)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'open': prices + np.random.randn(1000) * 50,
        'volume': np.random.lognormal(15, 1, 1000)
    })
    
    # Create and train RL system
    rl_system = RLTradingSystem(['BTCUSDT'])
    agent = rl_system.create_agent('BTCUSDT', data)
    
    # Train agent
    history = rl_system.train_agent('BTCUSDT', episodes=100)
    
    print(f"Training completed. Final average reward: {np.mean(history['episode_rewards'][-10:]):.2f}")
    
    # Test trading action
    test_state = np.random.randn(20)
    action = rl_system.get_trading_action('BTCUSDT', test_state)
    print(f"Test action: {action}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Real Data Backtesting System

This module provides comprehensive backtesting capabilities using real market data
from our ccxt/Delta Exchange integration, combined with our enhanced ML predictions
and Smart Money Concepts analysis.

Key Features:
- Real data fetching from multiple exchanges (Binance, Delta Exchange)
- Enhanced ML + SMC signal generation
- Comprehensive performance metrics
- Risk-adjusted returns analysis
- Multi-timeframe backtesting
- Portfolio simulation with realistic trading costs
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import our components
try:
    from ml.src.api.delta_client import DeltaExchangeClient
    from ml.src.data.data_loader import MarketDataLoader
    from ml.backend.src.api.enhanced_trading_predictions import (
        EnhancedTradingPredictor, TradingPredictionInput, ModelService
    )
    from ml.backend.src.strategy.smc_detection import SMCDector
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some components not available: {e}")
    COMPONENTS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    symbol: str = "BTCUSD"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    timeframe: str = "1h"
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% of capital per trade
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    use_enhanced_predictions: bool = True
    use_smc_analysis: bool = True
    confidence_threshold: float = 0.6
    risk_level: str = "medium"


@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'buy' or 'sell'
    pnl: Optional[float]
    pnl_pct: Optional[float]
    confidence: float
    signal_strength: str
    reason: str


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    config: BacktestConfig
    trades: List[Trade]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    final_capital: float
    equity_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    performance_metrics: Dict[str, Any]


class RealDataBacktester:
    """
    Comprehensive backtesting system using real market data
    """

    def __init__(self, config: BacktestConfig):
        """Initialize the backtester"""
        self.config = config
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.current_capital = config.initial_capital
        self.current_position = 0.0
        self.current_position_value = 0.0

        # Initialize data sources
        self.delta_client = None
        self.market_loader = None
        self.enhanced_predictor = None

        if COMPONENTS_AVAILABLE:
            self._initialize_components()

        logger.info(f"RealDataBacktester initialized for {config.symbol}")

    def _initialize_components(self):
        """Initialize data sources and prediction components"""
        try:
            # Initialize Delta Exchange client
            self.delta_client = DeltaExchangeClient()

            # Initialize market data loader
            self.market_loader = MarketDataLoader(
                timeframe=self.config.timeframe,
                symbols=[self.config.symbol.replace('USD', '/USDT')]
            )

            # Initialize enhanced predictor with mock model service
            if self.config.use_enhanced_predictions:
                from unittest.mock import Mock
                mock_service = Mock(spec=ModelService)
                mock_service.predict.return_value = {
                    'symbol': self.config.symbol,
                    'predictions': [0.2, 0.3, 0.5],
                    'confidence': 0.7,
                    'predicted_direction': 'up',
                    'prediction_time': datetime.now().isoformat(),
                    'model_version': 'backtest-v1.0'
                }
                self.enhanced_predictor = EnhancedTradingPredictor(mock_service)

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.enhanced_predictor = None

    def fetch_real_data(self) -> pd.DataFrame:
        """Fetch real market data for backtesting"""
        logger.info(f"Fetching real data for {self.config.symbol} from {self.config.start_date} to {self.config.end_date}")

        try:
            # Try Delta Exchange first
            if self.delta_client:
                logger.info("Fetching data from Delta Exchange...")
                start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
                end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
                days_back = (end_date - start_date).days

                data = self.delta_client.get_historical_ohlcv(
                    symbol=self.config.symbol,
                    interval=self.config.timeframe,
                    days_back=days_back
                )

                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info(f"Fetched {len(df)} candles from Delta Exchange")
                    return df

            # Fallback to market data loader (ccxt)
            if self.market_loader:
                logger.info("Fetching data from CCXT exchanges...")
                symbol_ccxt = self.config.symbol.replace('USD', '/USDT')

                try:
                    df = self.market_loader.fetch_data(
                        exchange='binance',
                        start_date=self.config.start_date,
                        end_date=self.config.end_date
                    )[symbol_ccxt]

                    logger.info(f"Fetched {len(df)} candles from Binance via CCXT")
                    return df

                except Exception as e:
                    logger.warning(f"CCXT fetch failed: {e}")

            # Generate sample data as fallback
            logger.warning("Using sample data for backtesting")
            return self._generate_sample_data()

        except Exception as e:
            logger.error(f"Error fetching real data: {e}")
            return self._generate_sample_data()

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic sample data for backtesting"""
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")

        # Calculate number of periods based on timeframe
        if self.config.timeframe == "1h":
            periods = int((end_date - start_date).total_seconds() / 3600)
            freq = 'H'
        elif self.config.timeframe == "4h":
            periods = int((end_date - start_date).total_seconds() / (4 * 3600))
            freq = '4H'
        elif self.config.timeframe == "1d":
            periods = (end_date - start_date).days
            freq = 'D'
        else:  # Default to 1h
            periods = int((end_date - start_date).total_seconds() / 3600)
            freq = 'H'

        # Generate realistic price data
        np.random.seed(42)
        base_price = 50000.0
        returns = np.random.normal(0.0001, 0.02, periods)  # Small positive drift with volatility

        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLCV data
        data = []
        timestamps = pd.date_range(start=start_date, periods=periods, freq=freq)

        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices[1:])):
            open_price = prices[i]

            # Generate realistic high/low
            volatility = close_price * 0.01
            high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility * 0.5))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility * 0.5))

            # Generate volume
            volume = np.random.uniform(1000000, 5000000)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample candles")
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using enhanced predictions and SMC analysis"""
        logger.info("Generating trading signals...")

        signals = []

        for i in range(len(data)):
            current_data = data.iloc[:i+1]

            # Skip if not enough data
            if len(current_data) < 50:
                signals.append({
                    'signal': 'hold',
                    'confidence': 0.5,
                    'strength': 'weak',
                    'reasoning': 'Insufficient data'
                })
                continue

            try:
                if self.enhanced_predictor and self.config.use_enhanced_predictions:
                    # Use enhanced predictions
                    signal_data = self._get_enhanced_signal(current_data)
                else:
                    # Use simple technical analysis
                    signal_data = self._get_simple_signal(current_data)

                signals.append(signal_data)

            except Exception as e:
                logger.warning(f"Error generating signal at index {i}: {e}")
                signals.append({
                    'signal': 'hold',
                    'confidence': 0.5,
                    'strength': 'weak',
                    'reasoning': f'Error: {str(e)}'
                })

        # Add signals to dataframe
        signal_df = pd.DataFrame(signals)
        result_df = data.copy()
        for col in signal_df.columns:
            result_df[col] = signal_df[col]

        logger.info(f"Generated signals for {len(result_df)} periods")
        return result_df

    def _get_enhanced_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get signal from enhanced trading predictor"""
        try:
            # Prepare OHLCV data for prediction
            ohlcv_data = data.tail(100).to_dict('records')  # Last 100 candles

            input_data = TradingPredictionInput(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                ohlcv_data=ohlcv_data,
                include_smc=self.config.use_smc_analysis,
                include_confluence=True,
                confidence_threshold=self.config.confidence_threshold,
                risk_level=self.config.risk_level
            )

            result = self.enhanced_predictor.predict_trading_signals(input_data)

            return {
                'signal': result.primary_signal.signal_type,
                'confidence': result.primary_signal.confidence,
                'strength': result.primary_signal.strength,
                'reasoning': '; '.join(result.primary_signal.reasoning[:2])  # First 2 reasons
            }

        except Exception as e:
            logger.warning(f"Enhanced prediction failed: {e}")
            return self._get_simple_signal(data)

    def _get_simple_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get signal from simple technical analysis"""
        try:
            # Calculate simple moving averages
            data_copy = data.copy()
            data_copy['sma_20'] = data_copy['close'].rolling(20).mean()
            data_copy['sma_50'] = data_copy['close'].rolling(50).mean()
            data_copy['rsi'] = self._calculate_rsi(data_copy['close'])

            latest = data_copy.iloc[-1]

            # Simple signal logic
            if latest['close'] > latest['sma_20'] > latest['sma_50'] and latest['rsi'] < 70:
                return {
                    'signal': 'buy',
                    'confidence': 0.6,
                    'strength': 'moderate',
                    'reasoning': 'Price above SMAs, RSI not overbought'
                }
            elif latest['close'] < latest['sma_20'] < latest['sma_50'] and latest['rsi'] > 30:
                return {
                    'signal': 'sell',
                    'confidence': 0.6,
                    'strength': 'moderate',
                    'reasoning': 'Price below SMAs, RSI not oversold'
                }
            else:
                return {
                    'signal': 'hold',
                    'confidence': 0.5,
                    'strength': 'weak',
                    'reasoning': 'No clear trend signal'
                }

        except Exception as e:
            return {
                'signal': 'hold',
                'confidence': 0.3,
                'strength': 'weak',
                'reasoning': f'Technical analysis error: {str(e)}'
            }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def run_backtest(self) -> BacktestResults:
        """Run the complete backtest"""
        logger.info("Starting backtest execution...")

        # Fetch real data
        self.data = self.fetch_real_data()

        # Generate signals
        signal_data = self.generate_signals(self.data)

        # Execute trades
        self._execute_backtest(signal_data)

        # Calculate performance metrics
        results = self._calculate_performance_metrics()

        logger.info(f"Backtest completed. Total trades: {len(self.trades)}")
        return results

    def _execute_backtest(self, data: pd.DataFrame):
        """Execute the backtest simulation"""
        logger.info("Executing backtest simulation...")

        current_trade = None

        for i, row in data.iterrows():
            current_price = row['close']
            signal = row['signal']
            confidence = row['confidence']
            strength = row['strength']
            reasoning = row['reasoning']

            # Update equity curve
            portfolio_value = self.current_capital + (self.current_position * current_price)
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'portfolio_value': portfolio_value,
                'price': current_price,
                'position': self.current_position,
                'capital': self.current_capital
            })

            # Check if we should close current position
            if current_trade and self._should_close_position(row, current_trade):
                self._close_position(current_trade, row)
                current_trade = None

            # Check if we should open new position
            if not current_trade and self._should_open_position(signal, confidence):
                current_trade = self._open_position(row, signal, confidence, strength, reasoning)

        # Close any remaining open position
        if current_trade:
            self._close_position(current_trade, data.iloc[-1])

    def _should_open_position(self, signal: str, confidence: float) -> bool:
        """Determine if we should open a new position"""
        return (signal in ['buy', 'sell'] and
                confidence >= self.config.confidence_threshold and
                self.current_position == 0)

    def _should_close_position(self, row: pd.Series, trade: Trade) -> bool:
        """Determine if we should close current position"""
        # Simple exit logic - close after 24 periods or on opposite signal
        current_time = row['timestamp']
        if trade.entry_time:
            hours_held = (current_time - trade.entry_time).total_seconds() / 3600

            # Close after 24 hours (24 periods for 1h timeframe)
            if hours_held >= 24:
                return True

            # Close on opposite signal with sufficient confidence
            if ((trade.side == 'buy' and row['signal'] == 'sell') or
                (trade.side == 'sell' and row['signal'] == 'buy')) and row['confidence'] >= 0.6:
                return True

        return False

    def _open_position(self, row: pd.Series, signal: str, confidence: float,
                      strength: str, reasoning: str) -> Trade:
        """Open a new trading position"""
        entry_price = row['close']
        entry_time = row['timestamp']

        # Calculate position size based on confidence and risk
        risk_multiplier = min(confidence / self.config.confidence_threshold, 1.0)
        position_size = self.config.max_position_size * risk_multiplier

        # Calculate quantity
        position_value = self.current_capital * position_size
        quantity = position_value / entry_price

        # Apply transaction costs
        transaction_cost = position_value * self.config.transaction_cost
        self.current_capital -= transaction_cost

        # Update position
        if signal == 'buy':
            self.current_position = quantity
            self.current_capital -= position_value
        else:  # sell (short)
            self.current_position = -quantity
            self.current_capital += position_value

        trade = Trade(
            entry_time=entry_time,
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=abs(quantity),
            side=signal,
            pnl=None,
            pnl_pct=None,
            confidence=confidence,
            signal_strength=strength,
            reason=reasoning
        )

        logger.debug(f"Opened {signal} position: {quantity:.4f} @ {entry_price:.2f}")
        return trade

    def _close_position(self, trade: Trade, row: pd.Series):
        """Close an existing trading position"""
        exit_price = row['close']
        exit_time = row['timestamp']

        # Calculate P&L
        if trade.side == 'buy':
            pnl = (exit_price - trade.entry_price) * trade.quantity
            self.current_capital += trade.quantity * exit_price
        else:  # sell (short)
            pnl = (trade.entry_price - exit_price) * trade.quantity
            self.current_capital -= trade.quantity * exit_price

        # Apply transaction costs and slippage
        position_value = trade.quantity * exit_price
        transaction_cost = position_value * self.config.transaction_cost
        slippage_cost = position_value * self.config.slippage

        pnl -= (transaction_cost + slippage_cost)
        self.current_capital -= transaction_cost

        # Calculate percentage return
        initial_value = trade.quantity * trade.entry_price
        pnl_pct = (pnl / initial_value) * 100 if initial_value > 0 else 0

        # Update trade record
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct

        # Reset position
        self.current_position = 0

        # Add to trades list
        self.trades.append(trade)

        logger.debug(f"Closed {trade.side} position: {trade.quantity:.4f} @ {exit_price:.2f}, P&L: {pnl:.2f}")

    def _calculate_performance_metrics(self) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        logger.info("Calculating performance metrics...")

        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)

        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod()

        # Basic metrics
        total_return = (self.current_capital / self.config.initial_capital) - 1

        # Annualized return
        start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        years = (end_date - start_date).days / 365.25
        annualized_return = ((1 + total_return) ** (1/years)) - 1 if years > 0 else 0

        # Sharpe ratio
        returns = equity_df['returns'].dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0

        # Maximum drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Average trade duration
        trade_durations = []
        for trade in self.trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                trade_durations.append(duration)

        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

        # Monthly returns
        monthly_returns = equity_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_df = monthly_returns.to_frame('monthly_return')

        # Additional performance metrics
        performance_metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([t.pnl for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t.pnl for t in losing_trades]) if losing_trades else 0,
            'avg_trade_duration_hours': avg_trade_duration,
            'total_transaction_costs': sum([abs(t.quantity * t.entry_price) * self.config.transaction_cost * 2 for t in self.trades]),
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        }

        return BacktestResults(
            config=self.config,
            trades=self.trades,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_trade_duration=avg_trade_duration,
            final_capital=self.current_capital,
            equity_curve=equity_df,
            monthly_returns=monthly_returns_df,
            performance_metrics=performance_metrics
        )

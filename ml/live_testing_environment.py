#!/usr/bin/env python3
"""
Live Testing Environment for Enhanced SmartMarketOOPS System
Configures paper trading mode with real-time signal quality monitoring
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.trading.multi_symbol_manager import MultiSymbolTradingManager
from src.risk.advanced_risk_manager import AdvancedRiskManager
from src.data.real_market_data_service import get_market_data_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveTestingEnvironment:
    """Live testing environment for enhanced signal quality system"""

    def __init__(self):
        """Initialize the live testing environment"""
        self.ml_api_url = "http://localhost:8000"
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        self.signal_history = []
        self.performance_metrics = {}
        self.is_running = False

        # Paper trading configuration
        self.paper_balance = 10000.0  # $10,000 starting balance
        self.position_size = 0.02     # 2% per trade
        self.max_positions = 3        # Maximum concurrent positions
        self.current_positions = {}

        logger.info("Live Testing Environment initialized")

    def generate_live_market_data(self, symbol: str) -> Dict[str, float]:
        """
        Generate simulated live market data
        In production, this would connect to real market data feeds
        """
        # Base prices for different symbols
        base_prices = {
            "BTCUSDT": 45000 + np.random.normal(0, 1000),
            "ETHUSDT": 2500 + np.random.normal(0, 100),
            "ADAUSDT": 0.5 + np.random.normal(0, 0.05)
        }

        base_price = base_prices.get(symbol, 100)

        # Generate realistic OHLCV data
        volatility = base_price * 0.001

        close = base_price + np.random.normal(0, volatility)
        open_price = close * (1 + np.random.normal(0, 0.001))
        high = max(open_price, close) + np.random.exponential(volatility)
        low = min(open_price, close) - np.random.exponential(volatility)
        volume = np.random.lognormal(15, 1)

        # Add technical indicators
        rsi = np.random.uniform(30, 70)  # Simulated RSI
        macd = np.random.normal(0, 0.1)  # Simulated MACD
        bb_upper = close * 1.02
        bb_lower = close * 0.98

        return {
            "open": float(open_price),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
            "rsi": float(rsi),
            "macd": float(macd),
            "bb_upper": float(bb_upper),
            "bb_lower": float(bb_lower),
            "timestamp": datetime.now().isoformat()
        }

    async def get_enhanced_signal(self, symbol: str, market_data: Dict[str, float]) -> Dict[str, Any]:
        """Get enhanced signal from the ML API"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "symbol": symbol,
                    "features": market_data,
                    "sequence_length": 60
                }

                async with session.post(
                    f"{self.ml_api_url}/api/models/enhanced/predict",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:

                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        logger.error(f"Enhanced signal request failed for {symbol}: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error getting enhanced signal for {symbol}: {e}")
            return None

    def process_signal_for_trading(self, symbol: str, signal: Dict[str, Any], market_data: Dict[str, float]) -> Dict[str, Any]:
        """Process enhanced signal for paper trading decisions"""
        if not signal or not signal.get('signal_valid', False):
            return {
                'action': 'HOLD',
                'reason': 'Signal not valid or not available',
                'confidence': 0.0
            }

        prediction = signal.get('prediction', 0.5)
        confidence = signal.get('confidence', 0.0)
        quality_score = signal.get('quality_score', 0.0)
        recommendation = signal.get('recommendation', 'NEUTRAL')

        # Enhanced trading logic based on signal quality
        if confidence < 0.7:
            return {
                'action': 'HOLD',
                'reason': f'Confidence too low: {confidence:.3f}',
                'confidence': confidence
            }

        if quality_score < 0.6:
            return {
                'action': 'HOLD',
                'reason': f'Quality score too low: {quality_score:.3f}',
                'confidence': confidence
            }

        # Determine trading action
        if prediction > 0.65 and 'BUY' in recommendation.upper():
            return {
                'action': 'BUY',
                'reason': f'Strong buy signal (pred: {prediction:.3f}, conf: {confidence:.3f})',
                'confidence': confidence,
                'target_price': market_data['close'] * 1.02,  # 2% target
                'stop_loss': market_data['close'] * 0.985     # 1.5% stop loss
            }
        elif prediction < 0.35 and 'SELL' in recommendation.upper():
            return {
                'action': 'SELL',
                'reason': f'Strong sell signal (pred: {prediction:.3f}, conf: {confidence:.3f})',
                'confidence': confidence,
                'target_price': market_data['close'] * 0.98,  # 2% target
                'stop_loss': market_data['close'] * 1.015     # 1.5% stop loss
            }
        else:
            return {
                'action': 'HOLD',
                'reason': f'Neutral signal (pred: {prediction:.3f})',
                'confidence': confidence
            }

    def execute_paper_trade(self, symbol: str, action: str, market_data: Dict[str, float], trade_decision: Dict[str, Any]) -> bool:
        """Execute paper trade based on signal"""
        if action == 'HOLD':
            return False

        current_price = market_data['close']
        position_value = self.paper_balance * self.position_size

        if action == 'BUY' and len(self.current_positions) < self.max_positions:
            # Open long position
            quantity = position_value / current_price

            self.current_positions[symbol] = {
                'type': 'LONG',
                'entry_price': current_price,
                'quantity': quantity,
                'target_price': trade_decision.get('target_price', current_price * 1.02),
                'stop_loss': trade_decision.get('stop_loss', current_price * 0.985),
                'entry_time': datetime.now(),
                'confidence': trade_decision.get('confidence', 0.0)
            }

            logger.info(f"📈 PAPER BUY {symbol}: {quantity:.6f} @ ${current_price:.2f} "
                       f"(conf: {trade_decision.get('confidence', 0):.3f})")
            return True

        elif action == 'SELL' and len(self.current_positions) < self.max_positions:
            # Open short position (simulated)
            quantity = position_value / current_price

            self.current_positions[symbol] = {
                'type': 'SHORT',
                'entry_price': current_price,
                'quantity': quantity,
                'target_price': trade_decision.get('target_price', current_price * 0.98),
                'stop_loss': trade_decision.get('stop_loss', current_price * 1.015),
                'entry_time': datetime.now(),
                'confidence': trade_decision.get('confidence', 0.0)
            }

            logger.info(f"📉 PAPER SELL {symbol}: {quantity:.6f} @ ${current_price:.2f} "
                       f"(conf: {trade_decision.get('confidence', 0):.3f})")
            return True

        return False

    def check_position_exits(self, market_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check if any positions should be closed"""
        exits = []

        for symbol, position in list(self.current_positions.items()):
            current_price = market_data['close']
            entry_price = position['entry_price']

            # Check for target or stop loss
            should_exit = False
            exit_reason = ""

            if position['type'] == 'LONG':
                if current_price >= position['target_price']:
                    should_exit = True
                    exit_reason = "Target reached"
                elif current_price <= position['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop loss hit"
            else:  # SHORT
                if current_price <= position['target_price']:
                    should_exit = True
                    exit_reason = "Target reached"
                elif current_price >= position['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop loss hit"

            # Check for time-based exit (24 hours max)
            if datetime.now() - position['entry_time'] > timedelta(hours=24):
                should_exit = True
                exit_reason = "Time limit reached"

            if should_exit:
                # Calculate P&L
                if position['type'] == 'LONG':
                    pnl = (current_price - entry_price) / entry_price
                else:  # SHORT
                    pnl = (entry_price - current_price) / entry_price

                pnl_amount = position['quantity'] * entry_price * pnl

                exits.append({
                    'symbol': symbol,
                    'type': position['type'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl_percent': pnl * 100,
                    'pnl_amount': pnl_amount,
                    'reason': exit_reason,
                    'confidence': position['confidence'],
                    'duration': datetime.now() - position['entry_time']
                })

                # Update paper balance
                self.paper_balance += pnl_amount

                # Remove position
                del self.current_positions[symbol]

                logger.info(f"🔄 PAPER EXIT {symbol}: {exit_reason} | "
                           f"P&L: {pnl*100:.2f}% (${pnl_amount:.2f}) | "
                           f"Balance: ${self.paper_balance:.2f}")

        return exits

    async def run_live_testing_cycle(self):
        """Run one cycle of live testing"""
        cycle_results = []

        for symbol in self.test_symbols:
            try:
                # Generate market data
                market_data = self.generate_live_market_data(symbol)

                # Get enhanced signal
                signal = await self.get_enhanced_signal(symbol, market_data)

                if signal:
                    # Process signal for trading
                    trade_decision = self.process_signal_for_trading(symbol, signal, market_data)

                    # Execute paper trade if applicable
                    trade_executed = self.execute_paper_trade(
                        symbol, trade_decision['action'], market_data, trade_decision
                    )

                    # Check for position exits
                    exits = self.check_position_exits(market_data)

                    # Record cycle result
                    cycle_result = {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'market_data': market_data,
                        'signal': signal,
                        'trade_decision': trade_decision,
                        'trade_executed': trade_executed,
                        'exits': exits,
                        'current_balance': self.paper_balance,
                        'active_positions': len(self.current_positions)
                    }

                    cycle_results.append(cycle_result)
                    self.signal_history.append(cycle_result)

                    # Update performance tracking
                    await self.update_performance_tracking(symbol, signal, exits)

            except Exception as e:
                logger.error(f"Error in live testing cycle for {symbol}: {e}")

        return cycle_results

    async def update_performance_tracking(self, symbol: str, signal: Dict[str, Any], exits: List[Dict[str, Any]]):
        """Update performance tracking with actual results"""
        try:
            for exit_trade in exits:
                # Determine actual outcome (1.0 for profit, 0.0 for loss)
                actual_outcome = 1.0 if exit_trade['pnl_percent'] > 0 else 0.0

                # Update ML model performance
                async with aiohttp.ClientSession() as session:
                    params = {
                        'prediction': signal.get('prediction', 0.5),
                        'actual_outcome': actual_outcome,
                        'confidence': signal.get('confidence', 0.0)
                    }

                    async with session.post(
                        f"{self.ml_api_url}/api/models/enhanced/models/{symbol}/performance",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            logger.debug(f"Performance updated for {symbol}")

        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")

    async def run_live_testing(self, duration_minutes: int = 60):
        """Run live testing for specified duration"""
        logger.info(f"🚀 Starting live testing for {duration_minutes} minutes")
        logger.info(f"Paper trading balance: ${self.paper_balance:.2f}")

        self.is_running = True
        start_time = datetime.now()
        cycle_count = 0

        try:
            while self.is_running and (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
                cycle_count += 1
                logger.info(f"\n--- Live Testing Cycle {cycle_count} ---")

                # Run testing cycle
                cycle_results = await self.run_live_testing_cycle()

                # Log summary
                valid_signals = sum(1 for r in cycle_results if r['signal'] and r['signal'].get('signal_valid', False))
                trades_executed = sum(1 for r in cycle_results if r['trade_executed'])

                logger.info(f"Cycle {cycle_count}: {valid_signals}/{len(cycle_results)} valid signals, "
                           f"{trades_executed} trades executed, "
                           f"{len(self.current_positions)} active positions")

                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second intervals

        except KeyboardInterrupt:
            logger.info("Live testing interrupted by user")
        except Exception as e:
            logger.error(f"Error in live testing: {e}")
        finally:
            self.is_running = False

        # Generate final report
        self.generate_live_testing_report()

    def generate_live_testing_report(self):
        """Generate comprehensive live testing report"""
        logger.info("\n" + "="*60)
        logger.info("LIVE TESTING REPORT")
        logger.info("="*60)

        # Calculate performance metrics
        total_signals = len(self.signal_history)
        valid_signals = sum(1 for h in self.signal_history if h['signal'] and h['signal'].get('signal_valid', False))
        total_trades = sum(1 for h in self.signal_history if h['trade_executed'])

        # Calculate P&L
        starting_balance = 10000.0
        total_pnl = self.paper_balance - starting_balance
        pnl_percent = (total_pnl / starting_balance) * 100

        logger.info(f"Testing Duration: {len(self.signal_history)} cycles")
        logger.info(f"Total Signals: {total_signals}")
        logger.info(f"Valid Signals: {valid_signals} ({valid_signals/total_signals*100:.1f}%)")
        logger.info(f"Trades Executed: {total_trades}")
        logger.info(f"Final Balance: ${self.paper_balance:.2f}")
        logger.info(f"Total P&L: ${total_pnl:.2f} ({pnl_percent:.2f}%)")
        logger.info(f"Active Positions: {len(self.current_positions)}")

        # Signal quality metrics
        if valid_signals > 0:
            avg_confidence = np.mean([h['signal']['confidence'] for h in self.signal_history
                                    if h['signal'] and h['signal'].get('signal_valid', False)])
            avg_quality = np.mean([h['signal']['quality_score'] for h in self.signal_history
                                 if h['signal'] and h['signal'].get('signal_valid', False)])

            logger.info(f"Average Confidence: {avg_confidence:.3f}")
            logger.info(f"Average Quality Score: {avg_quality:.3f}")

        logger.info("\n🎉 Live testing completed successfully!")


async def main():
    """Main function for live testing"""
    env = LiveTestingEnvironment()

    # Run live testing for 5 minutes (demo)
    await env.run_live_testing(duration_minutes=5)


if __name__ == "__main__":
    asyncio.run(main())

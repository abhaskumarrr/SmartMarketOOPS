#!/usr/bin/env python3
"""
Multi-Symbol Trading Manager for Enhanced SmartMarketOOPS System
Manages trading across multiple cryptocurrency pairs with symbol-specific optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json

from ..api.enhanced_model_service import EnhancedModelService
from ..data.real_market_data_service import get_market_data_service, MarketDataPoint
from ..utils.config import MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SymbolConfig:
    """Configuration for a specific trading symbol"""
    symbol: str
    enabled: bool = True
    model_type: str = 'enhanced_transformer'
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.6
    position_size_pct: float = 2.0  # Percentage of portfolio per trade
    max_positions: int = 3
    risk_multiplier: float = 1.0
    
    # Symbol-specific parameters
    volatility_adjustment: float = 1.0
    correlation_weight: float = 1.0
    market_cap_tier: str = 'large'  # large, mid, small
    
    # Trading parameters
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_hold_hours: int = 24


@dataclass
class SymbolPerformance:
    """Performance metrics for a specific symbol"""
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    last_updated: datetime = None


class MultiSymbolTradingManager:
    """Multi-symbol trading manager with enhanced optimization"""
    
    def __init__(self):
        """Initialize the multi-symbol trading manager"""
        self.enhanced_service = EnhancedModelService()
        self.market_data_service = None
        
        # Symbol configurations
        self.symbol_configs = self._initialize_symbol_configs()
        self.symbol_performance = {}
        self.correlation_matrix = pd.DataFrame()
        
        # Portfolio state
        self.portfolio_balance = 100000.0  # $100k starting balance
        self.active_positions = {}
        self.position_history = []
        
        # Risk management
        self.max_portfolio_risk = 0.1  # 10% max portfolio risk
        self.max_correlation_exposure = 0.6  # Max 60% in correlated assets
        
        logger.info("Multi-Symbol Trading Manager initialized")
    
    def _initialize_symbol_configs(self) -> Dict[str, SymbolConfig]:
        """Initialize symbol-specific configurations"""
        symbols = MODEL_CONFIG.get('supported_symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT'])
        
        configs = {}
        
        # Bitcoin - Large cap, lower volatility
        configs['BTCUSDT'] = SymbolConfig(
            symbol='BTCUSDT',
            confidence_threshold=0.75,
            quality_threshold=0.65,
            position_size_pct=3.0,
            volatility_adjustment=0.8,
            market_cap_tier='large',
            stop_loss_pct=1.5,
            take_profit_pct=3.0
        )
        
        # Ethereum - Large cap, moderate volatility
        configs['ETHUSDT'] = SymbolConfig(
            symbol='ETHUSDT',
            confidence_threshold=0.7,
            quality_threshold=0.6,
            position_size_pct=2.5,
            volatility_adjustment=1.0,
            market_cap_tier='large',
            stop_loss_pct=2.0,
            take_profit_pct=4.0
        )
        
        # Solana - Mid cap, higher volatility
        configs['SOLUSDT'] = SymbolConfig(
            symbol='SOLUSDT',
            confidence_threshold=0.65,
            quality_threshold=0.55,
            position_size_pct=2.0,
            volatility_adjustment=1.3,
            market_cap_tier='mid',
            stop_loss_pct=2.5,
            take_profit_pct=5.0
        )
        
        # Cardano - Mid cap, moderate volatility
        configs['ADAUSDT'] = SymbolConfig(
            symbol='ADAUSDT',
            confidence_threshold=0.65,
            quality_threshold=0.55,
            position_size_pct=1.5,
            volatility_adjustment=1.2,
            market_cap_tier='mid',
            stop_loss_pct=2.5,
            take_profit_pct=5.0
        )
        
        # Initialize performance tracking
        for symbol in configs.keys():
            self.symbol_performance[symbol] = SymbolPerformance(symbol=symbol)
        
        return configs
    
    async def initialize(self):
        """Initialize all services and load models"""
        logger.info("Initializing Multi-Symbol Trading Manager...")
        
        # Initialize market data service
        self.market_data_service = await get_market_data_service()
        
        # Initialize enhanced model service
        await self.enhanced_service.initialize_market_data_service()
        
        # Load models for all symbols
        for symbol in self.symbol_configs.keys():
            try:
                success = self.enhanced_service.load_model(symbol)
                if success:
                    logger.info(f"✅ Model loaded for {symbol}")
                else:
                    logger.warning(f"⚠️  Failed to load model for {symbol}")
            except Exception as e:
                logger.error(f"❌ Error loading model for {symbol}: {e}")
        
        # Calculate initial correlation matrix
        await self._update_correlation_matrix()
        
        logger.info("✅ Multi-Symbol Trading Manager initialized")
    
    async def _update_correlation_matrix(self):
        """Update correlation matrix between symbols"""
        try:
            # Get historical data for all symbols
            symbol_data = {}
            
            for symbol in self.symbol_configs.keys():
                try:
                    df = await self.market_data_service.get_historical_data(
                        symbol=symbol,
                        timeframe='1h',
                        limit=168  # 1 week of hourly data
                    )
                    
                    if not df.empty:
                        symbol_data[symbol] = df['close'].pct_change().dropna()
                
                except Exception as e:
                    logger.error(f"Error getting historical data for {symbol}: {e}")
            
            # Calculate correlation matrix
            if len(symbol_data) > 1:
                correlation_df = pd.DataFrame(symbol_data)
                self.correlation_matrix = correlation_df.corr()
                logger.info("✅ Correlation matrix updated")
            
        except Exception as e:
            logger.error(f"Error updating correlation matrix: {e}")
    
    async def generate_multi_symbol_signals(self) -> Dict[str, Any]:
        """Generate trading signals for all symbols"""
        signals = {}
        
        for symbol, config in self.symbol_configs.items():
            if not config.enabled:
                continue
            
            try:
                # Get enhanced prediction
                prediction = await self.enhanced_service.predict(
                    symbol=symbol,
                    features={},  # Will use real market data
                    sequence_length=60
                )
                
                # Apply symbol-specific filtering
                if self._should_generate_signal(symbol, prediction, config):
                    signals[symbol] = {
                        'prediction': prediction,
                        'config': config,
                        'timestamp': datetime.now().isoformat()
                    }
                    logger.info(f"📊 Signal generated for {symbol}: "
                               f"pred={prediction.get('prediction', 0):.3f}, "
                               f"conf={prediction.get('confidence', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _should_generate_signal(self, symbol: str, prediction: Dict[str, Any], config: SymbolConfig) -> bool:
        """Determine if a signal should be generated based on symbol-specific criteria"""
        
        # Check basic signal validity
        if not prediction.get('signal_valid', False):
            return False
        
        # Check confidence threshold
        confidence = prediction.get('confidence', 0)
        if confidence < config.confidence_threshold:
            return False
        
        # Check quality threshold
        quality_score = prediction.get('quality_score', 0)
        if quality_score < config.quality_threshold:
            return False
        
        # Check if we already have max positions for this symbol
        symbol_positions = sum(1 for pos in self.active_positions.values() 
                              if pos.get('symbol') == symbol)
        if symbol_positions >= config.max_positions:
            return False
        
        # Check portfolio-level risk
        if not self._check_portfolio_risk(symbol, config):
            return False
        
        return True
    
    def _check_portfolio_risk(self, symbol: str, config: SymbolConfig) -> bool:
        """Check if adding this position would exceed portfolio risk limits"""
        
        # Calculate current portfolio risk
        current_risk = sum(pos.get('risk_amount', 0) for pos in self.active_positions.values())
        current_risk_pct = current_risk / self.portfolio_balance
        
        # Calculate new position risk
        position_size = self.portfolio_balance * (config.position_size_pct / 100)
        position_risk = position_size * (config.stop_loss_pct / 100)
        new_risk_pct = (current_risk + position_risk) / self.portfolio_balance
        
        # Check max portfolio risk
        if new_risk_pct > self.max_portfolio_risk:
            logger.debug(f"Portfolio risk limit exceeded for {symbol}: {new_risk_pct:.2%} > {self.max_portfolio_risk:.2%}")
            return False
        
        # Check correlation exposure
        if not self._check_correlation_exposure(symbol, config):
            return False
        
        return True
    
    def _check_correlation_exposure(self, symbol: str, config: SymbolConfig) -> bool:
        """Check correlation exposure limits"""
        if self.correlation_matrix.empty:
            return True
        
        try:
            # Calculate exposure to correlated assets
            correlated_exposure = 0.0
            
            for pos_symbol, position in self.active_positions.items():
                if pos_symbol != symbol and symbol in self.correlation_matrix.index:
                    correlation = abs(self.correlation_matrix.loc[symbol, pos_symbol])
                    if correlation > 0.7:  # High correlation threshold
                        correlated_exposure += position.get('position_value', 0)
            
            # Add new position value
            new_position_value = self.portfolio_balance * (config.position_size_pct / 100)
            total_correlated_exposure = (correlated_exposure + new_position_value) / self.portfolio_balance
            
            if total_correlated_exposure > self.max_correlation_exposure:
                logger.debug(f"Correlation exposure limit exceeded for {symbol}: {total_correlated_exposure:.2%}")
                return False
            
        except Exception as e:
            logger.error(f"Error checking correlation exposure: {e}")
        
        return True
    
    async def execute_multi_symbol_trades(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute trades for multiple symbols with portfolio optimization"""
        executed_trades = []
        
        # Sort signals by confidence and quality
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: (x[1]['prediction'].get('confidence', 0) + 
                          x[1]['prediction'].get('quality_score', 0)) / 2,
            reverse=True
        )
        
        for symbol, signal_data in sorted_signals:
            try:
                trade = await self._execute_symbol_trade(symbol, signal_data)
                if trade:
                    executed_trades.append(trade)
                    
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
        
        return executed_trades
    
    async def _execute_symbol_trade(self, symbol: str, signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute trade for a specific symbol"""
        prediction = signal_data['prediction']
        config = signal_data['config']
        
        # Get current market data
        market_data = await self.market_data_service.get_latest_data(symbol)
        if not market_data:
            logger.warning(f"No market data available for {symbol}")
            return None
        
        current_price = market_data.close
        
        # Determine trade direction
        pred_value = prediction.get('prediction', 0.5)
        if pred_value > 0.6:
            direction = 'LONG'
        elif pred_value < 0.4:
            direction = 'SHORT'
        else:
            return None  # Neutral signal
        
        # Calculate position size
        position_value = self.portfolio_balance * (config.position_size_pct / 100)
        quantity = position_value / current_price
        
        # Calculate stop loss and take profit
        if direction == 'LONG':
            stop_loss = current_price * (1 - config.stop_loss_pct / 100)
            take_profit = current_price * (1 + config.take_profit_pct / 100)
        else:
            stop_loss = current_price * (1 + config.stop_loss_pct / 100)
            take_profit = current_price * (1 - config.take_profit_pct / 100)
        
        # Create trade record
        trade = {
            'id': f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': symbol,
            'direction': direction,
            'entry_price': current_price,
            'quantity': quantity,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': prediction.get('confidence', 0),
            'quality_score': prediction.get('quality_score', 0),
            'entry_time': datetime.now(),
            'status': 'ACTIVE',
            'risk_amount': position_value * (config.stop_loss_pct / 100)
        }
        
        # Add to active positions
        self.active_positions[trade['id']] = trade
        
        logger.info(f"🚀 {direction} trade executed for {symbol}: "
                   f"${position_value:.2f} @ ${current_price:.4f} "
                   f"(SL: ${stop_loss:.4f}, TP: ${take_profit:.4f})")
        
        return trade
    
    async def monitor_positions(self) -> List[Dict[str, Any]]:
        """Monitor and manage active positions"""
        closed_positions = []
        
        for trade_id, position in list(self.active_positions.items()):
            try:
                # Get current market data
                market_data = await self.market_data_service.get_latest_data(position['symbol'])
                if not market_data:
                    continue
                
                current_price = market_data.close
                
                # Check for exit conditions
                should_close, reason = self._should_close_position(position, current_price)
                
                if should_close:
                    closed_position = self._close_position(trade_id, current_price, reason)
                    closed_positions.append(closed_position)
                    
            except Exception as e:
                logger.error(f"Error monitoring position {trade_id}: {e}")
        
        return closed_positions
    
    def _should_close_position(self, position: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
        """Determine if a position should be closed"""
        
        # Check stop loss
        if position['direction'] == 'LONG':
            if current_price <= position['stop_loss']:
                return True, 'STOP_LOSS'
            if current_price >= position['take_profit']:
                return True, 'TAKE_PROFIT'
        else:  # SHORT
            if current_price >= position['stop_loss']:
                return True, 'STOP_LOSS'
            if current_price <= position['take_profit']:
                return True, 'TAKE_PROFIT'
        
        # Check time-based exit
        config = self.symbol_configs[position['symbol']]
        time_held = datetime.now() - position['entry_time']
        if time_held > timedelta(hours=config.max_hold_hours):
            return True, 'TIME_LIMIT'
        
        return False, ''
    
    def _close_position(self, trade_id: str, exit_price: float, reason: str) -> Dict[str, Any]:
        """Close a position and calculate P&L"""
        position = self.active_positions[trade_id]
        
        # Calculate P&L
        if position['direction'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:  # SHORT
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        pnl_pct = (pnl / position['position_value']) * 100
        
        # Update position
        position.update({
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'status': 'CLOSED'
        })
        
        # Update portfolio balance
        self.portfolio_balance += pnl
        
        # Update performance metrics
        self._update_symbol_performance(position)
        
        # Move to history and remove from active
        self.position_history.append(position)
        del self.active_positions[trade_id]
        
        logger.info(f"🔄 Position closed for {position['symbol']}: "
                   f"{reason} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
                   f"Balance: ${self.portfolio_balance:.2f}")
        
        return position
    
    def _update_symbol_performance(self, closed_position: Dict[str, Any]):
        """Update performance metrics for a symbol"""
        symbol = closed_position['symbol']
        perf = self.symbol_performance[symbol]
        
        perf.total_trades += 1
        perf.total_pnl += closed_position['pnl']
        
        if closed_position['pnl'] > 0:
            perf.winning_trades += 1
            perf.avg_win = ((perf.avg_win * (perf.winning_trades - 1)) + closed_position['pnl']) / perf.winning_trades
        else:
            perf.losing_trades += 1
            perf.avg_loss = ((perf.avg_loss * (perf.losing_trades - 1)) + closed_position['pnl']) / perf.losing_trades
        
        perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0
        perf.last_updated = datetime.now()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_position_value = sum(pos.get('position_value', 0) for pos in self.active_positions.values())
        available_balance = self.portfolio_balance - total_position_value
        
        return {
            'portfolio_balance': self.portfolio_balance,
            'available_balance': available_balance,
            'total_position_value': total_position_value,
            'active_positions': len(self.active_positions),
            'total_trades': len(self.position_history),
            'symbol_performance': {
                symbol: {
                    'total_trades': perf.total_trades,
                    'win_rate': perf.win_rate,
                    'total_pnl': perf.total_pnl,
                    'avg_win': perf.avg_win,
                    'avg_loss': perf.avg_loss
                }
                for symbol, perf in self.symbol_performance.items()
            },
            'correlation_matrix': self.correlation_matrix.to_dict() if not self.correlation_matrix.empty else {}
        }


async def main():
    """Test the multi-symbol trading manager"""
    manager = MultiSymbolTradingManager()
    await manager.initialize()
    
    # Generate signals
    signals = await manager.generate_multi_symbol_signals()
    print(f"Generated {len(signals)} signals")
    
    # Execute trades
    trades = await manager.execute_multi_symbol_trades(signals)
    print(f"Executed {len(trades)} trades")
    
    # Monitor positions
    await asyncio.sleep(5)  # Wait a bit
    closed = await manager.monitor_positions()
    print(f"Closed {len(closed)} positions")
    
    # Get portfolio summary
    summary = manager.get_portfolio_summary()
    print(f"Portfolio Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Real-Time Data Service for SmartMarketOOPS
Integrates with multiple exchanges for live market data
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import ta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    price: float
    timestamp: int
    indicators: Dict[str, float]
    source: str = "realtime_analysis"

class RealTimeDataService:
    """Real-time market data service with multiple exchange support"""
    
    def __init__(self):
        self.exchanges = {}
        self.market_data = {}
        self.trading_signals = {}
        self.is_running = False
        self.update_callbacks = []
        
        # Initialize exchanges
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Binance (free, no API key required for public data)
            self.exchanges['binance'] = ccxt.binance({
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,
            })
            logger.info("✅ Binance exchange initialized")
            
            # Coinbase (free, no API key required for public data)
            self.exchanges['coinbase'] = ccxt.coinbase({
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,
            })
            logger.info("✅ Coinbase exchange initialized")
            
            # Delta Exchange (if credentials available)
            delta_api_key = os.getenv('DELTA_EXCHANGE_API_KEY')
            delta_secret = os.getenv('DELTA_EXCHANGE_API_SECRET')
            
            if delta_api_key and delta_secret:
                # Note: CCXT doesn't have direct Delta Exchange support
                # We'll use it for other exchanges and implement Delta separately if needed
                logger.info("⚠️ Delta Exchange credentials found but not implemented in CCXT")
            else:
                logger.info("⚠️ Delta Exchange credentials not found")
                
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def start_data_feeds(self, symbols: List[str] = None):
        """Start real-time data feeds"""
        if symbols is None:
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

        self.is_running = True
        logger.info(f"🚀 Starting real-time data feeds for {symbols}")

        # Start data collection tasks
        tasks = []

        try:
            for symbol in symbols:
                for exchange_name, exchange in self.exchanges.items():
                    try:
                        if exchange.has.get('fetchTicker'):
                            task = asyncio.create_task(
                                self._fetch_ticker_loop(exchange_name, exchange, symbol)
                            )
                            tasks.append(task)
                            logger.info(f"✅ Started ticker task for {exchange_name}:{symbol}")

                        if exchange.has.get('fetchOHLCV'):
                            task = asyncio.create_task(
                                self._fetch_ohlcv_loop(exchange_name, exchange, symbol)
                            )
                            tasks.append(task)
                            logger.info(f"✅ Started OHLCV task for {exchange_name}:{symbol}")

                    except Exception as e:
                        logger.error(f"❌ Failed to start tasks for {exchange_name}:{symbol}: {e}")

            # Start signal generation
            signal_task = asyncio.create_task(self._generate_signals_loop(symbols))
            tasks.append(signal_task)
            logger.info(f"✅ Started signal generation task")

            logger.info(f"🎯 Total tasks started: {len(tasks)}")

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                logger.warning("⚠️ No tasks to run")

        except Exception as e:
            logger.error(f"❌ Error in data feeds: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            logger.info("🛑 Real-time data feeds stopped")
    
    async def _fetch_ticker_loop(self, exchange_name: str, exchange, symbol: str):
        """Fetch ticker data in a loop"""
        consecutive_errors = 0
        max_errors = 5

        logger.info(f"🔄 Starting ticker loop for {exchange_name}:{symbol}")

        while self.is_running and consecutive_errors < max_errors:
            try:
                # Fetch ticker with timeout
                ticker = await asyncio.wait_for(
                    asyncio.to_thread(exchange.fetch_ticker, symbol),
                    timeout=10.0
                )

                if ticker and ticker.get('close'):
                    # Convert to our MarketData format
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=ticker.get('timestamp', int(asyncio.get_event_loop().time() * 1000)),
                        open=ticker.get('open', 0) or 0,
                        high=ticker.get('high', 0) or 0,
                        low=ticker.get('low', 0) or 0,
                        close=ticker.get('close', 0) or 0,
                        volume=ticker.get('baseVolume', 0) or 0,
                        exchange=exchange_name
                    )

                    # Store the data
                    key = f"{exchange_name}:{symbol}"
                    self.market_data[key] = market_data

                    # Notify callbacks
                    await self._notify_callbacks('ticker', market_data)

                    logger.info(f"📊 {exchange_name} {symbol}: ${ticker['close']:.2f}")
                    consecutive_errors = 0  # Reset error count on success
                else:
                    logger.warning(f"⚠️ Invalid ticker data for {exchange_name}:{symbol}")

            except asyncio.TimeoutError:
                consecutive_errors += 1
                logger.warning(f"⏰ Timeout fetching ticker {exchange_name}:{symbol} (attempt {consecutive_errors})")
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ Error fetching ticker {exchange_name}:{symbol} (attempt {consecutive_errors}): {e}")

            await asyncio.sleep(5)  # Update every 5 seconds

        if consecutive_errors >= max_errors:
            logger.error(f"❌ Too many errors for {exchange_name}:{symbol} ticker, stopping loop")
    
    async def _fetch_ohlcv_loop(self, exchange_name: str, exchange, symbol: str):
        """Fetch OHLCV data in a loop"""
        while self.is_running:
            try:
                # Fetch recent candles (1-minute timeframe)
                ohlcv = await asyncio.to_thread(
                    exchange.fetch_ohlcv, symbol, '1m', limit=100
                )
                
                if ohlcv:
                    # Convert to DataFrame for analysis
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Store for signal generation
                    key = f"{exchange_name}:{symbol}:ohlcv"
                    self.market_data[key] = df
                    
                    logger.debug(f"📈 {exchange_name} {symbol}: {len(df)} candles updated")
                
            except Exception as e:
                logger.error(f"Error fetching OHLCV {exchange_name}:{symbol}: {e}")
            
            await asyncio.sleep(60)  # Update every minute
    
    async def _generate_signals_loop(self, symbols: List[str]):
        """Generate trading signals based on technical analysis"""
        while self.is_running:
            try:
                for symbol in symbols:
                    await self._analyze_symbol(symbol)
                
            except Exception as e:
                logger.error(f"Error generating signals: {e}")
            
            await asyncio.sleep(30)  # Generate signals every 30 seconds
    
    async def _analyze_symbol(self, symbol: str):
        """Analyze a symbol and generate trading signals"""
        try:
            # Get OHLCV data from the primary exchange (Binance)
            key = f"binance:{symbol}:ohlcv"
            if key not in self.market_data:
                return
            
            df = self.market_data[key].copy()
            if len(df) < 20:  # Need enough data for indicators
                return
            
            # Calculate technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Simple signal generation logic
            signals = []
            confidence = 0.5
            action = 'hold'
            
            # RSI signals
            if latest['rsi'] < 30:  # Oversold
                signals.append('rsi_oversold')
                confidence += 0.2
                action = 'buy'
            elif latest['rsi'] > 70:  # Overbought
                signals.append('rsi_overbought')
                confidence += 0.2
                action = 'sell'
            
            # MACD signals
            if latest['macd'] > latest['macd_signal']:
                signals.append('macd_bullish')
                confidence += 0.1
                if action == 'hold':
                    action = 'buy'
            elif latest['macd'] < latest['macd_signal']:
                signals.append('macd_bearish')
                confidence += 0.1
                if action == 'hold':
                    action = 'sell'
            
            # Bollinger Bands signals
            if latest['close'] < latest['bb_lower']:
                signals.append('bb_oversold')
                confidence += 0.15
                if action == 'hold':
                    action = 'buy'
            elif latest['close'] > latest['bb_upper']:
                signals.append('bb_overbought')
                confidence += 0.15
                if action == 'hold':
                    action = 'sell'
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=min(confidence, 0.95),  # Cap at 95%
                price=latest['close'],
                timestamp=int(latest['timestamp']),
                indicators={
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'macd_signal': latest['macd_signal'],
                    'bb_upper': latest['bb_upper'],
                    'bb_lower': latest['bb_lower'],
                    'sma_20': latest['sma_20'],
                    'ema_12': latest['ema_12'],
                    'signals': signals
                }
            )
            
            # Store the signal
            self.trading_signals[symbol] = signal
            
            # Notify callbacks
            await self._notify_callbacks('signal', signal)
            
            logger.info(f"🎯 {symbol}: {action.upper()} signal (confidence: {confidence:.1%})")
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    async def _notify_callbacks(self, data_type: str, data: Any):
        """Notify registered callbacks of new data"""
        for callback in self.update_callbacks:
            try:
                await callback(data_type, data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    def add_update_callback(self, callback):
        """Add a callback for data updates"""
        self.update_callbacks.append(callback)
    
    def get_latest_data(self, symbol: str = None) -> Dict:
        """Get latest market data"""
        if symbol:
            # Get data for specific symbol
            result = {}
            for key, data in self.market_data.items():
                if symbol in key:
                    result[key] = data
            return result
        else:
            # Get all data
            return self.market_data.copy()
    
    def get_latest_signals(self, symbol: str = None) -> Dict:
        """Get latest trading signals"""
        if symbol:
            return {symbol: self.trading_signals.get(symbol)}
        else:
            return self.trading_signals.copy()
    
    def stop(self):
        """Stop the data service"""
        self.is_running = False
        logger.info("🛑 Real-time data service stopped")

# Global instance
realtime_service = RealTimeDataService()

async def start_realtime_service():
    """Start the real-time data service"""
    await realtime_service.start_data_feeds()

if __name__ == "__main__":
    # Test the service
    async def test_callback(data_type, data):
        if data_type == 'signal':
            print(f"Signal: {data.symbol} - {data.action} ({data.confidence:.1%})")
        elif data_type == 'ticker':
            print(f"Ticker: {data.symbol} - ${data.close:.2f}")
    
    realtime_service.add_update_callback(test_callback)
    asyncio.run(start_realtime_service())

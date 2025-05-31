#!/usr/bin/env python3
"""
Multi-Asset Trading Engine for Enhanced SmartMarketOOPS
Supports crypto, forex, stocks, commodities, and cross-asset arbitrage
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
import ccxt
from forex_python.converter import CurrencyRates

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class enumeration"""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    COMMODITIES = "commodities"
    INDICES = "indices"


@dataclass
class AssetInfo:
    """Asset information structure"""
    symbol: str
    asset_class: AssetClass
    exchange: str
    base_currency: str
    quote_currency: str
    min_trade_size: float
    tick_size: float
    trading_hours: Dict[str, Any]
    margin_requirement: float = 1.0


class MultiAssetDataProvider:
    """Unified data provider for multiple asset classes"""

    def __init__(self):
        """Initialize multi-asset data provider"""
        self.crypto_exchanges = {}
        self.forex_converter = CurrencyRates()
        self.stock_provider = yf

        # Initialize crypto exchanges
        self.init_crypto_exchanges()

        # Asset configurations
        self.asset_configs = self.load_asset_configurations()

        logger.info("Multi-Asset Data Provider initialized")

    def init_crypto_exchanges(self):
        """Initialize cryptocurrency exchanges"""
        try:
            # Binance
            self.crypto_exchanges['binance'] = ccxt.binance({
                'apiKey': '',  # Add API keys as needed
                'secret': '',
                'sandbox': True,  # Use testnet
                'enableRateLimit': True,
            })

            # KuCoin
            self.crypto_exchanges['kucoin'] = ccxt.kucoin({
                'apiKey': '',
                'secret': '',
                'password': '',
                'sandbox': True,
                'enableRateLimit': True,
            })

            logger.info("Crypto exchanges initialized")

        except Exception as e:
            logger.error(f"Error initializing crypto exchanges: {e}")

    def load_asset_configurations(self) -> Dict[str, AssetInfo]:
        """Load asset configurations"""
        configs = {}

        # Cryptocurrency assets
        crypto_assets = [
            ('BTCUSDT', 'binance', 'BTC', 'USDT'),
            ('ETHUSDT', 'binance', 'ETH', 'USDT'),
            ('SOLUSDT', 'binance', 'SOL', 'USDT'),
            ('ADAUSDT', 'binance', 'ADA', 'USDT'),
            ('DOTUSDT', 'binance', 'DOT', 'USDT'),
            ('LINKUSDT', 'binance', 'LINK', 'USDT'),
        ]

        for symbol, exchange, base, quote in crypto_assets:
            configs[symbol] = AssetInfo(
                symbol=symbol,
                asset_class=AssetClass.CRYPTO,
                exchange=exchange,
                base_currency=base,
                quote_currency=quote,
                min_trade_size=0.001,
                tick_size=0.01,
                trading_hours={'24/7': True},
                margin_requirement=0.1
            )

        # Forex pairs
        forex_pairs = [
            ('EURUSD', 'EUR', 'USD'),
            ('GBPUSD', 'GBP', 'USD'),
            ('USDJPY', 'USD', 'JPY'),
            ('AUDUSD', 'AUD', 'USD'),
            ('USDCAD', 'USD', 'CAD'),
            ('USDCHF', 'USD', 'CHF'),
        ]

        for symbol, base, quote in forex_pairs:
            configs[symbol] = AssetInfo(
                symbol=symbol,
                asset_class=AssetClass.FOREX,
                exchange='forex',
                base_currency=base,
                quote_currency=quote,
                min_trade_size=1000,
                tick_size=0.0001,
                trading_hours={'weekdays': True},
                margin_requirement=0.02
            )

        # Stock assets
        stock_symbols = [
            ('AAPL', 'NASDAQ'),
            ('GOOGL', 'NASDAQ'),
            ('MSFT', 'NASDAQ'),
            ('TSLA', 'NASDAQ'),
            ('AMZN', 'NASDAQ'),
            ('NVDA', 'NASDAQ'),
        ]

        for symbol, exchange in stock_symbols:
            configs[symbol] = AssetInfo(
                symbol=symbol,
                asset_class=AssetClass.STOCKS,
                exchange=exchange,
                base_currency=symbol,
                quote_currency='USD',
                min_trade_size=1,
                tick_size=0.01,
                trading_hours={'market_hours': True},
                margin_requirement=0.25
            )

        # Commodity assets (via ETFs)
        commodity_etfs = [
            ('GLD', 'Gold'),
            ('SLV', 'Silver'),
            ('USO', 'Oil'),
            ('UNG', 'Natural Gas'),
            ('DBA', 'Agriculture'),
            ('PDBC', 'Commodities'),
        ]

        for symbol, commodity in commodity_etfs:
            configs[symbol] = AssetInfo(
                symbol=symbol,
                asset_class=AssetClass.COMMODITIES,
                exchange='NYSE',
                base_currency=symbol,
                quote_currency='USD',
                min_trade_size=1,
                tick_size=0.01,
                trading_hours={'market_hours': True},
                margin_requirement=0.5
            )

        return configs

    async def get_market_data(self, symbol: str, timeframe: str = '1h',
                            limit: int = 100) -> pd.DataFrame:
        """Get market data for any asset class"""

        if symbol not in self.asset_configs:
            raise ValueError(f"Unknown symbol: {symbol}")

        asset_info = self.asset_configs[symbol]

        try:
            if asset_info.asset_class == AssetClass.CRYPTO:
                return await self.get_crypto_data(symbol, timeframe, limit)
            elif asset_info.asset_class == AssetClass.FOREX:
                return await self.get_forex_data(symbol, timeframe, limit)
            elif asset_info.asset_class == AssetClass.STOCKS:
                return await self.get_stock_data(symbol, timeframe, limit)
            elif asset_info.asset_class == AssetClass.COMMODITIES:
                return await self.get_commodity_data(symbol, timeframe, limit)
            else:
                raise ValueError(f"Unsupported asset class: {asset_info.asset_class}")

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    async def get_crypto_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get cryptocurrency data"""
        asset_info = self.asset_configs[symbol]
        exchange = self.crypto_exchanges.get(asset_info.exchange)

        if not exchange:
            # Generate synthetic data as fallback
            return self.generate_synthetic_data(symbol, limit)

        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return self.generate_synthetic_data(symbol, limit)

    async def get_forex_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get forex data"""
        try:
            # Use yfinance for forex data
            ticker = f"{symbol}=X"
            data = yf.download(ticker, period="1mo", interval="1h", progress=False)

            if data.empty:
                return self.generate_synthetic_data(symbol, limit)

            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]
            data['volume'] = data.get('volume', 1000000)  # Default volume for forex

            return data.tail(limit)

        except Exception as e:
            logger.error(f"Error fetching forex data for {symbol}: {e}")
            return self.generate_synthetic_data(symbol, limit)

    async def get_stock_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get stock data"""
        try:
            # Use yfinance for stock data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo", interval="1h")

            if data.empty:
                return self.generate_synthetic_data(symbol, limit)

            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]

            return data.tail(limit)

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return self.generate_synthetic_data(symbol, limit)

    async def get_commodity_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get commodity data (via ETFs)"""
        # Commodity data is fetched same as stocks (ETFs)
        return await self.get_stock_data(symbol, timeframe, limit)

    def generate_synthetic_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Generate synthetic market data as fallback"""
        np.random.seed(hash(symbol) % 2**32)

        # Base price depends on asset class
        asset_info = self.asset_configs.get(symbol)
        if asset_info:
            if asset_info.asset_class == AssetClass.CRYPTO:
                base_price = 45000 if 'BTC' in symbol else 2500
            elif asset_info.asset_class == AssetClass.FOREX:
                base_price = 1.1
            elif asset_info.asset_class == AssetClass.STOCKS:
                base_price = 150
            else:
                base_price = 100
        else:
            base_price = 100

        # Generate price series
        returns = np.random.normal(0, 0.02, limit)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1H')

        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, limit))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, limit)
        }, index=dates)

        return data


class CrossAssetArbitrageDetector:
    """Detects arbitrage opportunities across different asset classes"""

    def __init__(self, data_provider: MultiAssetDataProvider):
        """Initialize arbitrage detector"""
        self.data_provider = data_provider
        self.correlation_matrix = {}
        self.arbitrage_opportunities = []

        # Arbitrage pairs and relationships
        self.arbitrage_pairs = [
            # Crypto-Stock correlations
            ('BTCUSDT', 'TSLA', 'correlation'),
            ('ETHUSDT', 'NVDA', 'correlation'),

            # Forex-Commodity relationships
            ('EURUSD', 'GLD', 'inverse_correlation'),
            ('USDJPY', 'USO', 'correlation'),

            # Cross-crypto arbitrage
            ('BTCUSDT', 'ETHUSDT', 'ratio_arbitrage'),

            # Stock-Commodity relationships
            ('AAPL', 'GLD', 'safe_haven'),
        ]

        logger.info("Cross-Asset Arbitrage Detector initialized")

    async def detect_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """Detect current arbitrage opportunities"""
        opportunities = []

        for asset1, asset2, relationship_type in self.arbitrage_pairs:
            try:
                opportunity = await self.analyze_pair_arbitrage(asset1, asset2, relationship_type)
                if opportunity:
                    opportunities.append(opportunity)

            except Exception as e:
                logger.error(f"Error analyzing arbitrage for {asset1}-{asset2}: {e}")

        self.arbitrage_opportunities = opportunities
        return opportunities

    async def analyze_pair_arbitrage(self, asset1: str, asset2: str,
                                   relationship_type: str) -> Optional[Dict[str, Any]]:
        """Analyze arbitrage opportunity between two assets"""

        # Get market data for both assets
        data1 = await self.data_provider.get_market_data(asset1, '1h', 100)
        data2 = await self.data_provider.get_market_data(asset2, '1h', 100)

        if data1.empty or data2.empty:
            return None

        # Align data by timestamp
        common_index = data1.index.intersection(data2.index)
        if len(common_index) < 20:
            return None

        data1_aligned = data1.loc[common_index]
        data2_aligned = data2.loc[common_index]

        # Calculate returns
        returns1 = data1_aligned['close'].pct_change().dropna()
        returns2 = data2_aligned['close'].pct_change().dropna()

        # Analyze based on relationship type
        if relationship_type == 'correlation':
            return self.analyze_correlation_arbitrage(asset1, asset2, returns1, returns2, data1_aligned, data2_aligned)
        elif relationship_type == 'inverse_correlation':
            return self.analyze_inverse_correlation_arbitrage(asset1, asset2, returns1, returns2, data1_aligned, data2_aligned)
        elif relationship_type == 'ratio_arbitrage':
            return self.analyze_ratio_arbitrage(asset1, asset2, data1_aligned, data2_aligned)
        elif relationship_type == 'safe_haven':
            return self.analyze_safe_haven_arbitrage(asset1, asset2, returns1, returns2, data1_aligned, data2_aligned)

        return None

    def analyze_correlation_arbitrage(self, asset1: str, asset2: str,
                                    returns1: pd.Series, returns2: pd.Series,
                                    data1: pd.DataFrame, data2: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze correlation-based arbitrage"""

        # Calculate correlation
        correlation = returns1.corr(returns2)

        if abs(correlation) < 0.5:  # Minimum correlation threshold
            return None

        # Calculate z-score of current spread
        spread = returns1.iloc[-20:] - returns2.iloc[-20:]
        spread_mean = spread.mean()
        spread_std = spread.std()

        if spread_std == 0:
            return None

        current_spread = returns1.iloc[-1] - returns2.iloc[-1]
        z_score = (current_spread - spread_mean) / spread_std

        # Check for arbitrage opportunity
        if abs(z_score) > 2.0:  # 2 standard deviations
            signal = 'SELL_1_BUY_2' if z_score > 0 else 'BUY_1_SELL_2'
            confidence = min(abs(z_score) / 3.0, 1.0)

            return {
                'type': 'correlation_arbitrage',
                'asset1': asset1,
                'asset2': asset2,
                'signal': signal,
                'confidence': confidence,
                'z_score': z_score,
                'correlation': correlation,
                'expected_return': abs(z_score) * spread_std,
                'timestamp': datetime.now()
            }

        return None

    def analyze_inverse_correlation_arbitrage(self, asset1: str, asset2: str,
                                            returns1: pd.Series, returns2: pd.Series,
                                            data1: pd.DataFrame, data2: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze inverse correlation arbitrage"""

        # Calculate correlation (should be negative)
        correlation = returns1.corr(returns2)

        if correlation > -0.3:  # Should be negatively correlated
            return None

        # Calculate z-score of current sum (should be near zero for inverse correlation)
        spread = returns1.iloc[-20:] + returns2.iloc[-20:]  # Sum instead of difference
        spread_mean = spread.mean()
        spread_std = spread.std()

        if spread_std == 0:
            return None

        current_spread = returns1.iloc[-1] + returns2.iloc[-1]
        z_score = (current_spread - spread_mean) / spread_std

        # Check for arbitrage opportunity
        if abs(z_score) > 1.5:
            signal = 'SELL_BOTH' if z_score > 0 else 'BUY_BOTH'
            confidence = min(abs(z_score) / 2.5, 1.0)

            return {
                'type': 'inverse_correlation_arbitrage',
                'asset1': asset1,
                'asset2': asset2,
                'signal': signal,
                'confidence': confidence,
                'z_score': z_score,
                'correlation': correlation,
                'expected_return': abs(z_score) * spread_std,
                'timestamp': datetime.now()
            }

        return None

    def analyze_ratio_arbitrage(self, asset1: str, asset2: str,
                              data1: pd.DataFrame, data2: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze ratio-based arbitrage"""

        # Calculate price ratio
        ratio = data1['close'] / data2['close']
        ratio_mean = ratio.iloc[-50:].mean()
        ratio_std = ratio.iloc[-50:].std()

        if ratio_std == 0:
            return None

        current_ratio = ratio.iloc[-1]
        z_score = (current_ratio - ratio_mean) / ratio_std

        # Check for arbitrage opportunity
        if abs(z_score) > 2.0:
            signal = 'SELL_1_BUY_2' if z_score > 0 else 'BUY_1_SELL_2'
            confidence = min(abs(z_score) / 3.0, 1.0)

            return {
                'type': 'ratio_arbitrage',
                'asset1': asset1,
                'asset2': asset2,
                'signal': signal,
                'confidence': confidence,
                'z_score': z_score,
                'current_ratio': current_ratio,
                'mean_ratio': ratio_mean,
                'expected_return': abs(z_score) * ratio_std / ratio_mean,
                'timestamp': datetime.now()
            }

        return None

    def analyze_safe_haven_arbitrage(self, asset1: str, asset2: str,
                                   returns1: pd.Series, returns2: pd.Series,
                                   data1: pd.DataFrame, data2: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze safe haven arbitrage (flight to quality)"""

        # Calculate volatility of asset1 (risk asset)
        volatility1 = returns1.iloc[-20:].std()
        volatility_threshold = returns1.iloc[-100:].std() * 1.5

        # Check if volatility is elevated (risk-off environment)
        if volatility1 > volatility_threshold:
            # In risk-off environment, expect asset2 (safe haven) to outperform
            recent_performance1 = returns1.iloc[-5:].sum()
            recent_performance2 = returns2.iloc[-5:].sum()

            performance_diff = recent_performance1 - recent_performance2

            if performance_diff > 0.02:  # Risk asset outperforming despite high volatility
                return {
                    'type': 'safe_haven_arbitrage',
                    'asset1': asset1,
                    'asset2': asset2,
                    'signal': 'SELL_1_BUY_2',
                    'confidence': min(volatility1 / volatility_threshold, 1.0),
                    'volatility_ratio': volatility1 / volatility_threshold,
                    'performance_diff': performance_diff,
                    'expected_return': performance_diff * 0.5,
                    'timestamp': datetime.now()
                }

        return None


class UnifiedAssetManager:
    """Unified asset management across all asset classes"""

    def __init__(self):
        """Initialize unified asset manager"""
        self.data_provider = MultiAssetDataProvider()
        self.arbitrage_detector = CrossAssetArbitrageDetector(self.data_provider)

        self.active_positions = {}
        self.portfolio_allocation = {}
        self.risk_limits = {
            AssetClass.CRYPTO: 0.4,      # 40% max allocation
            AssetClass.FOREX: 0.3,       # 30% max allocation
            AssetClass.STOCKS: 0.2,      # 20% max allocation
            AssetClass.COMMODITIES: 0.1, # 10% max allocation
        }

        logger.info("Unified Asset Manager initialized")

    async def get_multi_asset_signals(self) -> Dict[str, Any]:
        """Get trading signals across all asset classes"""
        signals = {}

        # Get signals for each asset class
        for asset_class in AssetClass:
            class_signals = await self.get_asset_class_signals(asset_class)
            signals[asset_class.value] = class_signals

        # Get arbitrage opportunities
        arbitrage_opportunities = await self.arbitrage_detector.detect_arbitrage_opportunities()
        signals['arbitrage'] = arbitrage_opportunities

        return signals

    async def get_asset_class_signals(self, asset_class: AssetClass) -> List[Dict[str, Any]]:
        """Get signals for specific asset class"""
        signals = []

        # Get assets for this class
        assets = [symbol for symbol, info in self.data_provider.asset_configs.items()
                 if info.asset_class == asset_class]

        for asset in assets[:5]:  # Limit to top 5 assets per class
            try:
                # Get market data
                data = await self.data_provider.get_market_data(asset, '1h', 50)

                if not data.empty:
                    # Generate simple technical signal
                    signal = self.generate_technical_signal(asset, data)
                    if signal:
                        signals.append(signal)

            except Exception as e:
                logger.error(f"Error generating signal for {asset}: {e}")

        return signals

    def generate_technical_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate technical trading signal"""
        if len(data) < 20:
            return None

        # Simple moving average crossover
        sma_short = data['close'].rolling(window=10).mean()
        sma_long = data['close'].rolling(window=20).mean()

        current_short = sma_short.iloc[-1]
        current_long = sma_long.iloc[-1]
        prev_short = sma_short.iloc[-2]
        prev_long = sma_long.iloc[-2]

        # Check for crossover
        if prev_short <= prev_long and current_short > current_long:
            signal = 'BUY'
            confidence = 0.7
        elif prev_short >= prev_long and current_short < current_long:
            signal = 'SELL'
            confidence = 0.7
        else:
            return None

        # Calculate additional metrics
        volatility = data['close'].pct_change().std()
        volume_trend = data['volume'].iloc[-5:].mean() / data['volume'].iloc[-20:-5].mean()

        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'price': data['close'].iloc[-1],
            'volatility': volatility,
            'volume_trend': volume_trend,
            'timestamp': datetime.now()
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary across all asset classes"""
        summary = {
            'total_assets': len(self.data_provider.asset_configs),
            'asset_classes': {ac.value: 0 for ac in AssetClass},
            'active_positions': len(self.active_positions),
            'allocation_by_class': self.portfolio_allocation.copy(),
            'risk_limits': self.risk_limits.copy()
        }

        # Count assets by class
        for asset_info in self.data_provider.asset_configs.values():
            summary['asset_classes'][asset_info.asset_class.value] += 1

        return summary


class OptionsGreeksCalculator:
    """Calculate options Greeks for derivatives trading"""

    def __init__(self):
        """Initialize options Greeks calculator"""
        self.risk_free_rate = 0.05  # 5% risk-free rate

    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price"""
        from scipy.stats import norm
        import math

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return call_price

    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put option price"""
        from scipy.stats import norm
        import math

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price

    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """Calculate all Greeks for an option"""
        from scipy.stats import norm
        import math

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))

        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                    - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                    + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

        # Vega
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100

        # Rho
        if option_type == 'call':
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }


class OptionsStrategy:
    """Options trading strategies"""

    def __init__(self):
        """Initialize options strategy"""
        self.greeks_calculator = OptionsGreeksCalculator()

    def covered_call_strategy(self, underlying_price: float, strike_price: float,
                            time_to_expiry: float, volatility: float) -> Dict[str, Any]:
        """Covered call strategy analysis"""

        # Calculate option price and Greeks
        call_price = self.greeks_calculator.black_scholes_call(
            underlying_price, strike_price, time_to_expiry, 0.05, volatility
        )

        greeks = self.greeks_calculator.calculate_greeks(
            underlying_price, strike_price, time_to_expiry, 0.05, volatility, 'call'
        )

        # Strategy analysis
        max_profit = strike_price - underlying_price + call_price
        max_loss = underlying_price - call_price
        breakeven = underlying_price - call_price

        return {
            'strategy': 'covered_call',
            'call_price': call_price,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'greeks': greeks,
            'profit_probability': 0.6 if strike_price > underlying_price else 0.4
        }

    def protective_put_strategy(self, underlying_price: float, strike_price: float,
                              time_to_expiry: float, volatility: float) -> Dict[str, Any]:
        """Protective put strategy analysis"""

        # Calculate option price and Greeks
        put_price = self.greeks_calculator.black_scholes_put(
            underlying_price, strike_price, time_to_expiry, 0.05, volatility
        )

        greeks = self.greeks_calculator.calculate_greeks(
            underlying_price, strike_price, time_to_expiry, 0.05, volatility, 'put'
        )

        # Strategy analysis
        max_loss = underlying_price - strike_price + put_price
        breakeven = underlying_price + put_price

        return {
            'strategy': 'protective_put',
            'put_price': put_price,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'greeks': greeks,
            'protection_level': strike_price / underlying_price
        }


async def main():
    """Test multi-asset trading engine"""
    # Initialize unified asset manager
    asset_manager = UnifiedAssetManager()

    # Get multi-asset signals
    signals = await asset_manager.get_multi_asset_signals()

    print("🌍 Multi-Asset Trading Signals:")
    for asset_class, class_signals in signals.items():
        if asset_class != 'arbitrage':
            print(f"\n{asset_class.upper()}:")
            for signal in class_signals:
                print(f"  {signal['symbol']}: {signal['signal']} (confidence: {signal['confidence']:.2f})")

    print(f"\nArbitrage Opportunities: {len(signals['arbitrage'])}")
    for arb in signals['arbitrage']:
        print(f"  {arb['type']}: {arb['asset1']}-{arb['asset2']} ({arb['signal']})")

    # Test options strategy
    options_strategy = OptionsStrategy()
    covered_call = options_strategy.covered_call_strategy(
        underlying_price=150, strike_price=155, time_to_expiry=0.25, volatility=0.25
    )
    print(f"\nCovered Call Strategy: Max Profit=${covered_call['max_profit']:.2f}")

    # Get portfolio summary
    summary = asset_manager.get_portfolio_summary()
    print(f"\nPortfolio Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())

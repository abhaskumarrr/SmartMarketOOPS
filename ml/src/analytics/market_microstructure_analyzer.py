#!/usr/bin/env python3
"""
Market Microstructure Analysis Engine for Enhanced SmartMarketOOPS
Analyzes order book dynamics, liquidity, and price discovery mechanisms
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import deque
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Order book snapshot data structure"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    mid_price: float
    spread: float
    total_bid_volume: float
    total_ask_volume: float


@dataclass
class LiquidityMetrics:
    """Liquidity measurement metrics"""
    timestamp: datetime
    symbol: str
    bid_ask_spread: float
    spread_bps: float
    market_depth: float
    price_impact_1pct: float
    price_impact_5pct: float
    order_book_imbalance: float
    effective_spread: float
    realized_spread: float
    liquidity_score: float


@dataclass
class MicrostructureSignal:
    """Microstructure trading signal"""
    timestamp: datetime
    symbol: str
    signal_type: str
    signal_strength: float
    confidence: float
    expected_duration: int  # minutes
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderBookAnalyzer:
    """Analyzes order book dynamics and liquidity"""
    
    def __init__(self, max_history: int = 1000):
        """Initialize order book analyzer"""
        self.max_history = max_history
        self.order_book_history = {}
        self.liquidity_history = {}
        
        logger.info("Order Book Analyzer initialized")
    
    def analyze_order_book(self, order_book: OrderBookSnapshot) -> LiquidityMetrics:
        """Analyze order book for liquidity metrics"""
        
        # Store order book
        if order_book.symbol not in self.order_book_history:
            self.order_book_history[order_book.symbol] = deque(maxlen=self.max_history)
        self.order_book_history[order_book.symbol].append(order_book)
        
        # Calculate liquidity metrics
        metrics = self._calculate_liquidity_metrics(order_book)
        
        # Store metrics
        if order_book.symbol not in self.liquidity_history:
            self.liquidity_history[order_book.symbol] = deque(maxlen=self.max_history)
        self.liquidity_history[order_book.symbol].append(metrics)
        
        return metrics
    
    def _calculate_liquidity_metrics(self, order_book: OrderBookSnapshot) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics"""
        
        # Basic spread metrics
        bid_ask_spread = order_book.spread
        mid_price = order_book.mid_price
        spread_bps = (bid_ask_spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Market depth analysis
        market_depth = self._calculate_market_depth(order_book)
        
        # Price impact analysis
        price_impact_1pct = self._calculate_price_impact(order_book, 0.01)
        price_impact_5pct = self._calculate_price_impact(order_book, 0.05)
        
        # Order book imbalance
        order_book_imbalance = self._calculate_order_book_imbalance(order_book)
        
        # Effective and realized spreads
        effective_spread = self._calculate_effective_spread(order_book)
        realized_spread = self._calculate_realized_spread(order_book)
        
        # Overall liquidity score
        liquidity_score = self._calculate_liquidity_score(
            spread_bps, market_depth, price_impact_1pct, order_book_imbalance
        )
        
        return LiquidityMetrics(
            timestamp=order_book.timestamp,
            symbol=order_book.symbol,
            bid_ask_spread=bid_ask_spread,
            spread_bps=spread_bps,
            market_depth=market_depth,
            price_impact_1pct=price_impact_1pct,
            price_impact_5pct=price_impact_5pct,
            order_book_imbalance=order_book_imbalance,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            liquidity_score=liquidity_score
        )
    
    def _calculate_market_depth(self, order_book: OrderBookSnapshot) -> float:
        """Calculate market depth (volume within 1% of mid price)"""
        mid_price = order_book.mid_price
        depth_threshold = mid_price * 0.01  # 1% from mid
        
        # Sum volume within threshold
        bid_depth = sum(size for price, size in order_book.bids 
                       if price >= mid_price - depth_threshold)
        ask_depth = sum(size for price, size in order_book.asks 
                       if price <= mid_price + depth_threshold)
        
        return bid_depth + ask_depth
    
    def _calculate_price_impact(self, order_book: OrderBookSnapshot, trade_size_pct: float) -> float:
        """Calculate price impact for given trade size percentage"""
        mid_price = order_book.mid_price
        trade_value = mid_price * trade_size_pct
        
        # Calculate impact for buy order (walking up the ask side)
        cumulative_volume = 0
        weighted_price = 0
        
        for price, size in sorted(order_book.asks):
            volume_needed = min(size, trade_value - cumulative_volume)
            weighted_price += price * volume_needed
            cumulative_volume += volume_needed
            
            if cumulative_volume >= trade_value:
                break
        
        if cumulative_volume > 0:
            avg_execution_price = weighted_price / cumulative_volume
            price_impact = (avg_execution_price - mid_price) / mid_price
        else:
            price_impact = float('inf')  # No liquidity
        
        return price_impact
    
    def _calculate_order_book_imbalance(self, order_book: OrderBookSnapshot) -> float:
        """Calculate order book imbalance"""
        total_bid_volume = order_book.total_bid_volume
        total_ask_volume = order_book.total_ask_volume
        
        if total_bid_volume + total_ask_volume == 0:
            return 0.0
        
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        return imbalance
    
    def _calculate_effective_spread(self, order_book: OrderBookSnapshot) -> float:
        """Calculate effective spread (actual transaction cost)"""
        # Simplified calculation - in practice would use actual trade data
        return order_book.spread * 0.8  # Assume 80% of quoted spread
    
    def _calculate_realized_spread(self, order_book: OrderBookSnapshot) -> float:
        """Calculate realized spread (post-trade price movement)"""
        # Simplified calculation - would need future price data
        symbol = order_book.symbol
        if symbol in self.order_book_history and len(self.order_book_history[symbol]) > 10:
            recent_spreads = [ob.spread for ob in list(self.order_book_history[symbol])[-10:]]
            return np.mean(recent_spreads) * 0.6  # Estimate
        return order_book.spread * 0.6
    
    def _calculate_liquidity_score(self, spread_bps: float, market_depth: float, 
                                 price_impact: float, imbalance: float) -> float:
        """Calculate overall liquidity score (0-100)"""
        
        # Normalize components (lower is better for spread and impact, higher for depth)
        spread_score = max(0, 100 - spread_bps * 10)  # Penalize wide spreads
        depth_score = min(100, market_depth * 10)  # Reward depth
        impact_score = max(0, 100 - price_impact * 10000)  # Penalize high impact
        balance_score = max(0, 100 - abs(imbalance) * 100)  # Penalize imbalance
        
        # Weighted average
        liquidity_score = (
            spread_score * 0.3 +
            depth_score * 0.3 +
            impact_score * 0.25 +
            balance_score * 0.15
        )
        
        return min(100, max(0, liquidity_score))
    
    def detect_microstructure_signals(self, symbol: str) -> List[MicrostructureSignal]:
        """Detect trading signals from microstructure analysis"""
        signals = []
        
        if symbol not in self.liquidity_history or len(self.liquidity_history[symbol]) < 10:
            return signals
        
        recent_metrics = list(self.liquidity_history[symbol])[-10:]
        latest_metrics = recent_metrics[-1]
        
        # Signal 1: Liquidity Shock
        liquidity_signal = self._detect_liquidity_shock(recent_metrics, latest_metrics)
        if liquidity_signal:
            signals.append(liquidity_signal)
        
        # Signal 2: Order Book Imbalance
        imbalance_signal = self._detect_imbalance_signal(recent_metrics, latest_metrics)
        if imbalance_signal:
            signals.append(imbalance_signal)
        
        # Signal 3: Spread Compression/Expansion
        spread_signal = self._detect_spread_signal(recent_metrics, latest_metrics)
        if spread_signal:
            signals.append(spread_signal)
        
        return signals
    
    def _detect_liquidity_shock(self, recent_metrics: List[LiquidityMetrics], 
                              latest: LiquidityMetrics) -> Optional[MicrostructureSignal]:
        """Detect sudden liquidity changes"""
        
        avg_liquidity = np.mean([m.liquidity_score for m in recent_metrics[:-1]])
        current_liquidity = latest.liquidity_score
        
        # Check for significant liquidity drop
        liquidity_change = (current_liquidity - avg_liquidity) / avg_liquidity
        
        if liquidity_change < -0.2:  # 20% drop in liquidity
            return MicrostructureSignal(
                timestamp=latest.timestamp,
                symbol=latest.symbol,
                signal_type='liquidity_shock',
                signal_strength=abs(liquidity_change),
                confidence=0.8,
                expected_duration=15,  # 15 minutes
                metadata={
                    'liquidity_change': liquidity_change,
                    'current_score': current_liquidity,
                    'average_score': avg_liquidity
                }
            )
        
        return None
    
    def _detect_imbalance_signal(self, recent_metrics: List[LiquidityMetrics], 
                               latest: LiquidityMetrics) -> Optional[MicrostructureSignal]:
        """Detect order book imbalance signals"""
        
        current_imbalance = latest.order_book_imbalance
        
        # Strong imbalance threshold
        if abs(current_imbalance) > 0.3:  # 30% imbalance
            signal_type = 'buy_pressure' if current_imbalance > 0 else 'sell_pressure'
            
            return MicrostructureSignal(
                timestamp=latest.timestamp,
                symbol=latest.symbol,
                signal_type=signal_type,
                signal_strength=abs(current_imbalance),
                confidence=0.7,
                expected_duration=10,  # 10 minutes
                metadata={
                    'imbalance': current_imbalance,
                    'direction': 'bullish' if current_imbalance > 0 else 'bearish'
                }
            )
        
        return None
    
    def _detect_spread_signal(self, recent_metrics: List[LiquidityMetrics], 
                            latest: LiquidityMetrics) -> Optional[MicrostructureSignal]:
        """Detect spread compression/expansion signals"""
        
        avg_spread = np.mean([m.spread_bps for m in recent_metrics[:-1]])
        current_spread = latest.spread_bps
        
        spread_change = (current_spread - avg_spread) / avg_spread
        
        # Significant spread changes
        if abs(spread_change) > 0.5:  # 50% change in spread
            signal_type = 'spread_expansion' if spread_change > 0 else 'spread_compression'
            
            return MicrostructureSignal(
                timestamp=latest.timestamp,
                symbol=latest.symbol,
                signal_type=signal_type,
                signal_strength=abs(spread_change),
                confidence=0.6,
                expected_duration=20,  # 20 minutes
                metadata={
                    'spread_change': spread_change,
                    'current_spread_bps': current_spread,
                    'average_spread_bps': avg_spread
                }
            )
        
        return None
    
    def get_liquidity_summary(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """Get liquidity summary for the last N hours"""
        
        if symbol not in self.liquidity_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.liquidity_history[symbol] 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            'symbol': symbol,
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'average_liquidity_score': np.mean([m.liquidity_score for m in recent_metrics]),
            'average_spread_bps': np.mean([m.spread_bps for m in recent_metrics]),
            'average_market_depth': np.mean([m.market_depth for m in recent_metrics]),
            'average_price_impact_1pct': np.mean([m.price_impact_1pct for m in recent_metrics]),
            'liquidity_volatility': np.std([m.liquidity_score for m in recent_metrics]),
            'min_liquidity_score': np.min([m.liquidity_score for m in recent_metrics]),
            'max_liquidity_score': np.max([m.liquidity_score for m in recent_metrics]),
            'current_metrics': {
                'liquidity_score': recent_metrics[-1].liquidity_score,
                'spread_bps': recent_metrics[-1].spread_bps,
                'market_depth': recent_metrics[-1].market_depth,
                'order_book_imbalance': recent_metrics[-1].order_book_imbalance
            }
        }


class PriceDiscoveryAnalyzer:
    """Analyzes price discovery mechanisms and efficiency"""
    
    def __init__(self):
        """Initialize price discovery analyzer"""
        self.price_history = {}
        self.trade_history = {}
        
    def analyze_price_discovery(self, symbol: str, price_data: pd.DataFrame, 
                              trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price discovery efficiency"""
        
        # Price discovery metrics
        price_efficiency = self._calculate_price_efficiency(price_data)
        information_share = self._calculate_information_share(price_data, trade_data)
        price_impact_decay = self._calculate_price_impact_decay(trade_data)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price_efficiency': price_efficiency,
            'information_share': information_share,
            'price_impact_decay': price_impact_decay,
            'discovery_quality_score': self._calculate_discovery_score(
                price_efficiency, information_share, price_impact_decay
            )
        }
    
    def _calculate_price_efficiency(self, price_data: pd.DataFrame) -> float:
        """Calculate price discovery efficiency using variance ratio"""
        if len(price_data) < 20:
            return 0.0
        
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return 0.0
        
        # Variance ratio test for random walk
        # Efficient markets should have variance ratio close to 1
        var_1 = returns.var()
        var_2 = returns.rolling(window=2).sum().var() / 2
        
        variance_ratio = var_2 / var_1 if var_1 > 0 else 1
        
        # Efficiency score (closer to 1 is more efficient)
        efficiency = 1 - abs(variance_ratio - 1)
        return max(0, min(1, efficiency))
    
    def _calculate_information_share(self, price_data: pd.DataFrame, 
                                   trade_data: pd.DataFrame) -> float:
        """Calculate information share of price discovery"""
        # Simplified information share calculation
        # In practice, this would use more sophisticated econometric models
        
        if len(trade_data) < 10:
            return 0.5  # Default neutral value
        
        # Correlation between trade flow and price changes
        price_changes = price_data['close'].pct_change().dropna()
        
        # Simulate trade flow impact (in real implementation, use actual trade data)
        trade_impact = np.random.normal(0, 0.001, len(price_changes))
        
        if len(price_changes) > 0 and len(trade_impact) > 0:
            correlation = np.corrcoef(price_changes[:len(trade_impact)], trade_impact)[0, 1]
            information_share = abs(correlation) if not np.isnan(correlation) else 0
        else:
            information_share = 0
        
        return min(1, max(0, information_share))
    
    def _calculate_price_impact_decay(self, trade_data: pd.DataFrame) -> float:
        """Calculate how quickly price impact decays"""
        # Simplified decay calculation
        # In practice, would analyze actual trade impact over time
        
        if len(trade_data) < 5:
            return 0.5  # Default value
        
        # Simulate impact decay (exponential decay is healthy)
        decay_rate = 0.8  # 80% decay per period
        return decay_rate
    
    def _calculate_discovery_score(self, efficiency: float, info_share: float, 
                                 decay_rate: float) -> float:
        """Calculate overall price discovery quality score"""
        
        # Weighted combination of metrics
        discovery_score = (
            efficiency * 0.4 +
            info_share * 0.3 +
            decay_rate * 0.3
        ) * 100
        
        return min(100, max(0, discovery_score))


def create_sample_order_book(symbol: str, base_price: float) -> OrderBookSnapshot:
    """Create sample order book for testing"""
    
    # Generate realistic order book
    spread = base_price * 0.001  # 0.1% spread
    bid_price = base_price - spread / 2
    ask_price = base_price + spread / 2
    
    # Create multiple levels
    bids = []
    asks = []
    
    for i in range(10):
        bid_level_price = bid_price - (i * spread * 0.1)
        ask_level_price = ask_price + (i * spread * 0.1)
        
        bid_size = np.random.uniform(1, 10) * (1 / (i + 1))  # Decreasing size
        ask_size = np.random.uniform(1, 10) * (1 / (i + 1))
        
        bids.append((bid_level_price, bid_size))
        asks.append((ask_level_price, ask_size))
    
    total_bid_volume = sum(size for _, size in bids)
    total_ask_volume = sum(size for _, size in asks)
    
    return OrderBookSnapshot(
        timestamp=datetime.now(),
        symbol=symbol,
        bids=bids,
        asks=asks,
        mid_price=base_price,
        spread=spread,
        total_bid_volume=total_bid_volume,
        total_ask_volume=total_ask_volume
    )


async def main():
    """Test market microstructure analyzer"""
    
    # Initialize analyzer
    analyzer = OrderBookAnalyzer()
    price_discovery = PriceDiscoveryAnalyzer()
    
    # Test with sample data
    symbols = ['BTCUSDT', 'ETHUSDT']
    base_prices = {'BTCUSDT': 45000, 'ETHUSDT': 2500}
    
    # Simulate order book updates
    for _ in range(20):
        for symbol in symbols:
            # Create sample order book
            base_price = base_prices[symbol] * (1 + np.random.normal(0, 0.001))
            order_book = create_sample_order_book(symbol, base_price)
            
            # Analyze liquidity
            metrics = analyzer.analyze_order_book(order_book)
            
            # Detect signals
            signals = analyzer.detect_microstructure_signals(symbol)
            
            if signals:
                for signal in signals:
                    logger.info(f"Microstructure signal: {signal.signal_type} for {symbol} "
                               f"(strength: {signal.signal_strength:.3f})")
        
        await asyncio.sleep(0.1)  # Simulate real-time updates
    
    # Get summary
    for symbol in symbols:
        summary = analyzer.get_liquidity_summary(symbol, hours=1)
        print(f"\n📊 Liquidity Summary for {symbol}:")
        print(f"Average Liquidity Score: {summary.get('average_liquidity_score', 0):.1f}")
        print(f"Average Spread (bps): {summary.get('average_spread_bps', 0):.1f}")
        print(f"Current Liquidity Score: {summary.get('current_metrics', {}).get('liquidity_score', 0):.1f}")


if __name__ == "__main__":
    asyncio.run(main())

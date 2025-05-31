#!/usr/bin/env python3
"""
Multi-Timeframe Confluence System for Smart Money Concepts

This module implements a sophisticated multi-timeframe analysis system that provides
institutional-grade confluence analysis across multiple timeframes. The system
establishes higher timeframe bias, validates cross-timeframe signals, identifies
discount/premium zones, and provides comprehensive confluence scoring.

Key Features:
- Higher Timeframe Bias Analysis (1D, 4H, 1H)
- Cross-Timeframe Signal Validation
- Discount/Premium Zone Identification
- Confluence Scoring Engine with weighted algorithms
- Multi-Timeframe SMC Pattern Alignment
- Institutional Trading Session Analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeframeType(Enum):
    """Enumeration for timeframe types"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class BiasDirection(Enum):
    """Enumeration for market bias directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    TRANSITIONING = "transitioning"


class ZoneType(Enum):
    """Enumeration for discount/premium zone types"""
    DISCOUNT = "discount"  # Below 50% of range
    PREMIUM = "premium"    # Above 50% of range
    EQUILIBRIUM = "equilibrium"  # Around 50% of range
    EXTREME_DISCOUNT = "extreme_discount"  # Below 25% of range
    EXTREME_PREMIUM = "extreme_premium"    # Above 75% of range


class ConfluenceStrength(Enum):
    """Enumeration for confluence strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    EXTREME = "extreme"


@dataclass
class TimeframeBias:
    """Data class representing bias for a specific timeframe"""
    timeframe: TimeframeType
    direction: BiasDirection
    strength: float
    confidence: float
    key_levels: List[float]
    trend_structure: Dict[str, Any]
    last_update: datetime

    def is_bullish(self) -> bool:
        """Check if bias is bullish"""
        return self.direction == BiasDirection.BULLISH

    def is_bearish(self) -> bool:
        """Check if bias is bearish"""
        return self.direction == BiasDirection.BEARISH


@dataclass
class DiscountPremiumZone:
    """Data class representing discount/premium zones"""
    zone_type: ZoneType
    price_range: Tuple[float, float]  # (low, high)
    current_position: float  # 0.0 to 1.0 (discount to premium)
    timeframe: TimeframeType
    strength: float
    key_levels: List[float]
    formation_timestamp: datetime

    def is_discount(self) -> bool:
        """Check if current position is in discount zone"""
        return self.zone_type in [ZoneType.DISCOUNT, ZoneType.EXTREME_DISCOUNT]

    def is_premium(self) -> bool:
        """Check if current position is in premium zone"""
        return self.zone_type in [ZoneType.PREMIUM, ZoneType.EXTREME_PREMIUM]


@dataclass
class ConfluenceSignal:
    """Data class representing a confluence trading signal"""
    signal_type: str  # 'buy' or 'sell'
    confluence_score: float
    strength: ConfluenceStrength
    timeframe_alignment: Dict[TimeframeType, float]
    smc_confluence: Dict[str, float]
    technical_confluence: Dict[str, float]
    market_timing_score: float
    entry_zone: DiscountPremiumZone
    key_levels: List[float]
    risk_reward_ratio: float
    timestamp: datetime

    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal"""
        return self.signal_type.lower() == 'buy'

    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal"""
        return self.signal_type.lower() == 'sell'


class MultiTimeframeAnalyzer:
    """
    Advanced Multi-Timeframe Confluence Analysis System

    Provides institutional-grade multi-timeframe analysis including:
    - Higher timeframe bias establishment
    - Cross-timeframe signal validation
    - Discount/premium zone identification
    - Comprehensive confluence scoring
    """

    def __init__(self, data_sources: Dict[TimeframeType, pd.DataFrame],
                 primary_timeframe: TimeframeType = TimeframeType.M15,
                 htf_timeframes: List[TimeframeType] = None):
        """
        Initialize the Multi-Timeframe Analyzer

        Args:
            data_sources: Dictionary mapping timeframes to OHLCV DataFrames
            primary_timeframe: Primary trading timeframe (default M15)
            htf_timeframes: Higher timeframe list for bias (default [H4, D1])
        """
        self.data_sources = data_sources
        self.primary_timeframe = primary_timeframe
        self.htf_timeframes = htf_timeframes or [TimeframeType.H4, TimeframeType.D1]

        # Validate data sources
        self._validate_data_sources()

        # Initialize analysis components
        self.timeframe_biases: Dict[TimeframeType, TimeframeBias] = {}
        self.discount_premium_zones: Dict[TimeframeType, DiscountPremiumZone] = {}
        self.confluence_signals: List[ConfluenceSignal] = []

        # Import SMC components if available
        self._initialize_smc_components()

        logger.info(f"MultiTimeframeAnalyzer initialized with {len(data_sources)} timeframes")

    def _validate_data_sources(self):
        """Validate that all data sources have required columns"""
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']

        for timeframe, df in self.data_sources.items():
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Timeframe {timeframe.value} missing columns: {missing}")

            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                # Create timestamp column if not present
                df['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(minutes=len(df) * 15),
                    periods=len(df),
                    freq='15T'
                )

    def _initialize_smc_components(self):
        """Initialize SMC components for each timeframe"""
        self.smc_analyzers = {}

        try:
            # Import SMC components
            from order_block_detection import OrderBlockDetector
            from fvg_detection import FVGDetector
            from liquidity_mapping import LiquidityMapper
            from market_structure_analysis import MarketStructureAnalyzer

            # Initialize for each timeframe
            for timeframe, data in self.data_sources.items():
                self.smc_analyzers[timeframe] = {
                    'order_blocks': OrderBlockDetector(data),
                    'fvg': FVGDetector(data),
                    'liquidity': LiquidityMapper(data),
                    'structure': MarketStructureAnalyzer(data)
                }

            self.smc_available = True
            logger.info("SMC components initialized for all timeframes")

        except ImportError as e:
            logger.warning(f"SMC components not available: {e}")
            self.smc_available = False

    def establish_htf_bias(self) -> Dict[TimeframeType, TimeframeBias]:
        """
        Establish higher timeframe bias for institutional direction

        Returns:
            Dictionary of timeframe biases
        """
        logger.info("Establishing higher timeframe bias...")

        for timeframe in self.htf_timeframes:
            if timeframe not in self.data_sources:
                logger.warning(f"No data available for timeframe {timeframe.value}")
                continue

            data = self.data_sources[timeframe]
            bias = self._analyze_timeframe_bias(timeframe, data)
            self.timeframe_biases[timeframe] = bias

        logger.info(f"Established bias for {len(self.timeframe_biases)} timeframes")
        return self.timeframe_biases

    def _analyze_timeframe_bias(self, timeframe: TimeframeType, data: pd.DataFrame) -> TimeframeBias:
        """Analyze bias for a specific timeframe"""
        # Calculate trend indicators
        data = data.copy()
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        data['ema_200'] = data['close'].ewm(span=200).mean()

        # Calculate price position relative to EMAs
        current_price = data['close'].iloc[-1]
        ema_20 = data['ema_20'].iloc[-1]
        ema_50 = data['ema_50'].iloc[-1]
        ema_200 = data['ema_200'].iloc[-1]

        # Determine bias direction
        if current_price > ema_20 > ema_50 > ema_200:
            direction = BiasDirection.BULLISH
            strength = 0.8
        elif current_price < ema_20 < ema_50 < ema_200:
            direction = BiasDirection.BEARISH
            strength = 0.8
        elif current_price > ema_20 and ema_20 > ema_50:
            direction = BiasDirection.BULLISH
            strength = 0.6
        elif current_price < ema_20 and ema_20 < ema_50:
            direction = BiasDirection.BEARISH
            strength = 0.6
        else:
            direction = BiasDirection.NEUTRAL
            strength = 0.3

        # Calculate confidence based on trend consistency
        price_changes = data['close'].pct_change().dropna()
        recent_changes = price_changes.tail(20)

        if direction == BiasDirection.BULLISH:
            positive_moves = (recent_changes > 0).sum()
            confidence = positive_moves / len(recent_changes)
        elif direction == BiasDirection.BEARISH:
            negative_moves = (recent_changes < 0).sum()
            confidence = negative_moves / len(recent_changes)
        else:
            confidence = 0.5

        # Identify key levels
        key_levels = self._identify_key_levels(data)

        # Analyze trend structure using SMC if available
        trend_structure = {}
        if self.smc_available and timeframe in self.smc_analyzers:
            try:
                structure_analyzer = self.smc_analyzers[timeframe]['structure']
                market_structure = structure_analyzer.analyze_market_structure()
                trend_structure = {
                    'current_trend': market_structure.current_trend.value,
                    'trend_strength': market_structure.trend_strength,
                    'structure_quality': market_structure.structure_quality
                }
            except Exception as e:
                logger.warning(f"Error analyzing structure for {timeframe.value}: {e}")

        return TimeframeBias(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            confidence=confidence,
            key_levels=key_levels,
            trend_structure=trend_structure,
            last_update=datetime.now()
        )

    def _identify_key_levels(self, data: pd.DataFrame, lookback: int = 50) -> List[float]:
        """Identify key support and resistance levels"""
        recent_data = data.tail(lookback)

        # Find swing highs and lows
        highs = []
        lows = []

        for i in range(2, len(recent_data) - 2):
            current_high = recent_data.iloc[i]['high']
            current_low = recent_data.iloc[i]['low']

            # Check for swing high
            if (current_high > recent_data.iloc[i-1]['high'] and
                current_high > recent_data.iloc[i+1]['high'] and
                current_high > recent_data.iloc[i-2]['high'] and
                current_high > recent_data.iloc[i+2]['high']):
                highs.append(current_high)

            # Check for swing low
            if (current_low < recent_data.iloc[i-1]['low'] and
                current_low < recent_data.iloc[i+1]['low'] and
                current_low < recent_data.iloc[i-2]['low'] and
                current_low < recent_data.iloc[i+2]['low']):
                lows.append(current_low)

        # Combine and sort key levels
        key_levels = sorted(set(highs + lows))

        # Return most significant levels (max 10)
        if len(key_levels) > 10:
            # Keep levels that are most frequently tested
            level_counts = {}
            for level in key_levels:
                touches = 0
                for _, row in recent_data.iterrows():
                    if abs(row['high'] - level) / level < 0.002 or abs(row['low'] - level) / level < 0.002:
                        touches += 1
                level_counts[level] = touches

            # Sort by touch count and take top 10
            sorted_levels = sorted(level_counts.items(), key=lambda x: x[1], reverse=True)
            key_levels = [level for level, _ in sorted_levels[:10]]

        return key_levels

    def identify_discount_premium_zones(self) -> Dict[TimeframeType, DiscountPremiumZone]:
        """
        Identify discount and premium zones across timeframes

        Returns:
            Dictionary of discount/premium zones by timeframe
        """
        logger.info("Identifying discount/premium zones...")

        for timeframe, data in self.data_sources.items():
            zone = self._calculate_discount_premium_zone(timeframe, data)
            self.discount_premium_zones[timeframe] = zone

        logger.info(f"Identified zones for {len(self.discount_premium_zones)} timeframes")
        return self.discount_premium_zones

    def _calculate_discount_premium_zone(self, timeframe: TimeframeType,
                                       data: pd.DataFrame, lookback: int = 100) -> DiscountPremiumZone:
        """Calculate discount/premium zone for a specific timeframe"""
        recent_data = data.tail(lookback)

        # Calculate range from recent swing high and low
        range_high = recent_data['high'].max()
        range_low = recent_data['low'].min()
        range_size = range_high - range_low

        # Current price position
        current_price = data['close'].iloc[-1]

        # Calculate position as percentage of range (0.0 = low, 1.0 = high)
        if range_size > 0:
            position = (current_price - range_low) / range_size
        else:
            position = 0.5  # Default to equilibrium if no range

        # Determine zone type based on position
        if position <= 0.25:
            zone_type = ZoneType.EXTREME_DISCOUNT
            strength = 0.9
        elif position <= 0.4:
            zone_type = ZoneType.DISCOUNT
            strength = 0.7
        elif position <= 0.6:
            zone_type = ZoneType.EQUILIBRIUM
            strength = 0.5
        elif position <= 0.75:
            zone_type = ZoneType.PREMIUM
            strength = 0.7
        else:
            zone_type = ZoneType.EXTREME_PREMIUM
            strength = 0.9

        # Identify key levels within the zone
        key_levels = []

        # Add Fibonacci levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for fib in fib_levels:
            level = range_low + (range_size * fib)
            key_levels.append(level)

        # Add psychological levels (round numbers)
        psychological_levels = self._find_psychological_levels(range_low, range_high)
        key_levels.extend(psychological_levels)

        # Remove duplicates and sort
        key_levels = sorted(set(key_levels))

        return DiscountPremiumZone(
            zone_type=zone_type,
            price_range=(range_low, range_high),
            current_position=position,
            timeframe=timeframe,
            strength=strength,
            key_levels=key_levels,
            formation_timestamp=datetime.now()
        )

    def _find_psychological_levels(self, range_low: float, range_high: float) -> List[float]:
        """Find psychological levels (round numbers) within the range"""
        psychological_levels = []

        # Determine the appropriate increment based on price level
        if range_high > 100000:
            increment = 10000  # For very high prices (like BTC)
        elif range_high > 10000:
            increment = 1000
        elif range_high > 1000:
            increment = 100
        elif range_high > 100:
            increment = 10
        else:
            increment = 1

        # Find round numbers within the range
        start = int(range_low / increment) * increment
        end = int(range_high / increment + 1) * increment

        current = start
        while current <= end:
            if range_low <= current <= range_high:
                psychological_levels.append(float(current))
            current += increment

        return psychological_levels

    def calculate_confluence_score(self, signal_type: str = 'buy') -> ConfluenceSignal:
        """
        Calculate comprehensive confluence score for trading signals

        Args:
            signal_type: 'buy' or 'sell' signal type

        Returns:
            ConfluenceSignal with comprehensive analysis
        """
        logger.info(f"Calculating confluence score for {signal_type} signal...")

        # Ensure we have bias and zones calculated
        if not self.timeframe_biases:
            self.establish_htf_bias()
        if not self.discount_premium_zones:
            self.identify_discount_premium_zones()

        # Calculate individual confluence components
        htf_score = self._calculate_htf_confluence(signal_type)
        smc_score = self._calculate_smc_confluence(signal_type)
        technical_score = self._calculate_technical_confluence(signal_type)
        timing_score = self._calculate_market_timing_score()

        # Weighted confluence calculation (as per requirements)
        # HTF bias 40%, SMC confluence 30%, technical indicators 20%, market timing 10%
        total_score = (
            htf_score * 0.4 +
            smc_score * 0.3 +
            technical_score * 0.2 +
            timing_score * 0.1
        )

        # Determine confluence strength
        if total_score >= 0.8:
            strength = ConfluenceStrength.EXTREME
        elif total_score >= 0.7:
            strength = ConfluenceStrength.VERY_STRONG
        elif total_score >= 0.6:
            strength = ConfluenceStrength.STRONG
        elif total_score >= 0.5:
            strength = ConfluenceStrength.MODERATE
        else:
            strength = ConfluenceStrength.WEAK

        # Get timeframe alignment scores
        timeframe_alignment = {}
        for timeframe, bias in self.timeframe_biases.items():
            if signal_type == 'buy' and bias.is_bullish():
                timeframe_alignment[timeframe] = bias.strength * bias.confidence
            elif signal_type == 'sell' and bias.is_bearish():
                timeframe_alignment[timeframe] = bias.strength * bias.confidence
            else:
                timeframe_alignment[timeframe] = 0.0

        # Get entry zone (primary timeframe)
        entry_zone = self.discount_premium_zones.get(
            self.primary_timeframe,
            self._calculate_discount_premium_zone(
                self.primary_timeframe,
                self.data_sources[self.primary_timeframe]
            )
        )

        # Calculate risk/reward ratio
        risk_reward = self._calculate_risk_reward_ratio(signal_type, entry_zone)

        # Get key levels for the signal
        key_levels = self._get_confluence_key_levels(signal_type)

        return ConfluenceSignal(
            signal_type=signal_type,
            confluence_score=total_score,
            strength=strength,
            timeframe_alignment=timeframe_alignment,
            smc_confluence={
                'order_blocks': smc_score,
                'liquidity': smc_score,
                'structure': smc_score,
                'fvg': smc_score
            },
            technical_confluence={
                'trend_alignment': technical_score,
                'momentum': technical_score,
                'support_resistance': technical_score
            },
            market_timing_score=timing_score,
            entry_zone=entry_zone,
            key_levels=key_levels,
            risk_reward_ratio=risk_reward,
            timestamp=datetime.now()
        )

    def _calculate_htf_confluence(self, signal_type: str) -> float:
        """Calculate higher timeframe confluence score"""
        if not self.timeframe_biases:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        # Weight higher timeframes more heavily
        timeframe_weights = {
            TimeframeType.D1: 0.5,
            TimeframeType.H4: 0.3,
            TimeframeType.H1: 0.2
        }

        for timeframe, bias in self.timeframe_biases.items():
            weight = timeframe_weights.get(timeframe, 0.1)

            if signal_type == 'buy' and bias.is_bullish():
                score = bias.strength * bias.confidence
            elif signal_type == 'sell' and bias.is_bearish():
                score = bias.strength * bias.confidence
            else:
                score = 0.0

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_smc_confluence(self, signal_type: str) -> float:
        """Calculate Smart Money Concepts confluence score"""
        if not self.smc_available:
            return 0.5  # Neutral score if SMC not available

        total_score = 0.0
        component_count = 0

        primary_data = self.data_sources[self.primary_timeframe]

        try:
            # Order Block confluence
            if self.primary_timeframe in self.smc_analyzers:
                ob_detector = self.smc_analyzers[self.primary_timeframe]['order_blocks']
                order_blocks = ob_detector.detect_order_blocks()

                ob_score = 0.0
                current_price = primary_data['close'].iloc[-1]

                for ob in order_blocks:
                    if signal_type == 'buy' and ob.type == 'bullish':
                        if ob.bottom <= current_price <= ob.top:
                            ob_score = max(ob_score, ob.strength)
                    elif signal_type == 'sell' and ob.type == 'bearish':
                        if ob.bottom <= current_price <= ob.top:
                            ob_score = max(ob_score, ob.strength)

                total_score += ob_score
                component_count += 1

            # Liquidity confluence
            if self.primary_timeframe in self.smc_analyzers:
                liquidity_mapper = self.smc_analyzers[self.primary_timeframe]['liquidity']
                liquidity_mapper.map_all_liquidity_levels()

                liq_score = 0.0
                current_price = primary_data['close'].iloc[-1]

                # Check for nearby liquidity levels
                active_levels = liquidity_mapper.get_active_levels_near_price(current_price, 0.02)

                for level in active_levels:
                    if signal_type == 'buy' and level.type.value in ['sell_side', 'equal_lows']:
                        liq_score = max(liq_score, level.strength)
                    elif signal_type == 'sell' and level.type.value in ['buy_side', 'equal_highs']:
                        liq_score = max(liq_score, level.strength)

                total_score += liq_score
                component_count += 1

            # Market Structure confluence
            if self.primary_timeframe in self.smc_analyzers:
                structure_analyzer = self.smc_analyzers[self.primary_timeframe]['structure']
                market_structure = structure_analyzer.analyze_market_structure()

                struct_score = 0.0

                if signal_type == 'buy' and market_structure.current_trend.value == 'bullish':
                    struct_score = market_structure.trend_strength
                elif signal_type == 'sell' and market_structure.current_trend.value == 'bearish':
                    struct_score = market_structure.trend_strength

                total_score += struct_score
                component_count += 1

        except Exception as e:
            logger.warning(f"Error calculating SMC confluence: {e}")
            return 0.5

        return total_score / component_count if component_count > 0 else 0.5

    def _calculate_technical_confluence(self, signal_type: str) -> float:
        """Calculate technical analysis confluence score"""
        primary_data = self.data_sources[self.primary_timeframe].copy()

        # Calculate technical indicators
        primary_data['rsi'] = self._calculate_rsi(primary_data['close'])
        primary_data['macd'], primary_data['macd_signal'] = self._calculate_macd(primary_data['close'])
        primary_data['bb_upper'], primary_data['bb_lower'] = self._calculate_bollinger_bands(primary_data['close'])

        current_price = primary_data['close'].iloc[-1]
        rsi = primary_data['rsi'].iloc[-1]
        macd = primary_data['macd'].iloc[-1]
        macd_signal = primary_data['macd_signal'].iloc[-1]
        bb_upper = primary_data['bb_upper'].iloc[-1]
        bb_lower = primary_data['bb_lower'].iloc[-1]

        score = 0.0

        if signal_type == 'buy':
            # RSI oversold
            if rsi < 30:
                score += 0.3
            elif rsi < 50:
                score += 0.1

            # MACD bullish
            if macd > macd_signal:
                score += 0.3

            # Bollinger Bands
            if current_price < bb_lower:
                score += 0.2

            # Support/Resistance
            support_score = self._calculate_support_resistance_score(primary_data, current_price, 'support')
            score += support_score * 0.2

        else:  # sell signal
            # RSI overbought
            if rsi > 70:
                score += 0.3
            elif rsi > 50:
                score += 0.1

            # MACD bearish
            if macd < macd_signal:
                score += 0.3

            # Bollinger Bands
            if current_price > bb_upper:
                score += 0.2

            # Support/Resistance
            resistance_score = self._calculate_support_resistance_score(primary_data, current_price, 'resistance')
            score += resistance_score * 0.2

        return min(1.0, score)

    def _calculate_market_timing_score(self) -> float:
        """Calculate market timing score based on session and volatility"""
        current_time = datetime.now()

        # Market session scoring (simplified)
        hour = current_time.hour

        # London session (8-16 UTC) and NY session (13-21 UTC) overlap
        if 13 <= hour <= 16:  # London-NY overlap
            session_score = 1.0
        elif 8 <= hour <= 21:  # Active sessions
            session_score = 0.8
        elif 21 <= hour <= 24 or 0 <= hour <= 8:  # Asian session
            session_score = 0.6
        else:
            session_score = 0.4

        # Day of week scoring
        weekday = current_time.weekday()
        if weekday in [1, 2, 3]:  # Tuesday, Wednesday, Thursday
            day_score = 1.0
        elif weekday in [0, 4]:  # Monday, Friday
            day_score = 0.8
        else:  # Weekend
            day_score = 0.3

        # Combine scores
        timing_score = (session_score * 0.7 + day_score * 0.3)

        return timing_score

    def _calculate_risk_reward_ratio(self, signal_type: str, entry_zone: DiscountPremiumZone) -> float:
        """Calculate risk/reward ratio for the signal"""
        current_price = self.data_sources[self.primary_timeframe]['close'].iloc[-1]

        # Use zone range for risk/reward calculation
        range_low, range_high = entry_zone.price_range
        range_size = range_high - range_low

        if signal_type == 'buy':
            # Risk: distance to range low
            # Reward: distance to range high
            risk = current_price - range_low
            reward = range_high - current_price
        else:  # sell
            # Risk: distance to range high
            # Reward: distance to range low
            risk = range_high - current_price
            reward = current_price - range_low

        if risk > 0:
            return reward / risk
        else:
            return 1.0  # Default ratio

    def _get_confluence_key_levels(self, signal_type: str) -> List[float]:
        """Get key levels relevant to the confluence signal"""
        key_levels = []

        # Collect key levels from all timeframes
        for timeframe, bias in self.timeframe_biases.items():
            key_levels.extend(bias.key_levels)

        # Add discount/premium zone levels
        for timeframe, zone in self.discount_premium_zones.items():
            key_levels.extend(zone.key_levels)

        # Remove duplicates and sort
        key_levels = sorted(set(key_levels))

        # Filter to most relevant levels (within reasonable range)
        current_price = self.data_sources[self.primary_timeframe]['close'].iloc[-1]
        relevant_levels = []

        for level in key_levels:
            distance_pct = abs(level - current_price) / current_price
            if distance_pct <= 0.1:  # Within 10% of current price
                relevant_levels.append(level)

        return relevant_levels[:10]  # Return top 10 most relevant levels

    # Technical indicator helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()

        return macd, macd_signal

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band, lower_band

    def _calculate_support_resistance_score(self, data: pd.DataFrame, current_price: float, level_type: str) -> float:
        """Calculate support/resistance score based on nearby levels"""
        score = 0.0

        # Look for recent swing points
        lookback = min(50, len(data))
        recent_data = data.tail(lookback)

        if level_type == 'support':
            # Find swing lows near current price
            for i in range(2, len(recent_data) - 2):
                low = recent_data.iloc[i]['low']
                if (low < recent_data.iloc[i-1]['low'] and
                    low < recent_data.iloc[i+1]['low'] and
                    low < recent_data.iloc[i-2]['low'] and
                    low < recent_data.iloc[i+2]['low']):

                    distance_pct = abs(current_price - low) / current_price
                    if distance_pct <= 0.02:  # Within 2%
                        score += 0.3
                    elif distance_pct <= 0.05:  # Within 5%
                        score += 0.1

        else:  # resistance
            # Find swing highs near current price
            for i in range(2, len(recent_data) - 2):
                high = recent_data.iloc[i]['high']
                if (high > recent_data.iloc[i-1]['high'] and
                    high > recent_data.iloc[i+1]['high'] and
                    high > recent_data.iloc[i-2]['high'] and
                    high > recent_data.iloc[i+2]['high']):

                    distance_pct = abs(current_price - high) / current_price
                    if distance_pct <= 0.02:  # Within 2%
                        score += 0.3
                    elif distance_pct <= 0.05:  # Within 5%
                        score += 0.1

        return min(1.0, score)

    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive multi-timeframe analysis

        Returns:
            Dictionary with complete analysis results
        """
        logger.info("Performing comprehensive multi-timeframe analysis...")

        # Establish HTF bias
        htf_biases = self.establish_htf_bias()

        # Identify discount/premium zones
        zones = self.identify_discount_premium_zones()

        # Calculate confluence for both directions
        buy_confluence = self.calculate_confluence_score('buy')
        sell_confluence = self.calculate_confluence_score('sell')

        # Determine best signal
        best_signal = buy_confluence if buy_confluence.confluence_score > sell_confluence.confluence_score else sell_confluence

        # Compile comprehensive analysis
        analysis = {
            'timestamp': datetime.now(),
            'primary_timeframe': self.primary_timeframe.value,
            'htf_biases': {tf.value: {
                'direction': bias.direction.value,
                'strength': bias.strength,
                'confidence': bias.confidence,
                'key_levels': bias.key_levels
            } for tf, bias in htf_biases.items()},
            'discount_premium_zones': {tf.value: {
                'zone_type': zone.zone_type.value,
                'current_position': zone.current_position,
                'strength': zone.strength,
                'price_range': zone.price_range,
                'key_levels': zone.key_levels
            } for tf, zone in zones.items()},
            'confluence_signals': {
                'buy': {
                    'score': buy_confluence.confluence_score,
                    'strength': buy_confluence.strength.value,
                    'timeframe_alignment': {tf.value: score for tf, score in buy_confluence.timeframe_alignment.items()},
                    'risk_reward_ratio': buy_confluence.risk_reward_ratio,
                    'key_levels': buy_confluence.key_levels
                },
                'sell': {
                    'score': sell_confluence.confluence_score,
                    'strength': sell_confluence.strength.value,
                    'timeframe_alignment': {tf.value: score for tf, score in sell_confluence.timeframe_alignment.items()},
                    'risk_reward_ratio': sell_confluence.risk_reward_ratio,
                    'key_levels': sell_confluence.key_levels
                }
            },
            'best_signal': {
                'type': best_signal.signal_type,
                'score': best_signal.confluence_score,
                'strength': best_signal.strength.value,
                'entry_zone': best_signal.entry_zone.zone_type.value,
                'risk_reward_ratio': best_signal.risk_reward_ratio
            },
            'market_timing_score': self._calculate_market_timing_score(),
            'smc_available': self.smc_available
        }

        logger.info(f"Comprehensive analysis complete. Best signal: {best_signal.signal_type} ({best_signal.confluence_score:.3f})")
        return analysis

    def get_confluence_statistics(self) -> Dict[str, Any]:
        """Get statistics about confluence analysis"""
        if not self.timeframe_biases or not self.discount_premium_zones:
            return {
                'total_timeframes': len(self.data_sources),
                'htf_biases_established': 0,
                'zones_identified': 0,
                'average_bias_strength': 0.0,
                'average_zone_strength': 0.0,
                'bullish_timeframes': 0,
                'bearish_timeframes': 0,
                'neutral_timeframes': 0
            }

        # Calculate bias statistics
        bias_strengths = [bias.strength for bias in self.timeframe_biases.values()]
        avg_bias_strength = np.mean(bias_strengths) if bias_strengths else 0.0

        bullish_count = sum(1 for bias in self.timeframe_biases.values() if bias.is_bullish())
        bearish_count = sum(1 for bias in self.timeframe_biases.values() if bias.is_bearish())
        neutral_count = len(self.timeframe_biases) - bullish_count - bearish_count

        # Calculate zone statistics
        zone_strengths = [zone.strength for zone in self.discount_premium_zones.values()]
        avg_zone_strength = np.mean(zone_strengths) if zone_strengths else 0.0

        return {
            'total_timeframes': len(self.data_sources),
            'htf_biases_established': len(self.timeframe_biases),
            'zones_identified': len(self.discount_premium_zones),
            'average_bias_strength': avg_bias_strength,
            'average_zone_strength': avg_zone_strength,
            'bullish_timeframes': bullish_count,
            'bearish_timeframes': bearish_count,
            'neutral_timeframes': neutral_count,
            'zone_distribution': {
                zone.zone_type.value: 1 for zone in self.discount_premium_zones.values()
            }
        }


# Utility functions for integration and testing
def create_sample_multi_timeframe_data() -> Dict[TimeframeType, pd.DataFrame]:
    """
    Create sample multi-timeframe OHLCV data for testing

    Returns:
        Dictionary mapping timeframes to sample DataFrames
    """
    np.random.seed(42)  # For reproducible results

    base_price = 50000
    base_volume = 1000000

    # Generate base 1-minute data
    num_minutes = 5000  # About 83 hours of data to ensure sufficient higher timeframe data
    m1_data = []

    # Create trending phases
    trend_phases = [
        {'type': 'bullish', 'length': 500, 'strength': 0.0002},
        {'type': 'bearish', 'length': 400, 'strength': -0.00015},
        {'type': 'consolidation', 'length': 600, 'strength': 0.00005},
        {'type': 'bullish', 'length': 500, 'strength': 0.00025}
    ]

    current_phase = 0
    phase_progress = 0

    for i in range(num_minutes):
        # Determine current trend phase
        if current_phase < len(trend_phases):
            phase = trend_phases[current_phase]

            if phase_progress >= phase['length']:
                current_phase += 1
                phase_progress = 0
                if current_phase < len(trend_phases):
                    phase = trend_phases[current_phase]

        # Generate price movement
        if current_phase < len(trend_phases):
            trend_move = phase['strength'] * (1 + np.random.normal(0, 0.5))
        else:
            trend_move = np.random.normal(0, 0.0001)

        # Add noise
        noise = np.random.normal(0, 0.0005)
        total_move = trend_move + noise

        new_price = base_price * (1 + total_move)

        # Generate realistic OHLC
        volatility = new_price * 0.002

        open_price = base_price + np.random.normal(0, volatility * 0.3)
        close_price = new_price

        high_wick = abs(np.random.normal(0, volatility * 0.4))
        low_wick = abs(np.random.normal(0, volatility * 0.4))

        high_price = max(open_price, close_price) + high_wick
        low_price = min(open_price, close_price) - low_wick

        # Generate volume
        volume = base_volume * np.random.uniform(0.5, 2.0)

        timestamp = pd.Timestamp.now() - pd.Timedelta(minutes=(num_minutes - i))

        m1_data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

        base_price = close_price
        phase_progress += 1

    m1_df = pd.DataFrame(m1_data)

    # Generate higher timeframe data by resampling
    timeframe_data = {}

    # Set timestamp as index for resampling
    m1_df.set_index('timestamp', inplace=True)

    # Generate different timeframes
    timeframes = {
        TimeframeType.M1: '1T',
        TimeframeType.M5: '5T',
        TimeframeType.M15: '15T',
        TimeframeType.M30: '30T',
        TimeframeType.H1: '1H',
        TimeframeType.H4: '4H'
    }

    for tf_type, resample_rule in timeframes.items():
        if tf_type == TimeframeType.M1:
            # Use original 1-minute data
            df = m1_df.copy()
        else:
            # Resample to higher timeframe
            df = m1_df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        # Reset index to have timestamp as column
        df.reset_index(inplace=True)
        timeframe_data[tf_type] = df

    return timeframe_data


def get_enhanced_multi_timeframe_analysis(data_sources: Dict[TimeframeType, pd.DataFrame],
                                        **kwargs) -> Dict[str, Any]:
    """
    Convenience function to get enhanced multi-timeframe analysis

    Args:
        data_sources: Dictionary mapping timeframes to OHLCV DataFrames
        **kwargs: Additional parameters for MultiTimeframeAnalyzer

    Returns:
        Dictionary with comprehensive multi-timeframe analysis
    """
    analyzer = MultiTimeframeAnalyzer(data_sources, **kwargs)
    analysis = analyzer.get_comprehensive_analysis()
    statistics = analyzer.get_confluence_statistics()

    # Add statistics to analysis
    analysis['statistics'] = statistics

    return analysis


# Example usage and testing
if __name__ == "__main__":
    print("Multi-Timeframe Confluence System - Testing")
    print("=" * 50)

    # Create sample multi-timeframe data
    print("Creating sample multi-timeframe OHLCV data...")
    data_sources = create_sample_multi_timeframe_data()

    print(f"Generated data for {len(data_sources)} timeframes:")
    for tf, df in data_sources.items():
        print(f"  {tf.value}: {len(df)} candles")

    # Initialize Multi-Timeframe Analyzer
    print("\nInitializing Multi-Timeframe Analyzer...")
    analyzer = MultiTimeframeAnalyzer(
        data_sources=data_sources,
        primary_timeframe=TimeframeType.M15,
        htf_timeframes=[TimeframeType.H4, TimeframeType.H1]
    )

    # Perform comprehensive analysis
    print("Performing comprehensive multi-timeframe analysis...")
    analysis = analyzer.get_comprehensive_analysis()

    # Get statistics
    stats = analyzer.get_confluence_statistics()

    # Print results
    print(f"\nMulti-Timeframe Analysis Results:")
    print(f"Primary timeframe: {analysis['primary_timeframe']}")
    print(f"Market timing score: {analysis['market_timing_score']:.3f}")
    print(f"SMC components available: {analysis['smc_available']}")

    print(f"\nHigher Timeframe Biases:")
    for tf, bias in analysis['htf_biases'].items():
        print(f"  {tf}: {bias['direction']} (strength: {bias['strength']:.3f}, confidence: {bias['confidence']:.3f})")

    print(f"\nDiscount/Premium Zones:")
    for tf, zone in analysis['discount_premium_zones'].items():
        print(f"  {tf}: {zone['zone_type']} (position: {zone['current_position']:.1%}, strength: {zone['strength']:.3f})")

    print(f"\nConfluence Signals:")
    buy_signal = analysis['confluence_signals']['buy']
    sell_signal = analysis['confluence_signals']['sell']
    print(f"  Buy: {buy_signal['score']:.3f} ({buy_signal['strength']}) - R:R {buy_signal['risk_reward_ratio']:.2f}")
    print(f"  Sell: {sell_signal['score']:.3f} ({sell_signal['strength']}) - R:R {sell_signal['risk_reward_ratio']:.2f}")

    best_signal = analysis['best_signal']
    print(f"\nBest Signal: {best_signal['type'].upper()}")
    print(f"  Score: {best_signal['score']:.3f}")
    print(f"  Strength: {best_signal['strength']}")
    print(f"  Entry zone: {best_signal['entry_zone']}")
    print(f"  Risk/Reward: {best_signal['risk_reward_ratio']:.2f}")

    print(f"\nStatistics:")
    print(f"Total timeframes: {stats['total_timeframes']}")
    print(f"HTF biases established: {stats['htf_biases_established']}")
    print(f"Zones identified: {stats['zones_identified']}")
    print(f"Average bias strength: {stats['average_bias_strength']:.3f}")
    print(f"Average zone strength: {stats['average_zone_strength']:.3f}")
    print(f"Bullish timeframes: {stats['bullish_timeframes']}")
    print(f"Bearish timeframes: {stats['bearish_timeframes']}")
    print(f"Neutral timeframes: {stats['neutral_timeframes']}")

    # Test convenience function
    print("\nTesting convenience function...")
    enhanced_analysis = get_enhanced_multi_timeframe_analysis(data_sources)
    print(f"Convenience function analysis complete. Best signal: {enhanced_analysis['best_signal']['type']} ({enhanced_analysis['best_signal']['score']:.3f})")

    print("\n✅ Multi-Timeframe Confluence System testing complete!")
    print("The system successfully analyzed multiple timeframes and provided confluence scoring.")
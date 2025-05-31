#!/usr/bin/env python3
"""
Fair Value Gap (FVG) Detection System for Smart Money Concepts

This module implements a sophisticated Fair Value Gap detection engine that identifies
institutional imbalances in price action. FVGs represent areas where price moved so
quickly that it left gaps in the market structure, creating zones of inefficiency
that often act as support and resistance levels.

Key Features:
- Advanced FVG identification algorithms
- Impulse strength validation
- FVG fill tracking and monitoring
- Multi-timeframe FVG analysis
- Reaction strength measurement
- Statistical analysis and reporting
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class FVGType(Enum):
    """Enumeration for Fair Value Gap types"""
    BULLISH = "bullish"
    BEARISH = "bearish"


class FVGStatus(Enum):
    """Enumeration for FVG status"""
    ACTIVE = "active"
    PARTIALLY_FILLED = "partially_filled"
    FULLY_FILLED = "fully_filled"
    EXPIRED = "expired"


@dataclass
class FairValueGap:
    """Data class representing a Fair Value Gap"""
    type: FVGType
    top: float
    bottom: float
    size: float
    percentage: float
    formation_index: int
    formation_timestamp: datetime
    candle_1_index: int
    candle_2_index: int
    candle_3_index: int
    impulse_strength: float
    volume_context: Dict[str, float]
    status: FVGStatus = FVGStatus.ACTIVE
    fill_percentage: float = 0.0
    first_touch_index: Optional[int] = None
    first_touch_timestamp: Optional[datetime] = None
    full_fill_index: Optional[int] = None
    full_fill_timestamp: Optional[datetime] = None
    reaction_strength: float = 0.0
    touches: int = 0
    max_reaction: float = 0.0
    validation_score: float = 0.0

    def get_midpoint(self) -> float:
        """Get the midpoint of the FVG"""
        return (self.top + self.bottom) / 2

    def get_size_percentage(self) -> float:
        """Get the size of the FVG as a percentage of price"""
        return self.percentage

    def is_price_in_fvg(self, price: float, tolerance: float = 0.0001) -> bool:
        """Check if a price is within the FVG boundaries"""
        tolerance_range = self.size * tolerance
        return (self.bottom - tolerance_range) <= price <= (self.top + tolerance_range)

    def calculate_fill_percentage(self, current_high: float, current_low: float) -> float:
        """Calculate how much of the FVG has been filled"""
        if self.type == FVGType.BULLISH:
            # For bullish FVG, filling happens from bottom up
            if current_low <= self.bottom:
                return 100.0  # Fully filled
            elif current_low < self.top:
                filled_amount = self.top - current_low
                return (filled_amount / self.size) * 100.0
        else:  # Bearish FVG
            # For bearish FVG, filling happens from top down
            if current_high >= self.top:
                return 100.0  # Fully filled
            elif current_high > self.bottom:
                filled_amount = current_high - self.bottom
                return (filled_amount / self.size) * 100.0

        return 0.0


class FVGDetector:
    """
    Advanced Fair Value Gap Detection Engine

    Detects institutional Fair Value Gaps based on:
    - Three-candle pattern analysis
    - Impulse strength validation
    - Volume confirmation
    - Market structure context
    - Fill tracking and monitoring
    """

    def __init__(self, ohlcv: pd.DataFrame, min_gap_percentage: float = 0.001,
                 min_impulse_strength: float = 0.015, volume_threshold_percentile: float = 70):
        """
        Initialize the FVG Detector

        Args:
            ohlcv: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            min_gap_percentage: Minimum gap size as percentage of price (default 0.1%)
            min_impulse_strength: Minimum impulse strength for FVG validation (default 1.5%)
            volume_threshold_percentile: Volume percentile threshold for validation (default 70th)
        """
        self.ohlcv = ohlcv.copy()
        self.min_gap_percentage = min_gap_percentage
        self.min_impulse_strength = min_impulse_strength
        self.volume_threshold_percentile = volume_threshold_percentile

        # Ensure timestamp column is datetime
        if 'timestamp' in self.ohlcv.columns:
            self.ohlcv['timestamp'] = pd.to_datetime(self.ohlcv['timestamp'])
        else:
            # Create timestamp column if not present
            self.ohlcv['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(minutes=len(self.ohlcv) * 15),
                periods=len(self.ohlcv),
                freq='15T'
            )

        # Calculate technical indicators
        self._calculate_indicators()

        # Storage for detected FVGs
        self.fvgs: List[FairValueGap] = []

    def _calculate_indicators(self):
        """Calculate technical indicators needed for FVG detection"""
        # Volume threshold for validation
        self.volume_threshold = np.percentile(self.ohlcv['volume'], self.volume_threshold_percentile)

        # Calculate price movement metrics
        self.ohlcv['price_change_pct'] = self.ohlcv['close'].pct_change()
        self.ohlcv['high_change_pct'] = self.ohlcv['high'].pct_change()
        self.ohlcv['low_change_pct'] = self.ohlcv['low'].pct_change()

        # Calculate Average True Range for context
        self._calculate_atr()

        # Calculate candle characteristics
        self.ohlcv['body_size'] = abs(self.ohlcv['close'] - self.ohlcv['open'])
        self.ohlcv['upper_wick'] = self.ohlcv['high'] - np.maximum(self.ohlcv['open'], self.ohlcv['close'])
        self.ohlcv['lower_wick'] = np.minimum(self.ohlcv['open'], self.ohlcv['close']) - self.ohlcv['low']
        self.ohlcv['total_range'] = self.ohlcv['high'] - self.ohlcv['low']

        # Calculate impulse strength indicators
        self.ohlcv['bullish_impulse'] = np.where(
            self.ohlcv['close'] > self.ohlcv['open'],
            (self.ohlcv['close'] - self.ohlcv['open']) / self.ohlcv['open'],
            0
        )
        self.ohlcv['bearish_impulse'] = np.where(
            self.ohlcv['close'] < self.ohlcv['open'],
            (self.ohlcv['open'] - self.ohlcv['close']) / self.ohlcv['open'],
            0
        )

    def _calculate_atr(self, period: int = 14):
        """Calculate Average True Range"""
        high_low = self.ohlcv['high'] - self.ohlcv['low']
        high_close_prev = abs(self.ohlcv['high'] - self.ohlcv['close'].shift(1))
        low_close_prev = abs(self.ohlcv['low'] - self.ohlcv['close'].shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        self.ohlcv['atr'] = true_range.rolling(window=period).mean()

    def identify_fvgs(self, validate_impulse: bool = True) -> List[FairValueGap]:
        """
        Identify Fair Value Gaps in the price data

        Args:
            validate_impulse: Whether to validate impulse strength (default True)

        Returns:
            List of detected Fair Value Gaps
        """
        self.fvgs = []

        # Need at least 3 candles to detect FVG
        for i in range(2, len(self.ohlcv)):
            # Get the three candles for FVG analysis
            candle_1 = self.ohlcv.iloc[i-2]  # First candle
            candle_2 = self.ohlcv.iloc[i-1]  # Middle candle (impulse candle)
            candle_3 = self.ohlcv.iloc[i]    # Third candle

            # Detect bullish FVG
            bullish_fvg = self._detect_bullish_fvg(candle_1, candle_2, candle_3, i-2, i-1, i)
            if bullish_fvg:
                if not validate_impulse or self._validate_fvg_impulse(bullish_fvg):
                    self.fvgs.append(bullish_fvg)

            # Detect bearish FVG
            bearish_fvg = self._detect_bearish_fvg(candle_1, candle_2, candle_3, i-2, i-1, i)
            if bearish_fvg:
                if not validate_impulse or self._validate_fvg_impulse(bearish_fvg):
                    self.fvgs.append(bearish_fvg)

        # Validate and score FVGs
        self._validate_and_score_fvgs()

        return self.fvgs

    def _detect_bullish_fvg(self, candle_1: pd.Series, candle_2: pd.Series, candle_3: pd.Series,
                           idx_1: int, idx_2: int, idx_3: int) -> Optional[FairValueGap]:
        """
        Detect bullish Fair Value Gap

        A bullish FVG occurs when:
        - The low of candle 3 is higher than the high of candle 1
        - Candle 2 is typically a strong bullish impulse candle
        - Creates an unfilled gap that acts as support
        """
        # Check for gap condition: candle_3_low > candle_1_high
        if candle_3['low'] <= candle_1['high']:
            return None

        gap_size = candle_3['low'] - candle_1['high']
        gap_percentage = gap_size / candle_1['high']

        # Check minimum gap size
        if gap_percentage < self.min_gap_percentage:
            return None

        # Calculate impulse strength of the middle candle
        impulse_strength = self._calculate_impulse_strength(candle_2, 'bullish')

        # Get volume context
        volume_context = self._get_volume_context([candle_1, candle_2, candle_3])

        # Create FVG object
        fvg = FairValueGap(
            type=FVGType.BULLISH,
            top=candle_3['low'],
            bottom=candle_1['high'],
            size=gap_size,
            percentage=gap_percentage,
            formation_index=idx_2,  # Middle candle index
            formation_timestamp=candle_2['timestamp'],
            candle_1_index=idx_1,
            candle_2_index=idx_2,
            candle_3_index=idx_3,
            impulse_strength=impulse_strength,
            volume_context=volume_context
        )

        return fvg

    def _detect_bearish_fvg(self, candle_1: pd.Series, candle_2: pd.Series, candle_3: pd.Series,
                           idx_1: int, idx_2: int, idx_3: int) -> Optional[FairValueGap]:
        """
        Detect bearish Fair Value Gap

        A bearish FVG occurs when:
        - The high of candle 3 is lower than the low of candle 1
        - Candle 2 is typically a strong bearish impulse candle
        - Creates an unfilled gap that acts as resistance
        """
        # Check for gap condition: candle_3_high < candle_1_low
        if candle_3['high'] >= candle_1['low']:
            return None

        gap_size = candle_1['low'] - candle_3['high']
        gap_percentage = gap_size / candle_1['low']

        # Check minimum gap size
        if gap_percentage < self.min_gap_percentage:
            return None

        # Calculate impulse strength of the middle candle
        impulse_strength = self._calculate_impulse_strength(candle_2, 'bearish')

        # Get volume context
        volume_context = self._get_volume_context([candle_1, candle_2, candle_3])

        # Create FVG object
        fvg = FairValueGap(
            type=FVGType.BEARISH,
            top=candle_1['low'],
            bottom=candle_3['high'],
            size=gap_size,
            percentage=gap_percentage,
            formation_index=idx_2,  # Middle candle index
            formation_timestamp=candle_2['timestamp'],
            candle_1_index=idx_1,
            candle_2_index=idx_2,
            candle_3_index=idx_3,
            impulse_strength=impulse_strength,
            volume_context=volume_context
        )

        return fvg

    def _calculate_impulse_strength(self, candle: pd.Series, direction: str) -> float:
        """Calculate the impulse strength of a candle"""
        if direction == 'bullish':
            # For bullish impulse, measure the strength of upward movement
            body_strength = (candle['close'] - candle['open']) / candle['open'] if candle['open'] > 0 else 0
            range_strength = (candle['high'] - candle['low']) / candle['low'] if candle['low'] > 0 else 0
            return max(body_strength, range_strength * 0.7)  # Weight body more than range
        else:
            # For bearish impulse, measure the strength of downward movement
            body_strength = (candle['open'] - candle['close']) / candle['open'] if candle['open'] > 0 else 0
            range_strength = (candle['high'] - candle['low']) / candle['high'] if candle['high'] > 0 else 0
            return max(body_strength, range_strength * 0.7)  # Weight body more than range

    def _get_volume_context(self, candles: List[pd.Series]) -> Dict[str, float]:
        """Get volume context for the FVG formation"""
        volumes = [candle['volume'] for candle in candles]
        avg_volume = np.mean(volumes)
        max_volume = max(volumes)

        return {
            'average_volume': avg_volume,
            'max_volume': max_volume,
            'volume_ratio': max_volume / self.volume_threshold if self.volume_threshold > 0 else 1.0,
            'above_threshold': max_volume > self.volume_threshold
        }

    def _validate_fvg_impulse(self, fvg: FairValueGap) -> bool:
        """Validate that the FVG has sufficient impulse strength"""
        return fvg.impulse_strength >= self.min_impulse_strength

    def _validate_and_score_fvgs(self):
        """Validate and score all detected FVGs"""
        for fvg in self.fvgs:
            score = self._calculate_fvg_validation_score(fvg)
            fvg.validation_score = score

    def _calculate_fvg_validation_score(self, fvg: FairValueGap) -> float:
        """
        Calculate validation score for an FVG based on multiple factors

        Scoring factors (0-1 scale):
        - Gap size relative to ATR (0-0.25)
        - Impulse strength (0-0.3)
        - Volume confirmation (0-0.25)
        - Gap percentage (0-0.2)
        """
        score = 0.0

        # Get ATR at formation time
        formation_candle = self.ohlcv.iloc[fvg.formation_index]
        atr_value = formation_candle['atr'] if not pd.isna(formation_candle['atr']) else fvg.size

        # Gap size factor (0-0.25)
        if atr_value > 0:
            gap_atr_ratio = fvg.size / atr_value
            gap_factor = min(0.25, gap_atr_ratio * 0.1)
            score += gap_factor

        # Impulse strength factor (0-0.3)
        impulse_factor = min(0.3, fvg.impulse_strength * 15)  # Scale to 0-0.3
        score += impulse_factor

        # Volume factor (0-0.25)
        volume_factor = min(0.25, fvg.volume_context['volume_ratio'] * 0.25)
        score += volume_factor

        # Gap percentage factor (0-0.2)
        percentage_factor = min(0.2, fvg.percentage * 100)  # Scale percentage to 0-0.2
        score += percentage_factor

        return min(1.0, score)

    def track_fvg_fills(self, start_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Track FVG fills and update their status

        Args:
            start_index: Index to start tracking from (default: formation index of each FVG)

        Returns:
            Dictionary with fill tracking statistics
        """
        fill_stats = {
            'total_fvgs': len(self.fvgs),
            'active_fvgs': 0,
            'partially_filled': 0,
            'fully_filled': 0,
            'expired': 0,
            'average_fill_time': 0,
            'fill_success_rate': 0
        }

        fill_times = []

        for fvg in self.fvgs:
            # Determine tracking start index
            track_start = start_index if start_index is not None else fvg.formation_index + 1

            # Track fills from formation onwards
            for i in range(track_start, len(self.ohlcv)):
                candle = self.ohlcv.iloc[i]

                # Calculate current fill percentage
                fill_pct = fvg.calculate_fill_percentage(candle['high'], candle['low'])

                # Update FVG status based on fill percentage
                if fill_pct >= 100.0:
                    if fvg.status != FVGStatus.FULLY_FILLED:
                        fvg.status = FVGStatus.FULLY_FILLED
                        fvg.full_fill_index = i
                        fvg.full_fill_timestamp = candle['timestamp']
                        fill_times.append(i - fvg.formation_index)
                    break
                elif fill_pct > 0:
                    if fvg.status == FVGStatus.ACTIVE:
                        fvg.status = FVGStatus.PARTIALLY_FILLED
                        fvg.first_touch_index = i
                        fvg.first_touch_timestamp = candle['timestamp']

                    # Update fill percentage and touches
                    if fill_pct > fvg.fill_percentage:
                        fvg.fill_percentage = fill_pct
                        fvg.touches += 1

                # Check for price reaction at FVG
                self._measure_fvg_reaction(fvg, i)

        # Calculate statistics
        for fvg in self.fvgs:
            if fvg.status == FVGStatus.ACTIVE:
                fill_stats['active_fvgs'] += 1
            elif fvg.status == FVGStatus.PARTIALLY_FILLED:
                fill_stats['partially_filled'] += 1
            elif fvg.status == FVGStatus.FULLY_FILLED:
                fill_stats['fully_filled'] += 1
            elif fvg.status == FVGStatus.EXPIRED:
                fill_stats['expired'] += 1

        # Calculate averages
        if fill_times:
            fill_stats['average_fill_time'] = np.mean(fill_times)

        if fill_stats['total_fvgs'] > 0:
            fill_stats['fill_success_rate'] = fill_stats['fully_filled'] / fill_stats['total_fvgs']

        return fill_stats

    def _measure_fvg_reaction(self, fvg: FairValueGap, current_index: int):
        """Measure price reaction strength at FVG levels"""
        if current_index >= len(self.ohlcv) - 1:
            return

        current_candle = self.ohlcv.iloc[current_index]

        # Check if price is interacting with FVG
        if fvg.is_price_in_fvg(current_candle['low']) or fvg.is_price_in_fvg(current_candle['high']):
            # Look ahead to measure reaction
            lookforward = min(10, len(self.ohlcv) - current_index - 1)

            if fvg.type == FVGType.BULLISH:
                # For bullish FVG, measure upward reaction
                max_high = current_candle['high']
                for j in range(1, lookforward + 1):
                    if current_index + j < len(self.ohlcv):
                        future_high = self.ohlcv.iloc[current_index + j]['high']
                        max_high = max(max_high, future_high)

                reaction = (max_high - current_candle['low']) / current_candle['low']
                fvg.max_reaction = max(fvg.max_reaction, reaction)

            else:  # Bearish FVG
                # For bearish FVG, measure downward reaction
                min_low = current_candle['low']
                for j in range(1, lookforward + 1):
                    if current_index + j < len(self.ohlcv):
                        future_low = self.ohlcv.iloc[current_index + j]['low']
                        min_low = min(min_low, future_low)

                reaction = (current_candle['high'] - min_low) / current_candle['high']
                fvg.max_reaction = max(fvg.max_reaction, reaction)

    def get_fvg_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about detected FVGs"""
        if not self.fvgs:
            return {
                'total_fvgs': 0,
                'bullish_fvgs': 0,
                'bearish_fvgs': 0,
                'average_gap_size': 0,
                'average_validation_score': 0,
                'average_impulse_strength': 0,
                'status_distribution': {},
                'size_distribution': {},
                'reaction_statistics': {}
            }

        # Basic counts
        total_fvgs = len(self.fvgs)
        bullish_fvgs = len([fvg for fvg in self.fvgs if fvg.type == FVGType.BULLISH])
        bearish_fvgs = len([fvg for fvg in self.fvgs if fvg.type == FVGType.BEARISH])

        # Average metrics
        avg_gap_size = np.mean([fvg.size for fvg in self.fvgs])
        avg_validation_score = np.mean([fvg.validation_score for fvg in self.fvgs])
        avg_impulse_strength = np.mean([fvg.impulse_strength for fvg in self.fvgs])

        # Status distribution
        status_counts = {}
        for status in FVGStatus:
            status_counts[status.value] = len([fvg for fvg in self.fvgs if fvg.status == status])

        # Size distribution
        gap_percentages = [fvg.percentage for fvg in self.fvgs]
        size_distribution = {
            'small': len([p for p in gap_percentages if p < 0.002]),  # < 0.2%
            'medium': len([p for p in gap_percentages if 0.002 <= p < 0.005]),  # 0.2% - 0.5%
            'large': len([p for p in gap_percentages if p >= 0.005])  # >= 0.5%
        }

        # Reaction statistics
        reactions = [fvg.max_reaction for fvg in self.fvgs if fvg.max_reaction > 0]
        reaction_stats = {
            'average_reaction': np.mean(reactions) if reactions else 0,
            'max_reaction': max(reactions) if reactions else 0,
            'fvgs_with_reaction': len(reactions),
            'reaction_rate': len(reactions) / total_fvgs if total_fvgs > 0 else 0
        }

        return {
            'total_fvgs': total_fvgs,
            'bullish_fvgs': bullish_fvgs,
            'bearish_fvgs': bearish_fvgs,
            'average_gap_size': avg_gap_size,
            'average_validation_score': avg_validation_score,
            'average_impulse_strength': avg_impulse_strength,
            'status_distribution': status_counts,
            'size_distribution': size_distribution,
            'reaction_statistics': reaction_stats
        }

    def get_active_fvgs(self, price_level: Optional[float] = None) -> List[FairValueGap]:
        """
        Get currently active FVGs, optionally filtered by proximity to price level

        Args:
            price_level: Optional price level to filter FVGs by proximity

        Returns:
            List of active FVGs
        """
        active_fvgs = [fvg for fvg in self.fvgs if fvg.status == FVGStatus.ACTIVE]

        if price_level is not None:
            # Filter by proximity to current price (within 5% by default)
            proximity_threshold = 0.05
            filtered_fvgs = []

            for fvg in active_fvgs:
                fvg_midpoint = fvg.get_midpoint()
                distance_pct = abs(price_level - fvg_midpoint) / price_level

                if distance_pct <= proximity_threshold:
                    filtered_fvgs.append(fvg)

            return filtered_fvgs

        return active_fvgs

    def validate_fvg_quality(self, fvg: FairValueGap) -> Dict[str, Any]:
        """
        Validate the quality of a specific FVG

        Args:
            fvg: FairValueGap to validate

        Returns:
            Dictionary with validation details
        """
        validation = {
            'overall_score': fvg.validation_score,
            'criteria': {
                'sufficient_gap_size': fvg.percentage >= self.min_gap_percentage,
                'strong_impulse': fvg.impulse_strength >= self.min_impulse_strength,
                'volume_confirmation': fvg.volume_context['above_threshold'],
                'good_validation_score': fvg.validation_score >= 0.5
            },
            'metrics': {
                'gap_percentage': fvg.percentage,
                'impulse_strength': fvg.impulse_strength,
                'volume_ratio': fvg.volume_context['volume_ratio'],
                'validation_score': fvg.validation_score
            },
            'status_info': {
                'current_status': fvg.status.value,
                'fill_percentage': fvg.fill_percentage,
                'touches': fvg.touches,
                'max_reaction': fvg.max_reaction
            }
        }

        # Overall quality assessment
        criteria_met = sum(validation['criteria'].values())
        validation['quality_level'] = 'high' if criteria_met >= 3 else 'medium' if criteria_met >= 2 else 'low'

        return validation


# Utility functions for integration with existing SMC system
def create_sample_fvg_data(num_candles: int = 500) -> pd.DataFrame:
    """
    Create sample OHLCV data with intentional FVGs for testing

    Args:
        num_candles: Number of candles to generate

    Returns:
        DataFrame with OHLCV data containing FVGs
    """
    np.random.seed(42)  # For reproducible results

    base_price = 50000
    data = []

    for i in range(num_candles):
        # Create normal price movement most of the time
        if i % 50 == 0 and i > 10:  # Create FVG every 50 candles
            # Create a gap by making an impulsive move
            if np.random.random() > 0.5:  # Bullish FVG
                # Candle 1: Normal candle
                open_1 = base_price
                close_1 = base_price * (1 + np.random.uniform(-0.01, 0.01))
                high_1 = max(open_1, close_1) * (1 + np.random.uniform(0, 0.005))
                low_1 = min(open_1, close_1) * (1 - np.random.uniform(0, 0.005))

                data.append({
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i) * 15),
                    'open': open_1,
                    'high': high_1,
                    'low': low_1,
                    'close': close_1,
                    'volume': np.random.uniform(800000, 1200000)
                })

                # Candle 2: Strong bullish impulse
                open_2 = close_1
                close_2 = open_2 * (1 + np.random.uniform(0.02, 0.04))  # Strong move up
                high_2 = close_2 * (1 + np.random.uniform(0, 0.01))
                low_2 = open_2 * (1 - np.random.uniform(0, 0.005))

                data.append({
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i - 1) * 15),
                    'open': open_2,
                    'high': high_2,
                    'low': low_2,
                    'close': close_2,
                    'volume': np.random.uniform(1500000, 2500000)  # High volume
                })

                # Candle 3: Gap up (creates FVG)
                gap_size = high_1 * np.random.uniform(0.002, 0.008)  # 0.2% to 0.8% gap
                open_3 = high_1 + gap_size
                close_3 = open_3 * (1 + np.random.uniform(-0.01, 0.02))
                high_3 = max(open_3, close_3) * (1 + np.random.uniform(0, 0.01))
                low_3 = min(open_3, close_3)

                data.append({
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i - 2) * 15),
                    'open': open_3,
                    'high': high_3,
                    'low': low_3,
                    'close': close_3,
                    'volume': np.random.uniform(1000000, 1500000)
                })

                base_price = close_3
                i += 2  # Skip next iterations as we've created 3 candles

            else:  # Bearish FVG
                # Similar logic for bearish FVG
                open_1 = base_price
                close_1 = base_price * (1 + np.random.uniform(-0.01, 0.01))
                high_1 = max(open_1, close_1) * (1 + np.random.uniform(0, 0.005))
                low_1 = min(open_1, close_1) * (1 - np.random.uniform(0, 0.005))

                data.append({
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i) * 15),
                    'open': open_1,
                    'high': high_1,
                    'low': low_1,
                    'close': close_1,
                    'volume': np.random.uniform(800000, 1200000)
                })

                # Strong bearish impulse
                open_2 = close_1
                close_2 = open_2 * (1 - np.random.uniform(0.02, 0.04))
                high_2 = open_2 * (1 + np.random.uniform(0, 0.005))
                low_2 = close_2 * (1 - np.random.uniform(0, 0.01))

                data.append({
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i - 1) * 15),
                    'open': open_2,
                    'high': high_2,
                    'low': low_2,
                    'close': close_2,
                    'volume': np.random.uniform(1500000, 2500000)
                })

                # Gap down
                gap_size = low_1 * np.random.uniform(0.002, 0.008)
                open_3 = low_1 - gap_size
                close_3 = open_3 * (1 + np.random.uniform(-0.02, 0.01))
                high_3 = max(open_3, close_3)
                low_3 = min(open_3, close_3) * (1 - np.random.uniform(0, 0.01))

                data.append({
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i - 2) * 15),
                    'open': open_3,
                    'high': high_3,
                    'low': low_3,
                    'close': close_3,
                    'volume': np.random.uniform(1000000, 1500000)
                })

                base_price = close_3
                i += 2

        else:
            # Normal candle
            volatility = base_price * 0.01
            open_price = base_price + np.random.normal(0, volatility * 0.5)
            close_price = open_price * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))

            data.append({
                'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i) * 15),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.uniform(800000, 1500000)
            })

            base_price = close_price

    return pd.DataFrame(data)


def get_enhanced_fvgs(ohlcv: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to get enhanced FVGs in dictionary format

    Args:
        ohlcv: OHLCV DataFrame
        **kwargs: Additional parameters for FVGDetector

    Returns:
        List of detected FVGs in dictionary format
    """
    detector = FVGDetector(ohlcv, **kwargs)
    fvgs = detector.identify_fvgs()

    # Track fills
    detector.track_fvg_fills()

    # Convert to dictionary format
    result = []
    for fvg in fvgs:
        result.append({
            'type': fvg.type.value,
            'top': fvg.top,
            'bottom': fvg.bottom,
            'size': fvg.size,
            'percentage': fvg.percentage,
            'midpoint': fvg.get_midpoint(),
            'formation_index': fvg.formation_index,
            'formation_timestamp': fvg.formation_timestamp,
            'impulse_strength': fvg.impulse_strength,
            'validation_score': fvg.validation_score,
            'status': fvg.status.value,
            'fill_percentage': fvg.fill_percentage,
            'touches': fvg.touches,
            'max_reaction': fvg.max_reaction,
            'volume_context': fvg.volume_context,
            'first_touch_index': fvg.first_touch_index,
            'first_touch_timestamp': fvg.first_touch_timestamp,
            'full_fill_index': fvg.full_fill_index,
            'full_fill_timestamp': fvg.full_fill_timestamp
        })

    return result


# Example usage and testing
if __name__ == "__main__":
    print("Fair Value Gap Detection System - Testing")
    print("=" * 50)

    # Create sample data with intentional FVGs
    print("Creating sample OHLCV data with FVGs...")
    ohlcv_data = create_sample_fvg_data(300)

    # Initialize FVG Detector
    print("Initializing FVG Detector...")
    fvg_detector = FVGDetector(
        ohlcv_data,
        min_gap_percentage=0.001,  # 0.1% minimum gap
        min_impulse_strength=0.015,  # 1.5% minimum impulse
        volume_threshold_percentile=70
    )

    # Detect FVGs
    print("Detecting Fair Value Gaps...")
    fvgs = fvg_detector.identify_fvgs()

    # Track fills
    print("Tracking FVG fills...")
    fill_stats = fvg_detector.track_fvg_fills()

    # Get statistics
    stats = fvg_detector.get_fvg_statistics()

    # Print results
    print(f"\nFVG Detection Results:")
    print(f"Total FVGs detected: {stats['total_fvgs']}")
    print(f"Bullish FVGs: {stats['bullish_fvgs']}")
    print(f"Bearish FVGs: {stats['bearish_fvgs']}")
    print(f"Average gap size: {stats['average_gap_size']:.2f}")
    print(f"Average validation score: {stats['average_validation_score']:.3f}")
    print(f"Average impulse strength: {stats['average_impulse_strength']:.3f}")

    print(f"\nFill Statistics:")
    print(f"Active FVGs: {fill_stats['active_fvgs']}")
    print(f"Partially filled: {fill_stats['partially_filled']}")
    print(f"Fully filled: {fill_stats['fully_filled']}")
    print(f"Fill success rate: {fill_stats['fill_success_rate']:.1%}")
    print(f"Average fill time: {fill_stats['average_fill_time']:.1f} candles")

    print(f"\nSize Distribution:")
    print(f"Small gaps (<0.2%): {stats['size_distribution']['small']}")
    print(f"Medium gaps (0.2%-0.5%): {stats['size_distribution']['medium']}")
    print(f"Large gaps (>0.5%): {stats['size_distribution']['large']}")

    print(f"\nReaction Statistics:")
    print(f"FVGs with price reaction: {stats['reaction_statistics']['fvgs_with_reaction']}")
    print(f"Average reaction strength: {stats['reaction_statistics']['average_reaction']:.3f}")
    print(f"Maximum reaction: {stats['reaction_statistics']['max_reaction']:.3f}")
    print(f"Reaction rate: {stats['reaction_statistics']['reaction_rate']:.1%}")

    # Show some example FVGs
    if fvgs:
        print(f"\nExample FVGs (first 3):")
        for i, fvg in enumerate(fvgs[:3]):
            print(f"FVG {i+1}: {fvg.type.value} gap at {fvg.get_midpoint():.2f}")
            print(f"  Size: {fvg.size:.2f} ({fvg.percentage:.3%})")
            print(f"  Validation score: {fvg.validation_score:.3f}")
            print(f"  Status: {fvg.status.value}")
            print(f"  Fill percentage: {fvg.fill_percentage:.1f}%")
            print(f"  Touches: {fvg.touches}")
            if fvg.max_reaction > 0:
                print(f"  Max reaction: {fvg.max_reaction:.3f}")
            print()

    # Test convenience function
    print("Testing convenience function...")
    enhanced_fvgs = get_enhanced_fvgs(ohlcv_data)
    print(f"Convenience function detected {len(enhanced_fvgs)} FVGs")

    print("\n✅ Fair Value Gap Detection System testing complete!")
    print("The system successfully detected and analyzed FVGs with comprehensive tracking.")


# Integration function for existing SMC system
def integrate_fvg_detector_with_smc(smc_detector_class):
    """
    Decorator to integrate FVG detection with existing SMC detector

    Args:
        smc_detector_class: Existing SMC detector class to enhance

    Returns:
        Enhanced SMC detector class with FVG capabilities
    """
    class EnhancedSMCDetector(smc_detector_class):
        def __init__(self, ohlcv: pd.DataFrame, **kwargs):
            super().__init__(ohlcv)
            # Initialize enhanced FVG detector
            self.fvg_detector = FVGDetector(ohlcv, **kwargs)

        def detect_fvg_enhanced(self, **kwargs) -> List[Dict]:
            """Enhanced FVG detection using the new FVGDetector"""
            return get_enhanced_fvgs(self.ohlcv, **kwargs)

        def get_fvg_statistics(self) -> Dict[str, Any]:
            """Get FVG statistics"""
            # Ensure FVGs are detected
            if not self.fvg_detector.fvgs:
                self.fvg_detector.identify_fvgs()
                self.fvg_detector.track_fvg_fills()

            return self.fvg_detector.get_fvg_statistics()

        def get_active_fvgs_near_price(self, price: float) -> List[FairValueGap]:
            """Get active FVGs near a specific price level"""
            # Ensure FVGs are detected
            if not self.fvg_detector.fvgs:
                self.fvg_detector.identify_fvgs()
                self.fvg_detector.track_fvg_fills()

            return self.fvg_detector.get_active_fvgs(price)

        def detect_all_enhanced(self) -> Dict[str, List[Dict]]:
            """Enhanced detect_all with improved FVG detection"""
            base_results = super().detect_all()

            # Replace basic FVG detection with enhanced version
            base_results['fvg'] = self.detect_fvg_enhanced()
            base_results['fvg_statistics'] = self.get_fvg_statistics()

            return base_results

    return EnhancedSMCDetector

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import the enhanced FVG detection system
try:
    from fvg_detection import FVGDetector, get_enhanced_fvgs
    FVG_DETECTION_AVAILABLE = True
except ImportError:
    FVG_DETECTION_AVAILABLE = False

# Import the enhanced Liquidity Mapping system
try:
    from liquidity_mapping import LiquidityMapper, get_enhanced_liquidity_levels
    LIQUIDITY_MAPPING_AVAILABLE = True
except ImportError:
    LIQUIDITY_MAPPING_AVAILABLE = False

# Import the enhanced Market Structure Analysis system
try:
    from market_structure_analysis import MarketStructureAnalyzer, get_enhanced_market_structure
    MARKET_STRUCTURE_AVAILABLE = True
except ImportError:
    MARKET_STRUCTURE_AVAILABLE = False

# Import the enhanced Multi-Timeframe Confluence system
try:
    from multi_timeframe_confluence import MultiTimeframeAnalyzer, get_enhanced_multi_timeframe_analysis
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False

@dataclass
class OrderBlock:
    """Data class representing an institutional order block"""
    type: str  # 'bullish' or 'bearish'
    top: float
    bottom: float
    high: float
    low: float
    volume: float
    timestamp: datetime
    index: int
    strength: float
    touches: int = 0
    last_touch: Optional[datetime] = None
    is_valid: bool = True
    formation_context: Dict = None

    def get_price_range(self) -> Tuple[float, float]:
        """Get the price range of the order block"""
        return (self.bottom, self.top)

    def is_price_in_block(self, price: float, tolerance: float = 0.001) -> bool:
        """Check if a price is within the order block with tolerance"""
        bottom, top = self.get_price_range()
        tolerance_range = (top - bottom) * tolerance
        return (bottom - tolerance_range) <= price <= (top + tolerance_range)

class OrderBlockDetector:
    """
    Advanced Order Block Detection Engine for Smart Money Concepts

    Detects institutional order blocks based on:
    - High volume candles near swing highs/lows
    - Strong impulsive moves (Break of Structure)
    - Subsequent price reactions and validations
    - Volume profile analysis
    """

    def __init__(self, ohlcv: pd.DataFrame, volume_threshold_percentile: float = 80,
                 min_impulse_strength: float = 0.02, swing_lookback: int = 20):
        """
        Initialize the Order Block Detector

        Args:
            ohlcv: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            volume_threshold_percentile: Percentile for high volume detection (default 80th percentile)
            min_impulse_strength: Minimum price movement percentage for impulse detection
            swing_lookback: Lookback period for swing high/low detection
        """
        self.ohlcv = ohlcv.copy()
        self.volume_threshold_percentile = volume_threshold_percentile
        self.min_impulse_strength = min_impulse_strength
        self.swing_lookback = swing_lookback

        # Ensure timestamp column is datetime
        if 'timestamp' in self.ohlcv.columns:
            self.ohlcv['timestamp'] = pd.to_datetime(self.ohlcv['timestamp'])

        # Calculate technical indicators
        self._calculate_indicators()

        # Detected order blocks storage
        self.order_blocks: List[OrderBlock] = []

    def _calculate_indicators(self):
        """Calculate technical indicators needed for order block detection"""
        # Volume threshold for high volume detection
        self.volume_threshold = np.percentile(self.ohlcv['volume'], self.volume_threshold_percentile)

        # Calculate swing highs and lows
        self.ohlcv['swing_high'] = self._detect_swing_highs()
        self.ohlcv['swing_low'] = self._detect_swing_lows()

        # Calculate candle body size and wicks
        self.ohlcv['body_size'] = abs(self.ohlcv['close'] - self.ohlcv['open'])
        self.ohlcv['upper_wick'] = self.ohlcv['high'] - np.maximum(self.ohlcv['open'], self.ohlcv['close'])
        self.ohlcv['lower_wick'] = np.minimum(self.ohlcv['open'], self.ohlcv['close']) - self.ohlcv['low']

        # Calculate price movement strength
        self.ohlcv['price_change_pct'] = self.ohlcv['close'].pct_change()

        # Calculate average true range for volatility context
        self.ohlcv['atr'] = self._calculate_atr(period=14)

    def _detect_swing_highs(self) -> pd.Series:
        """Detect swing highs using rolling window"""
        swing_highs = pd.Series(False, index=self.ohlcv.index)

        for i in range(self.swing_lookback, len(self.ohlcv) - self.swing_lookback):
            current_high = self.ohlcv.iloc[i]['high']
            left_highs = self.ohlcv.iloc[i-self.swing_lookback:i]['high']
            right_highs = self.ohlcv.iloc[i+1:i+self.swing_lookback+1]['high']

            if current_high > left_highs.max() and current_high > right_highs.max():
                swing_highs.iloc[i] = True

        return swing_highs

    def _detect_swing_lows(self) -> pd.Series:
        """Detect swing lows using rolling window"""
        swing_lows = pd.Series(False, index=self.ohlcv.index)

        for i in range(self.swing_lookback, len(self.ohlcv) - self.swing_lookback):
            current_low = self.ohlcv.iloc[i]['low']
            left_lows = self.ohlcv.iloc[i-self.swing_lookback:i]['low']
            right_lows = self.ohlcv.iloc[i+1:i+self.swing_lookback+1]['low']

            if current_low < left_lows.min() and current_low < right_lows.min():
                swing_lows.iloc[i] = True

        return swing_lows

    def _calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = self.ohlcv['high'] - self.ohlcv['low']
        high_close_prev = abs(self.ohlcv['high'] - self.ohlcv['close'].shift(1))
        low_close_prev = abs(self.ohlcv['low'] - self.ohlcv['close'].shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def detect_order_blocks(self, min_strength: float = 0.5) -> List[OrderBlock]:
        """
        Detect institutional order blocks based on Smart Money Concepts

        Args:
            min_strength: Minimum strength threshold for order block validation

        Returns:
            List of detected and validated order blocks
        """
        self.order_blocks = []

        # Iterate through the data to find potential order blocks
        for i in range(self.swing_lookback + 1, len(self.ohlcv) - 1):
            # Check for bullish order blocks
            bullish_ob = self._detect_bullish_order_block(i)
            if bullish_ob and bullish_ob.strength >= min_strength:
                self.order_blocks.append(bullish_ob)

            # Check for bearish order blocks
            bearish_ob = self._detect_bearish_order_block(i)
            if bearish_ob and bearish_ob.strength >= min_strength:
                self.order_blocks.append(bearish_ob)

        # Validate and filter order blocks
        self._validate_order_blocks()

        return [ob for ob in self.order_blocks if ob.is_valid]

    def _detect_bullish_order_block(self, index: int) -> Optional[OrderBlock]:
        """
        Detect bullish order block at given index

        A bullish order block is formed when:
        1. There's a bearish candle (or series of candles) at a swing low
        2. Followed by a strong bullish impulse (Break of Structure)
        3. High volume during the formation
        4. Price later returns to test the order block
        """
        current_candle = self.ohlcv.iloc[index]

        # Check if current candle is at or near a swing low
        if not self._is_near_swing_low(index):
            return None

        # Check for bearish candle formation
        if current_candle['close'] >= current_candle['open']:
            return None

        # Check for high volume
        if current_candle['volume'] < self.volume_threshold:
            return None

        # Look for subsequent bullish impulse
        impulse_strength = self._detect_bullish_impulse_after(index)
        if impulse_strength < self.min_impulse_strength:
            return None

        # Calculate order block strength
        strength = self._calculate_order_block_strength(index, 'bullish', impulse_strength)

        # Create order block
        order_block = OrderBlock(
            type='bullish',
            top=current_candle['open'],
            bottom=current_candle['close'],
            high=current_candle['high'],
            low=current_candle['low'],
            volume=current_candle['volume'],
            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else pd.Timestamp.now(),
            index=index,
            strength=strength,
            formation_context={
                'impulse_strength': impulse_strength,
                'swing_low': True,
                'volume_percentile': (current_candle['volume'] / self.volume_threshold) * 100
            }
        )

        return order_block

    def _detect_bearish_order_block(self, index: int) -> Optional[OrderBlock]:
        """
        Detect bearish order block at given index

        A bearish order block is formed when:
        1. There's a bullish candle (or series of candles) at a swing high
        2. Followed by a strong bearish impulse (Break of Structure)
        3. High volume during the formation
        4. Price later returns to test the order block
        """
        current_candle = self.ohlcv.iloc[index]

        # Check if current candle is at or near a swing high
        if not self._is_near_swing_high(index):
            return None

        # Check for bullish candle formation
        if current_candle['close'] <= current_candle['open']:
            return None

        # Check for high volume
        if current_candle['volume'] < self.volume_threshold:
            return None

        # Look for subsequent bearish impulse
        impulse_strength = self._detect_bearish_impulse_after(index)
        if impulse_strength < self.min_impulse_strength:
            return None

        # Calculate order block strength
        strength = self._calculate_order_block_strength(index, 'bearish', impulse_strength)

        # Create order block
        order_block = OrderBlock(
            type='bearish',
            top=current_candle['close'],
            bottom=current_candle['open'],
            high=current_candle['high'],
            low=current_candle['low'],
            volume=current_candle['volume'],
            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else pd.Timestamp.now(),
            index=index,
            strength=strength,
            formation_context={
                'impulse_strength': impulse_strength,
                'swing_high': True,
                'volume_percentile': (current_candle['volume'] / self.volume_threshold) * 100
            }
        )

        return order_block

    def _is_near_swing_low(self, index: int, tolerance: int = 3) -> bool:
        """Check if index is near a swing low within tolerance"""
        start_idx = max(0, index - tolerance)
        end_idx = min(len(self.ohlcv), index + tolerance + 1)

        for i in range(start_idx, end_idx):
            if i < len(self.ohlcv) and self.ohlcv.iloc[i]['swing_low']:
                return True
        return False

    def _is_near_swing_high(self, index: int, tolerance: int = 3) -> bool:
        """Check if index is near a swing high within tolerance"""
        start_idx = max(0, index - tolerance)
        end_idx = min(len(self.ohlcv), index + tolerance + 1)

        for i in range(start_idx, end_idx):
            if i < len(self.ohlcv) and self.ohlcv.iloc[i]['swing_high']:
                return True
        return False

    def _detect_bullish_impulse_after(self, start_index: int, lookforward: int = 10) -> float:
        """
        Detect bullish impulse strength after the given index

        Returns the maximum percentage gain within the lookforward period
        """
        if start_index + lookforward >= len(self.ohlcv):
            lookforward = len(self.ohlcv) - start_index - 1

        start_price = self.ohlcv.iloc[start_index]['close']
        max_price = start_price

        for i in range(start_index + 1, start_index + lookforward + 1):
            if i < len(self.ohlcv):
                current_high = self.ohlcv.iloc[i]['high']
                max_price = max(max_price, current_high)

        return (max_price - start_price) / start_price if start_price > 0 else 0

    def _detect_bearish_impulse_after(self, start_index: int, lookforward: int = 10) -> float:
        """
        Detect bearish impulse strength after the given index

        Returns the maximum percentage loss within the lookforward period
        """
        if start_index + lookforward >= len(self.ohlcv):
            lookforward = len(self.ohlcv) - start_index - 1

        start_price = self.ohlcv.iloc[start_index]['close']
        min_price = start_price

        for i in range(start_index + 1, start_index + lookforward + 1):
            if i < len(self.ohlcv):
                current_low = self.ohlcv.iloc[i]['low']
                min_price = min(min_price, current_low)

        return (start_price - min_price) / start_price if start_price > 0 else 0

    def _calculate_order_block_strength(self, index: int, ob_type: str, impulse_strength: float) -> float:
        """
        Calculate the strength of an order block based on multiple factors

        Factors considered:
        1. Volume relative to average
        2. Impulse strength following the order block
        3. Candle body size relative to ATR
        4. Position relative to swing points
        5. Time since formation

        Returns a strength score between 0 and 1
        """
        current_candle = self.ohlcv.iloc[index]

        # Volume factor (0-0.3)
        volume_factor = min(0.3, (current_candle['volume'] / self.volume_threshold) * 0.3)

        # Impulse factor (0-0.4)
        impulse_factor = min(0.4, impulse_strength * 20)  # Scale impulse to 0-0.4 range

        # Body size factor (0-0.2)
        atr_value = current_candle['atr'] if not pd.isna(current_candle['atr']) else 0.01
        body_size_ratio = current_candle['body_size'] / atr_value if atr_value > 0 else 0
        body_factor = min(0.2, body_size_ratio * 0.1)

        # Swing position factor (0-0.1)
        swing_factor = 0.1 if (ob_type == 'bullish' and self._is_near_swing_low(index)) or \
                              (ob_type == 'bearish' and self._is_near_swing_high(index)) else 0.05

        total_strength = volume_factor + impulse_factor + body_factor + swing_factor
        return min(1.0, total_strength)

    def _validate_order_blocks(self):
        """
        Validate detected order blocks and mark invalid ones

        Validation criteria:
        1. No overlapping order blocks of the same type
        2. Minimum distance between order blocks
        3. Order block must have been tested (price returned to it)
        4. Order block age validation
        """
        if not self.order_blocks:
            return

        # Sort order blocks by index
        self.order_blocks.sort(key=lambda x: x.index)

        # Remove overlapping order blocks (keep the stronger one)
        self._remove_overlapping_order_blocks()

        # Validate order block tests
        self._validate_order_block_tests()

        # Apply minimum distance filter
        self._apply_minimum_distance_filter()

    def _remove_overlapping_order_blocks(self):
        """Remove overlapping order blocks, keeping the stronger ones"""
        i = 0
        while i < len(self.order_blocks) - 1:
            current_ob = self.order_blocks[i]
            next_ob = self.order_blocks[i + 1]

            # Check if order blocks overlap
            if (current_ob.type == next_ob.type and
                self._order_blocks_overlap(current_ob, next_ob)):

                # Keep the stronger order block
                if current_ob.strength >= next_ob.strength:
                    next_ob.is_valid = False
                else:
                    current_ob.is_valid = False

            i += 1

    def _order_blocks_overlap(self, ob1: OrderBlock, ob2: OrderBlock, tolerance: float = 0.001) -> bool:
        """Check if two order blocks overlap within tolerance"""
        ob1_bottom, ob1_top = ob1.get_price_range()
        ob2_bottom, ob2_top = ob2.get_price_range()

        # Add tolerance
        tolerance_range1 = (ob1_top - ob1_bottom) * tolerance
        tolerance_range2 = (ob2_top - ob2_bottom) * tolerance

        ob1_bottom -= tolerance_range1
        ob1_top += tolerance_range1
        ob2_bottom -= tolerance_range2
        ob2_top += tolerance_range2

        # Check for overlap
        return not (ob1_top < ob2_bottom or ob2_top < ob1_bottom)

    def _validate_order_block_tests(self):
        """Validate that order blocks have been tested by price action"""
        for ob in self.order_blocks:
            if not ob.is_valid:
                continue

            # Look for price tests after order block formation
            test_found = False
            touches = 0

            for i in range(ob.index + 1, len(self.ohlcv)):
                candle = self.ohlcv.iloc[i]

                # Check if price tested the order block
                if ob.is_price_in_block(candle['low']) or ob.is_price_in_block(candle['high']):
                    test_found = True
                    touches += 1
                    ob.last_touch = candle['timestamp'] if 'timestamp' in candle else pd.Timestamp.now()

                # Stop looking after reasonable time period
                if i - ob.index > 50:  # Look ahead 50 candles max
                    break

            ob.touches = touches

            # Order block is invalid if never tested
            if not test_found:
                ob.is_valid = False

    def _apply_minimum_distance_filter(self, min_distance_pct: float = 0.005):
        """Apply minimum distance filter between order blocks"""
        valid_obs = [ob for ob in self.order_blocks if ob.is_valid]

        for i, ob1 in enumerate(valid_obs):
            for j, ob2 in enumerate(valid_obs[i+1:], i+1):
                if ob1.type == ob2.type:
                    # Calculate distance between order blocks
                    ob1_center = (ob1.top + ob1.bottom) / 2
                    ob2_center = (ob2.top + ob2.bottom) / 2
                    distance_pct = abs(ob1_center - ob2_center) / ob1_center

                    if distance_pct < min_distance_pct:
                        # Keep the stronger order block
                        if ob1.strength >= ob2.strength:
                            ob2.is_valid = False
                        else:
                            ob1.is_valid = False

    def validate_order_block(self, order_block: OrderBlock) -> Dict:
        """
        Validate a specific order block and return validation details

        Args:
            order_block: OrderBlock to validate

        Returns:
            Dictionary with validation results and metrics
        """
        validation_result = {
            'is_valid': order_block.is_valid,
            'strength': order_block.strength,
            'touches': order_block.touches,
            'age_candles': len(self.ohlcv) - order_block.index - 1,
            'volume_percentile': order_block.formation_context.get('volume_percentile', 0),
            'impulse_strength': order_block.formation_context.get('impulse_strength', 0),
            'validation_details': {}
        }

        # Check volume criteria
        validation_result['validation_details']['high_volume'] = \
            order_block.volume >= self.volume_threshold

        # Check impulse criteria
        validation_result['validation_details']['strong_impulse'] = \
            order_block.formation_context.get('impulse_strength', 0) >= self.min_impulse_strength

        # Check swing point criteria
        validation_result['validation_details']['at_swing_point'] = \
            order_block.formation_context.get('swing_low', False) or \
            order_block.formation_context.get('swing_high', False)

        # Check test criteria
        validation_result['validation_details']['price_tested'] = order_block.touches > 0

        # Overall validation
        validation_result['validation_details']['meets_all_criteria'] = all([
            validation_result['validation_details']['high_volume'],
            validation_result['validation_details']['strong_impulse'],
            validation_result['validation_details']['at_swing_point'],
            validation_result['validation_details']['price_tested']
        ])

        return validation_result

class SMCDector:
    def __init__(self, ohlcv: pd.DataFrame):
        self.ohlcv = ohlcv
        # Initialize the enhanced order block detector
        self.order_block_detector = OrderBlockDetector(ohlcv)

        # Initialize the enhanced FVG detector if available
        if FVG_DETECTION_AVAILABLE:
            self.fvg_detector = FVGDetector(ohlcv)
        else:
            self.fvg_detector = None

        # Initialize the enhanced Liquidity Mapper if available
        if LIQUIDITY_MAPPING_AVAILABLE:
            self.liquidity_mapper = LiquidityMapper(ohlcv)
        else:
            self.liquidity_mapper = None

        # Initialize the enhanced Market Structure Analyzer if available
        if MARKET_STRUCTURE_AVAILABLE:
            self.market_structure_analyzer = MarketStructureAnalyzer(ohlcv)
        else:
            self.market_structure_analyzer = None

    def detect_order_blocks(self, lookback=20) -> List[Dict]:
        """Enhanced order block detection using the new OrderBlockDetector"""
        # Use the new enhanced detector
        order_blocks = self.order_block_detector.detect_order_blocks()

        # Convert to the expected format for backward compatibility
        result = []
        for ob in order_blocks:
            result.append({
                'type': ob.type,
                'price': ob.bottom if ob.type == 'bullish' else ob.top,
                'top': ob.top,
                'bottom': ob.bottom,
                'volume': ob.volume,
                'strength': ob.strength,
                'timestamp': ob.timestamp,
                'touches': ob.touches,
                'is_valid': ob.is_valid
            })

        return result

    def detect_fvg(self, min_gap=0.002) -> List[Dict]:
        """Enhanced FVG detection using the new FVGDetector if available"""
        if FVG_DETECTION_AVAILABLE and self.fvg_detector is not None:
            # Use the enhanced FVG detection system
            return get_enhanced_fvgs(self.ohlcv, min_gap_percentage=min_gap)
        else:
            # Fallback to basic FVG detection
            fvg_list = []
            for i in range(1, len(self.ohlcv) - 1):
                prev_high = self.ohlcv['high'].iloc[i-1]
                next_low = self.ohlcv['low'].iloc[i+1]
                if next_low > prev_high and (next_low - prev_high) / prev_high > min_gap:
                    fvg_list.append({'type': 'bullish', 'price': (prev_high + next_low) / 2, 'timestamp': self.ohlcv['timestamp'].iloc[i]})
                prev_low = self.ohlcv['low'].iloc[i-1]
                next_high = self.ohlcv['high'].iloc[i+1]
                if prev_low > next_high and (prev_low - next_high) / prev_low > min_gap:
                    fvg_list.append({'type': 'bearish', 'price': (prev_low + next_high) / 2, 'timestamp': self.ohlcv['timestamp'].iloc[i]})
            return fvg_list

    def detect_liquidity_zones(self, window=20) -> List[Dict]:
        # Highs/lows with many touches = liquidity zones
        liquidity_zones = []
        highs = self.ohlcv['high'].rolling(window=window).apply(lambda x: (x == x.max()).sum(), raw=True)
        lows = self.ohlcv['low'].rolling(window=window).apply(lambda x: (x == x.min()).sum(), raw=True)
        for i in range(window, len(self.ohlcv)):
            if highs.iloc[i] >= 3:
                liquidity_zones.append({'type': 'high', 'price': self.ohlcv['high'].iloc[i], 'timestamp': self.ohlcv['timestamp'].iloc[i]})
            if lows.iloc[i] >= 3:
                liquidity_zones.append({'type': 'low', 'price': self.ohlcv['low'].iloc[i], 'timestamp': self.ohlcv['timestamp'].iloc[i]})
        return liquidity_zones

    def detect_all(self) -> Dict[str, List[Dict]]:
        """Detect all SMC patterns including enhanced order blocks and FVGs"""
        results = {
            'order_blocks': self.detect_order_blocks(),
            'fvg': self.detect_fvg(),
            'liquidity_zones': self.detect_liquidity_zones()
        }

        # Add enhanced statistics if available
        if FVG_DETECTION_AVAILABLE and self.fvg_detector is not None:
            results['fvg_statistics'] = self.get_fvg_statistics()

        # Add liquidity mapping statistics if available
        if LIQUIDITY_MAPPING_AVAILABLE and self.liquidity_mapper is not None:
            results['liquidity_statistics'] = self.get_liquidity_statistics()

        # Add market structure analysis if available
        if MARKET_STRUCTURE_AVAILABLE and self.market_structure_analyzer is not None:
            results['market_structure'] = self.get_market_structure_analysis()

        # Add multi-timeframe confluence analysis if available
        if MULTI_TIMEFRAME_AVAILABLE:
            results['multi_timeframe_confluence'] = self.get_multi_timeframe_confluence()

        return results

    def get_order_block_statistics(self) -> Dict:
        """Get statistics about detected order blocks"""
        order_blocks = self.order_block_detector.detect_order_blocks()

        if not order_blocks:
            return {
                'total_detected': 0,
                'valid_blocks': 0,
                'bullish_blocks': 0,
                'bearish_blocks': 0,
                'average_strength': 0,
                'average_touches': 0
            }

        valid_blocks = [ob for ob in order_blocks if ob.is_valid]
        bullish_blocks = [ob for ob in valid_blocks if ob.type == 'bullish']
        bearish_blocks = [ob for ob in valid_blocks if ob.type == 'bearish']

        return {
            'total_detected': len(order_blocks),
            'valid_blocks': len(valid_blocks),
            'bullish_blocks': len(bullish_blocks),
            'bearish_blocks': len(bearish_blocks),
            'average_strength': np.mean([ob.strength for ob in valid_blocks]) if valid_blocks else 0,
            'average_touches': np.mean([ob.touches for ob in valid_blocks]) if valid_blocks else 0,
            'strength_distribution': {
                'high': len([ob for ob in valid_blocks if ob.strength >= 0.8]),
                'medium': len([ob for ob in valid_blocks if 0.5 <= ob.strength < 0.8]),
                'low': len([ob for ob in valid_blocks if ob.strength < 0.5])
            }
        }

    def get_fvg_statistics(self) -> Dict:
        """Get statistics about detected FVGs using enhanced detector"""
        if not FVG_DETECTION_AVAILABLE or self.fvg_detector is None:
            return {
                'total_fvgs': 0,
                'bullish_fvgs': 0,
                'bearish_fvgs': 0,
                'average_gap_size': 0,
                'average_validation_score': 0,
                'fill_success_rate': 0,
                'enhanced_detection': False
            }

        # Ensure FVGs are detected and tracked
        if not self.fvg_detector.fvgs:
            self.fvg_detector.identify_fvgs()
            self.fvg_detector.track_fvg_fills()

        # Get comprehensive statistics
        stats = self.fvg_detector.get_fvg_statistics()
        fill_stats = self.fvg_detector.track_fvg_fills()

        # Combine and enhance statistics
        enhanced_stats = {
            'total_fvgs': stats['total_fvgs'],
            'bullish_fvgs': stats['bullish_fvgs'],
            'bearish_fvgs': stats['bearish_fvgs'],
            'average_gap_size': stats['average_gap_size'],
            'average_validation_score': stats['average_validation_score'],
            'average_impulse_strength': stats['average_impulse_strength'],
            'fill_success_rate': fill_stats['fill_success_rate'],
            'average_fill_time': fill_stats['average_fill_time'],
            'active_fvgs': fill_stats['active_fvgs'],
            'size_distribution': stats['size_distribution'],
            'reaction_statistics': stats['reaction_statistics'],
            'enhanced_detection': True
        }

        return enhanced_stats

    def get_liquidity_statistics(self) -> Dict:
        """Get statistics about detected liquidity levels using enhanced mapper"""
        if not LIQUIDITY_MAPPING_AVAILABLE or self.liquidity_mapper is None:
            return {
                'total_levels': 0,
                'equal_highs': 0,
                'equal_lows': 0,
                'buy_side_liquidity': 0,
                'sell_side_liquidity': 0,
                'average_level_strength': 0,
                'total_sweeps': 0,
                'stop_hunt_rate': 0,
                'enhanced_detection': False
            }

        # Ensure levels are mapped
        if not self.liquidity_mapper.liquidity_levels:
            self.liquidity_mapper.map_all_liquidity_levels()

        # Get comprehensive statistics
        stats = self.liquidity_mapper.get_liquidity_statistics()
        stop_hunt_analysis = self.liquidity_mapper.analyze_stop_hunt_patterns()

        # Combine and enhance statistics
        enhanced_stats = {
            'total_levels': stats['total_levels'],
            'equal_highs': stats['levels_by_type']['equal_highs'],
            'equal_lows': stats['levels_by_type']['equal_lows'],
            'buy_side_liquidity': stats['levels_by_type']['buy_side'],
            'sell_side_liquidity': stats['levels_by_type']['sell_side'],
            'average_level_strength': stats['average_level_strength'],
            'total_sweeps': stats['total_sweeps'],
            'stop_hunt_rate': stats['stop_hunt_rate'],
            'institutional_signature_avg': stats['institutional_signature_avg'],
            'active_levels': stats['active_levels'],
            'swept_levels': stats['swept_levels'],
            'stop_hunt_analysis': stop_hunt_analysis,
            'enhanced_detection': True
        }

        return enhanced_stats

    def get_market_structure_analysis(self) -> Dict:
        """Get comprehensive market structure analysis using enhanced analyzer"""
        if not MARKET_STRUCTURE_AVAILABLE or self.market_structure_analyzer is None:
            return {
                'current_trend': 'neutral',
                'trend_strength': 0.0,
                'structure_quality': 0.0,
                'total_swing_points': 0,
                'total_structure_breaks': 0,
                'bos_breaks': 0,
                'choch_breaks': 0,
                'enhanced_detection': False
            }

        # Perform comprehensive market structure analysis
        market_structure = self.market_structure_analyzer.analyze_market_structure()
        statistics = self.market_structure_analyzer.get_structure_statistics()

        # Get swing points and structure breaks
        swing_points = []
        for sp in self.market_structure_analyzer.swing_points:
            swing_points.append({
                'type': sp.type.value,
                'price': sp.price,
                'timestamp': sp.timestamp,
                'strength': sp.strength,
                'confirmed': sp.confirmed
            })

        structure_breaks = []
        for sb in self.market_structure_analyzer.structure_breaks:
            structure_breaks.append({
                'type': sb.type.value,
                'direction': sb.direction.value,
                'break_price': sb.break_price,
                'break_timestamp': sb.break_timestamp,
                'strength': sb.strength.value,
                'volume_confirmation': sb.volume_confirmation,
                'institutional_signature': sb.institutional_signature,
                'follow_through': sb.follow_through
            })

        # Combine comprehensive analysis
        enhanced_analysis = {
            'current_trend': market_structure.current_trend.value,
            'trend_strength': market_structure.trend_strength,
            'structure_quality': market_structure.structure_quality,
            'total_swing_points': statistics['total_swing_points'],
            'swing_highs': statistics['swing_highs'],
            'swing_lows': statistics['swing_lows'],
            'confirmed_swings': statistics['confirmed_swings'],
            'total_structure_breaks': statistics['total_structure_breaks'],
            'bos_breaks': statistics['bos_breaks'],
            'choch_breaks': statistics['choch_breaks'],
            'bullish_breaks': statistics['bullish_breaks'],
            'bearish_breaks': statistics['bearish_breaks'],
            'average_institutional_signature': statistics['average_institutional_signature'],
            'follow_through_rate': statistics['follow_through_rate'],
            'strength_distribution': statistics['strength_distribution'],
            'swing_points': swing_points,
            'structure_breaks': structure_breaks,
            'enhanced_detection': True
        }

        return enhanced_analysis

    def get_multi_timeframe_confluence(self) -> Dict:
        """Get multi-timeframe confluence analysis using enhanced analyzer"""
        if not MULTI_TIMEFRAME_AVAILABLE:
            return {
                'confluence_available': False,
                'best_signal': {'type': 'none', 'score': 0.0, 'strength': 'none'},
                'htf_bias': 'neutral',
                'entry_zone': 'equilibrium',
                'risk_reward_ratio': 1.0,
                'market_timing_score': 0.5
            }

        try:
            # Create multi-timeframe data sources from current data
            # For demonstration, we'll use the same data for different timeframes
            # In practice, you'd have actual multi-timeframe data
            from multi_timeframe_confluence import TimeframeType

            data_sources = {
                TimeframeType.M15: self.ohlcv.copy(),  # Primary timeframe
                TimeframeType.H1: self._resample_to_timeframe(self.ohlcv, '1H'),
                TimeframeType.H4: self._resample_to_timeframe(self.ohlcv, '4H')
            }

            # Initialize multi-timeframe analyzer
            from multi_timeframe_confluence import MultiTimeframeAnalyzer
            mtf_analyzer = MultiTimeframeAnalyzer(
                data_sources=data_sources,
                primary_timeframe=TimeframeType.M15,
                htf_timeframes=[TimeframeType.H4, TimeframeType.H1]
            )

            # Get comprehensive analysis
            analysis = mtf_analyzer.get_comprehensive_analysis()
            statistics = mtf_analyzer.get_confluence_statistics()

            # Extract key information for SMC integration
            best_signal = analysis['best_signal']
            htf_biases = analysis['htf_biases']
            zones = analysis['discount_premium_zones']

            # Determine overall HTF bias
            htf_bias = 'neutral'
            if htf_biases:
                bullish_count = sum(1 for bias in htf_biases.values() if bias['direction'] == 'bullish')
                bearish_count = sum(1 for bias in htf_biases.values() if bias['direction'] == 'bearish')

                if bullish_count > bearish_count:
                    htf_bias = 'bullish'
                elif bearish_count > bullish_count:
                    htf_bias = 'bearish'

            # Get primary timeframe zone
            primary_zone = zones.get('15m', {})
            entry_zone = primary_zone.get('zone_type', 'equilibrium')

            enhanced_confluence = {
                'confluence_available': True,
                'best_signal': {
                    'type': best_signal['type'],
                    'score': best_signal['score'],
                    'strength': best_signal['strength'],
                    'risk_reward_ratio': best_signal['risk_reward_ratio']
                },
                'htf_bias': htf_bias,
                'entry_zone': entry_zone,
                'market_timing_score': analysis['market_timing_score'],
                'timeframe_statistics': {
                    'total_timeframes': statistics['total_timeframes'],
                    'bullish_timeframes': statistics['bullish_timeframes'],
                    'bearish_timeframes': statistics['bearish_timeframes'],
                    'average_bias_strength': statistics['average_bias_strength']
                },
                'confluence_signals': analysis['confluence_signals'],
                'htf_biases': htf_biases,
                'discount_premium_zones': zones
            }

            return enhanced_confluence

        except Exception as e:
            logger.warning(f"Error in multi-timeframe confluence analysis: {e}")
            return {
                'confluence_available': False,
                'error': str(e),
                'best_signal': {'type': 'none', 'score': 0.0, 'strength': 'none'},
                'htf_bias': 'neutral',
                'entry_zone': 'equilibrium',
                'risk_reward_ratio': 1.0,
                'market_timing_score': 0.5
            }

    def _resample_to_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to a higher timeframe"""
        try:
            # Ensure timestamp is datetime and set as index
            df = data.copy()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # Resample to higher timeframe
            resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Reset index to have timestamp as column
            resampled.reset_index(inplace=True)

            return resampled

        except Exception as e:
            logger.warning(f"Error resampling to {timeframe}: {e}")
            # Return original data if resampling fails
            return data.copy()


def create_sample_data(num_candles: int = 1000) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing the Order Block Detection Engine

    Args:
        num_candles: Number of candles to generate

    Returns:
        DataFrame with OHLCV data and timestamps
    """
    np.random.seed(42)  # For reproducible results

    # Generate realistic price data with trends and volatility
    base_price = 50000  # Starting price (like BTC)
    prices = [base_price]
    volumes = []

    for i in range(num_candles):
        # Add some trend and noise
        trend = 0.0001 * np.sin(i / 100)  # Long-term trend
        noise = np.random.normal(0, 0.01)  # Random noise
        change = trend + noise

        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

        # Generate volume with some correlation to price movement
        base_volume = 1000000
        volume_multiplier = 1 + abs(change) * 10  # Higher volume on big moves
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
        volumes.append(volume)

    # Create OHLCV data
    data = []
    for i in range(num_candles):
        price = prices[i]
        volume = volumes[i]

        # Generate realistic OHLC from the price
        volatility = price * 0.005  # 0.5% volatility

        open_price = price + np.random.normal(0, volatility * 0.5)
        close_price = price + np.random.normal(0, volatility * 0.5)

        high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility * 0.3))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility * 0.3))

        timestamp = pd.Timestamp.now() - pd.Timedelta(minutes=(num_candles - i) * 15)

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    return pd.DataFrame(data)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    print("Creating sample OHLCV data...")
    ohlcv_data = create_sample_data(500)

    # Initialize SMC Detector
    print("Initializing SMC Detector...")
    smc = SMCDector(ohlcv_data)

    # Detect all patterns
    print("Detecting SMC patterns...")
    results = smc.detect_all()

    # Print results
    print(f"\nDetected {len(results['order_blocks'])} order blocks")
    print(f"Detected {len(results['fvg'])} Fair Value Gaps")
    print(f"Detected {len(results['liquidity_zones'])} liquidity zones")

    # Show enhanced FVG statistics if available
    if 'fvg_statistics' in results:
        fvg_stats = results['fvg_statistics']
        print(f"\nEnhanced FVG Statistics:")
        print(f"Total FVGs: {fvg_stats['total_fvgs']}")
        print(f"Bullish FVGs: {fvg_stats['bullish_fvgs']}")
        print(f"Bearish FVGs: {fvg_stats['bearish_fvgs']}")
        print(f"Average validation score: {fvg_stats['average_validation_score']:.3f}")
        print(f"Fill success rate: {fvg_stats['fill_success_rate']:.1%}")
        print(f"Active FVGs: {fvg_stats['active_fvgs']}")
        print(f"Enhanced detection: {fvg_stats['enhanced_detection']}")

    # Show enhanced liquidity statistics if available
    if 'liquidity_statistics' in results:
        liq_stats = results['liquidity_statistics']
        print(f"\nEnhanced Liquidity Statistics:")
        print(f"Total levels: {liq_stats['total_levels']}")
        print(f"Equal highs: {liq_stats['equal_highs']}")
        print(f"Equal lows: {liq_stats['equal_lows']}")
        print(f"Buy-side liquidity: {liq_stats['buy_side_liquidity']}")
        print(f"Sell-side liquidity: {liq_stats['sell_side_liquidity']}")
        print(f"Average level strength: {liq_stats['average_level_strength']:.3f}")
        print(f"Total sweeps: {liq_stats['total_sweeps']}")
        print(f"Stop hunt rate: {liq_stats['stop_hunt_rate']:.1%}")
        print(f"Enhanced detection: {liq_stats['enhanced_detection']}")

    # Show enhanced market structure analysis if available
    if 'market_structure' in results:
        ms_stats = results['market_structure']
        print(f"\nEnhanced Market Structure Analysis:")
        print(f"Current trend: {ms_stats['current_trend']}")
        print(f"Trend strength: {ms_stats['trend_strength']:.3f}")
        print(f"Structure quality: {ms_stats['structure_quality']:.3f}")
        print(f"Total swing points: {ms_stats['total_swing_points']}")
        print(f"Total structure breaks: {ms_stats['total_structure_breaks']}")
        print(f"BOS breaks: {ms_stats['bos_breaks']}")
        print(f"ChoCH breaks: {ms_stats['choch_breaks']}")
        print(f"Follow-through rate: {ms_stats['follow_through_rate']:.1%}")
        print(f"Enhanced detection: {ms_stats['enhanced_detection']}")

    # Show enhanced multi-timeframe confluence if available
    if 'multi_timeframe_confluence' in results:
        mtf_stats = results['multi_timeframe_confluence']
        print(f"\nEnhanced Multi-Timeframe Confluence:")
        print(f"Confluence available: {mtf_stats['confluence_available']}")
        if mtf_stats['confluence_available']:
            best_signal = mtf_stats['best_signal']
            print(f"Best signal: {best_signal['type']} (score: {best_signal['score']:.3f}, strength: {best_signal['strength']})")
            print(f"HTF bias: {mtf_stats['htf_bias']}")
            print(f"Entry zone: {mtf_stats['entry_zone']}")
            print(f"Risk/Reward: {best_signal['risk_reward_ratio']:.2f}")
            print(f"Market timing: {mtf_stats['market_timing_score']:.3f}")

            tf_stats = mtf_stats['timeframe_statistics']
            print(f"Timeframes: {tf_stats['total_timeframes']} (Bullish: {tf_stats['bullish_timeframes']}, Bearish: {tf_stats['bearish_timeframes']})")

    # Get order block statistics
    stats = smc.get_order_block_statistics()
    print(f"\nOrder Block Statistics:")
    print(f"Total detected: {stats['total_detected']}")
    print(f"Valid blocks: {stats['valid_blocks']}")
    print(f"Bullish blocks: {stats['bullish_blocks']}")
    print(f"Bearish blocks: {stats['bearish_blocks']}")
    print(f"Average strength: {stats['average_strength']:.3f}")
    print(f"Average touches: {stats['average_touches']:.1f}")

    # Show some example order blocks
    if results['order_blocks']:
        print(f"\nExample Order Blocks:")
        for i, ob in enumerate(results['order_blocks'][:3]):  # Show first 3
            print(f"Block {i+1}: {ob['type']} at {ob['price']:.2f}, strength: {ob['strength']:.3f}")

    print("\nOrder Block Detection Engine implementation complete!")


# For integration with existing pipeline
def get_enhanced_order_blocks(ohlcv: pd.DataFrame, **kwargs) -> List[Dict]:
    """
    Convenience function to get enhanced order blocks

    Args:
        ohlcv: OHLCV DataFrame
        **kwargs: Additional parameters for OrderBlockDetector

    Returns:
        List of detected order blocks in dictionary format
    """
    detector = OrderBlockDetector(ohlcv, **kwargs)
    order_blocks = detector.detect_order_blocks()

    # Convert to dictionary format
    result = []
    for ob in order_blocks:
        result.append({
            'type': ob.type,
            'price': ob.bottom if ob.type == 'bullish' else ob.top,
            'top': ob.top,
            'bottom': ob.bottom,
            'high': ob.high,
            'low': ob.low,
            'volume': ob.volume,
            'strength': ob.strength,
            'timestamp': ob.timestamp,
            'index': ob.index,
            'touches': ob.touches,
            'last_touch': ob.last_touch,
            'is_valid': ob.is_valid,
            'formation_context': ob.formation_context
        })

    return result
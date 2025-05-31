# Order Block Detection Engine

## Overview

The Order Block Detection Engine is a sophisticated implementation of Smart Money Concepts (SMC) for identifying institutional order blocks in cryptocurrency trading data. This engine detects high-probability zones where institutional traders have placed significant orders, providing valuable insights for trading decisions.

## Features

### Core Functionality
- **Advanced Order Block Detection**: Identifies bullish and bearish order blocks based on multiple criteria
- **Volume Analysis**: Incorporates volume thresholds and percentile analysis
- **Swing Point Detection**: Automatically identifies swing highs and lows
- **Impulse Strength Calculation**: Measures the strength of price movements following order blocks
- **Validation Logic**: Comprehensive validation system for order block quality
- **Strength Scoring**: Multi-factor strength calculation (0-1 scale)

### Smart Money Concepts Implementation
- **Break of Structure (BOS)**: Detects impulsive moves that validate order blocks
- **High Volume Confirmation**: Ensures order blocks form during high volume periods
- **Price Testing**: Validates that price returns to test the order block
- **Institutional Behavior**: Models how smart money operates in the market

## Architecture

### Main Classes

#### `OrderBlock` (Data Class)
Represents a detected institutional order block with the following properties:
- `type`: 'bullish' or 'bearish'
- `top/bottom`: Price boundaries of the order block
- `volume`: Volume during formation
- `strength`: Calculated strength score (0-1)
- `touches`: Number of times price has tested the block
- `timestamp`: Formation time
- `is_valid`: Validation status

#### `OrderBlockDetector`
Core detection engine with advanced algorithms:
- Swing high/low detection
- Volume analysis and thresholds
- Impulse strength calculation
- Order block validation
- Statistical analysis

#### `SMCDector` (Enhanced)
Backward-compatible wrapper that integrates the new detection engine with existing SMC functionality.

## Algorithm Details

### Detection Criteria

#### Bullish Order Blocks
1. **Formation**: Bearish candle at or near swing low
2. **Volume**: Above 80th percentile threshold
3. **Impulse**: Followed by bullish impulse ≥ 2% (configurable)
4. **Validation**: Price must return to test the block

#### Bearish Order Blocks
1. **Formation**: Bullish candle at or near swing high
2. **Volume**: Above 80th percentile threshold
3. **Impulse**: Followed by bearish impulse ≥ 2% (configurable)
4. **Validation**: Price must return to test the block

### Strength Calculation
Multi-factor scoring system (0-1 scale):
- **Volume Factor** (0-0.3): Relative to volume threshold
- **Impulse Factor** (0-0.4): Strength of subsequent price movement
- **Body Size Factor** (0-0.2): Candle body size relative to ATR
- **Swing Position Factor** (0-0.1): Proximity to swing points

### Validation Process
1. **Overlap Removal**: Eliminates overlapping blocks (keeps stronger)
2. **Distance Filter**: Ensures minimum distance between blocks
3. **Price Testing**: Validates that blocks have been tested
4. **Quality Metrics**: Comprehensive validation scoring

## Usage Examples

### Basic Usage
```python
from smc_detection import SMCDector, OrderBlockDetector
import pandas as pd

# Load your OHLCV data
ohlcv = pd.read_csv('your_data.csv')

# Initialize detector
smc = SMCDector(ohlcv)

# Detect all patterns
results = smc.detect_all()
order_blocks = results['order_blocks']

# Get statistics
stats = smc.get_order_block_statistics()
print(f"Detected {stats['valid_blocks']} valid order blocks")
```

### Advanced Usage
```python
# Direct use of enhanced detector
detector = OrderBlockDetector(
    ohlcv, 
    volume_threshold_percentile=85,  # Higher volume threshold
    min_impulse_strength=0.03,       # 3% minimum impulse
    swing_lookback=25                # Longer swing detection
)

order_blocks = detector.detect_order_blocks(min_strength=0.7)

# Validate specific order block
for ob in order_blocks:
    validation = detector.validate_order_block(ob)
    print(f"Block strength: {validation['strength']:.3f}")
    print(f"Meets all criteria: {validation['validation_details']['meets_all_criteria']}")
```

### Convenience Function
```python
from smc_detection import get_enhanced_order_blocks

# Quick access to enhanced order blocks
order_blocks = get_enhanced_order_blocks(
    ohlcv,
    volume_threshold_percentile=80,
    min_impulse_strength=0.025
)
```

## Configuration Parameters

### OrderBlockDetector Parameters
- `volume_threshold_percentile` (default: 80): Percentile for high volume detection
- `min_impulse_strength` (default: 0.02): Minimum 2% price movement for impulse
- `swing_lookback` (default: 20): Lookback period for swing detection

### Detection Parameters
- `min_strength` (default: 0.5): Minimum strength threshold for order blocks
- `tolerance` (default: 3): Tolerance for swing point proximity
- `lookforward` (default: 10): Candles to look ahead for impulse detection

## Output Format

### Order Block Dictionary
```python
{
    'type': 'bullish',           # or 'bearish'
    'price': 50000.0,            # Key price level
    'top': 50100.0,              # Upper boundary
    'bottom': 50000.0,           # Lower boundary
    'volume': 1500000.0,         # Formation volume
    'strength': 0.756,           # Strength score (0-1)
    'timestamp': '2024-01-01',   # Formation time
    'touches': 2,                # Number of tests
    'is_valid': True,            # Validation status
    'formation_context': {...}   # Additional context
}
```

### Statistics Output
```python
{
    'total_detected': 15,
    'valid_blocks': 12,
    'bullish_blocks': 6,
    'bearish_blocks': 6,
    'average_strength': 0.723,
    'average_touches': 1.8,
    'strength_distribution': {
        'high': 4,    # strength >= 0.8
        'medium': 6,  # 0.5 <= strength < 0.8
        'low': 2      # strength < 0.5
    }
}
```

## Testing

The implementation includes comprehensive tests:

```bash
cd ml/backend/src/strategy
python3 test_order_block_detection.py
```

Test coverage includes:
- Order block detection accuracy
- Validation logic
- Data structure integrity
- Integration with existing SMC system
- Edge cases and error handling

## Performance Considerations

- **Optimized Algorithms**: Efficient swing detection and validation
- **Memory Management**: Minimal memory footprint for large datasets
- **Configurable Parameters**: Adjustable for different market conditions
- **Scalable Design**: Handles datasets from minutes to daily timeframes

## Integration

The Order Block Detection Engine integrates seamlessly with:
- Existing SMC detection pipeline
- Trading strategy execution
- Risk management systems
- Backtesting frameworks
- Real-time trading systems

## Future Enhancements

Planned improvements include:
- Multi-timeframe order block analysis
- Machine learning-based strength scoring
- Real-time order block tracking
- Advanced visualization tools
- Performance optimization for high-frequency data

## Support

For questions or issues with the Order Block Detection Engine:
1. Check the test suite for usage examples
2. Review the comprehensive documentation
3. Examine the sample data generation for testing
4. Refer to the validation methods for quality assurance

The engine is designed to be robust, accurate, and production-ready for institutional-grade trading systems.

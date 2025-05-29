# TradingView Chart with Prediction Overlay

This directory contains components for an enhanced TradingView chart implementation with prediction overlay functionality. The chart is designed to display cryptocurrency price data alongside machine learning predictions and trading signals.

## Components

### 1. EnhancedTradingViewChart

The main component that combines the TradingView chart with controls. It manages state and coordinates interactions between the chart and controls.

**Key Features:**
- Symbol selection (cryptocurrency pairs)
- Time interval selection
- Dark/light mode toggle
- Prediction overlay toggle
- Technical indicators toggle
- Responsive design

### 2. TradingViewChartContainer

The core chart component that integrates with the TradingView JavaScript API. It handles:

- Chart initialization and configuration
- WebSocket connection for real-time data
- Prediction data overlay
- Trading signal markers
- Custom styling based on theme

### 3. ChartControls

A panel of controls for configuring the chart:

- Symbol selector
- Time interval selector
- Theme toggle
- Prediction overlay toggle
- Indicator controls

### 4. TradingViewConfig

Configuration file containing:

- Default indicator settings
- Chart theme styles
- Prediction overlay styles
- Signal marker styles
- Time intervals
- Utility functions for data formatting

## Usage

```jsx
import { EnhancedTradingViewChart } from '../components/charts/EnhancedTradingViewChart';

// Sample prediction data
const predictionData = [
  { time: 1642425322, value: 42000, confidence: 0.75 },
  { time: 1642511722, value: 42500, confidence: 0.72 },
  // ...more prediction points
];

// Sample trading signals
const signalData = [
  { time: 1642425322, type: 'buy', confidence: 0.82 },
  { time: 1642511722, type: 'sell', confidence: 0.78 },
  // ...more signals
];

function ChartPage() {
  return (
    <div className="chart-container">
      <EnhancedTradingViewChart 
        initialSymbol="BTCUSD"
        initialInterval="60" // 1h
        initialDarkMode={true}
        height={600}
        predictionsData={predictionData}
        signalsData={signalData}
      />
    </div>
  );
}
```

## Prediction Overlay

The prediction overlay displays future price projections based on machine learning models. Each prediction point includes:

- **Time**: When the prediction is for
- **Value**: Predicted price
- **Confidence**: Confidence level (0-1) of the prediction

The overlay uses a color gradient based on confidence levels, with higher confidence predictions displayed more prominently.

## Signal Markers

Trading signals are displayed as markers on the chart:

- **Buy signals**: Upward-pointing arrows (green)
- **Sell signals**: Downward-pointing arrows (red)
- **Hold signals**: Circles (gray)

Each marker includes the signal type and confidence level when available.

## Implementation Notes

1. The TradingView widget is loaded dynamically via CDN to ensure compatibility
2. WebSocket connections are established for real-time data updates
3. Prediction overlays are implemented using TradingView's built-in studies API
4. Signal markers are implemented using TradingView's shape drawing API
5. The component is optimized for SSR (Server-Side Rendering) using dynamic imports

## Dependencies

- React
- socket.io-client (for WebSocket connections)
- TradingView Charting Library (loaded via CDN) 
# API Contracts - Intelligent Trading Bot

## Overview

This document defines the API contracts for communication between microservices in the Intelligent Trading Bot system.

## 1. Data Collection Service API

### REST Endpoints

#### Get Market Data
```typescript
GET /api/v1/market-data/{symbol}
Query Parameters:
  - timeframe: string (1m, 5m, 15m, 1h, 4h, 1d)
  - limit: number (default: 100)
  - indicators: string[] (optional: atr, rsi, macd)

Response:
{
  symbol: string;
  timeframe: string;
  data: {
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    indicators?: {
      atr?: number;
      rsi?: number;
      macd?: { macd: number; signal: number; histogram: number };
    };
  }[];
}
```

#### Get Multi-Timeframe Data
```typescript
GET /api/v1/market-data/{symbol}/multi-timeframe
Query Parameters:
  - timeframes: string[] (comma-separated)
  - indicators: string[] (optional)

Response:
{
  symbol: string;
  timeframes: {
    [timeframe: string]: MarketDataPoint[];
  };
}
```

### WebSocket Events
```typescript
// Market Data Stream
{
  type: 'market_data';
  symbol: string;
  timeframe: string;
  data: MarketDataPoint;
}

// Indicator Update
{
  type: 'indicator_update';
  symbol: string;
  timeframe: string;
  indicators: IndicatorValues;
}
```

## 2. AI Intelligence Service API

### REST Endpoints

#### Generate Trading Signal
```typescript
POST /api/v1/ai/generate-signal
Body:
{
  symbol: string;
  marketData: MultiTimeframeData;
  currentPositions?: Position[];
}

Response:
{
  signal: {
    type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number; // 0-100
    strength: number; // 0-1
    timeframe: string;
    reasoning: string[];
  };
  predictions: {
    priceTarget: number;
    probability: number;
    timeHorizon: string;
  };
  riskAssessment: {
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
    maxPositionSize: number;
    suggestedStopLoss: number;
  };
}
```

#### Get Market Regime
```typescript
GET /api/v1/ai/market-regime/{symbol}
Response:
{
  regime: 'TRENDING_BULLISH' | 'TRENDING_BEARISH' | 'SIDEWAYS' | 'VOLATILE' | 'BREAKOUT';
  confidence: number;
  duration: number; // minutes
  characteristics: {
    volatility: number;
    trendStrength: number;
    momentum: number;
  };
  recommendations: {
    tradingStyle: 'SCALPING' | 'DAY_TRADING' | 'SWING_TRADING';
    riskMultiplier: number;
    timeframePreference: string[];
  };
}
```

#### Predict Position Outcome
```typescript
POST /api/v1/ai/predict-outcome
Body:
{
  position: Position;
  marketData: MultiTimeframeData;
  marketRegime: MarketRegime;
}

Response:
{
  prediction: {
    outcome: 'PROFIT' | 'LOSS' | 'BREAKEVEN';
    probability: number;
    expectedReturn: number;
    timeToTarget: number;
  };
  recommendations: {
    action: 'HOLD' | 'SCALE_IN' | 'SCALE_OUT' | 'CLOSE';
    confidence: number;
    reasoning: string[];
  };
}
```

## 3. Position Management Service API

### REST Endpoints

#### Get Position Health Score
```typescript
GET /api/v1/positions/{positionId}/health
Response:
{
  healthScore: number; // 0-100
  factors: {
    trendAlignment: number; // -1 to 1
    momentum: number;
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
    timeInPosition: number;
  };
  recommendations: {
    action: 'HOLD' | 'REDUCE' | 'CLOSE' | 'ADD';
    urgency: 'LOW' | 'MEDIUM' | 'HIGH';
    reasoning: string[];
  };
}
```

#### Update Position Strategy
```typescript
PUT /api/v1/positions/{positionId}/strategy
Body:
{
  stopLoss?: number;
  takeProfitLevels?: TakeProfitLevel[];
  trailingStop?: {
    enabled: boolean;
    distance: number;
    step: number;
  };
}

Response:
{
  success: boolean;
  updatedPosition: Position;
  changes: string[];
}
```

#### Execute Partial Exit
```typescript
POST /api/v1/positions/{positionId}/partial-exit
Body:
{
  percentage: number; // 0-100
  reason: string;
  targetPrice?: number;
}

Response:
{
  success: boolean;
  executedOrder: Order;
  remainingPosition: Position;
}
```

## 4. Risk Management Service API

### REST Endpoints

#### Assess Portfolio Risk
```typescript
GET /api/v1/risk/portfolio-assessment
Response:
{
  overallRisk: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  metrics: {
    totalExposure: number;
    maxDrawdown: number;
    sharpeRatio: number;
    correlationRisk: number;
  };
  limits: {
    maxPositionSize: number;
    maxTotalRisk: number;
    dailyLossLimit: number;
  };
  alerts: RiskAlert[];
}
```

#### Validate New Position
```typescript
POST /api/v1/risk/validate-position
Body:
{
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entryPrice: number;
  stopLoss?: number;
}

Response:
{
  approved: boolean;
  adjustedSize?: number;
  riskMetrics: {
    positionRisk: number;
    portfolioImpact: number;
    correlationRisk: number;
  };
  warnings: string[];
}
```

## 5. Event Schemas

### Market Data Events
```typescript
interface MarketDataEvent {
  type: 'MARKET_DATA';
  timestamp: number;
  symbol: string;
  timeframe: string;
  data: {
    price: number;
    volume: number;
    indicators: IndicatorValues;
  };
}
```

### Trading Signal Events
```typescript
interface TradingSignalEvent {
  type: 'TRADING_SIGNAL';
  timestamp: number;
  signal: {
    symbol: string;
    type: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    source: 'AI' | 'TECHNICAL' | 'SENTIMENT';
  };
  metadata: {
    modelVersion: string;
    marketRegime: string;
    reasoning: string[];
  };
}
```

### Position Events
```typescript
interface PositionEvent {
  type: 'POSITION_OPENED' | 'POSITION_UPDATED' | 'POSITION_CLOSED';
  timestamp: number;
  position: Position;
  trigger: {
    source: 'AI' | 'MANUAL' | 'RISK_MANAGEMENT';
    reason: string;
  };
}
```

### Risk Events
```typescript
interface RiskEvent {
  type: 'RISK_THRESHOLD_BREACH' | 'RISK_LIMIT_EXCEEDED';
  timestamp: number;
  severity: 'WARNING' | 'CRITICAL';
  details: {
    metric: string;
    currentValue: number;
    threshold: number;
    affectedPositions: string[];
  };
  recommendedActions: string[];
}
```

## 6. Authentication & Authorization

### API Key Authentication
```typescript
Headers:
{
  'Authorization': 'Bearer <api_key>';
  'X-Service-Name': '<service_name>';
  'X-Request-ID': '<unique_request_id>';
}
```

### Service-to-Service Authentication
```typescript
Headers:
{
  'X-Service-Token': '<jwt_token>';
  'X-Service-ID': '<service_identifier>';
}
```

## 7. Error Handling

### Standard Error Response
```typescript
{
  error: {
    code: string;
    message: string;
    details?: any;
    timestamp: number;
    requestId: string;
  };
}
```

### Error Codes
- `INVALID_REQUEST` - Malformed request
- `UNAUTHORIZED` - Authentication failed
- `FORBIDDEN` - Insufficient permissions
- `NOT_FOUND` - Resource not found
- `RATE_LIMITED` - Too many requests
- `SERVICE_UNAVAILABLE` - Service temporarily unavailable
- `INTERNAL_ERROR` - Internal server error

## 8. Rate Limiting

### Limits by Service
- **Data Collection**: 1000 requests/minute
- **AI Intelligence**: 100 requests/minute
- **Position Management**: 500 requests/minute
- **Risk Management**: 200 requests/minute

### Headers
```typescript
Response Headers:
{
  'X-RateLimit-Limit': '1000';
  'X-RateLimit-Remaining': '999';
  'X-RateLimit-Reset': '1640995200';
}
```

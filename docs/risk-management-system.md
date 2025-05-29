# Risk Management System Documentation

## Overview

The SmartMarketOOPS Risk Management System is a comprehensive solution designed to help traders manage their risk effectively. The system provides tools for position sizing, risk assessment, alert generation, and implementing circuit breakers to prevent excessive losses.

## Core Components

### 1. Risk Management Service

The Risk Management Service handles the core functionality related to position sizing and risk settings:

- **Position Sizing**: Calculates appropriate position sizes based on:
  - Account balance
  - Risk amount (fixed or percentage)
  - Entry and stop loss prices
  - Confidence level of signals
  
- **Risk Settings Management**: Stores and retrieves user-specific risk preferences:
  - Maximum position size
  - Maximum drawdown
  - Risk per trade
  - Position sizing method
  - Default stop loss and take profit settings
  
- **Stop Loss & Take Profit Calculation**: Determines optimal exit points using:
  - Fixed price values
  - Percentage-based calculations
  - ATR (Average True Range) multiples
  - Risk-reward ratios

### 2. Risk Assessment Service

The Risk Assessment Service evaluates trading risks and generates alerts:

- **Trade Risk Assessment**: Analyzes individual trades for:
  - Position size risk relative to account
  - Stop loss distance risk
  - Portfolio concentration risk
  
- **Portfolio Risk Assessment**: Provides overall portfolio metrics:
  - Total exposure by symbol and direction
  - Current drawdown
  - Diversification metrics
  - Risk distribution

- **Risk Alert Management**: Creates and manages risk alerts for:
  - Excessive position sizes
  - Approaching maximum drawdown
  - Symbol concentration
  - Correlated positions
  - Missing stop losses

### 3. Circuit Breaker Service

The Circuit Breaker Service implements trading halt mechanisms to prevent catastrophic losses:

- **Circuit Breaker Types**:
  - Maximum drawdown breaker
  - Daily loss breaker
  - Rapid loss breaker
  - Excessive exposure breaker
  
- **Circuit Breaker Management**:
  - Activation based on risk thresholds
  - Status tracking (active, reset)
  - Manual reset functionality
  - Historical tracking

## API Endpoints

The Risk Management System exposes the following RESTful endpoints:

### Risk Settings

- `GET /api/risk/settings` - Retrieve a user's risk settings
- `PUT /api/risk/settings` - Update a user's risk settings

### Position Sizing

- `POST /api/risk/position-size` - Calculate optimal position size

### Risk Assessment

- `POST /api/risk/assess-trade` - Assess risk for a potential trade
- `GET /api/risk/portfolio` - Get portfolio risk assessment
- `GET /api/risk/alerts` - List risk alerts for a user

### Circuit Breakers

- `GET /api/risk/circuit-breakers` - Get active circuit breakers
- `POST /api/risk/circuit-breakers/:id/reset` - Reset a specific circuit breaker

## Data Models

### RiskSettings

Stores user-specific risk preferences:

```
RiskSettings {
  id: string
  userId: string (optional)
  botId: string (optional)
  maxPositionSize: number
  maxDrawdown: number
  defaultStopLossType: StopLossType
  defaultStopLossValue: number
  defaultTakeProfitType: TakeProfitType
  defaultTakeProfitValue: number
  maxDailyLoss: number
  riskLevel: RiskLevel
  positionSizingMethod: PositionSizingMethod
  defaultRiskPerTrade: number
  maxOpenPositions: number
  maxPositionsPerSymbol: number
  enabledCircuitBreakers: boolean
  createdAt: Date
  updatedAt: Date
}
```

### RiskAlert

Records risk warnings:

```
RiskAlert {
  id: string
  userId: string
  type: RiskAlertType
  message: string
  severity: RiskLevel
  metadata: any (JSON)
  createdAt: Date
  resolvedAt: Date (optional)
}
```

### CircuitBreaker

Tracks trading halt mechanisms:

```
CircuitBreaker {
  id: string
  userId: string
  type: CircuitBreakerType
  status: CircuitBreakerStatus
  activationReason: string
  activatedAt: Date
  resetAt: Date (optional)
  resetReason: string (optional)
  metadata: any (JSON)
}
```

## Enums

- **RiskLevel**: `VERY_LOW`, `LOW`, `MEDIUM`, `HIGH`, `VERY_HIGH`
- **PositionSizingMethod**: `FIXED_AMOUNT`, `FIXED_PERCENTAGE`, `KELLY_CRITERION`, `OPTIMAL_F`
- **StopLossType**: `FIXED_PRICE`, `PERCENTAGE`, `ATR_MULTIPLE`, `VOLATILITY_BASED`
- **TakeProfitType**: `FIXED_PRICE`, `PERCENTAGE`, `RISK_REWARD_RATIO`, `ATR_MULTIPLE`
- **RiskAlertType**: `POSITION_SIZE_EXCEEDED`, `APPROACHING_MAX_DRAWDOWN`, `SYMBOL_CONCENTRATION_LIMIT`, etc.
- **CircuitBreakerType**: `MAX_DRAWDOWN_BREAKER`, `DAILY_LOSS_BREAKER`, `RAPID_LOSS_BREAKER`, etc.
- **CircuitBreakerStatus**: `ACTIVE`, `RESET`

## Usage Examples

### Calculating Position Size

```typescript
const riskManagementService = new RiskManagementService();

// Calculate position size with 1% risk
const result = await riskManagementService.calculatePositionSize(
  10000, // Account balance
  'BTC/USD', // Symbol
  1, // Risk 1% of account
  40000, // Entry price
  39000, // Stop loss price
  PositionSizingMethod.FIXED_PERCENTAGE
);

// Result: { positionSize: 0.0025, riskAmount: 100, riskPercentage: 1 }
```

### Assessing Trade Risk

```typescript
const riskAssessmentService = new RiskAssessmentService();

// Assess trade risk
const result = await riskAssessmentService.assessTradeRisk(
  'user123',
  {
    symbol: 'BTC/USD',
    direction: 'long',
    entryPrice: 40000,
    positionSize: 0.1,
    stopLossPrice: 39000,
    takeProfitPrice: 43000,
    accountBalance: 100000
  }
);

// Result includes risk level, alerts, and risk factors
```

### Activating Circuit Breakers

```typescript
const circuitBreakerService = new CircuitBreakerService();

// Check and activate circuit breakers if thresholds are exceeded
const result = await circuitBreakerService.checkAndActivateCircuitBreakers(
  'user123',
  100000, // Account balance
  11 // Current drawdown percentage
);

// Result indicates if circuit breakers were activated
```

## Testing

The Risk Management System includes comprehensive unit and integration tests:

- **Unit Tests**: Test individual service functionality in isolation
- **Integration Tests**: Test API endpoints and service interactions

Run the tests using:

```bash
npm run test:risk
```

## Implementation Best Practices

1. **Always set stop losses**: Ensure every position has an appropriate stop loss
2. **Use portfolio-based risk limits**: Consider total exposure, not just per-trade risk
3. **Adjust position size by confidence**: Scale position size based on signal quality
4. **Respect circuit breakers**: Honor trading halts when triggered
5. **Monitor drawdown**: Track drawdown relative to maximum allowed
6. **Set appropriate risk per trade**: Typically 1-2% per trade for most strategies
7. **Diversify by symbol and direction**: Avoid concentration in any single asset
8. **Use trailing stops for profitable trades**: Lock in profits when possible

## Future Enhancements

- Integration with ML-based risk models
- Correlation analysis between positions
- VaR (Value at Risk) calculations
- Custom risk rules engine
- Historical drawdown analysis
- Risk-adjusted performance metrics 
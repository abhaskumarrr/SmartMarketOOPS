# 🔌 API Reference - Ultimate Trading System

## 📊 Overview

The Ultimate Trading System provides comprehensive APIs for trading operations, market analysis, and performance monitoring. All APIs are designed for professional-grade applications with institutional-level reliability.

---

## 🚀 Quick Start

### **Base URL:**
```
Production: https://api.smartmarketoops.com
Development: http://localhost:3001/api
```

### **Authentication:**
```javascript
// API Key Authentication
headers: {
  'Authorization': 'Bearer YOUR_API_KEY',
  'Content-Type': 'application/json'
}
```

### **Rate Limits:**
- **Trading APIs:** 100 requests/minute
- **Market Data:** 1000 requests/minute  
- **Analytics:** 500 requests/minute

---

## 🎯 Trading Engine APIs

### **1. Start Ultimate Trading System**

**Endpoint:** `POST /trading/ultimate/start`

**Description:** Starts the ultimate high-performance trading system

**Request:**
```javascript
{
  "symbols": ["ETHUSD", "BTCUSD"],
  "riskPerTrade": 2.5,
  "maxConcurrentPositions": 2,
  "confluenceThreshold": 0.75
}
```

**Response:**
```javascript
{
  "success": true,
  "systemId": "ultimate_20250106_001",
  "status": "RUNNING",
  "config": {
    "targetWinRate": 68,
    "targetMonthlyReturn": 15,
    "maxDrawdown": 8
  },
  "timestamp": "2025-01-06T10:00:00Z"
}
```

### **2. Get System Status**

**Endpoint:** `GET /trading/ultimate/status`

**Response:**
```javascript
{
  "systemId": "ultimate_20250106_001",
  "status": "RUNNING",
  "uptime": "2h 15m 30s",
  "performance": {
    "currentWinRate": 82.1,
    "monthlyReturn": 7.8,
    "maxDrawdown": 0.06,
    "totalTrades": 15,
    "activePositions": 2
  },
  "lastUpdate": "2025-01-06T12:15:30Z"
}
```

### **3. Execute Manual Trade**

**Endpoint:** `POST /trading/execute`

**Request:**
```javascript
{
  "symbol": "ETHUSD",
  "side": "BUY",
  "strategy": "OHLC_ZONE",
  "confluenceScore": 0.85,
  "riskPercent": 2.5,
  "stopLoss": 2635.50,
  "takeProfit": 2685.75
}
```

**Response:**
```javascript
{
  "success": true,
  "tradeId": "ULTIMATE_1704537600_ETHUSD",
  "symbol": "ETHUSD",
  "side": "BUY",
  "entryPrice": 2653.25,
  "positionSize": 1.85,
  "riskAmount": 185.50,
  "expectedReturn": 3.2,
  "confluenceScore": 0.85,
  "timestamp": "2025-01-06T12:00:00Z"
}
```

---

## 📊 Market Analysis APIs

### **1. Daily OHLC Zone Analysis**

**Endpoint:** `GET /analysis/ohlc-zones/{symbol}`

**Response:**
```javascript
{
  "symbol": "ETHUSD",
  "currentPrice": 2653.25,
  "dailyLevels": {
    "PDH": 2680.50,
    "PDL": 2590.75,
    "PDC": 2645.25,
    "PP": 2638.83
  },
  "zoneAnalysis": {
    "inZone": true,
    "zoneName": "PDL_Support",
    "strength": 90,
    "distance": 0.12,
    "signal": {
      "action": "BUY",
      "confidence": 0.88,
      "stopLoss": 2635.50,
      "takeProfit": 2685.75
    }
  },
  "timestamp": "2025-01-06T12:00:00Z"
}
```

### **2. SMC Analysis**

**Endpoint:** `GET /analysis/smc/{symbol}`

**Response:**
```javascript
{
  "symbol": "ETHUSD",
  "orderBlocks": [
    {
      "type": "bullish",
      "price": 2650.25,
      "strength": 85,
      "fresh": true,
      "touches": 1
    }
  ],
  "fairValueGaps": [
    {
      "type": "bullish",
      "upper": 2655.50,
      "lower": 2652.25,
      "strength": 72,
      "filled": false
    }
  ],
  "liquidityZones": [
    {
      "type": "buy_side",
      "price": 2685.75,
      "strength": 90,
      "swept": false
    }
  ],
  "marketStructure": {
    "trend": "bullish",
    "lastBOS": 2660.25,
    "nextResistance": 2680.50
  }
}
```

### **3. AI Ensemble Prediction**

**Endpoint:** `GET /analysis/ai-prediction/{symbol}`

**Response:**
```javascript
{
  "symbol": "ETHUSD",
  "ensemble": {
    "signal": "BUY",
    "confidence": 0.831,
    "quality": 0.78
  },
  "models": {
    "lstm": {
      "signal": "BUY",
      "confidence": 0.82,
      "accuracy": 0.78
    },
    "transformer": {
      "signal": "BUY",
      "confidence": 0.87,
      "accuracy": 0.85
    },
    "smc": {
      "signal": "BUY",
      "confidence": 0.79,
      "accuracy": 0.75
    }
  },
  "prediction": {
    "direction": "UP",
    "magnitude": 1.2,
    "timeframe": "4h",
    "probability": 0.831
  }
}
```

---

## 🎯 Confluence Scoring API

### **Calculate Confluence Score**

**Endpoint:** `POST /analysis/confluence`

**Request:**
```javascript
{
  "symbol": "ETHUSD",
  "ohlcZone": {
    "strength": 90,
    "zoneName": "PDL_Support"
  },
  "smcEnhancement": {
    "present": true,
    "type": "bullish_OB",
    "bonus": 0.15
  },
  "aiConfirmation": {
    "confidence": 0.831,
    "alignment": true
  }
}
```

**Response:**
```javascript
{
  "confluenceScore": {
    "total": 0.851,
    "quality": "EXCELLENT",
    "breakdown": {
      "ohlcZone": 0.36,
      "smcEnhancement": 0.15,
      "aiConfirmation": 0.208,
      "alignmentBonus": 0.1
    },
    "meetsThreshold": true,
    "recommendation": "EXECUTE_TRADE"
  },
  "riskAssessment": {
    "winProbability": 0.78,
    "expectedReturn": 3.2,
    "riskReward": 3.5,
    "positionSizeRecommendation": 2.8
  }
}
```

---

## 📈 Performance Analytics APIs

### **1. Real-Time Performance**

**Endpoint:** `GET /analytics/performance/realtime`

**Response:**
```javascript
{
  "portfolio": {
    "currentBalance": 11250.75,
    "totalReturn": 12.51,
    "monthlyReturn": 7.8,
    "dailyPnL": 125.50,
    "maxDrawdown": 0.06
  },
  "trading": {
    "totalTrades": 25,
    "winningTrades": 21,
    "winRate": 84.0,
    "avgWin": 4.2,
    "avgLoss": -1.4,
    "profitFactor": 26.95,
    "sharpeRatio": 19.68
  },
  "activePositions": [
    {
      "tradeId": "ULTIMATE_1704537600_ETHUSD",
      "symbol": "ETHUSD",
      "side": "BUY",
      "entryPrice": 2653.25,
      "currentPrice": 2667.50,
      "unrealizedPnL": 2.1,
      "duration": "1h 15m"
    }
  ]
}
```

### **2. Feature Attribution**

**Endpoint:** `GET /analytics/attribution`

**Response:**
```javascript
{
  "featureAttribution": {
    "ohlcZones": {
      "contribution": 4.8,
      "winRate": 85.2,
      "tradesCount": 18
    },
    "smcEnhancement": {
      "contribution": 3.2,
      "winRate": 82.1,
      "tradesCount": 17
    },
    "aiConfirmation": {
      "contribution": 2.1,
      "winRate": 88.5,
      "tradesCount": 15
    },
    "riskManagement": {
      "contribution": 1.8,
      "capitalPreserved": 98.5,
      "maxDrawdownPrevented": 5.2
    }
  },
  "strategyBreakdown": {
    "dailyOHLC": 69.2,
    "smcOrderBlocks": 23.1,
    "aiEnsemble": 7.7
  }
}
```

---

## 🛡️ Risk Management APIs

### **1. Portfolio Risk Assessment**

**Endpoint:** `GET /risk/portfolio`

**Response:**
```javascript
{
  "portfolioRisk": {
    "totalRisk": 8.5,
    "availableRisk": 1.5,
    "riskUtilization": 85.0,
    "correlationRisk": 0.25
  },
  "positionRisk": [
    {
      "symbol": "ETHUSD",
      "riskPercent": 2.5,
      "riskAmount": 281.25,
      "stopLossDistance": 0.67,
      "timeAtRisk": "1h 15m"
    }
  ],
  "riskLimits": {
    "maxRiskPerTrade": 3.0,
    "maxPortfolioRisk": 10.0,
    "maxDrawdown": 8.0,
    "maxCorrelation": 0.7
  }
}
```

### **2. Kelly Criterion Calculation**

**Endpoint:** `POST /risk/kelly-sizing`

**Request:**
```javascript
{
  "winRate": 0.82,
  "avgWin": 0.042,
  "avgLoss": 0.014,
  "confluenceScore": 0.85,
  "currentBalance": 11250.75
}
```

**Response:**
```javascript
{
  "kellyFraction": 0.28,
  "cappedKelly": 0.25,
  "recommendedSize": 2.8,
  "riskAmount": 315.02,
  "expectedValue": 0.035,
  "confidenceAdjustment": 0.85,
  "portfolioAdjustment": 0.92
}
```

---

## 🔄 WebSocket APIs

### **Real-Time Updates**

**Connection:** `wss://api.smartmarketoops.com/ws`

**Subscribe to Trading Updates:**
```javascript
{
  "action": "subscribe",
  "channel": "trading_updates",
  "symbols": ["ETHUSD", "BTCUSD"]
}
```

**Real-Time Messages:**
```javascript
// Trade Execution
{
  "type": "TRADE_EXECUTED",
  "data": {
    "tradeId": "ULTIMATE_1704537600_ETHUSD",
    "symbol": "ETHUSD",
    "side": "BUY",
    "entryPrice": 2653.25,
    "confluenceScore": 0.85
  }
}

// Position Update
{
  "type": "POSITION_UPDATE",
  "data": {
    "tradeId": "ULTIMATE_1704537600_ETHUSD",
    "currentPrice": 2667.50,
    "unrealizedPnL": 2.1,
    "status": "ACTIVE"
  }
}

// Performance Update
{
  "type": "PERFORMANCE_UPDATE",
  "data": {
    "winRate": 84.0,
    "monthlyReturn": 7.8,
    "totalTrades": 25,
    "activePositions": 2
  }
}
```

---

## 🔧 Configuration APIs

### **Update System Parameters**

**Endpoint:** `PUT /config/trading-parameters`

**Request:**
```javascript
{
  "confluenceThreshold": 0.80,
  "maxRiskPerTrade": 3.0,
  "maxConcurrentPositions": 3,
  "stopLossPercent": 1.5,
  "takeProfitRatio": 3.0
}
```

**Response:**
```javascript
{
  "success": true,
  "message": "Trading parameters updated successfully",
  "newConfig": {
    "confluenceThreshold": 0.80,
    "maxRiskPerTrade": 3.0,
    "maxConcurrentPositions": 3,
    "stopLossPercent": 1.5,
    "takeProfitRatio": 3.0
  },
  "effectiveFrom": "2025-01-06T12:30:00Z"
}
```

---

## 📊 Error Handling

### **Standard Error Response:**
```javascript
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Insufficient balance for trade execution",
    "details": {
      "required": 500.00,
      "available": 450.25,
      "shortfall": 49.75
    }
  },
  "timestamp": "2025-01-06T12:00:00Z"
}
```

### **Common Error Codes:**
- `INSUFFICIENT_BALANCE` - Not enough funds for trade
- `INVALID_CONFLUENCE` - Confluence score below threshold
- `MAX_POSITIONS_REACHED` - Too many active positions
- `MARKET_CLOSED` - Trading not available
- `INVALID_SYMBOL` - Unsupported trading pair
- `RATE_LIMIT_EXCEEDED` - Too many API requests

---

## 🎯 SDK Examples

### **JavaScript/Node.js:**
```javascript
const SmartMarketAPI = require('@smartmarket/api-client');

const client = new SmartMarketAPI({
  apiKey: 'YOUR_API_KEY',
  baseURL: 'https://api.smartmarketoops.com'
});

// Start ultimate trading system
const result = await client.trading.ultimate.start({
  symbols: ['ETHUSD', 'BTCUSD'],
  riskPerTrade: 2.5
});

// Get real-time performance
const performance = await client.analytics.performance.realtime();
```

### **Python:**
```python
from smartmarket import SmartMarketAPI

client = SmartMarketAPI(
    api_key='YOUR_API_KEY',
    base_url='https://api.smartmarketoops.com'
)

# Execute manual trade
trade = client.trading.execute({
    'symbol': 'ETHUSD',
    'side': 'BUY',
    'strategy': 'OHLC_ZONE',
    'confluenceScore': 0.85
})
```

---

## 📄 Conclusion

The Ultimate Trading System APIs provide comprehensive access to all trading, analysis, and monitoring capabilities. The APIs are designed for:

✅ **Professional-grade reliability** with 99.9% uptime  
✅ **Real-time performance** with sub-100ms response times  
✅ **Comprehensive functionality** covering all system features  
✅ **Easy integration** with SDKs and examples  
✅ **Robust error handling** with detailed error messages  

**Ready for production integration!** 🚀

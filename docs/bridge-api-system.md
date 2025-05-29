# ML-Trading Bridge API System

## Overview

The Bridge API System provides a seamless connection between the Machine Learning (ML) system and the Trading System. It enables:

1. **ML Predictions**: Get price and direction predictions from trained ML models
2. **Signal Generation**: Convert ML predictions into actionable trading signals
3. **Strategy Integration**: Incorporate ML predictions into trading strategies
4. **Model Management**: Track and manage ML models
5. **Training Control**: Start, monitor, and manage model training processes
6. **Backtesting**: Test strategies with ML predictions using historical data

## Architecture

The Bridge API layer consists of several components:

- **MLBridgeService**: Communicates with the ML system API
- **BridgeService**: Core service that integrates ML predictions with trading signals
- **BridgeController**: HTTP controllers for the Bridge API endpoints
- **Prisma Models**: Database models for ML predictions, models, and bridge configurations

## API Endpoints

### Prediction Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/api/bridge/predict-and-signal` | POST | Get prediction and generate trading signal | Yes |
| `/api/bridge/predict` | POST | Get ML prediction only | Yes |
| `/api/bridge/predict-batch` | POST | Get batch predictions | Yes |
| `/api/bridge/predictions/:id` | GET | Get prediction by ID | Yes |

### Model Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/api/bridge/models` | GET | Get all available ML models | Yes |
| `/api/bridge/models/:id` | GET | Get model by ID | Yes |
| `/api/bridge/models/:id/features` | GET | Get feature importance for a model | Yes |

### Training Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/api/bridge/training` | POST | Start model training | Yes |
| `/api/bridge/training/:id` | GET | Get training status | Yes |
| `/api/bridge/training/:id` | DELETE | Cancel training | Yes |

### Backtest Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/api/bridge/backtest` | POST | Run backtest with ML predictions | Yes |

### Health Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|--------------|
| `/api/bridge/health` | GET | Get bridge health status | Yes |
| `/api/bridge/ml-health` | GET | Check ML system connection | Yes |

## Request/Response Examples

### Get Prediction and Generate Signal

**Request**:
```json
POST /api/bridge/predict-and-signal
{
  "symbol": "BTCUSD",
  "timeframe": "1h",
  "modelVersion": "v_20250523_023634",
  "confidenceThreshold": 75,
  "signalExpiry": 120
}
```

**Response**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "BTCUSD",
  "type": "ENTRY",
  "direction": "LONG",
  "strength": "STRONG",
  "timeframe": "SHORT",
  "price": 49582.35,
  "targetPrice": 51250.80,
  "stopLoss": 48590.70,
  "confidenceScore": 82,
  "expectedReturn": 2.46,
  "expectedRisk": 0.27,
  "riskRewardRatio": 9.11,
  "generatedAt": "2025-05-27T21:15:30.000Z",
  "expiresAt": "2025-05-27T23:15:30.000Z",
  "source": "ML:CNNLSTM:v_20250523_023634",
  "metadata": {
    "predictionId": "a1b2c3d4-e5f6-4a5b-8c7d-9e8f7a6b5c4d",
    "modelVersion": "v_20250523_023634",
    "modelName": "CNNLSTM",
    "modelPerformance": {
      "accuracy": 0.78,
      "precision": 0.82,
      "recall": 0.76
    },
    "timeframe": "1h",
    "originalTimestamps": ["2025-05-27T20:00:00.000Z", "2025-05-27T21:00:00.000Z"]
  },
  "predictionValues": {
    "raw": [51105.25, 51250.80],
    "timestamps": ["2025-05-27T22:00:00.000Z", "2025-05-27T23:00:00.000Z"],
    "confidences": [79, 82]
  },
  "createdAt": "2025-05-27T21:15:30.000Z",
  "updatedAt": "2025-05-27T21:15:30.000Z"
}
```

### Get Model Status

**Request**:
```
GET /api/bridge/models/cnnlstm-btcusd-1h-20250523
```

**Response**:
```json
{
  "id": "cnnlstm-btcusd-1h-20250523",
  "name": "CNNLSTM",
  "version": "v_20250523_023634",
  "modelType": "HYBRID",
  "symbol": "BTCUSD",
  "timeframe": "1h",
  "description": "CNN-LSTM hybrid model for BTC price prediction",
  "status": "ACTIVE",
  "accuracy": 0.78,
  "precision": 0.82,
  "recall": 0.76,
  "f1Score": 0.79,
  "trainedAt": "2025-05-23T02:36:34.000Z",
  "lastUsedAt": "2025-05-27T21:15:30.000Z",
  "location": "/models/registry/BTCUSD/v_20250523_023634",
  "params": {
    "layers": 4,
    "epochs": 100,
    "batchSize": 32,
    "learningRate": 0.001
  }
}
```

### Start Model Training

**Request**:
```json
POST /api/bridge/training
{
  "modelType": "LSTM",
  "symbol": "ETHUSD",
  "timeframe": "4h",
  "datasetSize": 5000,
  "epochs": 100,
  "params": {
    "layers": 3,
    "units": [64, 128, 64],
    "dropout": 0.2,
    "optimizer": "adam",
    "lossFn": "mse"
  },
  "features": ["close", "volume", "rsi", "macd", "bollinger"]
}
```

**Response**:
```json
{
  "id": "train_7a8b9c0d-1e2f-3a4b-5c6d-7e8f9a0b1c2d",
  "userId": "user123",
  "symbol": "ETHUSD",
  "timeframe": "4h",
  "modelType": "LSTM",
  "status": "QUEUED",
  "progress": 0,
  "startedAt": null,
  "completedAt": null,
  "params": {
    "layers": 3,
    "units": [64, 128, 64],
    "dropout": 0.2,
    "optimizer": "adam",
    "lossFn": "mse",
    "features": ["close", "volume", "rsi", "macd", "bollinger"],
    "datasetSize": 5000,
    "epochs": 100
  },
  "createdAt": "2025-05-27T21:30:00.000Z",
  "updatedAt": "2025-05-27T21:30:00.000Z"
}
```

## Bridge Health Monitoring

The Bridge API includes a health monitoring system that tracks:

1. **ML System Status**: Connection to the ML API
2. **Trading System Status**: Health of the trading system
3. **Latency Metrics**: Average latency for predictions and signal generation
4. **Error Tracking**: Errors encountered during bridge operations
5. **Usage Metrics**: Prediction and signal generation request volume

Health status can be checked via the `/api/bridge/health` endpoint, which returns a detailed health report.

## Configuration

The Bridge API is configured through environment variables:

```
# ML Bridge API Configuration
ML_API_URL=http://localhost:5000/api
ML_API_KEY=your-ml-api-key-here
ML_SYSTEM_RECONNECT_INTERVAL=30000 # 30 seconds
ML_HEALTH_CHECK_INTERVAL=300000 # 5 minutes
ML_BATCH_SIZE=20
ML_MAX_CONCURRENT_REQUESTS=5
ML_REQUEST_TIMEOUT=60000 # 60 seconds
ML_MAX_RETRIES=3
ML_AUTO_SIGNAL_GENERATION=false # Enable/disable automatic signal generation from ML predictions
```

## Database Models

The Bridge API uses several database models:

1. **MLModel**: Stores metadata about trained ML models
2. **MLPrediction**: Stores predictions generated by ML models
3. **MLTrainingJob**: Tracks model training jobs
4. **BridgeConfig**: Stores bridge configuration settings

## Integration with Trading System

The Bridge API integrates with the Trading System through:

1. **Signal Generation**: Converting ML predictions into trading signals
2. **Strategy Execution**: Providing predictions to the strategy execution service
3. **Risk Management**: Using prediction confidence for risk-adjusted position sizing
4. **Backtesting**: Running simulations using ML predictions on historical data

## Security

All Bridge API endpoints require authentication using JWT tokens. Users must have appropriate permissions to access prediction, training, and model management endpoints.

## Error Handling

The Bridge API implements robust error handling:

1. **Automatic Retries**: Failed requests to the ML system are retried with exponential backoff
2. **Graceful Degradation**: The system remains operational even if the ML system is temporarily unavailable
3. **Detailed Error Logging**: All errors are logged with context for debugging
4. **Client-Friendly Errors**: API clients receive clear error messages and status codes

## Testing

The Bridge API includes comprehensive tests:

1. **Unit Tests**: Testing individual service methods
2. **Integration Tests**: Testing API endpoints and database interactions
3. **Mock Tests**: Using mocked ML responses for deterministic testing

Run tests with:
```bash
./scripts/test-bridge-api.sh
``` 
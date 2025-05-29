# SMOOPs ML Service

This directory contains the machine learning components of the SMOOPs trading bot, focused on Smart Money Concepts detection and prediction.

## Structure

- `src/` - Source code for ML models and services
  - `api/` - FastAPI endpoints for model serving
  - `backtesting/` - Backtesting framework for strategy validation
  - `data/` - Data processing pipelines
  - `models/` - ML model definitions
  - `training/` - Training pipelines
  - `monitoring/` - Performance monitoring tools
  - `utils/` - Utility functions

- `data/` - Data storage
  - `raw/` - Raw market data
  - `processed/` - Processed datasets

- `models/` - Saved model checkpoints

- `logs/` - Training and evaluation logs

## Key Features

- **Smart Money Concepts Analysis**
  - Order Block Detection
  - Fair Value Gap Identification
  - Break of Structure Recognition
  - Liquidity Engineering Detection

- **PyTorch Models**
  - Optimized for Apple Silicon (MPS acceleration)
  - Transformer-based market prediction
  - Multi-timeframe analysis

- **Backtesting Framework**
  - Historical performance evaluation
  - Risk metrics calculation
  - Strategy optimization

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- FastAPI

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Service

```bash
# Start the ML service
python -m ml.backend.src.scripts.server
```

The service will be available at http://localhost:3002

## API Endpoints

- `/health` - Health check endpoint
- `/api/predict/{symbol}` - Get predictions for a trading symbol
- `/api/models` - List available models and their performance metrics

## Development Guidelines

1. **Model Training**
   - All models should be trained using the tools in `src/training/`
   - Model checkpoints should be saved to the `models/` directory
   - Training logs should be saved to `logs/`

2. **Backtesting**
   - Use the backtesting framework in `src/backtesting/` to validate strategies
   - Compare results against baseline models before deploying

3. **Performance Monitoring**
   - All deployed models should have continuous performance monitoring
   - Alert thresholds should be set for performance degradation

## License

This project is licensed under the MIT License. See the LICENSE file in the project root for details. 
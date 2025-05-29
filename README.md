# SMOOPs: Automated Crypto ML Trading Pipeline

## Overview
SMOOPs is a production-grade, fully automated machine learning pipeline for cryptocurrency trading. It features robust model training, strict preprocessor and feature alignment, reproducible inference, and a powerful backtesting engine. The system is managed and documented using Taskmaster for maximum reproducibility and team collaboration.

## Key Features
- **End-to-end ML pipeline**: Data ingestion, feature engineering, model training, evaluation, prediction, and backtesting.
- **Strict feature alignment**: Ensures that features used in training, prediction, and backtesting are always identical in name and order, preventing silent bugs.
- **Preprocessor persistence**: The exact fitted preprocessor (e.g., StandardScaler) is saved and loaded with each model checkpoint.
- **Robust backtesting**: Automated backtest engine with ML model integration, strict input dimension checks, and clear error reporting.
- **Taskmaster integration**: All tasks, subtasks, and workflow documentation are managed with Taskmaster for transparency and reproducibility.

## Workflow
1. **Train a Model**
   ```bash
   python3 -m ml.src.cli train --symbol BTCUSD --model-type lstm --data-path sample_data/BTCUSD_15m.csv --num-epochs 100 --batch-size 32 --sequence-length 60 --forecast-horizon 1
   ```
   - Saves model, preprocessor, and metadata in `models/registry/<SYMBOL>/<VERSION>/`.

2. **Make Predictions**
   ```bash
   python3 -m ml.src.cli predict --symbol BTCUSD --data-file sample_data/BTCUSD_15m.csv --output-file predictions.csv
   ```
   - Loads the latest model and preprocessor, applies exact feature engineering, and outputs predictions.

3. **Run Backtest**
   ```bash
   python3 -m ml.src.cli backtest --data-file sample_data/BTCUSD_15m.csv --strategy ml_model --symbol BTCUSD --model-type lstm --model-checkpoint models/registry/BTCUSD/<VERSION>/model.pt --preprocessor models/registry/BTCUSD/<VERSION>/preprocessor.pkl --output-dir runs/backtest/
   ```
   - Ensures feature and preprocessor alignment, outputs robust backtest results.

## Troubleshooting
- **Input Dimension Errors**: If you see errors like `size mismatch for lstm.weight_ih_l0`, check that your feature engineering and preprocessor match exactly between training and inference. See [PyTorch LSTM size mismatch discussion](https://discuss.pytorch.org/t/time-series-lstm-size-mismatch-beginner-question/4704).
- **Feature Mismatch**: The pipeline will log expected vs. actual feature columns and raise a clear error if they do not match.

## Taskmaster Usage
- All project tasks, subtasks, and workflow documentation are managed with Taskmaster.
- To regenerate markdown documentation and task files:
  ```bash
  task-master generate
  ```
- For more, see `.taskmasterconfig` and the `tasks/` directory.

## References
- [PyTorch LSTM: Size mismatch and feature alignment](https://discuss.pytorch.org/t/time-series-lstm-size-mismatch-beginner-question/4704)
- [Best practices for ML pipeline automation](https://www.markovml.com/blog/machine-learning-pipeline)
- [Taskmaster documentation](./.taskmasterconfig)

---
For questions or contributions, see the `docs/` directory or open an issue.

## Documentation

- **[Development Guide](docs/DEVELOPMENT.md)** - Comprehensive setup and development workflow
- **[Deployment Guide](docs/deployment-guide.md)** - Deployment options and procedures
- **[Environment Setup](docs/environment-setup.md)** - Environment configuration details
- **[Project Structure](docs/project-structure.md)** - Detailed breakdown of the codebase
- **[Linting Guide](docs/linting-guide.md)** - Code quality standards and linting tools

## Project Structure
```
SMOOPs_dev/
├── .github/            # GitHub workflows for CI/CD
├── backend/            # Node.js/Express backend API
│   ├── prisma/         # Database schema and migrations
│   ├── src/            # Backend source code
│   │   ├── controllers/# API controllers
│   │   ├── middleware/ # Express middleware
│   │   ├── routes/     # API routes
│   │   ├── services/   # Business logic
│   │   └── utils/      # Utility functions (encryption, etc.)
├── frontend/           # Next.js frontend application
│   ├── pages/          # Next.js pages
│   └── public/         # Static assets
├── ml/                 # Python ML models and services
│   ├── src/            # ML source code
│   │   ├── api/        # ML service API
│   │   ├── backtesting/# Backtesting framework
│   │   ├── data/       # Data processing pipelines
│   │   ├── models/     # ML model definitions
│   │   ├── training/   # Training pipelines
│   │   ├── monitoring/ # Performance monitoring
│   │   └── utils/      # Utility functions
│   ├── data/           # Data storage
│   │   ├── raw/        # Raw market data
│   │   └── processed/  # Processed datasets
│   ├── models/         # Saved model checkpoints
│   └── logs/           # Training and evaluation logs
├── scripts/            # Utility scripts and tooling
├── tasks/              # Task definitions and project management
├── docker-compose.yml  # Docker services configuration
└── README.md           # Project documentation
```

## Installation

### Prerequisites
- macOS (Apple Silicon recommended) or Linux
- Node.js 20+ and npm
- Python 3.10+
- Docker and Docker Compose (for containerized setup)
- Delta Exchange API credentials (testnet/real net)

### Quick Setup
The fastest way to get started is using our automated setup script:

```bash
# Clone the repository
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev

# Run the development setup script
npm run dev:setup
```

For detailed setup instructions, see the [Development Guide](docs/DEVELOPMENT.md).

### Docker Setup (Recommended)
The easiest way to run the project is using Docker Compose:
```bash
docker-compose up -d
```

This will start all services:
- PostgreSQL database
- Backend API (available at http://localhost:3001)
- Frontend dashboard (available at http://localhost:3000)
- ML service (available at http://localhost:3002)

### Development Tools

SMOOPs includes several helpful development tools:

```bash
# Run common development tasks
npm run dev:tasks

# View specific development task options
npm run dev:tasks help
```

## Usage

### Real-Time Delta Exchange Integration

- The backend securely stores your Delta Exchange API credentials (testnet or mainnet) and streams real-time market data and ML signals via WebSocket.
- The frontend dashboard connects to the backend WebSocket using `socket.io-client` and updates the TradingView-style chart in real time.
- To use your own credentials, add them via the dashboard or backend API (see API Key Management below).
- By default, the frontend connects to the backend WebSocket at `http://localhost:3001`. You can override this with the environment variable:
  ```env
  NEXT_PUBLIC_WS_URL=http://localhost:3001
  ```
  Add this to your `.env` file in the `frontend/` directory if needed.

#### How to Test Live Chart
1. Start all services (backend, frontend, ML, database) as described above.
2. Log in to the dashboard at `http://localhost:3000`.
3. The main chart will update in real time as new market data arrives from Delta Exchange testnet.
4. You can subscribe to different symbols or intervals as you build out the dashboard UI.

### Trading Dashboard
Access the trading dashboard at `http://localhost:3000` to:
- View real-time market data with SMC indicators (live chart updates via WebSocket)
- Monitor trading signals and executed trades
- Analyze performance metrics
- Configure trading strategies
- Manage API keys securely (testnet/mainnet)

### API Endpoints
The backend provides several API endpoints:

#### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - Create a new user account

#### API Key Management
- `GET /api/keys` - List all API keys for a user
- `POST /api/keys` - Add a new API key
- `DELETE /api/keys/:id` - Remove an API key

#### Trading
- `GET /api/delta/instruments` - Get available trading instruments
- `GET /api/delta/market/:symbol` - Get real-time market data
- `POST /api/delta/order` - Place a new order
- `GET /api/delta/orders` - Get order history

### ML Service
The ML service exposes endpoints for model training and prediction:

# SmartMarketOOPS

## Project Overview

SmartMarketOOPS is a comprehensive algorithmic trading platform that combines machine learning predictions with automated trading strategies to execute trades on cryptocurrency exchanges.

## Key Components

1. **Authentication System**: Secure JWT-based authentication with role-based access control
2. **ML Prediction System**: Advanced machine learning models for price prediction and trend analysis
3. **Trading Strategy Engine**: Configurable trading strategies with rule-based execution
4. **Risk Management System**: Comprehensive risk controls for position sizing and portfolio management
5. **Bridge API Layer**: Connection between ML predictions and trading execution
6. **Performance Testing Framework**: Evaluation and optimization of system performance
7. **Order Execution Service**: Smart order routing and execution on cryptocurrency exchanges

## Features

- User authentication and account management
- Machine learning model training and prediction
- Trading signal generation based on ML predictions
- Strategy creation, backtesting, and execution
- Real-time risk management and position sizing
- WebSocket-based real-time updates
- Performance monitoring and optimization
- Comprehensive API documentation

## Getting Started

### Prerequisites

- Node.js (v16+)
- PostgreSQL (v14+)
- Python 3.8+ (for ML components)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/SmartMarketOOPS.git
   cd SmartMarketOOPS
   ```

2. Install dependencies
   ```
   # Backend
   cd backend
   npm install
   
   # Frontend
   cd ../frontend
   npm install
   
   # ML components
   cd ../ml
   pip install -r requirements.txt
   ```

3. Set up environment variables
   ```
   cp example.env .env
   # Edit .env with your configuration
   ```

4. Set up database
   ```
   cd backend
   npx prisma migrate dev
   ```

5. Install k6 for performance testing
   ```
   cd backend
   ./scripts/install-k6.sh
   ```

### Running the Application

1. Start the backend server
   ```
   cd backend
   npm run dev
   ```

2. Start the frontend
   ```
   cd frontend
   npm run dev
   ```

3. Start the ML service
   ```
   cd ml
   python -m src.api.app
   ```

## Performance Testing

The platform includes a comprehensive performance testing and optimization framework:

### Running Tests via API

```bash
# Create and run a performance test
curl -X POST http://localhost:3001/api/performance/tests -H "Content-Type: application/json" -d '{
  "name": "API Latency Test",
  "testType": "API_LATENCY",
  "duration": 60,
  "concurrency": 20,
  "targetEndpoint": "/api/health"
}'
```

### Command Line Testing

```bash
# Run an API latency test
cd backend
npx ts-node scripts/run-performance-test.ts --type=API_LATENCY --endpoint=/api/health

# Run a load test
npx ts-node scripts/run-performance-test.ts --type=LOAD_TEST --endpoint=/api/bridge/predict
```

### Viewing Results

Performance test results are available in the database and through the API:

```bash
# Get all test results
curl http://localhost:3001/api/performance/tests
```

See [Performance Testing Documentation](docs/performance-testing-framework.md) for more details.

## API Documentation

API documentation is available at `/api/docs` when running the backend server.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Development Status

### ML Backend

We are currently working on Task 11: "Update ML Backend for Trading Predictions". Recent efforts have focused on:

- Updating the CNN-LSTM model for classification output.
- Implementing confidence score calculation.
- Debugging the model loading and prediction service, resolving issues with model parameters and feature engineering.
- Addressing class imbalance in the training data for 15-minute time frame (with limited success).

The next phase (Subtask 11.5) involves transitioning to larger time frames (1h, 4h, 1d) to capture "long and strong trades". This will require updates to the data pipeline for fetching and processing data for these time frames, redefining the target variable, and potentially revisiting technical indicators.
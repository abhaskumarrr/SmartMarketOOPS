# SmartMarketOOPS Installation Guide

This guide provides detailed instructions for setting up and running the SmartMarketOOPS trading system. It covers both development and production deployments.

## Prerequisites

Before beginning installation, ensure you have the following:

- **Node.js** (v18.x or later)
- **Python** (v3.10 or later)
- **Docker** and **Docker Compose**
- **Git**
- **PostgreSQL** (optional for local development without Docker)

## Quick Start with Setup Script

The fastest way to get started is using our setup script:

```bash
# Clone the repository
git clone https://github.com/yourusername/SmartMarketOOPS.git
cd SmartMarketOOPS

# Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The setup script automatically:
- Checks for required dependencies
- Sets up environment variables
- Installs backend and frontend dependencies
- Sets up Python environment
- Creates necessary directories
- Validates database connections

## Manual Installation

If you prefer to install manually, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SmartMarketOOPS.git
cd SmartMarketOOPS
```

### 2. Set Up Environment Variables

```bash
# Copy the example environment file
cp example.env .env

# Edit the .env file with your configuration
nano .env
```

Required variables to configure:
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET`: Secret for JWT token generation
- `DELTA_API_KEY` and `DELTA_API_SECRET`: Your Delta Exchange API credentials
- `NODE_ENV`: Set to "development" or "production"

### 3. Install Backend Dependencies

```bash
cd backend
npm install
```

### 4. Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

### 5. Set Up Python Environment

```bash
cd ../ml

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6. Set Up Database

```bash
# Using Docker Compose to start PostgreSQL
cd ..
docker-compose up -d postgres

# Run database migrations
cd backend
npx prisma migrate dev
```

## Development Mode

### Starting the Backend

```bash
cd backend
npm run dev
```

The backend will be available at http://localhost:3006

### Starting the Frontend

```bash
cd frontend
npm start
```

The frontend will be available at http://localhost:3000

### Starting the ML Service

```bash
cd ml
source venv/bin/activate  # On macOS/Linux
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The ML service will be available at http://localhost:8000

### Starting the Week 2 Integration Manager

```bash
cd ml
source venv/bin/activate  # On macOS/Linux
python week2_integration_launcher.py
```

## Production Deployment

### Using Docker Compose

For production deployment, we use Docker Compose:

```bash
# Build and start all services
docker-compose -f docker-compose.production.yml up -d
```

### Manual Production Setup

1. Build the frontend:
```bash
cd frontend
npm run build
```

2. Set up a production-ready database:
```bash
cd backend
npx prisma migrate deploy
```

3. Start the backend in production mode:
```bash
cd backend
npm run start:prod
```

4. Start the ML service with a production WSGI server:
```bash
cd ml
gunicorn -k uvicorn.workers.UvicornWorker -w 4 src.api.app:app
```

5. Use a reverse proxy like Nginx to serve the frontend build files and route API requests.

## Verifying Installation

After installation, verify your setup:

1. Access the frontend at http://localhost:3000
2. Check the backend health endpoint: http://localhost:3006/api/health
3. Verify the ML service: http://localhost:8000/health

## Configuring Delta Exchange

1. Create an account on Delta Exchange
2. Generate API keys (with trading permissions for production)
3. Add your API keys to the `.env` file
4. Test the connection using the dashboard's connection test feature

## Multi-Symbol Configuration

To configure trading for multiple symbols, edit the configuration in `ml/src/config/symbols.json`:

```json
{
  "BTCUSDT": {
    "confidence_threshold": 0.75,
    "position_size_pct": 3.0,
    "market_cap_tier": "large"
  },
  "ETHUSDT": {
    "confidence_threshold": 0.7,
    "position_size_pct": 2.5,
    "market_cap_tier": "large"
  },
  "SOLUSDT": {
    "confidence_threshold": 0.65,
    "position_size_pct": 2.0,
    "market_cap_tier": "mid"
  },
  "ADAUSDT": {
    "confidence_threshold": 0.65,
    "position_size_pct": 1.5,
    "market_cap_tier": "mid"
  }
}
```

## Troubleshooting

For common issues and solutions, please refer to our [Troubleshooting Guide](TROUBLESHOOTING.md).

## Next Steps

After installation:

1. Review the [User Guidelines](../USER_GUIDELINES.md)
2. Explore the [Trading Strategy Documentation](TRADING_STRATEGY.md)
3. Check out [API Reference](API_REFERENCE.md) for available endpoints 
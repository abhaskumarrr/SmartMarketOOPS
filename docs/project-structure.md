# Project Structure Documentation

## Overview
This document outlines the organization and structure of the SmartMarketOOPS project. The project follows a modular architecture with clear separation of concerns.

## Directory Structure

```
.
├── backend/                 # Backend server implementation
│   ├── src/                # Source code
│   ├── tests/              # All backend tests (unit, integration)
│   ├── prisma/             # Database schema and migrations
│   └── generated/          # Generated code (e.g., Prisma client)
│
├── frontend/               # Frontend application
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── components/        # Reusable React components
│
├── ml/                    # Machine Learning services
│   ├── src/              # ML model implementations
│   ├── data/             # Training and validation data
│   └── models/           # Trained model storage
│
├── monitoring/           # Monitoring and observability
│   ├── grafana/         # Grafana dashboards
│   └── prometheus/      # Prometheus configuration
│
├── scripts/             # Utility scripts
│   ├── setup.sh        # Main setup script
│   └── start.sh        # Main startup script
│
├── docs/               # Project documentation
├── data/              # Application data
└── config/            # Configuration files
```

## Key Components

### Backend
- `backend/src/`: Contains the main application logic
- `backend/tests/`: Comprehensive test suite
- `backend/prisma/`: Database schema and migrations

### Frontend
- `frontend/src/`: React application source code
- `frontend/components/`: Reusable UI components
- `frontend/public/`: Static assets and resources

### Machine Learning
- `ml/src/`: ML model implementations and training code
- `ml/data/`: Training and validation datasets
- `ml/models/`: Storage for trained models

### Monitoring
- Centralized monitoring solution using Grafana and Prometheus
- Custom dashboards for system metrics

### Scripts
The project includes two main scripts for setup and execution:

1. `setup.sh`: Consolidated setup script that:
   - Checks system requirements
   - Sets up Python environment
   - Sets up Node.js environment
   - Configures Docker environment
   - Sets up infrastructure
   - Validates the setup

2. `start.sh`: Consolidated start script that:
   - Starts all backend services
   - Launches frontend development server
   - Initializes monitoring
   - Runs pre-launch tests
   - Supports both development and production modes

## Configuration
- Environment variables are managed through `.env` files
- Configuration files are stored in the `config/` directory
- Each component can have its own specific configuration

## Data Management
- Raw data is stored in `data/raw/`
- Processed data in `data/processed/`
- Validation data in `data/validation/`

## Documentation
- Technical documentation in `docs/`
- API documentation generated from code
- Component-specific documentation in respective directories

## Development Workflow
1. Run `./scripts/setup.sh` to set up the development environment
2. Run `./scripts/start.sh` to start all services
3. Access the application at `http://localhost:3000`
4. Monitor the system at `http://localhost:9090`

## Testing
- Unit tests in respective `tests/` directories
- Integration tests in `backend/tests/integration/`
- End-to-end tests in `tests/`

## Deployment
- Docker containers for all services
- Kubernetes configurations in `k8s/` directory
- Monitoring setup included in deployment 
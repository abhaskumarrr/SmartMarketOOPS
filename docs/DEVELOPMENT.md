# SMOOPs Development Guide

This guide provides comprehensive information for setting up and working with the SMOOPs (Smart Money Order Blocks) trading bot codebase.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Architecture Overview](#architecture-overview)
- [Common Development Tasks](#common-development-tasks)
- [Testing](#testing)
- [Debugging](#debugging)
- [Deployment](#deployment)
- [Common Issues and Solutions](#common-issues-and-solutions)

## Prerequisites

Before you begin, make sure you have the following installed:

- **Node.js** (v18+) - [Download](https://nodejs.org/)
- **Python** (v3.10+) - [Download](https://www.python.org/downloads/)
- **Docker** and **Docker Compose** - [Download](https://docs.docker.com/get-docker/)
- **Git** - [Download](https://git-scm.com/downloads)

## Environment Setup

### Quick Setup

For a streamlined setup process, run:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev

# Run the development setup script
npm run dev:setup
```

This script will:
1. Check for required dependencies
2. Create a `.env` file from `example.env` if it doesn't exist
3. Generate an encryption key for securing API credentials
4. Install dependencies for all services
5. Initialize the database
6. Set up ML directories

### Manual Setup

If you prefer to set up your environment manually:

1. **Clone the repository**

```bash
git clone https://github.com/abhaskumarrr/SMOOPs_dev.git
cd SMOOPs_dev
```

2. **Create environment file**

```bash
cp example.env .env
# Edit .env with your credentials and settings
```

3. **Install dependencies**

```bash
# Root project dependencies
npm install

# Backend dependencies
cd backend && npm install

# Frontend dependencies
cd frontend && npm install

# ML dependencies
cd ml && pip install -r requirements.txt
```

4. **Generate encryption key**

```bash
npm run generate-key
```

5. **Initialize the database**

```bash
# Start PostgreSQL container
docker-compose up -d postgres

# Run database migrations
npm run db:migrate
```

### Environment Variables

See [Environment Setup Guide](./environment-setup.md) for detailed information about configuration options.

## Project Structure

The project follows a monorepo structure with three main components:

- **Backend** (`/backend`): Node.js/Express API with Prisma ORM
- **Frontend** (`/frontend`): Next.js web application
- **ML Service** (`/ml`): Python-based machine learning service

For a detailed breakdown of the directory structure, see [Project Structure Documentation](./project-structure.md).

## Development Workflow

### Starting the Development Servers

```bash
# Start all services in development mode
npm run dev

# Start individual services
npm run dev:backend
npm run dev:frontend
npm run dev:ml

# Start services using Docker
npm run docker:up
```

### Accessing the Services

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:3001
- **ML Service**: http://localhost:3002

### Development Tasks Script

We provide a convenient script for common development tasks:

```bash
# Show available tasks
npm run dev:tasks

# Or directly run a specific task
npm run dev:tasks start
npm run dev:tasks logs:backend
```

See all available tasks by running `npm run dev:tasks help`.

## Architecture Overview

### Backend (Node.js/Express)

The backend serves as the central API for all trading operations, data access, and user management.

- **API Endpoints**: RESTful endpoints for data access and trading operations
- **Database Access**: Uses Prisma ORM to interact with PostgreSQL
- **Authentication**: JWT-based authentication system
- **WebSockets**: Real-time data streaming for charts and trading signals
- **Encryption**: Secure storage for exchange API credentials

### Frontend (Next.js)

The frontend provides a rich dashboard UI for monitoring trading activities.

- **Dashboard**: Main interface for viewing market data and trading signals
- **Charts**: TradingView-style charts for price analysis
- **Trading Interface**: Controls for configuring and monitoring trading bots
- **Authentication**: User login, registration and profile management

### ML Service (Python/PyTorch)

The ML service handles all machine learning aspects of the trading system.

- **Model Training**: Training pipelines for Smart Money Concepts models
- **Prediction API**: Real-time signal generation for trading
- **Backtesting**: Framework for evaluating models on historical data
- **Data Processing**: Market data acquisition and preprocessing

## Common Development Tasks

### Database Operations

```bash
# Run migrations
npm run db:migrate

# Reset database (deletes all data)
npm run db:reset

# Generate Prisma client
npm run db:generate
```

### Docker Operations

```bash
# Start all services
npm run docker:up

# Stop all services
npm run docker:down

# View logs
npm run docker:logs

# Restart services
npm run docker:restart
```

### Code Quality

```bash
# Run linter
npm run lint

# Run linter with auto-fix
npm run lint:fix

# Run tests
npm run test
```

## Testing

### Running Tests

```bash
# Run all tests
npm run test

# Run tests for specific service
npm run test:backend
npm run test:frontend
npm run test:ml
```

### Test Structure

- Backend tests use Jest and SuperTest
- Frontend tests use Jest and React Testing Library
- ML tests use unittest or pytest

## Debugging

### Backend Debugging

- Use `console.log()` or debugger in your IDE
- Inspect Docker logs: `npm run docker:logs backend`

### Frontend Debugging

- Use browser developer tools
- React DevTools extension is recommended
- NextJS debugging is available at `http://localhost:3000/_next/webpack-hmr`

### ML Service Debugging

- Use logging: `import logging; logging.debug("message")`
- For Visual Studio Code debugging, a `launch.json` configuration is provided

## Deployment

See [Deployment Guide](./deployment-guide.md) for detailed instructions on deploying to various environments.

## Common Issues and Solutions

### Database Connection Issues

**Problem**: Cannot connect to database  
**Solution**: Ensure PostgreSQL container is running (`docker ps`). Check your `.env` file for correct database credentials.

### Node.js Memory Issues

**Problem**: Node.js crashes with "JavaScript heap out of memory"  
**Solution**: Increase Node.js memory limit with `NODE_OPTIONS=--max_old_space_size=4096` before the command.

### Python Package Conflicts

**Problem**: Conflicts between Python package versions  
**Solution**: Use a virtual environment (`python -m venv env`) and install packages from requirements.txt.

### Docker Permission Issues

**Problem**: Permission denied errors when running Docker  
**Solution**: Make sure your user is in the Docker group or use sudo (not recommended for development).

---

For more detailed documentation, see the other files in the `docs/` directory. If you encounter any issues not covered here, please check the project issues on GitHub or contact the maintainers. 
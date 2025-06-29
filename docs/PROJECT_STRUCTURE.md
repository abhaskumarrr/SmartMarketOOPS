# SmartMarketOOPS Project Structure

## Overview

This document provides a guide to the organized file structure of the SmartMarketOOPS project. The project is a monorepo containing three primary services: `frontend`, `backend`, and `ml`.

## High-Level Directory Structure

```
SmartMarketOOPS/
├── 📁 backend/          # Node.js/Express backend for API and orchestration
├── 📁 frontend/         # Next.js frontend for the user dashboard
├── 📁 ml/               # Python service for all ML and AI logic
├── 📁 data/             # All data, including market data, logs, and models
├── 📁 docs/             # Project documentation
├── 📁 scripts/          # Utility and startup scripts
├── 📄 .env              # Centralized environment variables (git-ignored)
├── 📄 example.env       # Template for environment variables
├── 📄 gemini-plan.md     # The development plan for the AI agent
└── 📄 package.json       # Root package.json for workspace management
```

## Key Directories Explained

### 📁 `backend/`
Contains the Node.js/Express backend application.
- **`src/`**: The main source code for the backend.
  - **`routes/`**: Defines all the API endpoints.
  - **`services/`**: Contains the core business logic, including exchange integration.
  - **`prisma/`**: The Prisma schema and database migrations.
- **`package.json`**: Defines dependencies and scripts for the backend service.

### 📁 `frontend/`
Contains the Next.js frontend application.
- **`src/`**: The main source code for the frontend.
  - **`app/`**: The core of the Next.js application, using the App Router.
  - **`components/`**: Reusable React components, including charts and UI elements.
  - **`lib/`**: Utility functions and hooks for the frontend.
- **`package.json`**: Defines dependencies and scripts for the frontend service.

### 📁 `ml/`
Contains the Python-based machine learning service.
- **`src/`**: The main source code for the ML service.
  - **`api/`**: The FastAPI application for serving the models.
  - **`training/`**: Scripts for training the various models (imitation, RL, etc.).
  - **`rl/`**: The reinforcement learning environment and agent.
  - **`meta_learning/`**: Scripts for meta-learning and model adaptation.
- **`models/`**: Saved model files.
- **`requirements.txt`**: Python dependencies for the ML service.

### 📁 `data/`
This directory is the central repository for all data used in the project.
- **`raw/`**: Raw, unprocessed market data (e.g., from CCXT).
- **`expert_trades/`**: Logs of expert trades for imitation learning.
- **`rlhf_feedback/`**: Human feedback data for the RLHF loop.
- **`backtest_results/`**: Output from backtesting runs.

### 📁 `scripts/`
Contains all the main scripts for managing the project.
- **`start_all.sh`**: The unified script for starting all services.
- **`download_data.py`**: The script for downloading real market data.
- Other utility scripts for deployment, testing, and maintenance.

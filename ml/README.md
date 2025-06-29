# Machine Learning Service

This directory contains the Python-based machine learning service for the SmartMarketOOPS project.

## Overview

This service is the "brain" of the trading agent. It is responsible for:
- Training models using imitation learning, reinforcement learning, and meta-learning.
- Serving the trained models via a FastAPI endpoint.
- Generating trading signals based on market data analysis.

## Getting Started

This service is started as part of the unified startup script in the root directory. To run it independently:

```bash
# From the ml/ directory
python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

For more detailed information on the architecture and project structure, please see the main documentation in the `docs/` directory at the root of the project.
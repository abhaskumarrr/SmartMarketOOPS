# Backend Service

This directory contains the Node.js/Express backend for the SmartMarketOOPS project.

## Overview

The backend service is the central orchestrator of the system. It is responsible for:
- Providing the API for the frontend.
- Managing user authentication and API keys.
- Connecting to cryptocurrency exchanges.
- Communicating with the ML service to get trading signals.

## Getting Started

This service is started as part of the unified startup script in the root directory. To run it independently:

```bash
# From the backend/ directory
npm run dev
```

For more detailed information on the architecture and project structure, please see the main documentation in the `docs/` directory at the root of the project.
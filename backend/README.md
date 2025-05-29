# SMOOPs Backend Service

This directory contains the Node.js/Express backend API for the SMOOPs trading bot.

## Structure

- `src/` - Source code for the backend API
  - `controllers/` - API endpoint controllers
  - `middleware/` - Express middleware
  - `routes/` - API route definitions
  - `services/` - Business logic and data access
  - `utils/` - Utility functions
  - `scripts/` - Utility scripts

- `prisma/` - Prisma ORM schema and migrations
  - `migrations/` - Database migration files
  - `schema.prisma` - Database schema definition

- `.keys/` - Storage for encrypted API keys (not in git)

## Key Features

- **RESTful API** for trading operations and data access
- **WebSocket support** for real-time market data and trading signals
- **Secure API key management** with encryption
- **Database access** via Prisma ORM

## Getting Started

### Prerequisites

- Node.js 20+
- PostgreSQL database

### Installation

```bash
# Install dependencies
npm install

# Generate Prisma client
npx prisma generate

# Set up the database
npx prisma migrate dev --name init
```

### Running the Service

```bash
# Start in development mode
npm run dev

# Start in production mode
npm start
```

## API Endpoints

Key API endpoints include:

- `/api/auth` - Authentication endpoints
- `/api/keys` - API key management
- `/api/delta` - Delta Exchange integration
- `/api/signals` - Trading signals

See the API documentation in `/docs/api-documentation.md` for details. 
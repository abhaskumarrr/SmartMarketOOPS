# Environment Setup Guide

This guide details how to set up and manage environment variables for the SMOOPs trading bot project.

## Overview

The project uses environment variables for configuration across all services:
- Backend (Node.js)
- Frontend (Next.js)
- ML Service (Python)
- Database (PostgreSQL)

## Quick Setup

For a standard development setup, run:

```bash
# Copy example environment file
cp example.env .env

# Run the setup script (interactive)
npm run setup

# Validate the environment configuration
npm run check-env

# Generate an encryption key (for API key storage)
npm run generate-key
```

## Environment Files

- `.env` - Main environment file for all services
- `example.env` - Example file with documented environment variables
- `.env.local` (optional) - Override values for local development (not checked into git)
- `.env.production` (optional) - Production-specific values

## Required Environment Variables

### Database Configuration
```bash
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=smoops
POSTGRES_PORT=5432
DATABASE_URL="postgresql://postgres:postgres@postgres:5432/smoops?schema=public"
```

### Service Ports
```bash
# Service Ports
PORT=3001          # Backend API port
ML_PORT=3002       # ML service port
FRONTEND_PORT=3000 # Frontend port
```

### Security Configuration
```bash
# Security Configuration
NODE_ENV=development  # 'development', 'production', or 'testing'
ENCRYPTION_MASTER_KEY=your_master_key_here  # Generate with 'npm run generate-key'
```

### Frontend Configuration
```bash
# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:3001  # Backend API URL for frontend
```

### ML Service Configuration
```bash
# ML Service Configuration
PYTHONUNBUFFERED=1     # Ensures Python logs appear immediately
TORCH_MPS_ENABLE=1     # For Apple Silicon GPU acceleration
```

### Exchange Configuration
```bash
# Delta Exchange Configuration
DELTA_EXCHANGE_TESTNET=true
DELTA_EXCHANGE_API_KEY=your_api_key
DELTA_EXCHANGE_API_SECRET=your_api_secret
```

## Environment Management Scripts

The project includes several scripts to help manage environment variables:

### `scripts/setup-env.sh`
Interactive script to set up the environment files for different deployment targets:
- Development (default)
- Production
- Testing

```bash
npm run setup
```

### `scripts/check-env.js`
Validates the environment configuration to ensure all required variables are set and formatted correctly:

```bash
npm run check-env
```

### `scripts/generate-encryption-key.js`
Generates a secure random key for encrypting sensitive data:

```bash
npm run generate-key
```

## Per-Service Environment Handling

### Backend
- Uses `backend/src/utils/env.js` to load and validate environment variables
- Automatically loads from root `.env` file
- Validates critical variables based on NODE_ENV
- Provides sensible defaults for development

### Frontend
- Uses `frontend/lib/env.js` for client-safe environment variables
- Only exposes variables with the `NEXT_PUBLIC_` prefix to the browser
- Validates client-side environment on startup

### ML Service
- Uses Python's `dotenv` module to load environment variables
- Validates required variables at startup
- Adjusts settings based on environment type

## Docker Environment

When using Docker Compose, environment variables flow from:
1. Host machine's environment
2. `.env` file in project root
3. Values set directly in `docker-compose.yml`

The `${VAR:-default}` syntax in docker-compose.yml provides fallbacks if variables are not set.

## Production Considerations

For production deployments:

1. **Never commit sensitive values to version control**
2. Use a secure method to manage production secrets:
   - Environment variables set on the host
   - Secret management service (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Docker secrets for Docker Swarm deployments
3. Set `NODE_ENV=production` to enable stricter validation
4. Generate unique keys for:
   - `ENCRYPTION_MASTER_KEY` - For API key encryption
   - `JWT_SECRET` - For authentication (if used)
5. Use complex passwords for database access
6. Restrict CORS settings by setting explicit origins

## Troubleshooting

### Common Issues

**Database Connection Problems**
- Check `DATABASE_URL` format and credentials
- Ensure the database server is running
- Verify network connectivity between services

**API Key Encryption Issues**
- If `ENCRYPTION_MASTER_KEY` changes, existing encrypted data can't be decrypted
- If needed, run migration scripts to re-encrypt data with new key

**Environment Not Loading**
- Verify `.env` file exists in project root
- Check file permissions
- Ensure Docker is using the correct env file 
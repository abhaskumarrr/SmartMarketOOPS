# Backend API Routes

This directory contains all API route definitions for the SmartMarketOOPS application.

## Delta Exchange API Routes

All Delta Exchange API routes are protected by authentication middleware.

### Market Data Endpoints

- `GET /api/delta/products` - Get all available products/markets
- `GET /api/delta/products/:id/orderbook` - Get order book for a specific product
- `GET /api/delta/products/:id/trades` - Get recent trades for a specific product
- `GET /api/delta/market-data?symbols=BTC,ETH` - Get combined market data for multiple symbols

### Account Endpoints

- `GET /api/delta/balance` - Get account wallet balance
- `GET /api/delta/positions` - Get active positions

### Order & Trading Endpoints

- `GET /api/delta/orders` - Get active orders (query params: status, symbol, etc.)
- `GET /api/delta/orders/history` - Get order history (query params: symbol, status, limit, etc.)
- `POST /api/delta/orders` - Create a new order
- `DELETE /api/delta/orders/:id` - Cancel a specific order
- `DELETE /api/delta/orders` - Cancel all orders (optional query params for filtering)
- `GET /api/delta/fills` - Get trade history/fills (query params: symbol, limit, etc.)

## API Key Routes

- `GET /api/keys` - Get user's API keys (masked)
- `POST /api/keys` - Add a new API key
- `DELETE /api/keys/:id` - Revoke/delete an API key

## Authentication Routes

- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login and get access token

## User Routes

- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update user profile

## Bot Routes

- `GET /api/bots` - Get all user's bots
- `GET /api/bots/:id` - Get a specific bot
- `POST /api/bots` - Create a new bot
- `PUT /api/bots/:id` - Update a bot
- `DELETE /api/bots/:id` - Delete a bot

## Health Routes

- `GET /api/health` - Get system health status
- `GET /api/health/db` - Get database connection status
- `GET /api/health/version` - Get API version information 
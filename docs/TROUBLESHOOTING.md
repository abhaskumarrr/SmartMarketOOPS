# SmartMarketOOPS Troubleshooting Guide

This guide provides solutions for common issues you might encounter when setting up, running, or deploying the SmartMarketOOPS trading system.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Connection Problems](#connection-problems)
3. [WebSocket Issues](#websocket-issues)
4. [Database Issues](#database-issues)
5. [Delta Exchange Integration](#delta-exchange-integration)
6. [ML Service Issues](#ml-service-issues)
7. [Docker Deployment Problems](#docker-deployment-problems)
8. [Performance Issues](#performance-issues)

## Installation Issues

### Node.js or Python Version Conflicts

**Symptoms**: Error messages about incompatible versions, syntax errors in code that should be valid.

**Solutions**:
- Verify Node.js version: `node --version` (should be v18.x or later)
- Verify Python version: `python --version` (should be v3.10 or later)
- Use nvm to switch Node.js versions: `nvm use 18`
- Use pyenv to switch Python versions: `pyenv local 3.10.x`

### Dependency Installation Failures

**Symptoms**: `npm install` or `pip install` fails with error messages.

**Solutions**:
- Clear NPM cache: `npm cache clean --force`
- Update NPM: `npm install -g npm@latest`
- Try using a different Python package installer: `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`
- Check for Python wheel issues and install required build tools for your OS

## Connection Problems

### API Connection Failures

**Symptoms**: "Cannot connect to server" errors, timeout issues.

**Solutions**:
- Verify all services are running (`npm run dev`, ML service, etc.)
- Check correct ports are being used and not blocked by firewall
- Verify proxy settings if using a corporate network
- Check CORS settings in the backend if frontend can't connect

### WebSocket Connection Issues

**Symptoms**: Real-time updates not working, "WebSocket connection failed" errors.

**Solutions**:
- Ensure WebSocket server is running
- Check for correct WebSocket protocol (ws:// or wss://)
- Verify the connection URL matches the environment (development/production)
- Implement the reconnection logic from `frontend/src/hooks/useWebSocket.ts`

```javascript
// Sample reconnection logic
const connectWebSocket = () => {
  socket = new WebSocket(wsUrl);
  
  socket.onopen = () => {
    console.log('WebSocket connected');
    backoffDelay = 1000; // Reset backoff delay on successful connection
  };
  
  socket.onclose = () => {
    console.log(`WebSocket disconnected. Reconnecting in ${backoffDelay/1000}s...`);
    setTimeout(connectWebSocket, backoffDelay);
    // Exponential backoff with max of 30 seconds
    backoffDelay = Math.min(backoffDelay * 1.5, 30000);
  };
};
```

## Database Issues

### Migration Failures

**Symptoms**: `prisma migrate dev` command fails, database tables not created.

**Solutions**:
- Verify PostgreSQL is running: `docker-compose ps`
- Check database connection string in `.env` file
- Ensure database exists: `createdb smartmarketoops`
- Reset migrations if needed: `npx prisma migrate reset`

### Connection Pooling Issues

**Symptoms**: "Too many clients already" errors, connection timeouts.

**Solutions**:
- Adjust connection pool settings in Prisma configuration
- Check for connection leaks in code (connections not being closed)
- Restart the PostgreSQL service: `docker-compose restart postgres`

## Delta Exchange Integration

### Authentication Failures

**Symptoms**: "Invalid API key" errors, 401 Unauthorized responses.

**Solutions**:
- Verify API key and secret are correctly set in `.env` file
- Check that API keys have appropriate permissions (read/trade)
- Ensure using testnet keys for testnet environment and production keys for production
- Verify API key has not expired or been revoked

### Order Placement Failures

**Symptoms**: Orders not being placed, error responses from Delta Exchange API.

**Solutions**:
- Check account has sufficient balance
- Verify product ID is correct for the symbol you're trading
- Ensure order parameters are valid (price, size, etc.)
- Check for rate limiting issues (too many requests)

## ML Service Issues

### Prediction Service Not Working

**Symptoms**: No predictions being generated, timeout errors when requesting predictions.

**Solutions**:
- Verify ML service is running: `curl http://localhost:8000/health`
- Check model files exist in the correct location
- Ensure Python environment has all required dependencies
- Check ML service logs for specific error messages

### Model Training Issues

**Symptoms**: Training fails, poor model performance.

**Solutions**:
- Verify training data is available and correctly formatted
- Check for GPU availability if using GPU-accelerated training
- Increase logging verbosity to debug training process
- Check for memory issues during training

## Docker Deployment Problems

### Container Startup Failures

**Symptoms**: Containers exit immediately, services not accessible.

**Solutions**:
- Check container logs: `docker-compose logs [service_name]`
- Verify environment variables are correctly set
- Check for port conflicts with other services
- Ensure volume mounts are correctly configured

### Volume Permission Issues

**Symptoms**: "Permission denied" errors in container logs.

**Solutions**:
- Check ownership of mounted directories
- Set appropriate permissions: `chmod -R 777 ./data` (for development only)
- Use Docker user mapping to match container user with host user

## Performance Issues

### Slow API Responses

**Symptoms**: Frontend feels sluggish, API requests take a long time.

**Solutions**:
- Enable API response caching
- Optimize database queries with proper indexes
- Implement pagination for large data sets
- Scale services horizontally if needed

### High CPU/Memory Usage

**Symptoms**: System resources maxed out, services becoming unresponsive.

**Solutions**:
- Implement request throttling
- Optimize heavy calculations
- Check for memory leaks in long-running processes
- Scale up hardware resources if needed

## WebSocket Reconnection Issues

### Disconnections in Production

**Symptoms**: WebSocket connections drop frequently in production.

**Solutions**:
- Implement heartbeat mechanism to keep connections alive
- Use WebSocket ping/pong frames
- Configure proper timeouts on load balancers and proxies
- Implement exponential backoff for reconnection attempts

## Real Market Data Integration

### Missing or Delayed Market Data

**Symptoms**: Charts not updating, stale prices, gaps in historical data.

**Solutions**:
- Check exchange API status
- Verify WebSocket subscriptions are correct
- Implement data validation to detect and fill gaps
- Use multiple data sources for redundancy

```python
# Sample code to check data freshness
def is_data_stale(last_update_time, max_staleness_seconds=60):
    """Check if data is stale"""
    current_time = datetime.utcnow()
    time_difference = (current_time - last_update_time).total_seconds()
    return time_difference > max_staleness_seconds
```

## Getting Additional Help

If you're still experiencing issues after trying these solutions:

1. Check the [GitHub Issues](https://github.com/yourusername/SmartMarketOOPS/issues) for similar problems and solutions
2. Join our [Discord community](https://discord.gg/smartmarketoops) for real-time support
3. Contact our support team at support@smartmarketoops.com

## Reporting Bugs

When reporting bugs, please include:

1. Detailed description of the issue
2. Steps to reproduce
3. Environment information (OS, Node.js version, Python version)
4. Relevant logs and error messages
5. Screenshots if applicable 
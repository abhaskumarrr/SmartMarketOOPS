#!/usr/bin/env node
/**
 * Simple SmartMarketOOPS Backend Server
 * Minimal implementation for testing
 */

import express from 'express';
import cors from 'cors';
import { createLogger } from './src/utils/logger';

const logger = createLogger('SimpleServer');
const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'SmartMarketOOPS Backend',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    environment: process.env.NODE_ENV || 'development',
    delta_exchange: {
      testnet: process.env.DELTA_TESTNET === 'true',
      api_key_configured: !!process.env.DELTA_API_KEY
    }
  });
});

// Test Delta Exchange connection
app.get('/api/delta/test', async (req, res) => {
  try {
    const fetch = (await import('node-fetch')).default;
    const response = await fetch('https://testnet-api.delta.exchange/v2/products');
    const data = await response.json();
    
    res.json({
      status: 'success',
      message: 'Delta Exchange connection working',
      products_count: data.result?.length || 0,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: 'Delta Exchange connection failed',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// ML service proxy
app.post('/api/ml/predict', async (req, res) => {
  try {
    const fetch = (await import('node-fetch')).default;
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(req.body)
    });
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: 'ML service connection failed',
      error: error.message
    });
  }
});

// System metrics
app.get('/api/metrics/system', (req, res) => {
  res.json({
    status: 'success',
    data: {
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      platform: process.platform,
      nodeVersion: process.version,
      timestamp: new Date().toISOString(),
      environment: {
        node_env: process.env.NODE_ENV,
        delta_testnet: process.env.DELTA_TESTNET,
        database_url: !!process.env.DATABASE_URL,
        redis_url: !!process.env.REDIS_URL
      }
    }
  });
});

// Trading configuration
app.get('/api/trading/config', (req, res) => {
  res.json({
    status: 'success',
    config: {
      exchange: 'Delta Exchange India Testnet',
      testnet: process.env.DELTA_TESTNET === 'true',
      max_position_size: process.env.MAX_POSITION_SIZE || 50,
      risk_per_trade: process.env.RISK_PER_TRADE || 0.015,
      max_positions: process.env.MAX_POSITIONS || 2,
      initial_capital: process.env.INITIAL_CAPITAL || 1000
    }
  });
});

// Error handling middleware
app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Server error:', error);
  res.status(500).json({
    status: 'error',
    message: 'Internal server error',
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    status: 'error',
    message: 'Endpoint not found',
    path: req.originalUrl,
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  logger.info(`🚀 SmartMarketOOPS Backend running on port ${PORT}`);
  logger.info(`📊 Health check: http://localhost:${PORT}/api/health`);
  logger.info(`🔗 Delta test: http://localhost:${PORT}/api/delta/test`);
  logger.info(`🧠 ML proxy: http://localhost:${PORT}/api/ml/predict`);
  logger.info(`📈 Environment: ${process.env.NODE_ENV || 'development'}`);
  logger.info(`🌐 Delta Testnet: ${process.env.DELTA_TESTNET === 'true' ? 'Enabled' : 'Disabled'}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});
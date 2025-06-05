/**
 * Main Server Entry Point
 * Sets up and starts the Express server with API routes
 */

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import path from 'path';
import dotenv from 'dotenv';
import http from 'http';
import { createWriteStream } from 'fs';
import compression from 'compression';
import prisma from './utils/prismaClient';
import cookieParser from 'cookie-parser';
import { errorHandler, notFoundHandler } from './middleware/errorHandler';
import { secureCookieParser, sessionActivity, setDeviceIdCookie } from './middleware/sessionMiddleware';
import optimizationMiddleware from './middleware/optimizationMiddleware';
import { createCacheService } from './services/cacheService';
import { createDatabaseOptimizationService } from './services/databaseOptimizationService';

// Load environment variables
dotenv.config({
  path: path.resolve(__dirname, '../../.env')
});

// Import routes - gradually re-enabling fixed routes
import healthRoutes from './routes/healthRoutes';
import authRoutes from './routes/authRoutes';
import userRoutes from './routes/userRoutes';
import apiKeyRoutes from './routes/apiKeyRoutes';
import metricsRoutes from './routes/metricsRoutes';
import roleRoutes from './routes/roleRoutes';
import sessionRoutes from './routes/sessionRoutes';
import signalRoutes from './routes/signalRoutes';
import riskRoutes from './routes/riskRoutes';
import strategyRoutes from './routes/strategyRoutes';
import bridgeRoutes from './routes/bridge/bridgeRoutes';
import performanceRoutes from './routes/performance/performanceRoutes';
import orderExecutionRoutes from './routes/trading/orderExecutionRoutes';
import tradingApiKeyRoutes from './routes/trading/apiKeyRoutes';
import botRoutes from './routes/botRoutes';
import auditRoutes from './routes/auditRoutes';
import tradesRoutes from './routes/trading/trades';
// import mlRoutes from './routes/mlRoutes';
// import marketDataRoutes from './routes/marketDataRoutes';
import tradingRoutes from './routes/tradingRoutes';
import tradingRoutesWorking from './routes/tradingRoutesWorking';
import deltaTradingRoutes from './routes/deltaTradingRoutes';
import mlRoutes from './routes/mlRoutes';
import marketDataRoutes from './routes/marketDataRoutes';
// Import other routes as needed

// Load socket initialization
const initializeWebsocketServer = require('./sockets/websocketServer').initializeWebsocketServer;

// Create Express app
const app = express();
const PORT = process.env.PORT || 3333;
const NODE_ENV = process.env.NODE_ENV || 'development';

// Initialize optimization services
const cacheService = createCacheService({
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
  keyPrefix: 'smartmarket:',
});

const dbOptimizationService = createDatabaseOptimizationService();

// Create HTTP server for Socket.IO
const server = http.createServer(app);

// Initialize WebSocket server
const io = initializeWebsocketServer(server);

// Logging middleware
const logStream = createWriteStream(path.join(__dirname, '../logs/server.log'), { flags: 'a' });
app.use((req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - start;
    const log = `${new Date().toISOString()} | ${req.method} ${req.url} ${res.statusCode} ${duration}ms\n`;

    logStream.write(log);

    if (NODE_ENV === 'development') {
      console.log(log);
    }
  });

  next();
});

// Performance and optimization middleware
app.use(optimizationMiddleware.performanceMonitor());
app.use(optimizationMiddleware.securityHeaders());
app.use(optimizationMiddleware.requestValidation());
app.use(optimizationMiddleware.requestTimeout(30000)); // 30 second timeout

// Compression middleware
app.use(compression({
  filter: (req, res) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return compression.filter(req, res);
  },
  threshold: 1024, // Only compress responses larger than 1KB
}));

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true,
  },
}));

// Rate limiting
app.use('/api/', optimizationMiddleware.createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
}));

// Stricter rate limiting for auth endpoints
app.use('/api/auth/', optimizationMiddleware.createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 50, // limit each IP to 50 auth requests per windowMs
}));

// CORS middleware
app.use(cors({
  origin: [
    process.env.CLIENT_URL || 'http://localhost:3000',
    'http://localhost:3001', // Frontend running on port 3001
    'http://localhost:3002',
    'http://localhost:3333'
  ],
  credentials: true,
  optionsSuccessStatus: 200,
}));

// Body parser middleware with size limits
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Cookie parser middleware (with signed cookies)
app.use(secureCookieParser);

// Set device ID cookie for session tracking
app.use(setDeviceIdCookie);

// Set trust proxy if behind a proxy
if (process.env.TRUST_PROXY === 'true') {
  app.set('trust proxy', 1);
}

// Track session activity for authenticated routes
app.use(sessionActivity);

// Root route handler - API welcome page
app.get('/', (req: Request, res: Response) => {
  res.status(200).json({
    name: 'SmartMarket OOPS API',
    version: '1.0.0',
    status: 'online',
    timestamp: new Date().toISOString(),
    routes: {
      health: '/api/health',
      auth: '/api/auth',
      users: '/api/users',
      apiKeys: '/api/api-keys',
      trading: {
        orders: '/api/orders',
        apiKeys: '/api/trading/api-keys',
        bots: '/api/bots',
        trades: '/api/trades'
      }
    }
  });
});

// Enhanced health check at root path
app.get('/health', optimizationMiddleware.healthCheck());

// Performance metrics endpoint
app.get('/metrics', (req: Request, res: Response) => {
  const metrics = optimizationMiddleware.getMetrics();
  const cacheStats = cacheService ? cacheService.getStats() : null;
  const dbStats = dbOptimizationService ? dbOptimizationService.getQueryStats() : null;

  res.json({
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    requests: metrics,
    cache: cacheStats,
    database: dbStats,
  });
});

// Cache management endpoints
app.post('/admin/cache/flush', async (req: Request, res: Response) => {
  try {
    if (cacheService) {
      await cacheService.flush();
      res.json({ success: true, message: 'Cache flushed successfully' });
    } else {
      res.status(503).json({ error: 'Cache service not available' });
    }
  } catch (error) {
    res.status(500).json({ error: 'Failed to flush cache' });
  }
});

// Use routes - gradually re-enabling fixed routes
app.use('/api/health', healthRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/api-keys', apiKeyRoutes);
app.use('/api', metricsRoutes);
app.use('/api/roles', roleRoutes);
app.use('/api/sessions', sessionRoutes);
app.use('/api/signals', signalRoutes);
app.use('/api/risk', riskRoutes);
app.use('/api/strategies', strategyRoutes);
app.use('/api/bridge', bridgeRoutes);
app.use('/api/performance', performanceRoutes);
app.use('/api/orders', orderExecutionRoutes);
app.use('/api/trading/api-keys', tradingApiKeyRoutes);
app.use('/api/bots', botRoutes);
app.use('/api/audit', auditRoutes);
app.use('/api/trades', tradesRoutes);
app.use('/api/ml', mlRoutes);
app.use('/api/market-data', marketDataRoutes);
app.use('/api/trading', tradingRoutes);
app.use('/api/trading-working', tradingRoutesWorking);
app.use('/api/delta-trading', deltaTradingRoutes);
// Use other routes as needed

// Not found middleware for undefined routes
app.use(notFoundHandler);

// Global error handling middleware
app.use(errorHandler);

// Start server
server.listen(PORT, () => {
  console.log(`Server running in ${NODE_ENV} mode on port ${PORT}`);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  console.error('Unhandled Promise Rejection:', err);
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully');

  // Close Prisma connection
  await prisma.$disconnect();

  // Close server
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });

  // Force close after timeout
  setTimeout(() => {
    console.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 10000);
});

// Export for testing
export { app, server, io, prisma };
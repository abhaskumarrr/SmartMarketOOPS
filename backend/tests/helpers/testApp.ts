/**
 * Test App Setup
 * Creates a test instance of the Express app for testing
 */

import express, { Express } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import cookieParser from 'cookie-parser';
import { PrismaClient } from '@prisma/client';
import optimizationMiddleware from '../../middleware/optimizationMiddleware';
import { errorHandler, notFoundHandler } from '../../middleware/errorHandler';

// Import routes
import authRoutes from '../../routes/authRoutes';
import botRoutes from '../../routes/botRoutes';
import userRoutes from '../../routes/userRoutes';

// Test database instance
let testPrisma: PrismaClient;

export const createTestApp = async (): Promise<Express> => {
  const app = express();

  // Initialize test database
  testPrisma = new PrismaClient({
    datasources: {
      db: {
        url: process.env.TEST_DATABASE_URL || 'postgresql://test:test@localhost:5432/smartmarket_test',
      },
    },
  });

  // Basic middleware
  app.use(helmet({
    contentSecurityPolicy: false, // Disable for testing
  }));

  app.use(cors({
    origin: true,
    credentials: true,
  }));

  app.use(compression());
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true, limit: '10mb' }));
  app.use(cookieParser());

  // Performance monitoring (but don't enforce rate limits in tests)
  app.use(optimizationMiddleware.performanceMonitor());
  app.use(optimizationMiddleware.securityHeaders());
  app.use(optimizationMiddleware.requestValidation());

  // Health check
  app.get('/health', (req, res) => {
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: 'test',
    });
  });

  // API routes
  app.use('/api/auth', authRoutes);
  app.use('/api/bots', botRoutes);
  app.use('/api/users', userRoutes);

  // Error handling
  app.use(notFoundHandler);
  app.use(optimizationMiddleware.errorHandler());

  return app;
};

export const getTestPrisma = (): PrismaClient => {
  return testPrisma;
};

export const closeTestApp = async (): Promise<void> => {
  if (testPrisma) {
    await testPrisma.$disconnect();
  }
};

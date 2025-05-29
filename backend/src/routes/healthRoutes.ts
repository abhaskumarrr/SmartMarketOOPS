/**
 * Health Check Routes
 * Endpoints for system monitoring and health status
 */

import express, { Request, Response } from 'express';
import { checkDbConnection } from '../utils/dbHealthCheck';

const router = express.Router();

// Basic health check
router.get('/', (req: Request, res: Response) => {
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Database connection health check
router.get('/db', async (req: Request, res: Response) => {
  try {
    const dbStatus = await checkDbConnection();
    
    res.status(200).json({
      status: dbStatus.success ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      database: dbStatus
    });
  } catch (error) {
    res.status(500).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: (error as Error).message
    });
  }
});

// Detailed system health check
router.get('/system', (req: Request, res: Response) => {
  const memoryUsage = process.memoryUsage();
  
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    system: {
      uptime: process.uptime(),
      nodeVersion: process.version,
      platform: process.platform,
      memory: {
        rss: `${Math.round(memoryUsage.rss / 1024 / 1024)}MB`,
        heapTotal: `${Math.round(memoryUsage.heapTotal / 1024 / 1024)}MB`,
        heapUsed: `${Math.round(memoryUsage.heapUsed / 1024 / 1024)}MB`,
        external: `${Math.round(memoryUsage.external / 1024 / 1024)}MB`
      },
      cpu: process.cpuUsage()
    }
  });
});

export default router; 
/**
 * Health Check Routes
 * Provides endpoints for system health monitoring
 */

const express = require('express');
const router = express.Router();
const os = require('os');
const { env } = require('../utils/env');
const prisma = require('../utils/prismaClient');

/**
 * Basic health check route
 * @route GET /api/health
 */
router.get('/', (req, res) => {
  res.status(200).json({
    success: true,
    status: 'healthy',
    timestamp: new Date().toISOString()
  });
});

/**
 * Detailed system health check
 * @route GET /api/health/system
 */
router.get('/system', async (req, res) => {
  try {
    const uptime = process.uptime();
    const memoryUsage = process.memoryUsage();
    
    // Format memory usage to MB
    const formatMemory = (bytes) => Math.round(bytes / 1024 / 1024 * 100) / 100;
    
    const systemInfo = {
      service: {
        status: 'healthy',
        environment: env.NODE_ENV,
        uptime: `${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m ${Math.floor(uptime % 60)}s`,
        timestamp: new Date().toISOString()
      },
      memory: {
        rss: `${formatMemory(memoryUsage.rss)} MB`,
        heapTotal: `${formatMemory(memoryUsage.heapTotal)} MB`,
        heapUsed: `${formatMemory(memoryUsage.heapUsed)} MB`,
        external: `${formatMemory(memoryUsage.external)} MB`
      },
      system: {
        platform: process.platform,
        arch: process.arch,
        cpus: os.cpus().length,
        totalMemory: `${Math.round(os.totalmem() / 1024 / 1024 / 1024 * 100) / 100} GB`,
        freeMemory: `${Math.round(os.freemem() / 1024 / 1024 / 1024 * 100) / 100} GB`,
        loadAvg: os.loadavg()
      }
    };
    
    res.status(200).json({
      success: true,
      data: systemInfo
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve system health information'
    });
  }
});

/**
 * Database health check
 * @route GET /api/health/db
 */
router.get('/db', async (req, res) => {
  try {
    // Simple query to check database connection
    const start = Date.now();
    await prisma.$queryRaw`SELECT 1`;
    const responseTime = Date.now() - start;
    
    res.status(200).json({
      success: true,
      data: {
        status: 'connected',
        responseTime: `${responseTime}ms`,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Database connection failed',
      details: error.message
    });
  }
});

module.exports = router; 
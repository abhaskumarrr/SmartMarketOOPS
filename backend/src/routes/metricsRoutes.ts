/**
 * Metrics Routes
 * Endpoints for system metrics and monitoring
 */

import express from 'express';

const router = express.Router();

// Health check for metrics routes
router.get('/metrics/health', (req, res) => {
  res.json({ 
    status: 'Metrics routes working', 
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage()
  });
});

// Basic system metrics
router.get('/metrics/system', (req, res) => {
  res.json({
    success: true,
    data: {
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      platform: process.platform,
      nodeVersion: process.version,
      timestamp: new Date().toISOString()
    }
  });
});

export default router;

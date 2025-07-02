import { Router } from 'express';
import { logger } from '../utils/logger';

const router = Router();

/**
 * GET /api/dashboard/dashboard-summary
 * Returns summary data for the main dashboard
 */
router.get('/dashboard-summary', async (req, res) => {
  try {
    logger.info('📊 Dashboard summary requested');
    
    // Mock data for now - in production this would come from the database
    const dashboardData = {
      portfolioValue: '125,430.50',
      dailyChange: '+2.34',
      activePositions: 5,
      profitablePositions: 3,
      aiAccuracy: 87,
      timestamp: new Date().toISOString()
    };
    
    res.json(dashboardData);
  } catch (error) {
    logger.error('❌ Dashboard summary error:', error as Error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch dashboard summary',
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;

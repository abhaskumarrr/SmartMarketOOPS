/**
 * Bot Routes
 * Simple placeholder implementation
 */

import { Router, Request, Response } from 'express';
import { authenticateJWT } from '../../middleware/authMiddleware';

const router = Router();

// Apply authentication middleware
router.use(authenticateJWT);

// Simple placeholder response for all bot routes
router.all('*', (req: Request, res: Response) => {
  res.status(501).json({
    success: false,
    message: 'Bot functionality not yet implemented'
  });
});

export default router; 
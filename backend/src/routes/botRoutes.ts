/**
 * Trading Bot Routes
 * Endpoints for configuring and managing trading bots
 */

import express from 'express';
import {
  createBot,
  getBots,
  getBot,
  updateBot,
  deleteBot,
  startBot,
  stopBot,
  getBotStatus,
  runBacktest,
  getBacktestHistory
} from '../controllers/botController';
import { auth } from '../middleware/auth';

const router = express.Router();

// All routes require authentication
router.use(auth);

// Bot configuration endpoints
router.post('/', createBot);
router.get('/', getBots);
router.get('/:id', getBot);
router.put('/:id', updateBot);
router.delete('/:id', deleteBot);

// Bot control endpoints
router.post('/:id/start', startBot);
router.post('/:id/stop', stopBot);
router.get('/:id/status', getBotStatus);

// Backtesting endpoints
router.post('/:id/backtest', runBacktest);
router.get('/:id/backtests', getBacktestHistory);

export default router;
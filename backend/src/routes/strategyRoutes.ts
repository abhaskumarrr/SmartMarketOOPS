/**
 * Strategy Routes
 * Routes for strategy management and execution
 */

import express from 'express';
import { authenticateJWT } from '../middleware/authMiddleware';
import {
  getAllStrategies,
  getStrategyById,
  createStrategy,
  updateStrategy,
  deleteStrategy,
  validateStrategy,
  startStrategyExecution,
  stopStrategyExecution,
  getUserExecutions,
  getExecutionById,
  getExecutionResults
} from '../controllers/strategyController';

const router = express.Router();

// Strategy routes
router.get('/strategies', authenticateJWT, getAllStrategies);
router.get('/strategies/:id', authenticateJWT, getStrategyById);
router.post('/strategies', authenticateJWT, createStrategy);
router.put('/strategies/:id', authenticateJWT, updateStrategy);
router.delete('/strategies/:id', authenticateJWT, deleteStrategy);
router.post('/strategies/validate', authenticateJWT, validateStrategy);
router.post('/strategies/:id/execute', authenticateJWT, startStrategyExecution);

// Execution routes
router.get('/executions', authenticateJWT, getUserExecutions);
router.get('/executions/:id', authenticateJWT, getExecutionById);
router.get('/executions/:id/results', authenticateJWT, getExecutionResults);
router.post('/executions/:id/stop', authenticateJWT, stopStrategyExecution);

export default router; 
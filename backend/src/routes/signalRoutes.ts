/**
 * Signal Routes
 * API routes for the signal generation system
 */

import express from 'express';
import * as signalController from '../controllers/signalController';
import { protect } from '../middleware/auth';

const router = express.Router();

// Apply authentication middleware to all signal routes
router.use(protect);

/**
 * @route GET /api/signals
 * @description Get signals based on filter criteria
 * @access Private
 */
router.get('/', signalController.getSignals);

/**
 * @route GET /api/signals/:symbol/latest
 * @description Get the latest signal for a specific symbol
 * @access Private
 */
router.get('/:symbol/latest', signalController.getLatestSignal);

/**
 * @route POST /api/signals/generate
 * @description Generate new trading signals for a symbol
 * @access Private
 */
router.post('/generate', signalController.generateSignals);

export default router; 
/**
 * Signal Routes
 * Provides endpoints for trading signals and strategies
 */

const express = require('express');
const router = express.Router();
const { auth } = require('../middleware/auth');
const signalController = require('../controllers/signalController');

/**
 * Get trading signals (public route)
 * @route GET /api/signals
 */
router.get('/', signalController.getLatestSignals);

/**
 * Get the latest trading signals
 * @route GET /api/signals/latest
 */
router.get('/latest', auth, signalController.getLatestSignals);

/**
 * Get trading signals history
 * @route GET /api/signals/history
 */
router.get('/history', auth, signalController.getSignalHistory);

/**
 * Get available trading strategies
 * @route GET /api/signals/strategies
 */
router.get('/strategies', auth, signalController.getStrategies);

module.exports = router; 
/**
 * Risk Management Routes
 * API routes for risk management functionality
 */

import { Router } from 'express';
import * as riskController from '../controllers/riskController';
import { authenticateJWT } from '../middleware/authMiddleware';

const router = Router();

/**
 * Apply authentication middleware to all routes
 */
router.use(authenticateJWT);

/**
 * @route   GET /api/risk/settings
 * @desc    Get risk settings for a user or bot
 * @access  Private
 * @query   botId (optional) - Bot ID for bot-specific settings
 */
router.get('/settings', riskController.getRiskSettings);

/**
 * @route   POST /api/risk/settings
 * @desc    Create or update risk settings
 * @access  Private
 * @body    RiskSettings object
 */
router.post('/settings', riskController.saveRiskSettings);

/**
 * @route   POST /api/risk/position-size
 * @desc    Calculate position size based on risk parameters
 * @access  Private
 * @body    PositionSizingRequest object
 */
router.post('/position-size', riskController.calculatePositionSize);

/**
 * @route   GET /api/risk/report
 * @desc    Generate a comprehensive risk report
 * @access  Private
 */
router.get('/report', riskController.generateRiskReport);

/**
 * @route   GET /api/risk/analysis/:positionId
 * @desc    Analyze risk for a specific position
 * @access  Private
 * @param   positionId - Position ID to analyze
 */
router.get('/analysis/:positionId', riskController.analyzeTradeRisk);

/**
 * @route   GET /api/risk/alerts
 * @desc    Get risk alerts for the user
 * @access  Private
 * @query   acknowledged (optional) - Filter by acknowledgement status
 * @query   level (optional) - Filter by alert level
 * @query   limit (optional) - Limit number of results
 */
router.get('/alerts', riskController.getRiskAlerts);

/**
 * @route   PUT /api/risk/alerts/:alertId/acknowledge
 * @desc    Acknowledge a risk alert
 * @access  Private
 * @param   alertId - Alert ID to acknowledge
 */
router.put('/alerts/:alertId/acknowledge', riskController.acknowledgeRiskAlert);

/**
 * @route   GET /api/risk/circuit-breaker
 * @desc    Check circuit breaker status
 * @access  Private
 * @query   botId (optional) - Bot ID to check status for
 */
router.get('/circuit-breaker', riskController.getCircuitBreakerStatus);

/**
 * @route   POST /api/risk/circuit-breaker/reset
 * @desc    Reset circuit breaker
 * @access  Private
 * @body    { botId } (optional) - Bot ID to reset circuit breaker for
 */
router.post('/circuit-breaker/reset', riskController.resetCircuitBreaker);

/**
 * @route   GET /api/risk/trading-allowed
 * @desc    Check if trading is allowed
 * @access  Private
 * @query   botId (optional) - Bot ID to check
 * @query   symbol (optional) - Trading symbol to check
 */
router.get('/trading-allowed', riskController.checkTradingAllowed);

export default router; 
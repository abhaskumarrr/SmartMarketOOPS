"use strict";
/**
 * Risk Management Routes
 * API routes for risk management functionality
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const riskController = __importStar(require("../controllers/riskController"));
const authMiddleware_1 = require("../middleware/authMiddleware");
const router = (0, express_1.Router)();
/**
 * Apply authentication middleware to all routes
 */
router.use(authMiddleware_1.authenticateJWT);
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
exports.default = router;
//# sourceMappingURL=riskRoutes.js.map
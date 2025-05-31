"use strict";
/**
 * Circuit Breaker Service
 * Implements circuit breaker patterns to halt trading in abnormal conditions
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.CircuitBreakerService = void 0;
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
const logger_1 = require("../../utils/logger");
const riskAssessmentService_1 = __importDefault(require("./riskAssessmentService"));
// Create logger
const logger = (0, logger_1.createLogger)('CircuitBreakerService');
/**
 * Default circuit breaker configuration
 */
const DEFAULT_CIRCUIT_BREAKER_CONFIG = {
    enabled: true,
    maxDailyLoss: 5.0, // 5% max daily loss
    maxDrawdown: 10.0, // 10% max drawdown
    volatilityMultiplier: 3.0, // 3x normal volatility
    consecutiveLosses: 3, // 3 consecutive losses
    tradingPause: 3600, // 1 hour pause
    marketWideEnabled: true, // React to market-wide circuit breakers
    enableManualOverride: true // Allow manual reset
};
/**
 * Circuit Breaker Service class
 * Handles trading halt mechanisms during abnormal conditions
 */
class CircuitBreakerService {
    /**
     * Creates a new Circuit Breaker Service instance
     */
    constructor() {
        // Store circuit breaker status
        this.circuitBreakerStatus = new Map();
        // Consecutive losses counter
        this.consecutiveLosses = new Map();
        // Store market volatility
        this.marketVolatility = new Map();
        logger.info('Circuit Breaker Service initialized');
        // Setup cleanup interval to remove expired circuit breakers
        setInterval(() => this._cleanupExpiredCircuitBreakers(), 60000); // Every minute
    }
    /**
     * Check if trading is allowed
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @param symbol - Trading symbol
     * @returns Object with allowed status and reason if not allowed
     */
    async isTradingAllowed(userId, botId, symbol) {
        try {
            // Get circuit breaker status
            const circuitBreakerId = this._getCircuitBreakerId(userId, botId);
            const status = this.circuitBreakerStatus.get(circuitBreakerId);
            // If circuit breaker is tripped and cooldown hasn't expired
            if (status && status.isTripped) {
                const now = new Date();
                if (status.cooldownUntil && status.cooldownUntil > now) {
                    logger.info(`Trading halted for user ${userId}${botId ? ` bot ${botId}` : ''}: ${status.reason}`);
                    return {
                        allowed: false,
                        reason: status.reason || 'Circuit breaker tripped'
                    };
                }
                else {
                    // Cooldown expired, auto-reset circuit breaker
                    await this.resetCircuitBreaker(userId, botId);
                    logger.info(`Circuit breaker auto-reset for user ${userId}${botId ? ` bot ${botId}` : ''}`);
                    return { allowed: true };
                }
            }
            // Check for market-wide circuit breakers if symbol provided
            if (symbol && await this._checkMarketWideCircuitBreaker(symbol)) {
                const reason = 'Market-wide circuit breaker in effect';
                await this._tripCircuitBreaker(userId, reason, botId);
                return {
                    allowed: false,
                    reason
                };
            }
            // Get risk settings
            const riskSettings = await this._getRiskSettings(userId, botId);
            if (!riskSettings.circuitBreakerEnabled) {
                return { allowed: true };
            }
            // Check drawdown
            const report = await riskAssessmentService_1.default.generateRiskReport(userId);
            if (report.currentDrawdown > riskSettings.maxDrawdownBreaker) {
                const reason = `Maximum drawdown exceeded: ${report.currentDrawdown.toFixed(2)}%`;
                await this._tripCircuitBreaker(userId, reason, botId);
                return {
                    allowed: false,
                    reason
                };
            }
            // Check daily loss
            if (Math.abs(report.dailyPnLPercentage) > riskSettings.maxDailyLossBreaker && report.dailyPnLPercentage < 0) {
                const reason = `Maximum daily loss exceeded: ${Math.abs(report.dailyPnLPercentage).toFixed(2)}%`;
                await this._tripCircuitBreaker(userId, reason, botId);
                return {
                    allowed: false,
                    reason
                };
            }
            // Check consecutive losses
            const consecutiveLossesKey = this._getCircuitBreakerId(userId, botId);
            const consecutiveLosses = this.consecutiveLosses.get(consecutiveLossesKey) || 0;
            if (consecutiveLosses >= riskSettings.consecutiveLossesBreaker) {
                const reason = `Too many consecutive losses: ${consecutiveLosses}`;
                await this._tripCircuitBreaker(userId, reason, botId);
                return {
                    allowed: false,
                    reason
                };
            }
            // Check volatility if symbol provided
            if (symbol) {
                const volatility = this.marketVolatility.get(symbol) || 0;
                const normalVolatility = await this._getNormalVolatility(symbol);
                if (volatility > normalVolatility * riskSettings.volatilityMultiplier) {
                    const reason = `Abnormal market volatility: ${volatility.toFixed(2)}%`;
                    await this._tripCircuitBreaker(userId, reason, botId);
                    return {
                        allowed: false,
                        reason
                    };
                }
            }
            // All checks passed
            return { allowed: true };
        }
        catch (error) {
            const logData = {
                userId,
                botId,
                symbol,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error checking if trading is allowed for user ${userId}`, logData);
            // Default to allowed in case of error (configurable)
            return { allowed: true };
        }
    }
    /**
     * Record a trade result (win/loss) for consecutive loss tracking
     * @param userId - User ID
     * @param isWin - Whether the trade was a win
     * @param botId - Optional Bot ID
     */
    recordTradeResult(userId, isWin, botId) {
        const key = this._getCircuitBreakerId(userId, botId);
        if (isWin) {
            // Reset consecutive losses on win
            this.consecutiveLosses.set(key, 0);
        }
        else {
            // Increment consecutive losses on loss
            const currentLosses = this.consecutiveLosses.get(key) || 0;
            this.consecutiveLosses.set(key, currentLosses + 1);
            logger.debug(`Consecutive losses for ${key}: ${currentLosses + 1}`);
        }
    }
    /**
     * Update market volatility for a symbol
     * @param symbol - Trading symbol
     * @param volatilityValue - Volatility percentage
     */
    updateMarketVolatility(symbol, volatilityValue) {
        this.marketVolatility.set(symbol, volatilityValue);
        logger.debug(`Updated volatility for ${symbol}: ${volatilityValue.toFixed(2)}%`);
    }
    /**
     * Manually trip circuit breaker
     * @param userId - User ID
     * @param reason - Reason for tripping
     * @param botId - Optional Bot ID
     * @returns Trip success status
     */
    async manuallyTripCircuitBreaker(userId, reason = 'Manually triggered', botId) {
        try {
            await this._tripCircuitBreaker(userId, reason, botId);
            return true;
        }
        catch (error) {
            logger.error(`Error manually tripping circuit breaker for user ${userId}`, {
                userId,
                botId,
                reason,
                error: error instanceof Error ? error.message : String(error)
            });
            return false;
        }
    }
    /**
     * Reset circuit breaker
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Reset success status
     */
    async resetCircuitBreaker(userId, botId) {
        try {
            const circuitBreakerId = this._getCircuitBreakerId(userId, botId);
            const status = this.circuitBreakerStatus.get(circuitBreakerId);
            if (!status) {
                return true; // No circuit breaker to reset
            }
            // Check if it's resetable
            if (!status.resetable) {
                logger.warn(`Cannot reset circuit breaker for ${circuitBreakerId}: not resetable`);
                return false;
            }
            // Reset circuit breaker
            this.circuitBreakerStatus.delete(circuitBreakerId);
            // Reset consecutive losses
            this.consecutiveLosses.set(circuitBreakerId, 0);
            logger.info(`Circuit breaker reset for ${circuitBreakerId}`);
            return true;
        }
        catch (error) {
            logger.error(`Error resetting circuit breaker for user ${userId}`, {
                userId,
                botId,
                error: error instanceof Error ? error.message : String(error)
            });
            return false;
        }
    }
    /**
     * Get circuit breaker status
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Circuit breaker status
     */
    getCircuitBreakerStatus(userId, botId) {
        const circuitBreakerId = this._getCircuitBreakerId(userId, botId);
        return this.circuitBreakerStatus.get(circuitBreakerId) || null;
    }
    /**
     * Trip circuit breaker
     * @private
     * @param userId - User ID
     * @param reason - Reason for tripping
     * @param botId - Optional Bot ID
     */
    async _tripCircuitBreaker(userId, reason, botId) {
        // Get circuit breaker ID
        const circuitBreakerId = this._getCircuitBreakerId(userId, botId);
        // Get risk settings
        const riskSettings = await this._getRiskSettings(userId, botId);
        // Calculate cooldown period
        const now = new Date();
        const cooldownUntil = new Date(now.getTime() + riskSettings.tradingPause * 1000);
        // Create circuit breaker status
        const status = {
            userId,
            botId,
            isTripped: true,
            reason,
            trippedAt: now,
            cooldownUntil,
            resetable: riskSettings.enableManualOverride
        };
        // Store circuit breaker status
        this.circuitBreakerStatus.set(circuitBreakerId, status);
        // Log circuit breaker trip
        logger.warn(`Circuit breaker tripped for ${circuitBreakerId}: ${reason}`);
        // Create risk alert
        await riskAssessmentService_1.default.createAlert(userId, 'CIRCUIT_BREAKER', // Type assertion for now, should add to RiskAlertType enum
        'critical', `Trading halted: ${reason}`, {
            botId,
            reason,
            trippedAt: now.toISOString(),
            cooldownUntil: cooldownUntil.toISOString()
        });
    }
    /**
     * Check if market-wide circuit breaker is in effect
     * @private
     * @param symbol - Trading symbol
     * @returns Whether market-wide circuit breaker is in effect
     */
    async _checkMarketWideCircuitBreaker(symbol) {
        // In a real system, this would check exchange API for market-wide circuit breakers
        // For now, return false (no market-wide circuit breaker)
        return false;
    }
    /**
     * Get risk settings
     * @private
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Risk settings
     */
    async _getRiskSettings(userId, botId) {
        try {
            // Query risk settings from database
            const riskSettings = await prismaClient_1.default.riskSettings.findFirst({
                where: {
                    userId,
                    botId: botId || null,
                    isActive: true
                }
            });
            if (riskSettings) {
                return riskSettings;
            }
            // If no settings found, return default circuit breaker settings
            return {
                circuitBreakerEnabled: DEFAULT_CIRCUIT_BREAKER_CONFIG.enabled,
                maxDailyLossBreaker: DEFAULT_CIRCUIT_BREAKER_CONFIG.maxDailyLoss,
                maxDrawdownBreaker: DEFAULT_CIRCUIT_BREAKER_CONFIG.maxDrawdown,
                volatilityMultiplier: DEFAULT_CIRCUIT_BREAKER_CONFIG.volatilityMultiplier,
                consecutiveLossesBreaker: DEFAULT_CIRCUIT_BREAKER_CONFIG.consecutiveLosses,
                tradingPause: DEFAULT_CIRCUIT_BREAKER_CONFIG.tradingPause,
                marketWideEnabled: DEFAULT_CIRCUIT_BREAKER_CONFIG.marketWideEnabled,
                enableManualOverride: DEFAULT_CIRCUIT_BREAKER_CONFIG.enableManualOverride
            };
        }
        catch (error) {
            logger.error(`Error getting risk settings for user ${userId}`, {
                userId,
                botId,
                error: error instanceof Error ? error.message : String(error)
            });
            // Return default settings in case of error
            return {
                circuitBreakerEnabled: DEFAULT_CIRCUIT_BREAKER_CONFIG.enabled,
                maxDailyLossBreaker: DEFAULT_CIRCUIT_BREAKER_CONFIG.maxDailyLoss,
                maxDrawdownBreaker: DEFAULT_CIRCUIT_BREAKER_CONFIG.maxDrawdown,
                volatilityMultiplier: DEFAULT_CIRCUIT_BREAKER_CONFIG.volatilityMultiplier,
                consecutiveLossesBreaker: DEFAULT_CIRCUIT_BREAKER_CONFIG.consecutiveLosses,
                tradingPause: DEFAULT_CIRCUIT_BREAKER_CONFIG.tradingPause,
                marketWideEnabled: DEFAULT_CIRCUIT_BREAKER_CONFIG.marketWideEnabled,
                enableManualOverride: DEFAULT_CIRCUIT_BREAKER_CONFIG.enableManualOverride
            };
        }
    }
    /**
     * Get normal volatility for a symbol
     * @private
     * @param symbol - Trading symbol
     * @returns Normal volatility percentage
     */
    async _getNormalVolatility(symbol) {
        // In a real system, this would fetch from a market data service
        // For now, return a placeholder value of 2.0%
        return 2.0;
    }
    /**
     * Get circuit breaker ID
     * @private
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Circuit breaker ID
     */
    _getCircuitBreakerId(userId, botId) {
        return botId ? `${userId}:${botId}` : userId;
    }
    /**
     * Clean up expired circuit breakers
     * @private
     */
    _cleanupExpiredCircuitBreakers() {
        const now = new Date();
        for (const [id, status] of this.circuitBreakerStatus.entries()) {
            if (status.cooldownUntil && status.cooldownUntil < now) {
                this.circuitBreakerStatus.delete(id);
                logger.debug(`Cleaned up expired circuit breaker for ${id}`);
            }
        }
    }
}
exports.CircuitBreakerService = CircuitBreakerService;
// Create default instance
const circuitBreakerService = new CircuitBreakerService();
exports.default = circuitBreakerService;
//# sourceMappingURL=circuitBreakerService.js.map
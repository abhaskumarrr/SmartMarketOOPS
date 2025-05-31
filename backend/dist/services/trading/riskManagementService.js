"use strict";
/**
 * Risk Management Service
 * Handles position sizing, risk assessment, and protection mechanisms
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RiskManagementService = void 0;
const uuid_1 = require("uuid");
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
const logger_1 = require("../../utils/logger");
const risk_1 = require("../../types/risk");
// Create logger
const logger = (0, logger_1.createLogger)('RiskManagementService');
/**
 * Default risk settings
 */
const DEFAULT_RISK_SETTINGS = {
    positionSizingMethod: risk_1.PositionSizingMethod.FIXED_FRACTIONAL,
    riskPercentage: 1.0, // 1% of account per trade
    maxPositionSize: 100000, // 100k units max position
    kellyFraction: 0.5, // Half Kelly for safety
    stopLossType: risk_1.StopLossType.PERCENTAGE,
    stopLossValue: 2.0, // 2% stop loss
    takeProfitType: risk_1.TakeProfitType.RISK_REWARD,
    takeProfitValue: 2.0, // 2:1 risk-reward ratio
    maxRiskPerTrade: 2.0, // Max 2% risk per trade
    maxRiskPerSymbol: 5.0, // Max 5% risk per symbol
    maxRiskPerDirection: 10.0, // Max 10% risk in same direction
    maxTotalRisk: 20.0, // Max 20% total risk
    maxDrawdown: 20.0, // Max 20% drawdown
    maxPositions: 10, // Max 10 open positions
    maxDailyLoss: 5.0, // Max 5% daily loss
    cooldownPeriod: 3600, // 1 hour cooldown
    volatilityLookback: 14, // 14 periods for volatility
};
/**
 * Risk Management Service class
 * Provides methods for managing trading risk
 */
class RiskManagementService {
    /**
     * Creates a new Risk Management Service instance
     */
    constructor() {
        logger.info('Risk Management Service initialized');
    }
    /**
     * Get risk settings for a user or bot
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Risk settings or default settings if none found
     */
    async getRiskSettings(userId, botId) {
        try {
            // Query risk settings from database
            const riskSettings = await prismaClient_1.default.riskSettings.findFirst({
                where: {
                    userId,
                    botId: botId || null,
                    isActive: true
                },
                orderBy: {
                    updatedAt: 'desc'
                }
            });
            if (riskSettings) {
                logger.debug(`Found risk settings for user ${userId}${botId ? ` and bot ${botId}` : ''}`);
                return riskSettings;
            }
            // If no settings found, get default settings
            logger.info(`No risk settings found for user ${userId}${botId ? ` and bot ${botId}` : ''}, using defaults`);
            return this.getDefaultRiskSettings(userId, botId);
        }
        catch (error) {
            const logData = {
                userId,
                botId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting risk settings for user ${userId}`, logData);
            throw error;
        }
    }
    /**
     * Create default risk settings for a user or bot
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Default risk settings
     */
    async getDefaultRiskSettings(userId, botId) {
        // Create default risk settings
        return {
            id: (0, uuid_1.v4)(),
            name: botId ? 'Default Bot Risk Settings' : 'Default User Risk Settings',
            description: 'Automatically generated default risk settings',
            userId,
            botId: botId || null,
            isActive: true,
            positionSizingMethod: DEFAULT_RISK_SETTINGS.positionSizingMethod,
            riskPercentage: DEFAULT_RISK_SETTINGS.riskPercentage,
            maxPositionSize: DEFAULT_RISK_SETTINGS.maxPositionSize,
            kellyFraction: DEFAULT_RISK_SETTINGS.kellyFraction,
            stopLossType: DEFAULT_RISK_SETTINGS.stopLossType,
            stopLossValue: DEFAULT_RISK_SETTINGS.stopLossValue,
            takeProfitType: DEFAULT_RISK_SETTINGS.takeProfitType,
            takeProfitValue: DEFAULT_RISK_SETTINGS.takeProfitValue,
            maxRiskPerTrade: DEFAULT_RISK_SETTINGS.maxRiskPerTrade,
            maxRiskPerSymbol: DEFAULT_RISK_SETTINGS.maxRiskPerSymbol,
            maxRiskPerDirection: DEFAULT_RISK_SETTINGS.maxRiskPerDirection,
            maxTotalRisk: DEFAULT_RISK_SETTINGS.maxTotalRisk,
            maxDrawdown: DEFAULT_RISK_SETTINGS.maxDrawdown,
            maxPositions: DEFAULT_RISK_SETTINGS.maxPositions,
            maxDailyLoss: DEFAULT_RISK_SETTINGS.maxDailyLoss,
            cooldownPeriod: DEFAULT_RISK_SETTINGS.cooldownPeriod,
            volatilityLookback: DEFAULT_RISK_SETTINGS.volatilityLookback,
            createdAt: new Date(),
            updatedAt: new Date()
        };
    }
    /**
     * Create or update risk settings
     * @param settings - Risk settings to create or update
     * @returns Created or updated risk settings
     */
    async saveRiskSettings(settings) {
        try {
            const { id, ...data } = settings;
            // Check if settings exist
            const existingSettings = id ? await prismaClient_1.default.riskSettings.findUnique({
                where: { id }
            }) : null;
            if (existingSettings) {
                // Update existing settings
                logger.info(`Updating risk settings with ID ${id}`);
                return await prismaClient_1.default.riskSettings.update({
                    where: { id },
                    data: {
                        ...data,
                        updatedAt: new Date()
                    }
                });
            }
            else {
                // Create new settings
                logger.info(`Creating new risk settings for user ${settings.userId}`);
                return await prismaClient_1.default.riskSettings.create({
                    data: {
                        ...data,
                        id: (0, uuid_1.v4)()
                    }
                });
            }
        }
        catch (error) {
            const logData = {
                settings,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error('Error saving risk settings', logData);
            throw error;
        }
    }
    /**
     * Calculate position size based on risk parameters
     * @param request - Position sizing request
     * @returns Position sizing result
     */
    async calculatePositionSize(request) {
        try {
            const { userId, botId, symbol, direction, entryPrice, stopLossPrice, stopLossPercentage, confidence = 50 // Default to moderate confidence
             } = request;
            // Get risk settings
            const settings = await this.getRiskSettings(userId, botId);
            // Get account balance
            const accountBalance = await this._getAccountBalance(userId);
            // Get current open positions
            const openPositions = await this._getOpenPositions(userId);
            // Calculate existing risk exposure
            const existingRisk = this._calculateExistingRisk(openPositions, accountBalance);
            // Check if new position would exceed risk limits
            const availableRisk = Math.max(0, settings.maxTotalRisk - existingRisk.totalRisk);
            // Calculate maximum risk amount for this trade
            const maxRiskAmount = Math.min((accountBalance * settings.maxRiskPerTrade) / 100, (accountBalance * availableRisk) / 100);
            // Calculate stop loss amount if not provided
            let actualStopLossPercentage = stopLossPercentage;
            let actualStopLossPrice = stopLossPrice;
            if (!actualStopLossPercentage && !actualStopLossPrice) {
                // Use default from settings
                actualStopLossPercentage = settings.stopLossValue;
            }
            if (!actualStopLossPrice && actualStopLossPercentage) {
                // Calculate stop loss price from percentage
                const slippageFactor = 1.1; // Add 10% for slippage
                actualStopLossPrice = direction === 'long'
                    ? entryPrice * (1 - (actualStopLossPercentage * slippageFactor) / 100)
                    : entryPrice * (1 + (actualStopLossPercentage * slippageFactor) / 100);
            }
            if (!actualStopLossPercentage && actualStopLossPrice) {
                // Calculate percentage from price
                actualStopLossPercentage = direction === 'long'
                    ? ((entryPrice - actualStopLossPrice) / entryPrice) * 100
                    : ((actualStopLossPrice - entryPrice) / entryPrice) * 100;
            }
            // Adjust risk based on confidence score
            const confidenceAdjustment = Math.min(confidence / 50, 1.5); // 0.2 - 1.5 range
            const adjustedRiskPercentage = settings.riskPercentage * confidenceAdjustment;
            // Calculate risk amount
            const riskAmount = Math.min((accountBalance * adjustedRiskPercentage) / 100, maxRiskAmount);
            // Calculate position size based on risk amount and stop loss
            const positionSize = (riskAmount * 100) / (actualStopLossPercentage || 1);
            // Apply maximum position size limit
            const limitedPositionSize = Math.min(positionSize, settings.maxPositionSize);
            // Calculate take profit price based on settings
            let takeProfitPrice;
            if (settings.takeProfitType === risk_1.TakeProfitType.RISK_REWARD) {
                const rewardMultiple = settings.takeProfitValue;
                const stopDistance = Math.abs(entryPrice - (actualStopLossPrice || entryPrice));
                takeProfitPrice = direction === 'long'
                    ? entryPrice + (stopDistance * rewardMultiple)
                    : entryPrice - (stopDistance * rewardMultiple);
            }
            else if (settings.takeProfitType === risk_1.TakeProfitType.PERCENTAGE) {
                takeProfitPrice = direction === 'long'
                    ? entryPrice * (1 + settings.takeProfitValue / 100)
                    : entryPrice * (1 - settings.takeProfitValue / 100);
            }
            // Calculate potential profit and loss
            const potentialLoss = (limitedPositionSize * actualStopLossPercentage) / 100;
            const potentialProfit = takeProfitPrice
                ? direction === 'long'
                    ? ((takeProfitPrice - entryPrice) / entryPrice) * limitedPositionSize
                    : ((entryPrice - takeProfitPrice) / entryPrice) * limitedPositionSize
                : 0;
            // Calculate risk-reward ratio
            const riskRewardRatio = potentialLoss > 0 ? potentialProfit / potentialLoss : 0;
            // Generate warnings
            const warnings = [];
            if (limitedPositionSize < positionSize) {
                warnings.push(`Position size reduced from ${positionSize.toFixed(2)} to ${limitedPositionSize.toFixed(2)} due to maximum position size limit`);
            }
            if (riskAmount < (accountBalance * adjustedRiskPercentage) / 100) {
                warnings.push(`Risk amount reduced due to risk limits (current exposure: ${existingRisk.totalRisk.toFixed(2)}%, max: ${settings.maxTotalRisk}%)`);
            }
            if (existingRisk.symbolRisk[symbol] && existingRisk.symbolRisk[symbol] + adjustedRiskPercentage > settings.maxRiskPerSymbol) {
                warnings.push(`Adding this position would exceed the maximum risk per symbol (${settings.maxRiskPerSymbol}%) for ${symbol}`);
            }
            if (existingRisk.directionRisk[direction] && existingRisk.directionRisk[direction] + adjustedRiskPercentage > settings.maxRiskPerDirection) {
                warnings.push(`Adding this position would exceed the maximum risk per direction (${settings.maxRiskPerDirection}%) for ${direction} positions`);
            }
            // Return position sizing result
            return {
                positionSize: limitedPositionSize,
                maxPositionSize: settings.maxPositionSize,
                riskAmount,
                riskPercentage: adjustedRiskPercentage,
                leverage: 1, // Default leverage, can be adjusted
                margin: limitedPositionSize / 100, // Simple margin calculation
                potentialLoss,
                potentialProfit,
                riskRewardRatio,
                adjustedForRisk: limitedPositionSize < positionSize,
                warnings
            };
        }
        catch (error) {
            const logData = {
                request,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error('Error calculating position size', logData);
            throw error;
        }
    }
    /**
     * Get account balance for a user
     * @private
     * @param userId - User ID
     * @returns Account balance
     */
    async _getAccountBalance(userId) {
        // In a real system, this would fetch from a balance service or exchange API
        // For now, return a placeholder balance of 10,000
        return 10000;
    }
    /**
     * Get open positions for a user
     * @private
     * @param userId - User ID
     * @returns Open positions
     */
    async _getOpenPositions(userId) {
        try {
            return await prismaClient_1.default.position.findMany({
                where: {
                    userId,
                    status: 'Open'
                }
            });
        }
        catch (error) {
            logger.error(`Error getting open positions for user ${userId}`, {
                userId,
                error: error instanceof Error ? error.message : String(error)
            });
            return [];
        }
    }
    /**
     * Calculate existing risk exposure
     * @private
     * @param positions - Open positions
     * @param accountBalance - Account balance
     * @returns Risk exposure details
     */
    _calculateExistingRisk(positions, accountBalance) {
        const symbolRisk = {};
        const directionRisk = {};
        let totalRisk = 0;
        for (const position of positions) {
            // Calculate risk for this position
            const positionRisk = this._calculatePositionRisk(position, accountBalance);
            // Add to symbol risk
            symbolRisk[position.symbol] = (symbolRisk[position.symbol] || 0) + positionRisk;
            // Add to direction risk
            directionRisk[position.side.toLowerCase()] = (directionRisk[position.side.toLowerCase()] || 0) + positionRisk;
            // Add to total risk
            totalRisk += positionRisk;
        }
        return {
            symbolRisk,
            directionRisk,
            totalRisk
        };
    }
    /**
     * Calculate risk for a single position
     * @private
     * @param position - Position
     * @param accountBalance - Account balance
     * @returns Risk percentage
     */
    _calculatePositionRisk(position, accountBalance) {
        if (!position.stopLossPrice || !position.entryPrice) {
            // If no stop loss, assume maximum risk (position size / account balance)
            return (position.amount * position.entryPrice) / accountBalance * 100;
        }
        // Calculate risk based on stop loss
        const riskAmount = position.side.toLowerCase() === 'long'
            ? (position.entryPrice - position.stopLossPrice) * position.amount
            : (position.stopLossPrice - position.entryPrice) * position.amount;
        return (riskAmount / accountBalance) * 100;
    }
}
exports.RiskManagementService = RiskManagementService;
// Create default instance
const riskManagementService = new RiskManagementService();
exports.default = riskManagementService;
//# sourceMappingURL=riskManagementService.js.map
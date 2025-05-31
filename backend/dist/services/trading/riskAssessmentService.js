"use strict";
/**
 * Risk Assessment Service
 * Handles portfolio risk analysis, monitoring, and alerts
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RiskAssessmentService = void 0;
const uuid_1 = require("uuid");
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
const logger_1 = require("../../utils/logger");
const risk_1 = require("../../types/risk");
// Create logger
const logger = (0, logger_1.createLogger)('RiskAssessmentService');
/**
 * Risk Assessment Service class
 * Provides methods for analyzing and monitoring trading risk
 */
class RiskAssessmentService {
    /**
     * Creates a new Risk Assessment Service instance
     */
    constructor() {
        // Keep track of equity high watermark for drawdown calculations
        this.equityHighWatermark = {};
        logger.info('Risk Assessment Service initialized');
    }
    /**
     * Generate a comprehensive risk report for a user
     * @param userId - User ID
     * @returns Risk report
     */
    async generateRiskReport(userId) {
        try {
            logger.info(`Generating risk report for user ${userId}`);
            // Get account balance and equity
            const accountBalance = await this._getAccountBalance(userId);
            const accountEquity = await this._getAccountEquity(userId);
            // Get open positions
            const openPositions = await this._getOpenPositions(userId);
            // Calculate metrics
            const totalMargin = this._calculateTotalMargin(openPositions);
            const freeMargin = accountBalance - totalMargin;
            const marginLevel = totalMargin > 0 ? (accountEquity / totalMargin) * 100 : 100;
            // Calculate exposure
            const { exposureBySymbol, exposureByDirection, totalRisk } = this._calculateExposure(openPositions, accountBalance);
            // Calculate drawdown
            const { currentDrawdown, maxDrawdown } = this._calculateDrawdown(userId, accountEquity);
            // Calculate daily P&L
            const { dailyPnL, dailyPnLPercentage } = await this._calculateDailyPnL(userId, accountBalance);
            // Generate alerts
            const alerts = await this._generateAlerts(userId, {
                marginLevel,
                currentDrawdown,
                dailyPnLPercentage,
                exposureBySymbol,
                exposureByDirection,
                totalRisk
            });
            // Determine overall risk level
            const riskLevel = this._determineRiskLevel({
                marginLevel,
                currentDrawdown,
                dailyPnLPercentage,
                totalRisk
            });
            return {
                userId,
                timestamp: new Date().toISOString(),
                totalBalance: accountBalance,
                totalEquity: accountEquity,
                totalMargin,
                freeMargin,
                marginLevel,
                openPositions: openPositions.length,
                openPositionsRisk: totalRisk,
                maxDrawdown,
                currentDrawdown,
                dailyPnL,
                dailyPnLPercentage,
                exposureBySymbol,
                exposureByDirection,
                riskLevel,
                alerts
            };
        }
        catch (error) {
            const logData = {
                userId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error generating risk report for user ${userId}`, logData);
            throw error;
        }
    }
    /**
     * Analyze risk for a specific trade/position
     * @param userId - User ID
     * @param positionId - Position ID
     * @returns Trade risk analysis
     */
    async analyzeTradeRisk(userId, positionId) {
        try {
            logger.info(`Analyzing trade risk for position ${positionId}`);
            // Get position
            const position = await prismaClient_1.default.position.findFirst({
                where: {
                    id: positionId,
                    userId
                }
            });
            if (!position) {
                throw new Error(`Position ${positionId} not found for user ${userId}`);
            }
            // Get account balance
            const accountBalance = await this._getAccountBalance(userId);
            // Calculate metrics
            const unrealizedPnL = position.currentPrice
                ? (position.side.toLowerCase() === 'long'
                    ? (position.currentPrice - position.entryPrice) * position.amount
                    : (position.entryPrice - position.currentPrice) * position.amount)
                : 0;
            const unrealizedPnLPercentage = (unrealizedPnL / accountBalance) * 100;
            const distanceToStopLoss = position.stopLossPrice
                ? Math.abs((position.stopLossPrice - (position.currentPrice || position.entryPrice)) / (position.currentPrice || position.entryPrice)) * 100
                : 0;
            const distanceToTakeProfit = position.takeProfitPrice
                ? Math.abs((position.takeProfitPrice - (position.currentPrice || position.entryPrice)) / (position.currentPrice || position.entryPrice)) * 100
                : undefined;
            // Calculate breakeven price (including fees, assuming 0.1% fee)
            const feePercentage = 0.1;
            const feesAmount = (position.entryPrice * position.amount * feePercentage) / 100;
            const breakEvenPrice = position.side.toLowerCase() === 'long'
                ? position.entryPrice + (feesAmount / position.amount)
                : position.entryPrice - (feesAmount / position.amount);
            // Calculate risk-reward ratio
            const riskRewardRatio = (position.takeProfitPrice && position.stopLossPrice)
                ? Math.abs((position.takeProfitPrice - position.entryPrice) / (position.stopLossPrice - position.entryPrice))
                : 0;
            // Calculate time in trade
            const timeInTrade = Math.floor((Date.now() - position.openedAt.getTime()) / 1000);
            // Calculate exposure and risk
            const exposure = (position.amount * position.entryPrice) / accountBalance * 100;
            // Calculate risk amount based on stop loss
            const riskAmount = position.stopLossPrice
                ? Math.abs(position.stopLossPrice - position.entryPrice) * position.amount
                : position.amount * position.entryPrice; // If no stop loss, assume full position at risk
            const riskPercentage = (riskAmount / accountBalance) * 100;
            // Calculate margin and liquidation price (assuming leverage)
            const margin = (position.amount * position.entryPrice) / position.leverage;
            // Simple liquidation price calculation (can be refined based on exchange rules)
            const liquidationPrice = position.side.toLowerCase() === 'long'
                ? position.entryPrice * (1 - (1 / position.leverage) * 0.8) // 80% of margin used triggers liquidation
                : position.entryPrice * (1 + (1 / position.leverage) * 0.8);
            // Calculate normalized risk score (0-100)
            const riskFactors = [
                Math.min(riskPercentage * 5, 30), // Risk percentage contributes up to 30 points
                Math.min(exposure * 2, 20), // Exposure contributes up to 20 points
                position.stopLossPrice ? 0 : 20, // No stop loss adds 20 points
                position.leverage > 5 ? Math.min((position.leverage - 5) * 5, 20) : 0, // High leverage adds up to 20 points
                distanceToStopLoss < 1 ? 10 : 0 // Close to stop loss adds 10 points
            ];
            // Return analysis
            return {
                tradeId: position.id,
                symbol: position.symbol,
                direction: position.side.toLowerCase(),
                entryPrice: position.entryPrice,
                currentPrice: position.currentPrice || position.entryPrice,
                positionSize: position.amount,
                exposure,
                riskAmount,
                riskPercentage,
                stopLossPrice: position.stopLossPrice || 0,
                takeProfitPrice: position.takeProfitPrice || undefined,
                unrealizedPnL,
                unrealizedPnLPercentage,
                distanceToStopLoss,
                distanceToTakeProfit,
                breakEvenPrice,
                riskRewardRatio,
                timeInTrade,
                margin,
                liquidationPrice,
                riskScore: Math.min(100, riskFactors.reduce((sum, value) => sum + value, 0)),
                riskLevel: this._determinePositionRiskLevel(riskFactors.reduce((sum, value) => sum + value, 0))
            };
        }
        catch (error) {
            const logData = {
                userId,
                positionId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error analyzing trade risk for position ${positionId}`, logData);
            throw error;
        }
    }
    /**
     * Create a risk alert
     * @param userId - User ID
     * @param type - Alert type
     * @param level - Alert level
     * @param message - Alert message
     * @param details - Alert details
     * @returns Created alert
     */
    async createAlert(userId, type, level, message, details) {
        try {
            logger.info(`Creating ${level} risk alert for user ${userId}: ${type}`);
            const alert = await prismaClient_1.default.riskAlert.create({
                data: {
                    id: (0, uuid_1.v4)(),
                    userId,
                    type,
                    level,
                    message,
                    details: details,
                    timestamp: new Date(),
                    acknowledged: false
                }
            });
            return {
                id: alert.id,
                userId: alert.userId,
                type: alert.type,
                level: alert.level,
                message: alert.message,
                details: alert.details,
                timestamp: alert.timestamp.toISOString(),
                acknowledged: alert.acknowledged,
                resolvedAt: alert.resolvedAt?.toISOString()
            };
        }
        catch (error) {
            const logData = {
                userId,
                type,
                level,
                message,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error creating risk alert for user ${userId}`, logData);
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
     * Get account equity for a user
     * @private
     * @param userId - User ID
     * @returns Account equity
     */
    async _getAccountEquity(userId) {
        // Get account balance
        const accountBalance = await this._getAccountBalance(userId);
        // Get open positions
        const openPositions = await this._getOpenPositions(userId);
        // Calculate unrealized P&L
        let unrealizedPnL = 0;
        for (const position of openPositions) {
            if (position.currentPrice) {
                const positionPnL = position.side.toLowerCase() === 'long'
                    ? (position.currentPrice - position.entryPrice) * position.amount
                    : (position.entryPrice - position.currentPrice) * position.amount;
                unrealizedPnL += positionPnL;
            }
        }
        // Return equity (balance + unrealized P&L)
        return accountBalance + unrealizedPnL;
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
     * Calculate total margin used
     * @private
     * @param positions - Open positions
     * @returns Total margin
     */
    _calculateTotalMargin(positions) {
        return positions.reduce((total, position) => {
            return total + ((position.amount * position.entryPrice) / position.leverage);
        }, 0);
    }
    /**
     * Calculate exposure by symbol and direction
     * @private
     * @param positions - Open positions
     * @param accountBalance - Account balance
     * @returns Exposure details
     */
    _calculateExposure(positions, accountBalance) {
        const exposureBySymbol = {};
        const exposureByDirection = {};
        let totalRisk = 0;
        for (const position of positions) {
            // Calculate exposure for this position (position size / account balance)
            const positionExposure = (position.amount * position.entryPrice) / accountBalance * 100;
            // Calculate risk for this position
            const positionRisk = position.stopLossPrice
                ? Math.abs(position.stopLossPrice - position.entryPrice) * position.amount / accountBalance * 100
                : positionExposure; // If no stop loss, assume full position at risk
            // Add to symbol exposure
            exposureBySymbol[position.symbol] = (exposureBySymbol[position.symbol] || 0) + positionExposure;
            // Add to direction exposure
            const direction = position.side.toLowerCase();
            exposureByDirection[direction] = (exposureByDirection[direction] || 0) + positionExposure;
            // Add to total risk
            totalRisk += positionRisk;
        }
        return {
            exposureBySymbol,
            exposureByDirection,
            totalRisk
        };
    }
    /**
     * Calculate drawdown
     * @private
     * @param userId - User ID
     * @param currentEquity - Current equity
     * @returns Drawdown details
     */
    _calculateDrawdown(userId, currentEquity) {
        // Initialize high watermark if not exists
        if (!this.equityHighWatermark[userId]) {
            this.equityHighWatermark[userId] = currentEquity;
        }
        // Update high watermark if current equity is higher
        if (currentEquity > this.equityHighWatermark[userId]) {
            this.equityHighWatermark[userId] = currentEquity;
        }
        // Calculate current drawdown
        const currentDrawdown = this.equityHighWatermark[userId] > 0
            ? ((this.equityHighWatermark[userId] - currentEquity) / this.equityHighWatermark[userId]) * 100
            : 0;
        // For max drawdown, we should ideally store this in the database
        // For now, we'll just return current drawdown as max drawdown
        const maxDrawdown = currentDrawdown;
        return {
            currentDrawdown,
            maxDrawdown
        };
    }
    /**
     * Calculate daily P&L
     * @private
     * @param userId - User ID
     * @param accountBalance - Account balance
     * @returns Daily P&L details
     */
    async _calculateDailyPnL(userId, accountBalance) {
        // Get today's closed positions
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const closedPositions = await prismaClient_1.default.position.findMany({
            where: {
                userId,
                status: 'Closed',
                closedAt: {
                    gte: today
                }
            }
        });
        // Calculate realized P&L
        let dailyPnL = 0;
        for (const position of closedPositions) {
            if (position.pnl) {
                dailyPnL += position.pnl;
            }
        }
        // Calculate P&L percentage
        const dailyPnLPercentage = (dailyPnL / accountBalance) * 100;
        return {
            dailyPnL,
            dailyPnLPercentage
        };
    }
    /**
     * Generate risk alerts
     * @private
     * @param userId - User ID
     * @param metrics - Risk metrics
     * @returns Risk alerts
     */
    async _generateAlerts(userId, metrics) {
        const alerts = [];
        // Check margin level
        if (metrics.marginLevel < 150) {
            alerts.push(await this.createAlert(userId, risk_1.RiskAlertType.MARGIN_CALL, metrics.marginLevel < 120 ? 'critical' : 'warning', `Margin level is getting low: ${metrics.marginLevel.toFixed(2)}%`, { marginLevel: metrics.marginLevel }));
        }
        // Check drawdown
        if (metrics.currentDrawdown > 10) {
            alerts.push(await this.createAlert(userId, risk_1.RiskAlertType.DRAWDOWN_WARNING, metrics.currentDrawdown > 15 ? 'critical' : 'warning', `Account drawdown is significant: ${metrics.currentDrawdown.toFixed(2)}%`, { drawdown: metrics.currentDrawdown }));
        }
        // Check daily loss
        if (metrics.dailyPnLPercentage < -3) {
            alerts.push(await this.createAlert(userId, risk_1.RiskAlertType.DAILY_LOSS_WARNING, metrics.dailyPnLPercentage < -5 ? 'critical' : 'warning', `Significant daily loss: ${Math.abs(metrics.dailyPnLPercentage).toFixed(2)}%`, { dailyLoss: metrics.dailyPnLPercentage }));
        }
        // Check high exposure
        for (const [symbol, exposure] of Object.entries(metrics.exposureBySymbol)) {
            if (exposure > 20) {
                alerts.push(await this.createAlert(userId, risk_1.RiskAlertType.HIGH_EXPOSURE, exposure > 30 ? 'critical' : 'warning', `High exposure to ${symbol}: ${exposure.toFixed(2)}%`, { symbol, exposure }));
            }
        }
        // Check concentration risk
        const symbolCount = Object.keys(metrics.exposureBySymbol).length;
        if (symbolCount === 1 && metrics.totalRisk > 10) {
            alerts.push(await this.createAlert(userId, risk_1.RiskAlertType.CONCENTRATION_RISK, 'warning', 'Portfolio is concentrated in a single asset', { symbolCount, totalRisk: metrics.totalRisk }));
        }
        return alerts;
    }
    /**
     * Determine overall risk level
     * @private
     * @param metrics - Risk metrics
     * @returns Risk level
     */
    _determineRiskLevel(metrics) {
        // Calculate a risk score based on various factors
        let riskScore = 0;
        // Margin level contribution
        if (metrics.marginLevel < 110) {
            riskScore += 40;
        }
        else if (metrics.marginLevel < 150) {
            riskScore += 30;
        }
        else if (metrics.marginLevel < 200) {
            riskScore += 20;
        }
        else if (metrics.marginLevel < 300) {
            riskScore += 10;
        }
        // Drawdown contribution
        if (metrics.currentDrawdown > 20) {
            riskScore += 40;
        }
        else if (metrics.currentDrawdown > 15) {
            riskScore += 30;
        }
        else if (metrics.currentDrawdown > 10) {
            riskScore += 20;
        }
        else if (metrics.currentDrawdown > 5) {
            riskScore += 10;
        }
        // Daily P&L contribution
        if (metrics.dailyPnLPercentage < -7) {
            riskScore += 30;
        }
        else if (metrics.dailyPnLPercentage < -5) {
            riskScore += 20;
        }
        else if (metrics.dailyPnLPercentage < -3) {
            riskScore += 10;
        }
        // Total risk contribution
        if (metrics.totalRisk > 30) {
            riskScore += 40;
        }
        else if (metrics.totalRisk > 20) {
            riskScore += 30;
        }
        else if (metrics.totalRisk > 15) {
            riskScore += 20;
        }
        else if (metrics.totalRisk > 10) {
            riskScore += 10;
        }
        // Determine risk level based on score
        if (riskScore >= 80) {
            return risk_1.RiskLevel.VERY_HIGH;
        }
        else if (riskScore >= 60) {
            return risk_1.RiskLevel.HIGH;
        }
        else if (riskScore >= 40) {
            return risk_1.RiskLevel.MODERATE;
        }
        else if (riskScore >= 20) {
            return risk_1.RiskLevel.LOW;
        }
        else {
            return risk_1.RiskLevel.VERY_LOW;
        }
    }
    /**
     * Determine position risk level
     * @private
     * @param riskScore - Risk score
     * @returns Risk level
     */
    _determinePositionRiskLevel(riskScore) {
        if (riskScore >= 80) {
            return risk_1.RiskLevel.VERY_HIGH;
        }
        else if (riskScore >= 60) {
            return risk_1.RiskLevel.HIGH;
        }
        else if (riskScore >= 40) {
            return risk_1.RiskLevel.MODERATE;
        }
        else if (riskScore >= 20) {
            return risk_1.RiskLevel.LOW;
        }
        else {
            return risk_1.RiskLevel.VERY_LOW;
        }
    }
}
exports.RiskAssessmentService = RiskAssessmentService;
// Create default instance
const riskAssessmentService = new RiskAssessmentService();
exports.default = riskAssessmentService;
//# sourceMappingURL=riskAssessmentService.js.map
"use strict";
/**
 * ML-Powered Position Management System
 * Advanced position management using ML models for dynamic stop/take profit optimization,
 * position sizing, and exit timing prediction
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MLPositionManager = void 0;
const EnhancedTradingDecisionEngine_1 = require("./EnhancedTradingDecisionEngine");
const DataCollectorIntegration_1 = require("./DataCollectorIntegration");
const DeltaTradingBot_1 = require("./DeltaTradingBot");
const logger_1 = require("../utils/logger");
const ioredis_1 = __importDefault(require("ioredis"));
class MLPositionManager {
    constructor() {
        // Active positions tracking
        this.activePositions = new Map();
        this.positionHistory = [];
        // ML model training data
        this.trainingData = [];
        // Configuration optimized for small capital + high leverage
        this.config = {
            // ML thresholds for precise exits
            exitPredictionThreshold: 0.75,
            riskAdjustmentFactor: 0.3,
            // Dynamic trailing stops for profit protection
            trailingStopEnabled: true,
            trailingStopDistance: 0.008, // 0.8% trailing distance
            maxStopLossAdjustment: 0.005, // Max 0.5% adjustment
            // Aggressive take profit optimization
            dynamicTakeProfitEnabled: true,
            profitLockingThreshold: 0.6, // Lock at 60% of target
            maxTakeProfitExtension: 0.02, // Max 2% extension
            // Position sizing for small capital
            maxPositionAdjustment: 0.2, // Max 20% adjustment
            riskBasedSizing: true,
            // Hold time optimization for active trading
            holdTimeOptimization: true,
            maxHoldTime: 4 * 60 * 60 * 1000, // 4 hours
            minHoldTime: 2 * 60 * 1000 // 2 minutes
        };
        // Performance tracking
        this.performanceMetrics = {
            totalPositions: 0,
            winningPositions: 0,
            totalPnL: 0,
            maxDrawdown: 0,
            averageHoldTime: 0,
            mlAccuracy: 0
        };
        this.decisionEngine = new EnhancedTradingDecisionEngine_1.EnhancedTradingDecisionEngine();
        this.dataIntegration = new DataCollectorIntegration_1.DataCollectorIntegration();
        this.tradingBot = new DeltaTradingBot_1.DeltaTradingBot();
        this.redis = new ioredis_1.default({
            host: process.env.REDIS_HOST || 'localhost',
            port: parseInt(process.env.REDIS_PORT || '6379')
        });
    }
    /**
     * Initialize ML Position Manager
     */
    async initialize() {
        try {
            logger_1.logger.info('🤖 Initializing ML Position Manager...');
            // Initialize dependencies
            await this.decisionEngine.initialize();
            await this.dataIntegration.initialize();
            // Load existing positions from Redis
            await this.loadActivePositions();
            // Load historical training data
            await this.loadTrainingData();
            logger_1.logger.info('✅ ML Position Manager initialized successfully');
            logger_1.logger.info(`📊 Configuration: Exit Threshold ${(this.config.exitPredictionThreshold * 100).toFixed(0)}%, Trailing ${this.config.trailingStopEnabled ? 'ON' : 'OFF'}, Dynamic TP ${this.config.dynamicTakeProfitEnabled ? 'ON' : 'OFF'}`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize ML Position Manager:', error.message);
            throw error;
        }
    }
    /**
     * Create new position from trading decision
     */
    async createPosition(decision, currentPrice) {
        try {
            logger_1.logger.info(`📈 Creating position for ${decision.symbol} ${decision.action}`);
            const position = {
                id: `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                symbol: decision.symbol,
                side: decision.action === 'buy' ? 'long' : 'short',
                entryPrice: currentPrice,
                currentPrice: currentPrice,
                quantity: this.calculatePositionQuantity(decision, currentPrice),
                leverage: decision.leverage,
                // Initial levels from decision
                stopLoss: decision.stopLoss,
                takeProfit: decision.takeProfit,
                // ML predictions (initial)
                exitProbability: 0.1, // Low initial exit probability
                optimalExitPrice: decision.takeProfit,
                riskScore: decision.riskScore,
                // Performance tracking
                unrealizedPnL: 0,
                maxDrawdown: 0,
                maxProfit: 0,
                holdingTime: 0,
                // Metadata
                entryTimestamp: Date.now(),
                lastUpdate: Date.now(),
                decisionId: `${decision.symbol}_${decision.timestamp}`
            };
            // Store position
            this.activePositions.set(position.id, position);
            await this.savePositionToRedis(position);
            // Update performance metrics
            this.performanceMetrics.totalPositions++;
            logger_1.logger.info(`✅ Position created: ${position.id}`);
            logger_1.logger.info(`   Entry: $${position.entryPrice} | SL: $${position.stopLoss} | TP: $${position.takeProfit}`);
            logger_1.logger.info(`   Quantity: ${position.quantity} | Leverage: ${position.leverage}x`);
            return position;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to create position for ${decision.symbol}:`, error.message);
            return null;
        }
    }
    /**
     * Update position with current market data and ML predictions
     */
    async updatePosition(positionId, currentPrice) {
        try {
            const position = this.activePositions.get(positionId);
            if (!position) {
                logger_1.logger.warn(`⚠️ Position ${positionId} not found`);
                return null;
            }
            // Update basic position data
            position.currentPrice = currentPrice;
            position.holdingTime = Date.now() - position.entryTimestamp;
            position.lastUpdate = Date.now();
            // Calculate unrealized P&L
            position.unrealizedPnL = this.calculateUnrealizedPnL(position);
            // Update max profit/drawdown
            if (position.unrealizedPnL > position.maxProfit) {
                position.maxProfit = position.unrealizedPnL;
            }
            if (position.unrealizedPnL < position.maxDrawdown) {
                position.maxDrawdown = position.unrealizedPnL;
            }
            // Get ML features for position management
            const features = await this.extractPositionFeatures(position);
            if (features) {
                // Update ML predictions
                await this.updateMLPredictions(position, features);
                // Apply dynamic position management
                await this.applyDynamicManagement(position, features);
            }
            // Save updated position
            await this.savePositionToRedis(position);
            logger_1.logger.debug(`📊 Position ${positionId} updated: P&L ${position.unrealizedPnL.toFixed(4)}, Exit Prob ${(position.exitProbability * 100).toFixed(1)}%`);
            return position;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to update position ${positionId}:`, error.message);
            return null;
        }
    }
    /**
     * Check if position should be closed based on ML predictions
     */
    async shouldClosePosition(positionId) {
        try {
            const position = this.activePositions.get(positionId);
            if (!position) {
                return { shouldClose: false, reason: 'Position not found', urgency: 'low' };
            }
            // Check ML exit probability
            if (position.exitProbability > this.config.exitPredictionThreshold) {
                return {
                    shouldClose: true,
                    reason: `ML exit signal: ${(position.exitProbability * 100).toFixed(1)}% probability`,
                    urgency: 'high'
                };
            }
            // Check stop loss hit
            const stopHit = this.isStopLossHit(position);
            if (stopHit) {
                return {
                    shouldClose: true,
                    reason: `Stop loss hit: ${position.currentPrice} vs ${position.stopLoss}`,
                    urgency: 'high'
                };
            }
            // Check take profit hit
            const tpHit = this.isTakeProfitHit(position);
            if (tpHit) {
                return {
                    shouldClose: true,
                    reason: `Take profit hit: ${position.currentPrice} vs ${position.takeProfit}`,
                    urgency: 'medium'
                };
            }
            // Check maximum hold time
            if (this.config.holdTimeOptimization && position.holdingTime > this.config.maxHoldTime) {
                return {
                    shouldClose: true,
                    reason: `Maximum hold time exceeded: ${Math.round(position.holdingTime / 60000)} minutes`,
                    urgency: 'medium'
                };
            }
            // Check risk-based exit
            if (position.riskScore > 0.8) {
                return {
                    shouldClose: true,
                    reason: `High risk score: ${(position.riskScore * 100).toFixed(0)}%`,
                    urgency: 'medium'
                };
            }
            return { shouldClose: false, reason: 'Position within parameters', urgency: 'low' };
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to check position closure for ${positionId}:`, error.message);
            return { shouldClose: false, reason: 'Error checking position', urgency: 'low' };
        }
    }
    /**
     * Close position and record training data
     */
    async closePosition(positionId, exitPrice, reason) {
        try {
            const position = this.activePositions.get(positionId);
            if (!position) {
                logger_1.logger.warn(`⚠️ Position ${positionId} not found for closure`);
                return false;
            }
            // Calculate final P&L
            const finalPnL = this.calculateRealizedPnL(position, exitPrice);
            // Record training data for ML improvement
            await this.recordTrainingData(position, exitPrice, finalPnL);
            // Update performance metrics
            this.updatePerformanceMetrics(position, finalPnL);
            // Move to history
            position.currentPrice = exitPrice;
            position.unrealizedPnL = finalPnL;
            this.positionHistory.push({ ...position });
            // Remove from active positions
            this.activePositions.delete(positionId);
            await this.redis.del(`position:${positionId}`);
            logger_1.logger.info(`🔒 Position closed: ${positionId}`);
            logger_1.logger.info(`   Exit: $${exitPrice} | P&L: ${finalPnL.toFixed(4)} | Reason: ${reason}`);
            logger_1.logger.info(`   Hold Time: ${Math.round(position.holdingTime / 60000)} minutes`);
            return true;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to close position ${positionId}:`, error.message);
            return false;
        }
    }
    /**
     * Get all active positions
     */
    getActivePositions() {
        return Array.from(this.activePositions.values());
    }
    /**
     * Get position by ID
     */
    getPosition(positionId) {
        return this.activePositions.get(positionId) || null;
    }
    /**
     * Get performance metrics
     */
    getPerformanceMetrics() {
        const winRate = this.performanceMetrics.totalPositions > 0 ?
            (this.performanceMetrics.winningPositions / this.performanceMetrics.totalPositions) * 100 : 0;
        return {
            ...this.performanceMetrics,
            winRate: winRate.toFixed(1),
            averagePnL: this.performanceMetrics.totalPositions > 0 ?
                (this.performanceMetrics.totalPnL / this.performanceMetrics.totalPositions).toFixed(4) : '0.0000',
            activePositions: this.activePositions.size
        };
    }
    /**
     * Update configuration
     */
    updateConfiguration(newConfig) {
        this.config = { ...this.config, ...newConfig };
        logger_1.logger.info('🔧 ML Position Manager configuration updated');
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger_1.logger.info('🧹 Cleaning up ML Position Manager...');
            // Save all active positions
            for (const position of this.activePositions.values()) {
                await this.savePositionToRedis(position);
            }
            // Save training data
            await this.saveTrainingData();
            await this.redis.quit();
            logger_1.logger.info('✅ ML Position Manager cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during ML Position Manager cleanup:', error.message);
        }
    }
    // Private methods for ML position management
    /**
     * Calculate position quantity based on decision and current price
     */
    calculatePositionQuantity(decision, currentPrice) {
        // This would integrate with actual account balance and position sizing logic
        // For now, return a calculated quantity based on position size percentage
        const notionalValue = 1000; // Placeholder - should get actual account balance
        const positionValue = notionalValue * decision.positionSize;
        return positionValue / currentPrice;
    }
    /**
     * Calculate unrealized P&L for position
     */
    calculateUnrealizedPnL(position) {
        const priceDiff = position.side === 'long' ?
            position.currentPrice - position.entryPrice :
            position.entryPrice - position.currentPrice;
        return (priceDiff / position.entryPrice) * position.leverage;
    }
    /**
     * Calculate realized P&L at exit
     */
    calculateRealizedPnL(position, exitPrice) {
        const priceDiff = position.side === 'long' ?
            exitPrice - position.entryPrice :
            position.entryPrice - exitPrice;
        return (priceDiff / position.entryPrice) * position.leverage;
    }
    /**
     * Extract ML features for position management (45 features)
     */
    async extractPositionFeatures(position) {
        try {
            // Get market features from data integration
            const marketFeatures = await this.dataIntegration.getRealTimeTradingFeatures(position.symbol);
            if (!marketFeatures)
                return null;
            // Position-specific features (15 features)
            const positionFeatures = [
                // Position metrics (5)
                position.unrealizedPnL,
                position.maxProfit,
                position.maxDrawdown,
                position.holdingTime / (60 * 60 * 1000), // Hours
                position.leverage / 200, // Normalized leverage
                // Price relationship features (5)
                (position.currentPrice - position.entryPrice) / position.entryPrice,
                (position.currentPrice - position.stopLoss) / position.entryPrice,
                (position.takeProfit - position.currentPrice) / position.entryPrice,
                position.side === 'long' ? 1 : -1, // Side encoding
                position.riskScore,
                // Time-based features (5)
                this.getTimeOfDayFeature(),
                this.getMarketSessionFeature(),
                Math.min(1, position.holdingTime / this.config.maxHoldTime), // Hold time ratio
                position.exitProbability, // Previous ML prediction
                marketFeatures.dataQuality
            ];
            // Combine with market features (36) + position features (15) = 51 total features
            return [
                // Market features from data integration (36)
                ...marketFeatures.fibonacciProximity,
                marketFeatures.nearestFibLevel,
                marketFeatures.fibStrength,
                marketFeatures.bias4h,
                marketFeatures.bias1h,
                marketFeatures.bias15m,
                marketFeatures.bias5m,
                marketFeatures.overallBias,
                marketFeatures.biasAlignment,
                marketFeatures.bodyPercentage,
                marketFeatures.wickPercentage,
                marketFeatures.buyingPressure,
                marketFeatures.sellingPressure,
                marketFeatures.candleType,
                marketFeatures.momentum,
                marketFeatures.volatility,
                marketFeatures.volume,
                marketFeatures.volumeRatio,
                marketFeatures.timeOfDay,
                marketFeatures.marketSession,
                marketFeatures.pricePosition,
                // Position-specific features (15)
                ...positionFeatures
            ];
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to extract position features for ${position.id}:`, error.message);
            return null;
        }
    }
    /**
     * Update ML predictions for position
     */
    async updateMLPredictions(position, features) {
        try {
            // Simulate ML model predictions (in production, this would call actual ML models)
            // Exit probability prediction (0-1)
            const exitProb = this.predictExitProbability(features, position);
            position.exitProbability = exitProb;
            // Optimal exit price prediction
            const optimalExit = this.predictOptimalExitPrice(features, position);
            position.optimalExitPrice = optimalExit;
            // Risk score update
            const riskScore = this.predictRiskScore(features, position);
            position.riskScore = riskScore;
            logger_1.logger.debug(`🤖 ML predictions updated for ${position.id}: Exit ${(exitProb * 100).toFixed(1)}%, Risk ${(riskScore * 100).toFixed(1)}%`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to update ML predictions for ${position.id}:`, error.message);
        }
    }
    /**
     * Apply dynamic position management based on ML insights
     */
    async applyDynamicManagement(position, features) {
        try {
            // Dynamic trailing stop adjustment
            if (this.config.trailingStopEnabled) {
                this.updateTrailingStop(position);
            }
            // Dynamic take profit optimization
            if (this.config.dynamicTakeProfitEnabled) {
                this.updateDynamicTakeProfit(position, features);
            }
            // Risk-based stop loss adjustment
            this.updateRiskBasedStopLoss(position);
            logger_1.logger.debug(`⚙️ Dynamic management applied to ${position.id}: SL ${position.stopLoss}, TP ${position.takeProfit}`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to apply dynamic management to ${position.id}:`, error.message);
        }
    }
    /**
     * Update trailing stop based on current price movement
     */
    updateTrailingStop(position) {
        const trailingDistance = this.config.trailingStopDistance;
        if (position.side === 'long') {
            // For long positions, trail stop loss upward
            const newStopLoss = position.currentPrice * (1 - trailingDistance);
            if (newStopLoss > position.stopLoss) {
                position.stopLoss = Math.min(newStopLoss, position.stopLoss + this.config.maxStopLossAdjustment * position.entryPrice);
                position.trailingStop = newStopLoss;
            }
        }
        else {
            // For short positions, trail stop loss downward
            const newStopLoss = position.currentPrice * (1 + trailingDistance);
            if (newStopLoss < position.stopLoss) {
                position.stopLoss = Math.max(newStopLoss, position.stopLoss - this.config.maxStopLossAdjustment * position.entryPrice);
                position.trailingStop = newStopLoss;
            }
        }
    }
    /**
     * Update dynamic take profit based on ML predictions
     */
    updateDynamicTakeProfit(position, features) {
        // Check if we should lock in profits
        const profitRatio = position.unrealizedPnL / (position.takeProfit - position.entryPrice) * position.entryPrice;
        if (profitRatio > this.config.profitLockingThreshold) {
            // Lock in some profits by moving take profit closer
            const lockingAdjustment = 0.3; // Lock 30% of remaining profit
            if (position.side === 'long') {
                const remainingProfit = position.takeProfit - position.currentPrice;
                position.takeProfit = position.currentPrice + (remainingProfit * (1 - lockingAdjustment));
            }
            else {
                const remainingProfit = position.currentPrice - position.takeProfit;
                position.takeProfit = position.currentPrice - (remainingProfit * (1 - lockingAdjustment));
            }
        }
        // Extend take profit if ML predicts continued movement
        if (position.exitProbability < 0.3 && position.unrealizedPnL > 0) {
            const extension = this.config.maxTakeProfitExtension * position.entryPrice;
            if (position.side === 'long') {
                position.takeProfit = Math.min(position.takeProfit + extension, position.optimalExitPrice);
            }
            else {
                position.takeProfit = Math.max(position.takeProfit - extension, position.optimalExitPrice);
            }
        }
    }
    /**
     * Update stop loss based on risk assessment
     */
    updateRiskBasedStopLoss(position) {
        if (position.riskScore > 0.7) {
            // Tighten stop loss for high risk
            const riskAdjustment = this.config.riskAdjustmentFactor * position.riskScore;
            const adjustment = riskAdjustment * this.config.maxStopLossAdjustment * position.entryPrice;
            if (position.side === 'long') {
                position.stopLoss = Math.max(position.stopLoss, position.currentPrice - adjustment);
            }
            else {
                position.stopLoss = Math.min(position.stopLoss, position.currentPrice + adjustment);
            }
        }
    }
    /**
     * Check if stop loss is hit
     */
    isStopLossHit(position) {
        if (position.side === 'long') {
            return position.currentPrice <= position.stopLoss;
        }
        else {
            return position.currentPrice >= position.stopLoss;
        }
    }
    /**
     * Check if take profit is hit
     */
    isTakeProfitHit(position) {
        if (position.side === 'long') {
            return position.currentPrice >= position.takeProfit;
        }
        else {
            return position.currentPrice <= position.takeProfit;
        }
    }
    /**
     * Predict exit probability using ML features
     */
    predictExitProbability(features, position) {
        // Simplified ML prediction logic (in production, use trained models)
        // Base probability from hold time
        const holdTimeRatio = position.holdingTime / this.config.maxHoldTime;
        let exitProb = holdTimeRatio * 0.3; // 30% weight for hold time
        // Add market volatility factor
        const volatility = features[15] || 0; // Volatility feature
        exitProb += volatility * 0.2; // 20% weight for volatility
        // Add bias alignment factor (less alignment = higher exit probability)
        const biasAlignment = features[8] || 0.5; // Bias alignment feature
        exitProb += (1 - biasAlignment) * 0.2; // 20% weight for bias misalignment
        // Add unrealized P&L factor
        if (position.unrealizedPnL < -0.02) { // -2% loss
            exitProb += 0.3; // Increase exit probability for losses
        }
        else if (position.unrealizedPnL > 0.03) { // +3% profit
            exitProb += 0.2; // Increase exit probability for large profits
        }
        // Add risk score factor
        exitProb += position.riskScore * 0.1; // 10% weight for risk
        return Math.min(1.0, Math.max(0.0, exitProb));
    }
    /**
     * Predict optimal exit price using ML features
     */
    predictOptimalExitPrice(features, position) {
        // Simplified optimal exit prediction
        // Base on current take profit
        let optimalExit = position.takeProfit;
        // Adjust based on momentum
        const momentum = features[14] || 0; // Momentum feature
        const momentumAdjustment = momentum * 0.01 * position.entryPrice; // 1% max adjustment
        if (position.side === 'long') {
            optimalExit += momentumAdjustment;
        }
        else {
            optimalExit -= momentumAdjustment;
        }
        // Adjust based on Fibonacci levels
        const fibStrength = features[6] || 0; // Fibonacci strength
        const fibAdjustment = fibStrength * 0.005 * position.entryPrice; // 0.5% max adjustment
        if (position.side === 'long') {
            optimalExit += fibAdjustment;
        }
        else {
            optimalExit -= fibAdjustment;
        }
        return optimalExit;
    }
    /**
     * Predict risk score using ML features
     */
    predictRiskScore(features, position) {
        // Simplified risk prediction
        let riskScore = 0;
        // Market volatility risk
        const volatility = features[15] || 0;
        riskScore += volatility * 0.3;
        // Time-based risk (longer holds = higher risk)
        const holdTimeRatio = position.holdingTime / this.config.maxHoldTime;
        riskScore += holdTimeRatio * 0.2;
        // Drawdown risk
        if (position.maxDrawdown < -0.01) { // -1% drawdown
            riskScore += Math.abs(position.maxDrawdown) * 2; // 2x weight for drawdown
        }
        // Market session risk (Asian session = higher risk)
        const marketSession = features[19] || 1;
        if (marketSession === 0) { // Asian session
            riskScore += 0.2;
        }
        // Data quality risk
        const dataQuality = features[features.length - 1] || 1;
        riskScore += (1 - dataQuality) * 0.3;
        return Math.min(1.0, Math.max(0.0, riskScore));
    }
    /**
     * Record training data for ML model improvement
     */
    async recordTrainingData(position, exitPrice, finalPnL) {
        try {
            const features = await this.extractPositionFeatures(position);
            if (!features)
                return;
            const trainingData = {
                features,
                exitPrice,
                exitTime: position.holdingTime / (60 * 1000), // Minutes
                profitLoss: finalPnL,
                wasOptimal: this.wasExitOptimal(position, exitPrice, finalPnL)
            };
            this.trainingData.push(trainingData);
            // Keep only last 1000 training samples
            if (this.trainingData.length > 1000) {
                this.trainingData = this.trainingData.slice(-1000);
            }
            logger_1.logger.debug(`📚 Training data recorded for position ${position.id}`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to record training data for ${position.id}:`, error.message);
        }
    }
    /**
     * Determine if exit was optimal
     */
    wasExitOptimal(position, exitPrice, finalPnL) {
        // Simple heuristic: exit is optimal if P&L > 2% or loss < -1%
        return finalPnL > 0.02 || (finalPnL < 0 && finalPnL > -0.01);
    }
    /**
     * Update performance metrics
     */
    updatePerformanceMetrics(position, finalPnL) {
        if (finalPnL > 0) {
            this.performanceMetrics.winningPositions++;
        }
        this.performanceMetrics.totalPnL += finalPnL;
        if (finalPnL < this.performanceMetrics.maxDrawdown) {
            this.performanceMetrics.maxDrawdown = finalPnL;
        }
        // Update average hold time
        const totalHoldTime = this.performanceMetrics.averageHoldTime * (this.performanceMetrics.totalPositions - 1);
        this.performanceMetrics.averageHoldTime = (totalHoldTime + position.holdingTime) / this.performanceMetrics.totalPositions;
    }
    /**
     * Save position to Redis
     */
    async savePositionToRedis(position) {
        try {
            await this.redis.setex(`position:${position.id}`, 86400, JSON.stringify(position)); // 24 hour TTL
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to save position ${position.id} to Redis:`, error.message);
        }
    }
    /**
     * Load active positions from Redis
     */
    async loadActivePositions() {
        try {
            const keys = await this.redis.keys('position:*');
            for (const key of keys) {
                const positionData = await this.redis.get(key);
                if (positionData) {
                    const position = JSON.parse(positionData);
                    this.activePositions.set(position.id, position);
                }
            }
            logger_1.logger.info(`📊 Loaded ${this.activePositions.size} active positions from Redis`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to load active positions from Redis:', error.message);
        }
    }
    /**
     * Load training data from Redis
     */
    async loadTrainingData() {
        try {
            const trainingDataStr = await this.redis.get('ml_position_training_data');
            if (trainingDataStr) {
                this.trainingData = JSON.parse(trainingDataStr);
                logger_1.logger.info(`📚 Loaded ${this.trainingData.length} training samples`);
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to load training data from Redis:', error.message);
        }
    }
    /**
     * Save training data to Redis
     */
    async saveTrainingData() {
        try {
            await this.redis.setex('ml_position_training_data', 86400 * 7, JSON.stringify(this.trainingData)); // 7 day TTL
            logger_1.logger.debug(`💾 Saved ${this.trainingData.length} training samples to Redis`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to save training data to Redis:', error.message);
        }
    }
    /**
     * Get time of day feature (0-1)
     */
    getTimeOfDayFeature() {
        const now = new Date();
        const hours = now.getUTCHours();
        const minutes = now.getUTCMinutes();
        const totalMinutes = hours * 60 + minutes;
        return totalMinutes / (24 * 60);
    }
    /**
     * Get market session feature
     */
    getMarketSessionFeature() {
        const now = new Date();
        const utcHours = now.getUTCHours();
        if (utcHours >= 0 && utcHours < 7)
            return 0; // Asian
        if (utcHours >= 7 && utcHours < 13)
            return 1; // European
        if (utcHours >= 13 && utcHours < 16)
            return 2; // Euro-American overlap
        if (utcHours >= 16 && utcHours < 22)
            return 1.5; // American
        return 0.5; // Late American/Early Asian
    }
}
exports.MLPositionManager = MLPositionManager;
//# sourceMappingURL=MLPositionManager.js.map
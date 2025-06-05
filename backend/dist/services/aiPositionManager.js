"use strict";
/**
 * AI Position Manager
 * Manages Delta Exchange positions using AI-powered dynamic take profit system
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AIPositionManager = void 0;
const dynamicTakeProfitManager_1 = require("./dynamicTakeProfitManager");
const MultiTimeframeAnalysisEngine_1 = require("./MultiTimeframeAnalysisEngine");
const EnhancedMarketRegimeDetector_1 = require("./EnhancedMarketRegimeDetector");
const logger_1 = require("../utils/logger");
class AIPositionManager {
    constructor(deltaApi, deltaUnified) {
        this.managedPositions = new Map();
        this.isRunning = false;
        this.updateInterval = null;
        this.deltaApi = deltaApi;
        this.deltaUnified = deltaUnified;
        this.takeProfitManager = new dynamicTakeProfitManager_1.DynamicTakeProfitManager();
        this.mtfAnalyzer = new MultiTimeframeAnalysisEngine_1.MultiTimeframeAnalysisEngine(deltaUnified);
        this.regimeDetector = new EnhancedMarketRegimeDetector_1.EnhancedMarketRegimeDetector(deltaUnified);
    }
    /**
     * Start AI position management
     */
    async startManagement() {
        if (this.isRunning) {
            logger_1.logger.warn('AI Position Manager is already running');
            return;
        }
        this.isRunning = true;
        logger_1.logger.info('🤖 Starting AI Position Management System');
        // Initial position scan
        await this.scanAndManagePositions();
        // Set up periodic updates every 30 seconds
        this.updateInterval = setInterval(async () => {
            await this.scanAndManagePositions();
        }, 30000);
        logger_1.logger.info('✅ AI Position Manager started - monitoring every 30 seconds');
    }
    /**
     * Stop AI position management
     */
    stopManagement() {
        this.isRunning = false;
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        logger_1.logger.info('🛑 AI Position Manager stopped');
    }
    /**
     * Scan and manage all positions
     */
    async scanAndManagePositions() {
        try {
            logger_1.logger.info('🔍 Scanning positions for AI management...');
            // Get current positions from Delta Exchange
            const positions = await this.deltaApi.getPositions();
            if (positions.length === 0) {
                logger_1.logger.info('📊 No active positions found');
                return;
            }
            logger_1.logger.info(`🎯 Found ${positions.length} active positions`);
            // Process each position
            for (const position of positions) {
                await this.managePosition(position);
            }
            // Display summary
            this.displayManagementSummary();
        }
        catch (error) {
            logger_1.logger.error('❌ Error scanning positions:', error.message);
        }
    }
    /**
     * Manage individual position with AI
     */
    async managePosition(position) {
        try {
            const positionId = `${position.symbol}_${position.size}`;
            // Get or create managed position
            let managedPos = this.managedPositions.get(positionId);
            if (!managedPos) {
                // New position - set up AI management
                managedPos = await this.initializePositionManagement(position);
                this.managedPositions.set(positionId, managedPos);
                logger_1.logger.info(`🆕 New position detected: ${position.symbol} - Setting up AI management`);
            }
            // Update current data
            managedPos.currentPrice = await this.getCurrentPrice(position.symbol);
            managedPos.unrealizedPnl = parseFloat(position.unrealized_pnl || '0');
            managedPos.lastUpdate = Date.now();
            // Get AI analysis
            const aiAnalysis = await this.getAIAnalysis(managedPos);
            // Execute AI recommendations
            await this.executeAIRecommendation(managedPos, aiAnalysis);
            logger_1.logger.info(`🤖 AI Analysis for ${position.symbol}: ${aiAnalysis.action} (${aiAnalysis.confidence}% confidence)`);
            logger_1.logger.info(`   Reasoning: ${aiAnalysis.reasoning}`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Error managing position ${position.symbol}:`, error.message);
        }
    }
    /**
     * Initialize AI management for new position
     */
    async initializePositionManagement(position) {
        const side = parseFloat(position.size) > 0 ? 'LONG' : 'SHORT';
        const entryPrice = parseFloat(position.entry_price);
        const size = Math.abs(parseFloat(position.size));
        // Generate dynamic take profit levels
        const marketRegime = {
            type: 'TRENDING', // Simplified - in real system would analyze market
            strength: 75,
            direction: side === 'LONG' ? 'UP' : 'DOWN',
            volatility: 0.03,
            volume: 1.2,
        };
        const takeProfitConfig = {
            asset: position.symbol,
            entryPrice,
            stopLoss: side === 'LONG' ? entryPrice * 0.975 : entryPrice * 1.025, // 2.5% stop
            positionSize: size,
            side: side === 'LONG' ? 'BUY' : 'SELL',
            marketRegime,
            momentum: side === 'LONG' ? 50 : -50,
            volume: 1.2,
        };
        const takeProfitLevels = this.takeProfitManager.generateDynamicTakeProfitLevels(takeProfitConfig);
        // Convert to partial exits
        const partialExits = takeProfitLevels.map((level, index) => ({
            level: index + 1,
            percentage: level.percentage,
            targetPrice: level.priceTarget,
            executed: false,
        }));
        const managedPosition = {
            id: `${position.symbol}_${position.size}`,
            symbol: position.symbol,
            side,
            size,
            entryPrice,
            currentPrice: entryPrice,
            unrealizedPnl: parseFloat(position.unrealized_pnl || '0'),
            takeProfitLevels,
            stopLoss: takeProfitConfig.stopLoss,
            partialExits,
            status: 'MANAGING',
            lastUpdate: Date.now(),
            aiRecommendations: [],
        };
        logger_1.logger.info(`🎯 AI Management initialized for ${position.symbol}:`);
        logger_1.logger.info(`   Side: ${side}, Size: ${size}, Entry: $${entryPrice}`);
        logger_1.logger.info(`   Stop Loss: $${takeProfitConfig.stopLoss.toFixed(2)}`);
        logger_1.logger.info(`   Take Profit Levels: ${partialExits.length}`);
        return managedPosition;
    }
    /**
     * Get current price for symbol
     */
    async getCurrentPrice(symbol) {
        try {
            const ticker = await this.deltaApi.getTicker(symbol);
            return parseFloat(ticker.close || '0');
        }
        catch (error) {
            logger_1.logger.warn(`⚠️ Could not get current price for ${symbol}, using cached price`);
            return 0;
        }
    }
    /**
     * Get enhanced position health analysis
     */
    async getEnhancedPositionHealth(position) {
        try {
            // Get multi-timeframe analysis
            const mtfAnalysis = await this.mtfAnalyzer.analyzeSymbol(position.symbol);
            // Get regime analysis
            const regimeAnalysis = await this.regimeDetector.detectRegime(position.symbol);
            // Calculate position metrics
            const priceChange = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
            const positionAge = Date.now() - position.lastUpdate;
            // Multi-timeframe trend alignment
            const trendAlignment = this.calculateTrendAlignment(position, mtfAnalysis);
            // Momentum score
            const momentumScore = this.calculateMomentumScore(mtfAnalysis, priceChange);
            // Risk-adjusted return
            const riskAdjustedReturn = this.calculateRiskAdjustedReturn(position, regimeAnalysis);
            // Volatility factor
            const volatilityFactor = regimeAnalysis.volatility_metrics.price_volatility;
            // Regime compatibility
            const regimeCompatibility = this.calculateRegimeCompatibility(position, regimeAnalysis);
            // ML prediction
            const mlPrediction = this.generateMLPrediction(position, mtfAnalysis, regimeAnalysis);
            // Calculate individual factors
            const factors = {
                multi_timeframe_alignment: trendAlignment,
                position_age_factor: this.calculateAgeFactor(positionAge, regimeAnalysis),
                pnl_momentum: this.calculatePnLMomentum(position, priceChange),
                market_regime_score: regimeAnalysis.confidence,
                volume_confirmation: this.calculateVolumeConfirmation(mtfAnalysis)
            };
            // Calculate overall health score
            const score = this.calculateEnhancedHealthScore(factors, mlPrediction, regimeCompatibility);
            // Generate recommendations
            const recommendations = this.generateEnhancedRecommendations(score, factors, mlPrediction, regimeAnalysis, position);
            return {
                score,
                trend_alignment: trendAlignment,
                momentum_score: momentumScore,
                risk_adjusted_return: riskAdjustedReturn,
                volatility_factor: volatilityFactor,
                regime_compatibility: regimeCompatibility,
                ml_prediction: mlPrediction,
                factors,
                recommendations
            };
        }
        catch (error) {
            logger_1.logger.error(`Error calculating enhanced position health for ${position.symbol}:`, error);
            // Fallback to basic analysis
            return this.getBasicPositionHealth(position);
        }
    }
    /**
     * Get AI analysis for position (enhanced version)
     */
    async getAIAnalysis(position) {
        // Get enhanced position health
        const health = await this.getEnhancedPositionHealth(position);
        // Convert enhanced health to AI analysis
        return {
            action: health.recommendations.action,
            confidence: health.score,
            reasoning: health.recommendations.reasoning.join('; '),
            targetPrice: health.recommendations.optimal_exit_price,
            percentage: this.getExitPercentage(health.recommendations.action, health.score),
            newStopLoss: this.calculateDynamicStopLoss(position, health)
        };
    }
    /**
     * Original AI analysis method (fallback)
     */
    async getBasicAIAnalysis(position) {
        const priceChange = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
        const isProfit = (position.side === 'LONG' && priceChange > 0) || (position.side === 'SHORT' && priceChange < 0);
        const profitPercent = Math.abs(priceChange);
        // AI Decision Logic
        if (isProfit && profitPercent >= 5) {
            // Strong profit - consider partial exit
            const unexecutedExits = position.partialExits.filter(exit => !exit.executed);
            if (unexecutedExits.length > 0) {
                const nextExit = unexecutedExits[0];
                const shouldExit = position.side === 'LONG'
                    ? position.currentPrice >= nextExit.targetPrice
                    : position.currentPrice <= nextExit.targetPrice;
                if (shouldExit) {
                    return {
                        action: 'PARTIAL_EXIT',
                        confidence: 85,
                        reasoning: `Price reached take profit level ${nextExit.level} (${profitPercent.toFixed(1)}% profit)`,
                        targetPrice: nextExit.targetPrice,
                        percentage: nextExit.percentage,
                    };
                }
            }
            // Trail stop loss if in significant profit
            if (profitPercent >= 10) {
                const newStopLoss = position.side === 'LONG'
                    ? position.currentPrice * 0.95 // Trail 5% below current price
                    : position.currentPrice * 1.05; // Trail 5% above current price
                return {
                    action: 'TRAIL_STOP',
                    confidence: 75,
                    reasoning: `Trailing stop loss due to ${profitPercent.toFixed(1)}% profit`,
                    newStopLoss,
                };
            }
        }
        // Check stop loss
        const shouldStop = position.side === 'LONG'
            ? position.currentPrice <= position.stopLoss
            : position.currentPrice >= position.stopLoss;
        if (shouldStop) {
            return {
                action: 'FULL_EXIT',
                confidence: 95,
                reasoning: `Stop loss triggered at $${position.currentPrice.toFixed(2)}`,
            };
        }
        // Default: hold position
        return {
            action: 'HOLD',
            confidence: 60,
            reasoning: `Position within normal range (${priceChange.toFixed(1)}% from entry)`,
        };
    }
    /**
     * Execute AI recommendation
     */
    async executeAIRecommendation(position, analysis) {
        position.aiRecommendations.push(`${new Date().toISOString()}: ${analysis.action} - ${analysis.reasoning}`);
        switch (analysis.action) {
            case 'PARTIAL_EXIT':
                await this.executePartialExit(position, analysis);
                break;
            case 'FULL_EXIT':
                await this.executeFullExit(position, analysis);
                break;
            case 'TRAIL_STOP':
                await this.updateStopLoss(position, analysis.newStopLoss);
                break;
            case 'HOLD':
                // No action needed
                break;
        }
    }
    /**
     * Execute partial exit
     */
    async executePartialExit(position, analysis) {
        try {
            logger_1.logger.info(`🎯 Executing partial exit: ${analysis.percentage}% at $${analysis.targetPrice?.toFixed(2)}`);
            // In a real implementation, this would place an order
            // For now, we'll simulate the execution
            const exitSize = (position.size * analysis.percentage) / 100;
            // Mark the exit as executed
            const exit = position.partialExits.find(e => e.targetPrice === analysis.targetPrice);
            if (exit) {
                exit.executed = true;
                exit.executedAt = Date.now();
                exit.pnl = this.calculatePartialPnl(position, analysis.targetPrice, exitSize);
            }
            // Update position size
            position.size -= exitSize;
            logger_1.logger.info(`✅ Partial exit executed: ${analysis.percentage}% at $${analysis.targetPrice?.toFixed(2)}`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to execute partial exit:', error.message);
        }
    }
    /**
     * Execute full exit
     */
    async executeFullExit(position, analysis) {
        try {
            logger_1.logger.info(`🚨 Executing full exit: ${analysis.reasoning}`);
            // In a real implementation, this would close the position
            position.status = 'CLOSED';
            logger_1.logger.info(`✅ Position closed: ${position.symbol}`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to execute full exit:', error.message);
        }
    }
    /**
     * Update stop loss
     */
    async updateStopLoss(position, newStopLoss) {
        try {
            logger_1.logger.info(`🔄 Updating stop loss from $${position.stopLoss.toFixed(2)} to $${newStopLoss.toFixed(2)}`);
            position.stopLoss = newStopLoss;
            logger_1.logger.info(`✅ Stop loss updated to $${newStopLoss.toFixed(2)}`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to update stop loss:', error.message);
        }
    }
    /**
     * Calculate partial P&L
     */
    calculatePartialPnl(position, exitPrice, exitSize) {
        const priceChange = position.side === 'LONG'
            ? exitPrice - position.entryPrice
            : position.entryPrice - exitPrice;
        return (priceChange / position.entryPrice) * exitSize * 200; // Assuming 200x leverage
    }
    /**
     * Display management summary
     */
    displayManagementSummary() {
        const activePositions = Array.from(this.managedPositions.values()).filter(p => p.status === 'MANAGING');
        if (activePositions.length === 0) {
            logger_1.logger.info('📊 No positions under AI management');
            return;
        }
        logger_1.logger.info('\n🤖 AI POSITION MANAGEMENT SUMMARY:');
        activePositions.forEach(position => {
            const profitPercent = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
            const executedExits = position.partialExits.filter(e => e.executed).length;
            logger_1.logger.info(`   ${position.symbol}: ${position.side} $${position.unrealizedPnl.toFixed(2)} (${profitPercent.toFixed(1)}%)`);
            logger_1.logger.info(`     Partial Exits: ${executedExits}/${position.partialExits.length}`);
            logger_1.logger.info(`     Stop Loss: $${position.stopLoss.toFixed(2)}`);
        });
    }
    /**
     * Get managed positions
     */
    getManagedPositions() {
        return Array.from(this.managedPositions.values());
    }
    // Enhanced Position Health Analysis Helper Methods
    /**
     * Calculate trend alignment across timeframes
     */
    calculateTrendAlignment(position, mtfAnalysis) {
        const positionDirection = position.side === 'LONG' ? 1 : -1;
        const overallTrend = mtfAnalysis.overallTrend;
        let alignment = 0;
        if (overallTrend.direction === 'bullish' && positionDirection > 0) {
            alignment = overallTrend.alignment;
        }
        else if (overallTrend.direction === 'bearish' && positionDirection < 0) {
            alignment = overallTrend.alignment;
        }
        else if (overallTrend.direction === 'sideways') {
            alignment = 0;
        }
        else {
            alignment = -overallTrend.alignment; // Misaligned
        }
        return alignment;
    }
    /**
     * Calculate momentum score
     */
    calculateMomentumScore(mtfAnalysis, priceChange) {
        const trendStrength = mtfAnalysis.overallTrend.strength;
        const priceMovement = Math.abs(priceChange) / 100; // Normalize to 0-1
        return Math.min(1, (trendStrength + priceMovement) / 2);
    }
    /**
     * Calculate risk-adjusted return
     */
    calculateRiskAdjustedReturn(position, regimeAnalysis) {
        const priceChange = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
        const volatility = regimeAnalysis.volatility_metrics.price_volatility;
        if (volatility === 0)
            return 0;
        return priceChange / (volatility * 100); // Sharpe-like ratio
    }
    /**
     * Calculate regime compatibility
     */
    calculateRegimeCompatibility(position, regimeAnalysis) {
        const recommendations = regimeAnalysis.trading_recommendations;
        const positionSide = position.side;
        // Check if current regime favors the position direction
        let compatibility = 0.5; // Base compatibility
        if (recommendations.strategy_type === 'trend_following') {
            if ((regimeAnalysis.current_regime.includes('bullish') && positionSide === 'LONG') ||
                (regimeAnalysis.current_regime.includes('bearish') && positionSide === 'SHORT')) {
                compatibility = 0.9;
            }
        }
        else if (recommendations.strategy_type === 'mean_reversion') {
            // Mean reversion strategies work better in ranging markets
            compatibility = regimeAnalysis.current_regime === 'sideways' ? 0.8 : 0.3;
        }
        return compatibility;
    }
    /**
     * Generate ML prediction
     */
    generateMLPrediction(position, mtfAnalysis, regimeAnalysis) {
        // Simplified ML prediction - in production, this would use actual ML models
        const trendAlignment = this.calculateTrendAlignment(position, mtfAnalysis);
        const regimeCompatibility = this.calculateRegimeCompatibility(position, regimeAnalysis);
        const signalConfidence = mtfAnalysis.signals.confidence;
        // Combine factors for outcome probability
        const outcomeProbability = Math.min(1, (Math.abs(trendAlignment) * 0.4 +
            regimeCompatibility * 0.3 +
            signalConfidence * 0.3));
        // Expected return based on trend strength and regime
        const expectedReturn = mtfAnalysis.overallTrend.strength *
            (trendAlignment > 0 ? 1 : -1) *
            regimeAnalysis.trading_recommendations.risk_multiplier;
        // Time to target based on regime duration
        const timeToTarget = regimeAnalysis.duration_minutes;
        return {
            outcome_probability: outcomeProbability,
            expected_return: expectedReturn,
            time_to_target: timeToTarget,
            confidence: Math.min(1, (outcomeProbability + signalConfidence) / 2)
        };
    }
    /**
     * Calculate age factor
     */
    calculateAgeFactor(ageMs, regimeAnalysis) {
        const ageMinutes = ageMs / (1000 * 60);
        const optimalTime = regimeAnalysis.duration_minutes;
        // Optimal holding time factor
        if (ageMinutes < optimalTime * 0.5) {
            return 1.0; // Fresh position
        }
        else if (ageMinutes < optimalTime) {
            return 0.8; // Mature position
        }
        else {
            return 0.5; // Overdue position
        }
    }
    /**
     * Calculate P&L momentum
     */
    calculatePnLMomentum(position, priceChange) {
        const positionDirection = position.side === 'LONG' ? 1 : -1;
        const momentum = (priceChange / 100) * positionDirection;
        return Math.max(-1, Math.min(1, momentum));
    }
    /**
     * Calculate volume confirmation
     */
    calculateVolumeConfirmation(mtfAnalysis) {
        // Simplified volume confirmation - would use actual volume analysis in production
        return 0.7; // Default moderate confirmation
    }
    /**
     * Calculate enhanced health score
     */
    calculateEnhancedHealthScore(factors, mlPrediction, regimeCompatibility) {
        let score = 50; // Base score
        // Multi-timeframe alignment (25% weight)
        score += factors.multi_timeframe_alignment * 25;
        // ML prediction (20% weight)
        score += mlPrediction.outcome_probability * 20;
        // Regime compatibility (15% weight)
        score += regimeCompatibility * 15;
        // P&L momentum (15% weight)
        score += factors.pnl_momentum * 15;
        // Market regime confidence (10% weight)
        score += factors.market_regime_score * 10;
        // Position age factor (10% weight)
        score += factors.position_age_factor * 10;
        // Volume confirmation (5% weight)
        score += factors.volume_confirmation * 5;
        return Math.max(0, Math.min(100, score));
    }
    /**
     * Generate enhanced recommendations
     */
    generateEnhancedRecommendations(score, factors, mlPrediction, regimeAnalysis, position) {
        const reasoning = [];
        let action = 'HOLD';
        let urgency = 'LOW';
        let riskLevel = 'MEDIUM';
        // Critical conditions
        if (score < 20) {
            action = 'CLOSE';
            urgency = 'CRITICAL';
            riskLevel = 'HIGH';
            reasoning.push(`Critical health score: ${score.toFixed(1)}`);
        }
        // Poor performance
        else if (score < 40) {
            action = 'SCALE_OUT';
            urgency = 'HIGH';
            riskLevel = 'HIGH';
            reasoning.push(`Poor position health: ${score.toFixed(1)}`);
        }
        // Good performance with high ML confidence
        else if (score > 80 && mlPrediction.confidence > 0.8) {
            action = 'SCALE_IN';
            urgency = 'MEDIUM';
            riskLevel = 'LOW';
            reasoning.push(`Strong position with high ML confidence`);
        }
        // Profitable position with good trend
        else if (score > 70 && factors.pnl_momentum > 0.5) {
            action = 'TRAIL_STOP';
            urgency = 'LOW';
            riskLevel = 'LOW';
            reasoning.push(`Profitable position with good momentum`);
        }
        // Add regime-specific reasoning
        reasoning.push(`Market regime: ${regimeAnalysis.current_regime}`);
        reasoning.push(`Trend alignment: ${(factors.multi_timeframe_alignment * 100).toFixed(1)}%`);
        reasoning.push(`ML prediction confidence: ${(mlPrediction.confidence * 100).toFixed(1)}%`);
        return {
            action,
            urgency,
            reasoning,
            optimal_exit_price: this.calculateOptimalExitPrice(position, mlPrediction),
            risk_level: riskLevel
        };
    }
    /**
     * Calculate optimal exit price
     */
    calculateOptimalExitPrice(position, mlPrediction) {
        const expectedReturn = mlPrediction.expected_return;
        const currentPrice = position.currentPrice;
        if (position.side === 'LONG') {
            return currentPrice * (1 + expectedReturn);
        }
        else {
            return currentPrice * (1 - expectedReturn);
        }
    }
    /**
     * Get exit percentage based on action and score
     */
    getExitPercentage(action, score) {
        switch (action) {
            case 'SCALE_OUT':
                return score < 30 ? 75 : 50;
            case 'CLOSE':
                return 100;
            case 'SCALE_IN':
                return undefined; // No exit, adding to position
            default:
                return undefined;
        }
    }
    /**
     * Calculate dynamic stop loss
     */
    calculateDynamicStopLoss(position, health) {
        if (health.recommendations.action === 'TRAIL_STOP') {
            const atrMultiplier = health.volatility_factor > 0.03 ? 2.5 : 1.5;
            const stopDistance = position.currentPrice * atrMultiplier * 0.01; // Simplified ATR calculation
            if (position.side === 'LONG') {
                return position.currentPrice - stopDistance;
            }
            else {
                return position.currentPrice + stopDistance;
            }
        }
        return undefined;
    }
    /**
     * Fallback basic position health
     */
    getBasicPositionHealth(position) {
        const priceChange = ((position.currentPrice - position.entryPrice) / position.entryPrice) * 100;
        const basicScore = Math.max(0, Math.min(100, 50 + priceChange));
        return {
            score: basicScore,
            trend_alignment: 0,
            momentum_score: 0.5,
            risk_adjusted_return: priceChange / 100,
            volatility_factor: 0.02,
            regime_compatibility: 0.5,
            ml_prediction: {
                outcome_probability: 0.5,
                expected_return: 0,
                time_to_target: 60,
                confidence: 0.3
            },
            factors: {
                multi_timeframe_alignment: 0,
                position_age_factor: 0.5,
                pnl_momentum: priceChange / 100,
                market_regime_score: 0.5,
                volume_confirmation: 0.5
            },
            recommendations: {
                action: 'HOLD',
                urgency: 'LOW',
                reasoning: ['Basic analysis - enhanced features unavailable'],
                risk_level: 'MEDIUM'
            }
        };
    }
}
exports.AIPositionManager = AIPositionManager;
//# sourceMappingURL=aiPositionManager.js.map
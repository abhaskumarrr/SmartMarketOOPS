"use strict";
/**
 * Enhanced Trading Decision Engine
 * Core ML-driven trading logic with ensemble models, confidence scoring, and intelligent entry/exit decisions
 * Optimized for small capital + high leverage + pinpoint entry/exit precision
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.EnhancedTradingDecisionEngine = void 0;
const DataCollectorIntegration_1 = require("./DataCollectorIntegration");
const MLTradingDecisionEngine_1 = require("./MLTradingDecisionEngine");
const MultiTimeframeAnalysisEngine_1 = require("./MultiTimeframeAnalysisEngine");
const EnhancedMLIntegrationService_1 = require("./EnhancedMLIntegrationService");
const DeltaTradingBot_1 = require("./DeltaTradingBot");
const logger_1 = require("../utils/logger");
class EnhancedTradingDecisionEngine {
    constructor() {
        // Enhanced configuration for small capital + high leverage
        this.config = {
            // Higher confidence thresholds for pinpoint entries
            minConfidenceThreshold: 0.70,
            highConfidenceThreshold: 0.85,
            // Optimized position sizing for small capital
            basePositionSize: 0.03, // 3% base position
            maxPositionSize: 0.08, // 8% maximum position
            confidenceMultiplier: 1.5, // Scale with confidence
            // Enhanced leverage for maximum profit
            baseLeverage: 100, // 100x base leverage
            maxLeverage: 200, // 200x maximum leverage
            stopLossBase: 0.012, // 1.2% stop loss (tight for pinpoint entries)
            takeProfitBase: 0.040, // 4% take profit (maximize profit)
            // Optimized model weights based on backtesting
            modelWeights: {
                lstm: 0.35, // LSTM for sequential patterns
                transformer: 0.40, // Transformer for attention-based analysis
                ensemble: 0.25 // Ensemble for stability
            },
            // Feature importance weights
            featureWeights: {
                fibonacci: 0.30, // High weight for Fibonacci levels
                bias: 0.25, // Multi-timeframe bias importance
                candles: 0.20, // Candle formation analysis
                volume: 0.15, // Volume confirmation
                timing: 0.10 // Market timing
            }
        };
        // Active decisions cache
        this.activeDecisions = new Map();
        this.decisionHistory = [];
        this.dataIntegration = new DataCollectorIntegration_1.DataCollectorIntegration();
        this.mtfAnalyzer = new MultiTimeframeAnalysisEngine_1.MultiTimeframeAnalysisEngine();
        this.mlService = new EnhancedMLIntegrationService_1.EnhancedMLIntegrationService();
        this.tradingBot = new DeltaTradingBot_1.DeltaTradingBot();
        this.mlEngine = new MLTradingDecisionEngine_1.MLTradingDecisionEngine(this.mtfAnalyzer, this.mlService, this.tradingBot);
    }
    /**
     * Initialize the enhanced trading decision engine
     */
    async initialize() {
        try {
            logger_1.logger.info('🧠 Initializing Enhanced Trading Decision Engine...');
            // Initialize all components
            await this.dataIntegration.initialize();
            await this.mlEngine.initialize();
            // Connect data integration to ML engine
            this.dataIntegration.setMLEngine(this.mlEngine);
            logger_1.logger.info('✅ Enhanced Trading Decision Engine initialized successfully');
            logger_1.logger.info(`🎯 Configuration: Min Confidence ${(this.config.minConfidenceThreshold * 100).toFixed(0)}%, Max Position ${(this.config.maxPositionSize * 100).toFixed(0)}%, Max Leverage ${this.config.maxLeverage}x`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Enhanced Trading Decision Engine:', error.message);
            throw error;
        }
    }
    /**
     * Generate comprehensive trading decision for a symbol
     */
    async generateTradingDecision(symbol) {
        try {
            logger_1.logger.debug(`🧠 Generating trading decision for ${symbol}...`);
            // Step 1: Extract ML features from multi-timeframe data
            const features = await this.dataIntegration.getRealTimeTradingFeatures(symbol);
            if (!features || features.dataQuality < 0.8) {
                logger_1.logger.warn(`⚠️ Insufficient data quality for ${symbol}: ${features?.dataQuality || 0}`);
                return null;
            }
            // Step 2: Get ML model predictions
            const modelPredictions = await this.getMLModelPredictions(features);
            if (!modelPredictions) {
                logger_1.logger.warn(`⚠️ Failed to get ML predictions for ${symbol}`);
                return null;
            }
            // Step 3: Calculate ensemble confidence and action
            const ensembleDecision = this.calculateEnsembleDecision(modelPredictions);
            // Step 4: Check confidence threshold
            if (ensembleDecision.confidence < this.config.minConfidenceThreshold) {
                logger_1.logger.debug(`📊 Confidence too low for ${symbol}: ${(ensembleDecision.confidence * 100).toFixed(1)}%`);
                return null;
            }
            // Step 5: Analyze key features for decision support
            const keyFeatures = this.analyzeKeyFeatures(features);
            // Step 6: Calculate risk assessment
            const riskAssessment = this.calculateRiskAssessment(features, ensembleDecision);
            // Step 7: Determine position sizing and leverage
            const positionDetails = this.calculatePositionDetails(ensembleDecision.confidence, riskAssessment);
            // Step 8: Calculate stop loss and take profit levels
            const { stopLoss, takeProfit } = await this.calculateStopLossAndTakeProfit(symbol, ensembleDecision.action, ensembleDecision.confidence, features);
            // Step 9: Generate comprehensive trading decision
            const decision = {
                action: ensembleDecision.action,
                confidence: ensembleDecision.confidence,
                symbol,
                timestamp: Date.now(),
                // Position details
                stopLoss,
                takeProfit,
                positionSize: positionDetails.size,
                leverage: positionDetails.leverage,
                // ML insights
                modelVotes: modelPredictions,
                keyFeatures,
                // Risk assessment
                riskScore: riskAssessment.score,
                maxDrawdown: riskAssessment.maxDrawdown,
                winProbability: riskAssessment.winProbability,
                // Execution details
                urgency: this.determineUrgency(ensembleDecision.confidence, keyFeatures),
                timeToLive: this.calculateTimeToLive(ensembleDecision.action, keyFeatures),
                reasoning: this.generateReasoningExplanation(features, ensembleDecision, keyFeatures)
            };
            // Cache the decision
            this.activeDecisions.set(symbol, decision);
            this.decisionHistory.push(decision);
            logger_1.logger.info(`🎯 Trading decision generated for ${symbol}:`);
            logger_1.logger.info(`   Action: ${decision.action.toUpperCase()}`);
            logger_1.logger.info(`   Confidence: ${(decision.confidence * 100).toFixed(1)}%`);
            logger_1.logger.info(`   Position Size: ${(decision.positionSize * 100).toFixed(1)}%`);
            logger_1.logger.info(`   Leverage: ${decision.leverage}x`);
            logger_1.logger.info(`   Risk Score: ${(decision.riskScore * 100).toFixed(1)}%`);
            return decision;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to generate trading decision for ${symbol}:`, error.message);
            return null;
        }
    }
    /**
     * Get the latest trading decision for a symbol
     */
    getLatestDecision(symbol) {
        return this.activeDecisions.get(symbol) || null;
    }
    /**
     * Get decision history
     */
    getDecisionHistory(limit = 100) {
        return this.decisionHistory.slice(-limit);
    }
    /**
     * Update configuration
     */
    updateConfiguration(newConfig) {
        this.config = { ...this.config, ...newConfig };
        logger_1.logger.info('🔧 Enhanced Trading Decision Engine configuration updated');
    }
    /**
     * Get current configuration
     */
    getConfiguration() {
        return { ...this.config };
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger_1.logger.info('🧹 Cleaning up Enhanced Trading Decision Engine...');
            await this.dataIntegration.cleanup();
            this.activeDecisions.clear();
            logger_1.logger.info('✅ Enhanced Trading Decision Engine cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during Enhanced Trading Decision Engine cleanup:', error.message);
        }
    }
    // Private methods for decision logic
    /**
     * Get ML model predictions from all models
     */
    async getMLModelPredictions(features) {
        try {
            // Convert features to ML input format
            const mlInput = this.convertFeaturesToMLInput(features);
            // Get predictions from all models
            const lstmPrediction = await this.mlService.predictWithLSTM(mlInput);
            const transformerPrediction = await this.mlService.predictWithTransformer(mlInput);
            const ensemblePrediction = await this.mlService.predictWithEnsemble(mlInput);
            return {
                lstm: {
                    action: this.convertPredictionToAction(lstmPrediction),
                    confidence: lstmPrediction.confidence || 0.5
                },
                transformer: {
                    action: this.convertPredictionToAction(transformerPrediction),
                    confidence: transformerPrediction.confidence || 0.5
                },
                ensemble: {
                    action: this.convertPredictionToAction(ensemblePrediction),
                    confidence: ensemblePrediction.confidence || 0.5
                }
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to get ML model predictions:', error.message);
            return null;
        }
    }
    /**
     * Calculate ensemble decision from model votes
     */
    calculateEnsembleDecision(modelPredictions) {
        const { lstm, transformer, ensemble } = modelPredictions;
        const weights = this.config.modelWeights;
        // Calculate weighted confidence
        const weightedConfidence = (lstm.confidence * weights.lstm) +
            (transformer.confidence * weights.transformer) +
            (ensemble.confidence * weights.ensemble);
        // Determine action based on weighted voting
        const actionVotes = {
            buy: 0,
            sell: 0,
            hold: 0
        };
        // Weight the votes
        if (lstm.action === 'buy')
            actionVotes.buy += weights.lstm;
        else if (lstm.action === 'sell')
            actionVotes.sell += weights.lstm;
        else
            actionVotes.hold += weights.lstm;
        if (transformer.action === 'buy')
            actionVotes.buy += weights.transformer;
        else if (transformer.action === 'sell')
            actionVotes.sell += weights.transformer;
        else
            actionVotes.hold += weights.transformer;
        if (ensemble.action === 'buy')
            actionVotes.buy += weights.ensemble;
        else if (ensemble.action === 'sell')
            actionVotes.sell += weights.ensemble;
        else
            actionVotes.hold += weights.ensemble;
        // Determine final action
        let finalAction = 'hold';
        let maxVote = actionVotes.hold;
        if (actionVotes.buy > maxVote) {
            finalAction = 'buy';
            maxVote = actionVotes.buy;
        }
        if (actionVotes.sell > maxVote) {
            finalAction = 'sell';
        }
        // Adjust confidence based on vote consensus
        const consensusBonus = maxVote > 0.7 ? 0.1 : 0; // Bonus for strong consensus
        const finalConfidence = Math.min(1.0, weightedConfidence + consensusBonus);
        return { action: finalAction, confidence: finalConfidence };
    }
    /**
     * Analyze key features for decision support
     */
    analyzeKeyFeatures(features) {
        const weights = this.config.featureWeights;
        // Fibonacci signal strength
        const fibonacciSignal = this.calculateFibonacciSignal(features);
        // Bias alignment strength
        const biasAlignment = features.biasAlignment;
        // Candle formation strength
        const candleStrength = this.calculateCandleStrength(features);
        // Volume confirmation
        const volumeConfirmation = Math.min(1.0, features.volumeRatio / 2); // Normalize volume ratio
        // Market timing score
        const marketTiming = this.calculateMarketTiming(features);
        return {
            fibonacciSignal,
            biasAlignment,
            candleStrength,
            volumeConfirmation,
            marketTiming
        };
    }
    /**
     * Calculate risk assessment
     */
    calculateRiskAssessment(features, decision) {
        // Base risk from volatility
        const volatilityRisk = Math.min(1.0, features.volatility * 2);
        // Risk from data quality
        const dataQualityRisk = 1 - features.dataQuality;
        // Risk from confidence level
        const confidenceRisk = 1 - decision.confidence;
        // Risk from market timing
        const timingRisk = features.marketSession === 0 ? 0.2 : 0; // Higher risk during Asian session
        // Combined risk score
        const riskScore = Math.min(1.0, (volatilityRisk * 0.4) +
            (dataQualityRisk * 0.3) +
            (confidenceRisk * 0.2) +
            (timingRisk * 0.1));
        // Estimate maximum drawdown based on risk
        const maxDrawdown = riskScore * 0.05; // 0-5% based on risk
        // Estimate win probability based on confidence and risk
        const winProbability = Math.max(0.5, decision.confidence * (1 - riskScore * 0.3));
        return {
            score: riskScore,
            maxDrawdown,
            winProbability
        };
    }
    /**
     * Calculate position details based on confidence and risk
     */
    calculatePositionDetails(confidence, riskAssessment) {
        // Position size based on confidence (higher confidence = larger position)
        const confidenceMultiplier = 1 + ((confidence - this.config.minConfidenceThreshold) * this.config.confidenceMultiplier);
        let positionSize = this.config.basePositionSize * confidenceMultiplier;
        // Adjust for risk (higher risk = smaller position)
        positionSize *= (1 - riskAssessment.score * 0.5);
        // Cap at maximum
        positionSize = Math.min(positionSize, this.config.maxPositionSize);
        // Leverage based on confidence and risk
        let leverage = this.config.baseLeverage;
        // Increase leverage for high confidence
        if (confidence > this.config.highConfidenceThreshold) {
            leverage = Math.min(this.config.maxLeverage, leverage * 1.5);
        }
        // Reduce leverage for high risk
        leverage *= (1 - riskAssessment.score * 0.3);
        // Ensure minimum leverage
        leverage = Math.max(50, Math.round(leverage));
        return {
            size: Math.round(positionSize * 10000) / 10000, // Round to 4 decimals
            leverage: Math.round(leverage)
        };
    }
    /**
     * Calculate stop loss and take profit levels
     */
    async calculateStopLossAndTakeProfit(symbol, action, confidence, features) {
        // Get current price (using close price from features)
        const currentPrice = features.pricePosition; // This needs to be actual price
        // Base stop loss and take profit
        let stopLossPercent = this.config.stopLossBase;
        let takeProfitPercent = this.config.takeProfitBase;
        // Adjust based on confidence (higher confidence = tighter stops, higher targets)
        if (confidence > this.config.highConfidenceThreshold) {
            stopLossPercent *= 0.8; // Tighter stop loss
            takeProfitPercent *= 1.3; // Higher take profit
        }
        // Adjust based on volatility
        const volatilityMultiplier = 1 + (features.volatility * 0.5);
        stopLossPercent *= volatilityMultiplier;
        takeProfitPercent *= volatilityMultiplier;
        // Calculate actual levels
        let stopLoss;
        let takeProfit;
        if (action === 'buy') {
            stopLoss = currentPrice * (1 - stopLossPercent);
            takeProfit = currentPrice * (1 + takeProfitPercent);
        }
        else { // sell
            stopLoss = currentPrice * (1 + stopLossPercent);
            takeProfit = currentPrice * (1 - takeProfitPercent);
        }
        return {
            stopLoss: Math.round(stopLoss * 100) / 100,
            takeProfit: Math.round(takeProfit * 100) / 100
        };
    }
    /**
     * Convert features to ML input format
     */
    convertFeaturesToMLInput(features) {
        return [
            // Fibonacci features (7)
            ...features.fibonacciProximity,
            features.nearestFibLevel,
            features.fibStrength,
            // Multi-timeframe bias features (6)
            features.bias4h,
            features.bias1h,
            features.bias15m,
            features.bias5m,
            features.overallBias,
            features.biasAlignment,
            // Candle formation features (7)
            features.bodyPercentage,
            features.wickPercentage,
            features.buyingPressure,
            features.sellingPressure,
            features.candleType,
            features.momentum,
            features.volatility,
            // Market context features (5)
            features.volume,
            features.volumeRatio,
            features.timeOfDay,
            features.marketSession,
            features.pricePosition
        ];
    }
    /**
     * Convert ML prediction to trading action
     */
    convertPredictionToAction(prediction) {
        if (!prediction || typeof prediction.action !== 'string') {
            return 'hold';
        }
        const action = prediction.action.toLowerCase();
        if (['buy', 'long'].includes(action))
            return 'buy';
        if (['sell', 'short'].includes(action))
            return 'sell';
        return 'hold';
    }
    /**
     * Calculate Fibonacci signal strength
     */
    calculateFibonacciSignal(features) {
        // Find the strongest Fibonacci level proximity
        const maxProximity = Math.max(...features.fibonacciProximity);
        // Combine with Fibonacci strength
        const signal = (maxProximity + features.fibStrength) / 2;
        // Convert to -1 to 1 range (negative for sell, positive for buy)
        return (signal - 0.5) * 2;
    }
    /**
     * Calculate candle formation strength
     */
    calculateCandleStrength(features) {
        // Combine multiple candle factors
        const bodyStrength = features.bodyPercentage;
        const pressureBalance = Math.abs(features.buyingPressure - features.sellingPressure);
        const momentumStrength = Math.abs(features.momentum);
        return (bodyStrength + pressureBalance + momentumStrength) / 3;
    }
    /**
     * Calculate market timing score
     */
    calculateMarketTiming(features) {
        // Higher score for active trading sessions
        let timingScore = 0.5; // Base score
        // Boost for European and American sessions
        if (features.marketSession >= 1) {
            timingScore += 0.3;
        }
        // Boost for overlap periods
        if (features.marketSession === 2) {
            timingScore += 0.2;
        }
        return Math.min(1.0, timingScore);
    }
    /**
     * Determine trade urgency
     */
    determineUrgency(confidence, keyFeatures) {
        if (confidence > this.config.highConfidenceThreshold && keyFeatures.biasAlignment > 0.8) {
            return 'high';
        }
        if (confidence > (this.config.minConfidenceThreshold + 0.1)) {
            return 'medium';
        }
        return 'low';
    }
    /**
     * Calculate time to live for decision
     */
    calculateTimeToLive(action, keyFeatures) {
        // Base TTL: 5 minutes
        let ttl = 5 * 60 * 1000;
        // Shorter TTL for high urgency trades
        if (keyFeatures.biasAlignment > 0.8) {
            ttl = 2 * 60 * 1000; // 2 minutes
        }
        // Longer TTL for hold actions
        if (action === 'hold') {
            ttl = 10 * 60 * 1000; // 10 minutes
        }
        return ttl;
    }
    /**
     * Generate human-readable reasoning explanation
     */
    generateReasoningExplanation(features, decision, keyFeatures) {
        const reasoning = [];
        // Confidence explanation
        reasoning.push(`ML ensemble confidence: ${(decision.confidence * 100).toFixed(1)}%`);
        // Fibonacci analysis
        if (keyFeatures.fibonacciSignal > 0.3) {
            reasoning.push(`Strong Fibonacci support detected (${(keyFeatures.fibonacciSignal * 100).toFixed(0)}%)`);
        }
        else if (keyFeatures.fibonacciSignal < -0.3) {
            reasoning.push(`Strong Fibonacci resistance detected (${Math.abs(keyFeatures.fibonacciSignal * 100).toFixed(0)}%)`);
        }
        // Bias alignment
        if (keyFeatures.biasAlignment > 0.7) {
            reasoning.push(`Excellent multi-timeframe bias alignment (${(keyFeatures.biasAlignment * 100).toFixed(0)}%)`);
        }
        else if (keyFeatures.biasAlignment < 0.4) {
            reasoning.push(`Poor timeframe alignment - conflicting signals`);
        }
        // Candle strength
        if (keyFeatures.candleStrength > 0.7) {
            reasoning.push(`Strong candle formation pattern detected`);
        }
        // Volume confirmation
        if (keyFeatures.volumeConfirmation > 0.8) {
            reasoning.push(`High volume confirmation (${(features.volumeRatio).toFixed(1)}x average)`);
        }
        else if (keyFeatures.volumeConfirmation < 0.3) {
            reasoning.push(`Low volume - weak confirmation`);
        }
        // Market timing
        if (keyFeatures.marketTiming > 0.8) {
            reasoning.push(`Optimal market timing - active trading session`);
        }
        else if (keyFeatures.marketTiming < 0.4) {
            reasoning.push(`Suboptimal timing - low activity session`);
        }
        // Data quality
        if (features.dataQuality < 0.9) {
            reasoning.push(`Data quality: ${(features.dataQuality * 100).toFixed(0)}% - proceed with caution`);
        }
        return reasoning;
    }
}
exports.EnhancedTradingDecisionEngine = EnhancedTradingDecisionEngine;
//# sourceMappingURL=EnhancedTradingDecisionEngine.js.map
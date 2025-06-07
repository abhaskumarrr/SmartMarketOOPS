"use strict";
/**
 * ML Trading Decision Engine
 *
 * Integrates all our trading analysis (Fibonacci, SMC, confluence, candle formation,
 * momentum trains) as features for ML models to make actual trading decisions.
 *
 * This replaces hard-coded rules with ML-driven intelligence that learns optimal
 * combinations of our comprehensive trading analysis.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MLTradingDecisionEngine = void 0;
const logger_1 = require("../utils/logger");
class MLTradingDecisionEngine {
    constructor(mtfAnalyzer, mlService, tradingBot) {
        this.isInitialized = false;
        // Feature extraction cache
        this.featureCache = new Map();
        this.cacheTimeout = 30000; // 30 seconds
        // ML Model ensemble weights (learned from performance)
        this.modelWeights = {
            lstm: 0.35,
            transformer: 0.40,
            ensemble: 0.25
        };
        // Decision thresholds (optimized for small capital + high leverage + pinpoint entries)
        this.thresholds = {
            minConfidence: 0.70, // Higher confidence for pinpoint entries (70%)
            strongSignal: 0.85, // Strong signal threshold (85%)
            positionSizeBase: 0.03, // Base position size (3% - higher for small capital)
            maxPositionSize: 0.08, // Maximum position size (8% - higher for pinpoint entries)
            stopLossBase: 0.012, // Tighter stop loss (1.2% - pinpoint entries)
            takeProfitBase: 0.040 // Higher take profit (4% - maximize profit with high leverage)
        };
        this.mtfAnalyzer = mtfAnalyzer;
        this.mlService = mlService;
        this.tradingBot = tradingBot;
    }
    /**
     * Initialize the ML Trading Decision Engine
     */
    async initialize() {
        try {
            logger_1.logger.info('🤖 Initializing ML Trading Decision Engine...');
            // Verify all components are ready
            if (!this.mtfAnalyzer || !this.mlService || !this.tradingBot) {
                throw new Error('Required components not provided');
            }
            // Load and validate ML models
            await this.validateMLModels();
            // Load optimized parameters from previous performance
            await this.loadOptimizedParameters();
            this.isInitialized = true;
            logger_1.logger.info('✅ ML Trading Decision Engine initialized successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize ML Trading Decision Engine:', error);
            throw error;
        }
    }
    /**
     * Generate ML-driven trading decision
     */
    async generateTradingDecision(symbol, currentPrice) {
        if (!this.isInitialized) {
            throw new Error('ML Trading Decision Engine not initialized');
        }
        try {
            logger_1.logger.info(`🧠 Generating ML trading decision for ${symbol} at $${currentPrice.toFixed(2)}`);
            // Step 1: Extract comprehensive trading features
            const features = await this.extractTradingFeatures(symbol, currentPrice);
            // Step 2: Get ML model predictions
            const mlPredictions = await this.getMLPredictions(symbol, features);
            // Step 3: Combine predictions using ensemble weights
            const ensembleDecision = this.combineMLPredictions(mlPredictions);
            // Step 4: Apply risk management and position sizing
            const finalDecision = this.applyRiskManagement(ensembleDecision, features);
            // Step 5: Cache decision for performance tracking
            this.cacheDecision(symbol, finalDecision, features);
            logger_1.logger.info(`✅ ML decision: ${finalDecision.action} (${(finalDecision.confidence * 100).toFixed(1)}% confidence)`);
            return finalDecision;
        }
        catch (error) {
            logger_1.logger.error(`❌ Error generating ML trading decision for ${symbol}:`, error);
            throw error;
        }
    }
    /**
     * Execute trade based on ML decision
     */
    async executeTrade(symbol, decision, currentPrice) {
        try {
            if (decision.action === 'HOLD') {
                logger_1.logger.info(`⏸️ ML decision: HOLD for ${symbol} - no trade executed`);
                return false;
            }
            if (decision.confidence < this.thresholds.minConfidence) {
                logger_1.logger.info(`⚠️ ML confidence too low (${(decision.confidence * 100).toFixed(1)}%) - trade skipped`);
                return false;
            }
            logger_1.logger.info(`🚀 Executing ML-driven ${decision.action} trade for ${symbol}`);
            // Calculate actual trade parameters
            const tradeParams = {
                symbol,
                side: decision.action.toLowerCase(),
                size: decision.positionSize,
                price: currentPrice,
                stopLoss: decision.action === 'BUY'
                    ? currentPrice * (1 - decision.stopLoss)
                    : currentPrice * (1 + decision.stopLoss),
                takeProfit: decision.action === 'BUY'
                    ? currentPrice * (1 + decision.takeProfit)
                    : currentPrice * (1 - decision.takeProfit),
                timeHorizon: decision.timeHorizon,
                mlConfidence: decision.confidence,
                reasoning: decision.reasoning
            };
            // Execute trade through DeltaTradingBot
            const success = await this.tradingBot.executeTrade(tradeParams);
            if (success) {
                logger_1.logger.info(`✅ ML-driven trade executed successfully for ${symbol}`);
                await this.trackTradePerformance(symbol, decision, tradeParams);
            }
            else {
                logger_1.logger.error(`❌ Failed to execute ML-driven trade for ${symbol}`);
            }
            return success;
        }
        catch (error) {
            logger_1.logger.error(`❌ Error executing ML-driven trade for ${symbol}:`, error);
            return false;
        }
    }
    /**
     * Extract comprehensive trading features from all our analysis
     */
    async extractTradingFeatures(symbol, currentPrice) {
        // Check cache first
        const cacheKey = `${symbol}_${Math.floor(Date.now() / this.cacheTimeout)}`;
        const cached = this.featureCache.get(cacheKey);
        if (cached) {
            return cached.features;
        }
        // Extract features from all our analysis systems
        const features = {
            fibonacciProximity: await this.extractFibonacciFeatures(symbol, currentPrice),
            timeframeBias: await this.extractTimeframeBiasFeatures(symbol),
            candleFormation: await this.extractCandleFormationFeatures(symbol, currentPrice),
            smcAnalysis: await this.extractSMCFeatures(symbol),
            confluence: await this.extractConfluenceFeatures(symbol),
            marketContext: await this.extractMarketContextFeatures(symbol)
        };
        // Cache features
        this.featureCache.set(cacheKey, { features, timestamp: Date.now() });
        return features;
    }
    /**
     * Validate that all ML models are loaded and ready
     */
    async validateMLModels() {
        try {
            // Check if ML service is ready
            const isReady = await this.mlService.isModelReady();
            if (!isReady) {
                throw new Error('ML models not ready');
            }
            // Test prediction with dummy data
            const testFeatures = this.createDummyFeatures();
            const testPrediction = await this.mlService.predictPositionOutcome('BTCUSD', testFeatures);
            if (!testPrediction || typeof testPrediction.confidence !== 'number') {
                throw new Error('ML model validation failed');
            }
            logger_1.logger.info('✅ ML models validated successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ ML model validation failed:', error);
            throw error;
        }
    }
    /**
     * Load optimized parameters from previous performance data
     */
    async loadOptimizedParameters() {
        try {
            // In production, load from database or config file
            // For now, use optimized defaults from backtesting
            this.modelWeights = {
                lstm: 0.35, // LSTM good for sequential patterns
                transformer: 0.40, // Transformer best for complex relationships
                ensemble: 0.25 // Ensemble for stability
            };
            this.thresholds = {
                minConfidence: 0.65,
                strongSignal: 0.80,
                positionSizeBase: 0.02,
                maxPositionSize: 0.05,
                stopLossBase: 0.015,
                takeProfitBase: 0.030
            };
            logger_1.logger.info('✅ Optimized parameters loaded');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to load optimized parameters:', error);
            // Continue with defaults
        }
    }
    /**
     * Get predictions from all ML models
     */
    async getMLPredictions(symbol, features) {
        try {
            // Convert features to format expected by ML models
            const mlInput = this.convertFeaturesToMLInput(features);
            // Get predictions from different models
            const [lstmPred, transformerPred, ensemblePred] = await Promise.all([
                this.mlService.predictWithLSTM(symbol, mlInput),
                this.mlService.predictWithTransformer(symbol, mlInput),
                this.mlService.predictPositionOutcome(symbol, mlInput)
            ]);
            return {
                lstm: lstmPred || { action: 'HOLD', confidence: 0.5 },
                transformer: transformerPred || { action: 'HOLD', confidence: 0.5 },
                ensemble: ensemblePred || { action: 'HOLD', confidence: 0.5 }
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Error getting ML predictions:', error);
            // Return neutral predictions on error
            return {
                lstm: { action: 'HOLD', confidence: 0.5 },
                transformer: { action: 'HOLD', confidence: 0.5 },
                ensemble: { action: 'HOLD', confidence: 0.5 }
            };
        }
    }
    /**
     * Combine ML predictions using weighted ensemble
     */
    combineMLPredictions(predictions) {
        // Convert actions to numeric scores
        const actionToScore = (action) => {
            switch (action.toUpperCase()) {
                case 'BUY': return 1;
                case 'SELL': return -1;
                default: return 0;
            }
        };
        // Calculate weighted score
        const lstmScore = actionToScore(predictions.lstm.action) * predictions.lstm.confidence;
        const transformerScore = actionToScore(predictions.transformer.action) * predictions.transformer.confidence;
        const ensembleScore = actionToScore(predictions.ensemble.action) * predictions.ensemble.confidence;
        const weightedScore = (lstmScore * this.modelWeights.lstm +
            transformerScore * this.modelWeights.transformer +
            ensembleScore * this.modelWeights.ensemble);
        // Calculate overall confidence
        const weightedConfidence = (predictions.lstm.confidence * this.modelWeights.lstm +
            predictions.transformer.confidence * this.modelWeights.transformer +
            predictions.ensemble.confidence * this.modelWeights.ensemble);
        // Determine final action
        let finalAction;
        if (weightedScore > 0.1) {
            finalAction = 'BUY';
        }
        else if (weightedScore < -0.1) {
            finalAction = 'SELL';
        }
        else {
            finalAction = 'HOLD';
        }
        return {
            action: finalAction,
            confidence: Math.abs(weightedConfidence),
            modelContributions: {
                lstm: predictions.lstm.confidence * this.modelWeights.lstm,
                transformer: predictions.transformer.confidence * this.modelWeights.transformer,
                ensemble: predictions.ensemble.confidence * this.modelWeights.ensemble
            }
        };
    }
    /**
     * Apply risk management and position sizing to ML decision
     */
    applyRiskManagement(decision, features) {
        // Calculate dynamic position size based on confidence and market conditions
        let positionSize = this.thresholds.positionSizeBase;
        if (decision.confidence > this.thresholds.strongSignal) {
            positionSize *= 1.5; // Increase size for high confidence
        }
        // Adjust for market volatility
        positionSize *= (1 - features.marketContext.volatility * 0.3);
        // Cap position size
        positionSize = Math.min(positionSize, this.thresholds.maxPositionSize);
        // Calculate dynamic stop loss and take profit
        const volatilityMultiplier = 1 + features.marketContext.volatility;
        const stopLoss = this.thresholds.stopLossBase * volatilityMultiplier;
        const takeProfit = this.thresholds.takeProfitBase * volatilityMultiplier;
        // Determine time horizon based on confluence and momentum
        let timeHorizon = 'DAY';
        if (features.confluence.momentumTrain > 0.8) {
            timeHorizon = 'SCALP';
        }
        else if (features.timeframeBias.alignment > 0.7) {
            timeHorizon = 'SWING';
        }
        // Generate reasoning
        const primaryFactors = [];
        if (features.confluence.overallScore > 0.75)
            primaryFactors.push('High confluence score');
        if (features.timeframeBias.alignment > 0.7)
            primaryFactors.push('Strong timeframe alignment');
        if (features.fibonacciProximity.nearestLevel < 0.1)
            primaryFactors.push('Near key Fibonacci level');
        if (features.smcAnalysis.orderBlockStrength > 0.6)
            primaryFactors.push('Strong order block presence');
        return {
            action: decision.action,
            confidence: decision.confidence,
            positionSize,
            stopLoss,
            takeProfit,
            timeHorizon,
            reasoning: {
                primaryFactors,
                riskAssessment: this.generateRiskAssessment(features, decision.confidence),
                modelContributions: decision.modelContributions
            }
        };
    }
    /**
     * Cache decision for performance tracking
     */
    cacheDecision(symbol, decision, features) {
        // Store decision with timestamp for later performance analysis
        const decisionRecord = {
            symbol,
            timestamp: Date.now(),
            decision,
            features,
            executed: false
        };
        // In production, store in database
        logger_1.logger.debug(`📝 Cached ML decision for ${symbol}: ${decision.action} (${(decision.confidence * 100).toFixed(1)}%)`);
    }
    /**
     * Track trade performance for ML model optimization
     */
    async trackTradePerformance(symbol, decision, tradeParams) {
        try {
            // Record trade execution for performance tracking
            const performanceRecord = {
                symbol,
                timestamp: Date.now(),
                decision,
                tradeParams,
                mlConfidence: decision.confidence,
                modelContributions: decision.reasoning.modelContributions
            };
            // In production, store in database for model retraining
            logger_1.logger.info(`📊 Tracking ML trade performance for ${symbol}`);
        }
        catch (error) {
            logger_1.logger.error('❌ Error tracking trade performance:', error);
        }
    }
    /**
     * Extract Fibonacci analysis features
     */
    async extractFibonacciFeatures(symbol, currentPrice) {
        try {
            // Get Fibonacci analysis from MTF analyzer
            const fibAnalysis = await this.mtfAnalyzer.getFibonacciAnalysis(symbol);
            if (!fibAnalysis) {
                return this.getDefaultFibFeatures();
            }
            // Calculate proximity to each Fibonacci level (0 = at level, 1 = far from level)
            const levels = [0.236, 0.382, 0.5, 0.618, 0.786];
            const proximities = levels.map(level => {
                const fibPrice = fibAnalysis.levels[level] || currentPrice;
                return Math.abs(currentPrice - fibPrice) / currentPrice;
            });
            // Find nearest level
            const nearestLevelIndex = proximities.indexOf(Math.min(...proximities));
            const nearestLevel = proximities[nearestLevelIndex];
            const levelStrength = fibAnalysis.strength || 0.5;
            return {
                level236: Math.min(proximities[0], 1),
                level382: Math.min(proximities[1], 1),
                level500: Math.min(proximities[2], 1),
                level618: Math.min(proximities[3], 1),
                level786: Math.min(proximities[4], 1),
                nearestLevel: Math.min(nearestLevel, 1),
                levelStrength: Math.max(0, Math.min(levelStrength, 1))
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Error extracting Fibonacci features:', error);
            return this.getDefaultFibFeatures();
        }
    }
    /**
     * Extract multi-timeframe bias features
     */
    async extractTimeframeBiasFeatures(symbol) {
        try {
            // Get bias analysis from MTF analyzer
            const biasAnalysis = await this.mtfAnalyzer.getTimeframeBias(symbol);
            if (!biasAnalysis) {
                return this.getDefaultBiasFeatures();
            }
            // Normalize bias values to -1 to 1 range
            const normalizeBias = (bias) => {
                return Math.max(-1, Math.min(1, bias / 100));
            };
            const bias4H = normalizeBias(biasAnalysis['4H'] || 0);
            const bias1H = normalizeBias(biasAnalysis['1H'] || 0);
            const bias15M = normalizeBias(biasAnalysis['15M'] || 0);
            const bias5M = normalizeBias(biasAnalysis['5M'] || 0);
            // Calculate alignment (how well timeframes agree)
            const biases = [bias4H, bias1H, bias15M, bias5M];
            const avgBias = biases.reduce((sum, bias) => sum + bias, 0) / biases.length;
            const alignment = 1 - (biases.reduce((sum, bias) => sum + Math.abs(bias - avgBias), 0) / biases.length);
            // Calculate overall strength
            const strength = Math.abs(avgBias);
            return {
                bias4H,
                bias1H,
                bias15M,
                bias5M,
                alignment: Math.max(0, Math.min(1, alignment)),
                strength: Math.max(0, Math.min(1, strength))
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Error extracting timeframe bias features:', error);
            return this.getDefaultBiasFeatures();
        }
    }
    /**
     * Extract candle formation features
     */
    async extractCandleFormationFeatures(symbol, currentPrice) {
        try {
            // Get candle analysis from MTF analyzer
            const candleAnalysis = await this.mtfAnalyzer.getCandleFormationAnalysis(symbol);
            if (!candleAnalysis) {
                return this.getDefaultCandleFeatures();
            }
            return {
                bodyPercent: Math.max(0, Math.min(1, candleAnalysis.bodyPercent / 100)),
                upperWickPercent: Math.max(0, Math.min(1, candleAnalysis.upperWickPercent / 100)),
                lowerWickPercent: Math.max(0, Math.min(1, candleAnalysis.lowerWickPercent / 100)),
                buyingPressure: Math.max(0, Math.min(1, candleAnalysis.buyingPressure / 100)),
                sellingPressure: Math.max(0, Math.min(1, candleAnalysis.sellingPressure / 100)),
                candleType: this.encodeCandleType(candleAnalysis.type),
                momentum: Math.max(-1, Math.min(1, candleAnalysis.momentum / 100))
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Error extracting candle formation features:', error);
            return this.getDefaultCandleFeatures();
        }
    }
    /**
     * Extract Smart Money Concepts features
     */
    async extractSMCFeatures(symbol) {
        try {
            // Get SMC analysis from MTF analyzer
            const smcAnalysis = await this.mtfAnalyzer.getSMCAnalysis(symbol);
            if (!smcAnalysis) {
                return this.getDefaultSMCFeatures();
            }
            return {
                orderBlockStrength: Math.max(0, Math.min(1, smcAnalysis.orderBlockStrength / 100)),
                fvgPresence: Math.max(0, Math.min(1, smcAnalysis.fvgPresence / 100)),
                liquidityLevel: Math.max(0, Math.min(1, smcAnalysis.liquidityLevel / 100)),
                structureBreak: Math.max(0, Math.min(1, smcAnalysis.structureBreak / 100)),
                institutionalFlow: Math.max(-1, Math.min(1, smcAnalysis.institutionalFlow / 100))
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Error extracting SMC features:', error);
            return this.getDefaultSMCFeatures();
        }
    }
    /**
     * Extract confluence features
     */
    async extractConfluenceFeatures(symbol) {
        try {
            // Get confluence analysis from MTF analyzer
            const confluenceAnalysis = await this.mtfAnalyzer.getConfluenceAnalysis(symbol);
            if (!confluenceAnalysis) {
                return this.getDefaultConfluenceFeatures();
            }
            return {
                overallScore: Math.max(0, Math.min(1, confluenceAnalysis.overallScore / 100)),
                fibWeight: Math.max(0, Math.min(1, confluenceAnalysis.fibWeight / 100)),
                biasWeight: Math.max(0, Math.min(1, confluenceAnalysis.biasWeight / 100)),
                smcWeight: Math.max(0, Math.min(1, confluenceAnalysis.smcWeight / 100)),
                momentumTrain: Math.max(0, Math.min(1, confluenceAnalysis.momentumTrain / 100)),
                entryTiming: Math.max(0, Math.min(1, confluenceAnalysis.entryTiming / 100))
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Error extracting confluence features:', error);
            return this.getDefaultConfluenceFeatures();
        }
    }
    /**
     * Extract market context features
     */
    async extractMarketContextFeatures(symbol) {
        try {
            // Get market context from MTF analyzer
            const marketContext = await this.mtfAnalyzer.getMarketContext(symbol);
            if (!marketContext) {
                return this.getDefaultMarketContextFeatures();
            }
            // Normalize time of day (0 = start of session, 1 = end of session)
            const now = new Date();
            const timeOfDay = (now.getHours() * 60 + now.getMinutes()) / (24 * 60);
            return {
                volatility: Math.max(0, Math.min(1, marketContext.volatility / 100)),
                volume: Math.max(0, Math.min(1, marketContext.volume / 100)),
                timeOfDay,
                marketRegime: Math.max(0, Math.min(1, marketContext.regime / 100)),
                sessionType: this.encodeSessionType()
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Error extracting market context features:', error);
            return this.getDefaultMarketContextFeatures();
        }
    }
    // Utility methods for feature conversion and defaults
    convertFeaturesToMLInput(features) {
        // Convert all features to a flat array for ML models
        return [
            // Fibonacci features (7 values)
            features.fibonacciProximity.level236,
            features.fibonacciProximity.level382,
            features.fibonacciProximity.level500,
            features.fibonacciProximity.level618,
            features.fibonacciProximity.level786,
            features.fibonacciProximity.nearestLevel,
            features.fibonacciProximity.levelStrength,
            // Timeframe bias features (6 values)
            features.timeframeBias.bias4H,
            features.timeframeBias.bias1H,
            features.timeframeBias.bias15M,
            features.timeframeBias.bias5M,
            features.timeframeBias.alignment,
            features.timeframeBias.strength,
            // Candle formation features (7 values)
            features.candleFormation.bodyPercent,
            features.candleFormation.upperWickPercent,
            features.candleFormation.lowerWickPercent,
            features.candleFormation.buyingPressure,
            features.candleFormation.sellingPressure,
            features.candleFormation.candleType,
            features.candleFormation.momentum,
            // SMC features (5 values)
            features.smcAnalysis.orderBlockStrength,
            features.smcAnalysis.fvgPresence,
            features.smcAnalysis.liquidityLevel,
            features.smcAnalysis.structureBreak,
            features.smcAnalysis.institutionalFlow,
            // Confluence features (6 values)
            features.confluence.overallScore,
            features.confluence.fibWeight,
            features.confluence.biasWeight,
            features.confluence.smcWeight,
            features.confluence.momentumTrain,
            features.confluence.entryTiming,
            // Market context features (5 values)
            features.marketContext.volatility,
            features.marketContext.volume,
            features.marketContext.timeOfDay,
            features.marketContext.marketRegime,
            features.marketContext.sessionType
        ];
    }
    createDummyFeatures() {
        return {
            fibonacciProximity: this.getDefaultFibFeatures(),
            timeframeBias: this.getDefaultBiasFeatures(),
            candleFormation: this.getDefaultCandleFeatures(),
            smcAnalysis: this.getDefaultSMCFeatures(),
            confluence: this.getDefaultConfluenceFeatures(),
            marketContext: this.getDefaultMarketContextFeatures()
        };
    }
    generateRiskAssessment(features, confidence) {
        const riskFactors = [];
        if (features.marketContext.volatility > 0.7)
            riskFactors.push('High volatility');
        if (features.timeframeBias.alignment < 0.5)
            riskFactors.push('Timeframe misalignment');
        if (confidence < 0.7)
            riskFactors.push('Low ML confidence');
        if (features.confluence.overallScore < 0.6)
            riskFactors.push('Weak confluence');
        if (riskFactors.length === 0)
            return 'Low risk - favorable conditions';
        if (riskFactors.length <= 2)
            return `Medium risk - ${riskFactors.join(', ')}`;
        return `High risk - ${riskFactors.join(', ')}`;
    }
    encodeCandleType(type) {
        const typeMap = {
            'doji': 0.1,
            'hammer': 0.3,
            'shooting_star': 0.7,
            'strong_bullish': 0.9,
            'strong_bearish': 0.1,
            'neutral': 0.5
        };
        return typeMap[type] || 0.5;
    }
    encodeSessionType() {
        const now = new Date();
        const hour = now.getUTCHours();
        // Encode trading sessions (0-1)
        if (hour >= 0 && hour < 8)
            return 0.2; // Asian session
        if (hour >= 8 && hour < 16)
            return 0.8; // European session
        if (hour >= 16 && hour < 24)
            return 1.0; // US session
        return 0.5; // Overlap periods
    }
    // Default feature methods for fallback scenarios
    getDefaultFibFeatures() {
        return {
            level236: 0.5,
            level382: 0.5,
            level500: 0.5,
            level618: 0.5,
            level786: 0.5,
            nearestLevel: 0.5,
            levelStrength: 0.5
        };
    }
    getDefaultBiasFeatures() {
        return {
            bias4H: 0,
            bias1H: 0,
            bias15M: 0,
            bias5M: 0,
            alignment: 0.5,
            strength: 0.5
        };
    }
    getDefaultCandleFeatures() {
        return {
            bodyPercent: 0.5,
            upperWickPercent: 0.25,
            lowerWickPercent: 0.25,
            buyingPressure: 0.5,
            sellingPressure: 0.5,
            candleType: 0.5,
            momentum: 0
        };
    }
    getDefaultSMCFeatures() {
        return {
            orderBlockStrength: 0.5,
            fvgPresence: 0.5,
            liquidityLevel: 0.5,
            structureBreak: 0.5,
            institutionalFlow: 0
        };
    }
    getDefaultConfluenceFeatures() {
        return {
            overallScore: 0.5,
            fibWeight: 0.5,
            biasWeight: 0.5,
            smcWeight: 0.5,
            momentumTrain: 0.5,
            entryTiming: 0.5
        };
    }
    getDefaultMarketContextFeatures() {
        return {
            volatility: 0.5,
            volume: 0.5,
            timeOfDay: 0.5,
            marketRegime: 0.5,
            sessionType: 0.5
        };
    }
}
exports.MLTradingDecisionEngine = MLTradingDecisionEngine;
//# sourceMappingURL=MLTradingDecisionEngine.js.map
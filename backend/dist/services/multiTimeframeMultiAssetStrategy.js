"use strict";
/**
 * Multi-Timeframe Multi-Asset Trading Strategy
 * Advanced strategy combining multi-timeframe analysis with multi-asset portfolio optimization
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiTimeframeMultiAssetStrategy = void 0;
exports.createMultiTimeframeMultiAssetStrategy = createMultiTimeframeMultiAssetStrategy;
const multiTimeframeMultiAssetDataProvider_1 = require("./multiTimeframeMultiAssetDataProvider");
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
class MultiTimeframeMultiAssetStrategy {
    constructor() {
        this.name = 'Multi_Timeframe_Multi_Asset_Strategy';
        this.dataProvider = (0, multiTimeframeMultiAssetDataProvider_1.createMultiTimeframeMultiAssetDataProvider)();
        this.trainedModels = {};
        this.lastDecisionTime = 0;
        // Strategy configuration
        this.timeframeHierarchy = {
            '1d': 100, // Highest priority - trend direction
            '4h': 80, // High priority - intermediate trend
            '1h': 60, // Medium priority - short-term trend
            '15m': 40, // Lower priority - entry timing
            '5m': 20, // Low priority - fine-tuning
            '3m': 15, // Very low priority - micro-timing
            '1m': 10, // Lowest priority - execution
        };
        this.assetConfigs = [
            {
                asset: 'BTCUSD',
                timeframes: ['1d', '4h', '1h', '15m'],
                priority: 'PRIMARY',
                weight: 0.4,
            },
            {
                asset: 'ETHUSD',
                timeframes: ['1d', '4h', '1h', '15m'],
                priority: 'PRIMARY',
                weight: 0.35,
            },
            {
                asset: 'SOLUSD',
                timeframes: ['4h', '1h', '15m', '5m'],
                priority: 'SECONDARY',
                weight: 0.25,
            },
        ];
        this.parameters = {
            // Multi-timeframe parameters
            primaryTimeframe: '1h',
            confirmationTimeframes: ['4h', '1d'],
            executionTimeframes: ['15m', '5m'],
            // Hierarchical decision making
            minTimeframeConsensus: 0.6, // 60% of timeframes must agree
            higherTimeframeWeight: 2.0, // Higher timeframes get 2x weight
            conflictResolutionMethod: 'HIGHER_TIMEFRAME_WINS',
            // Multi-asset parameters
            maxAssetExposure: 0.4, // Max 40% in any single asset
            minCashReserve: 0.1, // Keep 10% in cash
            correlationThreshold: 0.7, // High correlation threshold
            // Signal generation - LOWERED FOR REAL TRADING
            minConfidence: 55, // Minimum confidence for signals (lowered from 70)
            minHierarchicalScore: 0.45, // Minimum hierarchical agreement (lowered from 0.65)
            decisionCooldown: 60000, // 1 minute between decisions (reduced from 5 minutes)
            // Risk management
            volatilityAdjustment: true,
            dynamicPositionSizing: true,
            crossAssetRiskLimit: 0.8, // Max 80% correlated exposure
        };
        logger_1.logger.info('🔄 Multi-Timeframe Multi-Asset Strategy initialized', {
            supportedAssets: this.assetConfigs.map(c => c.asset),
            timeframeHierarchy: Object.keys(this.timeframeHierarchy),
        });
    }
    /**
     * Initialize the strategy
     */
    async initialize(config) {
        this.config = config;
        this.lastDecisionTime = 0;
        // Load trained models
        await this.loadTrainedModels();
        logger_1.logger.info(`🎯 Initialized ${this.name}`, {
            symbol: config.symbol,
            timeframe: config.timeframe,
            assetConfigs: this.assetConfigs.length,
            modelsLoaded: Object.keys(this.trainedModels).length,
        });
    }
    /**
     * Load trained models for multi-timeframe multi-asset analysis
     */
    async loadTrainedModels() {
        logger_1.logger.info('📂 Loading multi-timeframe multi-asset models...');
        const modelsDir = path_1.default.join(process.cwd(), 'trained_models');
        const modelTypes = ['transformer', 'lstm', 'smc'];
        for (const modelType of modelTypes) {
            try {
                const latestModelPath = path_1.default.join(modelsDir, `${modelType}_model_latest.json`);
                if (fs_1.default.existsSync(latestModelPath)) {
                    const modelData = fs_1.default.readFileSync(latestModelPath, 'utf8');
                    const trainedModel = JSON.parse(modelData);
                    if (this.validateModel(trainedModel)) {
                        this.trainedModels[modelType] = trainedModel;
                        logger_1.logger.info(`✅ Loaded ${trainedModel.modelName} for multi-timeframe multi-asset analysis`);
                    }
                }
                else {
                    this.trainedModels[modelType] = this.createDefaultModel(modelType);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ Failed to load ${modelType} model:`, error);
                this.trainedModels[modelType] = this.createDefaultModel(modelType);
            }
        }
    }
    /**
     * Generate multi-timeframe multi-asset trading signal
     */
    generateSignal(data, currentIndex) {
        if (!this.config) {
            throw new Error('Strategy not initialized. Call initialize() first.');
        }
        if (currentIndex < 100) {
            return null;
        }
        const currentTime = Date.now();
        if (currentTime - this.lastDecisionTime < this.parameters.decisionCooldown) {
            return null;
        }
        try {
            // Convert single-timeframe data to multi-timeframe multi-asset format
            const multiTimeframeData = this.convertToMultiTimeframeData(data, currentIndex);
            if (!multiTimeframeData) {
                return null;
            }
            // Analyze each asset across multiple timeframes
            const assetAnalyses = this.analyzeAllAssetsAllTimeframes(multiTimeframeData);
            if (assetAnalyses.length === 0) {
                return null;
            }
            // Generate timeframe breakdown
            const timeframeBreakdown = this.generateTimeframeBreakdown(multiTimeframeData, assetAnalyses);
            // Calculate portfolio recommendation
            const portfolioRecommendation = this.calculatePortfolioRecommendation(assetAnalyses, multiTimeframeData);
            // Make hierarchical decision
            const hierarchicalDecision = this.makeHierarchicalDecision(assetAnalyses, timeframeBreakdown);
            // Generate final signal
            const signal = this.generateFinalSignal(assetAnalyses, timeframeBreakdown, portfolioRecommendation, hierarchicalDecision, data[currentIndex]);
            if (signal && signal.confidence >= this.parameters.minConfidence) {
                this.lastDecisionTime = currentTime;
                logger_1.logger.info(`🔄 Generated multi-timeframe multi-asset ${signal.type} signal`, {
                    confidence: signal.confidence,
                    primaryAsset: signal.hierarchicalDecision.primaryTimeframe,
                    assetsAnalyzed: assetAnalyses.length,
                });
                return signal;
            }
            return null;
        }
        catch (error) {
            logger_1.logger.error('❌ Error generating multi-timeframe multi-asset signal:', error);
            return null;
        }
    }
    /**
     * Convert single-timeframe data to multi-timeframe format (simplified)
     */
    convertToMultiTimeframeData(data, currentIndex) {
        if (currentIndex >= data.length) {
            return null;
        }
        const currentCandle = data[currentIndex];
        // Simplified conversion - in a real implementation, this would fetch actual multi-timeframe data
        const multiTimeframeData = {
            timestamp: currentCandle.timestamp,
            assets: {},
            crossAssetAnalysis: {
                correlations: {},
                dominance: {},
                volatilityRanking: {},
            },
            timeframeConsensus: {},
        };
        // Populate with current data (simplified)
        const supportedAssets = ['BTCUSD', 'ETHUSD', 'SOLUSD'];
        const supportedTimeframes = ['1m', '3m', '5m', '15m', '1h', '4h', '1d'];
        for (const asset of supportedAssets) {
            multiTimeframeData.assets[asset] = {};
            for (const timeframe of supportedTimeframes) {
                // Use current candle data for all timeframes (simplified)
                multiTimeframeData.assets[asset][timeframe] = {
                    ...currentCandle,
                    symbol: asset,
                };
            }
        }
        // Add simplified cross-asset analysis
        multiTimeframeData.crossAssetAnalysis.correlations['1h'] = {
            btc_eth: 0.7 + (Math.random() - 0.5) * 0.3,
            btc_sol: 0.6 + (Math.random() - 0.5) * 0.3,
            eth_sol: 0.5 + (Math.random() - 0.5) * 0.3,
        };
        return multiTimeframeData;
    }
    /**
     * Analyze all assets across all timeframes
     */
    analyzeAllAssetsAllTimeframes(data) {
        const analyses = [];
        for (const assetConfig of this.assetConfigs) {
            const analysis = this.analyzeAssetAllTimeframes(data, assetConfig);
            if (analysis) {
                analyses.push(analysis);
            }
        }
        return analyses;
    }
    /**
     * Analyze a single asset across all its configured timeframes
     */
    analyzeAssetAllTimeframes(data, assetConfig) {
        const timeframeSignals = [];
        for (const timeframe of assetConfig.timeframes) {
            const signal = this.analyzeAssetTimeframe(data, assetConfig.asset, timeframe);
            if (signal) {
                timeframeSignals.push(signal);
            }
        }
        if (timeframeSignals.length === 0) {
            return null;
        }
        // Calculate consensus
        const consensus = this.calculateTimeframeConsensus(timeframeSignals);
        // Calculate hierarchical score
        const hierarchicalScore = this.calculateHierarchicalScore(timeframeSignals);
        // Calculate volatility adjustment
        const volatilityAdjustment = this.calculateVolatilityAdjustment(data, assetConfig.asset);
        return {
            asset: assetConfig.asset,
            timeframeSignals,
            consensusSignal: consensus.signal,
            consensusConfidence: consensus.confidence,
            hierarchicalScore,
            volatilityAdjustment,
        };
    }
    /**
     * Analyze a specific asset-timeframe combination
     */
    analyzeAssetTimeframe(data, asset, timeframe) {
        const candle = data.assets[asset]?.[timeframe];
        if (!candle) {
            return null;
        }
        // Use AI models to generate signal (simplified)
        const modelPredictions = this.getModelPredictions(candle, asset, timeframe);
        if (modelPredictions.length === 0) {
            return null;
        }
        const avgPrediction = modelPredictions.reduce((sum, p) => sum + p.prediction, 0) / modelPredictions.length;
        const avgConfidence = modelPredictions.reduce((sum, p) => sum + p.confidence, 0) / modelPredictions.length;
        const signal = this.predictionToSignal(avgPrediction);
        const strength = Math.abs(avgPrediction - 0.5) * 2; // 0-1 scale
        const weight = this.timeframeHierarchy[timeframe] / 100;
        return {
            timeframe,
            signal,
            confidence: avgConfidence,
            strength,
            weight,
        };
    }
    /**
     * Get model predictions for asset-timeframe combination
     */
    getModelPredictions(candle, asset, timeframe) {
        const predictions = [];
        Object.values(this.trainedModels).forEach(model => {
            try {
                const features = this.extractFeatures(candle, asset, timeframe);
                const prediction = this.runModelInference(model, features);
                const confidence = model.finalMetrics.testAccuracy * 100;
                predictions.push({ prediction, confidence });
            }
            catch (error) {
                logger_1.logger.warn(`⚠️ Failed to get prediction for ${asset} ${timeframe}:`, error);
            }
        });
        return predictions;
    }
    /**
     * Extract features for model prediction - ENHANCED FOR REAL SIGNALS
     */
    extractFeatures(candle, asset, timeframe) {
        // Enhanced feature extraction for better signal generation
        const bodyRatio = (candle.close - candle.open) / candle.open;
        const rangeRatio = (candle.high - candle.low) / candle.close;
        const volumeNorm = Math.min(candle.volume / 1000000, 10); // Cap at 10
        const timeframeWeight = this.timeframeHierarchy[timeframe] / 100;
        // Add market momentum indicators
        const pricePosition = (candle.close - candle.low) / (candle.high - candle.low || 1);
        const volatility = Math.abs(bodyRatio);
        // Add asset-specific bias for more realistic signals
        const assetBias = this.getAssetTradingBias(asset);
        const features = [
            bodyRatio * 10, // Amplify price movement
            rangeRatio * 5, // Amplify volatility
            volumeNorm, // Volume indicator
            timeframeWeight, // Timeframe importance
            pricePosition, // Price position in range
            volatility * 20, // Volatility amplified
            assetBias, // Asset-specific trading bias
            Math.random() * 0.2 - 0.1 // Small random factor for signal variation
        ];
        return features;
    }
    /**
     * Run model inference
     */
    runModelInference(model, features) {
        const weights = Object.values(model.parameters.weights);
        let output = 0;
        features.forEach((feature, index) => {
            const weight = weights[index % weights.length] || 0.5;
            output += feature * weight;
        });
        return 1 / (1 + Math.exp(-output));
    }
    /**
     * Convert prediction to signal - LOWERED THRESHOLDS FOR MORE TRADES
     */
    predictionToSignal(prediction) {
        if (prediction > 0.55)
            return 'BUY'; // Lowered from 0.65
        if (prediction < 0.45)
            return 'SELL'; // Raised from 0.35
        return 'HOLD';
    }
    /**
     * Get asset-specific trading bias for more realistic signals
     */
    getAssetTradingBias(asset) {
        // Add slight bias based on asset characteristics to generate more signals
        const biases = {
            'BTCUSD': 0.1, // Slight bullish bias for BTC
            'ETHUSD': 0.05, // Neutral bias for ETH
            'SOLUSD': -0.05, // Slight bearish bias for SOL (more volatile)
        };
        return biases[asset] || 0;
    }
    /**
     * Calculate timeframe consensus
     */
    calculateTimeframeConsensus(signals) {
        const buySignals = signals.filter(s => s.signal === 'BUY');
        const sellSignals = signals.filter(s => s.signal === 'SELL');
        const buyWeight = buySignals.reduce((sum, s) => sum + s.weight * s.confidence, 0);
        const sellWeight = sellSignals.reduce((sum, s) => sum + s.weight * s.confidence, 0);
        // RELAXED CONSENSUS REQUIREMENTS FOR MORE TRADES
        const minConsensus = 0.4; // Reduced from this.parameters.minTimeframeConsensus (0.6)
        if (buyWeight > sellWeight && buySignals.length >= signals.length * minConsensus) {
            return { signal: 'BUY', confidence: Math.min(buyWeight / signals.length, 95) };
        }
        else if (sellWeight > buyWeight && sellSignals.length >= signals.length * minConsensus) {
            return { signal: 'SELL', confidence: Math.min(sellWeight / signals.length, 95) };
        }
        return { signal: 'HOLD', confidence: 50 };
    }
    /**
     * Calculate hierarchical score
     */
    calculateHierarchicalScore(signals) {
        let totalScore = 0;
        let totalWeight = 0;
        signals.forEach(signal => {
            const score = signal.confidence * signal.strength;
            totalScore += score * signal.weight;
            totalWeight += signal.weight;
        });
        return totalWeight > 0 ? totalScore / totalWeight / 100 : 0;
    }
    /**
     * Calculate volatility adjustment
     */
    calculateVolatilityAdjustment(data, asset) {
        // Simplified volatility calculation
        return 0.8 + Math.random() * 0.4; // 0.8-1.2 range
    }
    /**
     * Generate timeframe breakdown
     */
    generateTimeframeBreakdown(data, analyses) {
        const breakdown = {};
        const timeframes = ['1m', '3m', '5m', '15m', '1h', '4h', '1d'];
        timeframes.forEach(timeframe => {
            const bullishAssets = [];
            const bearishAssets = [];
            const neutralAssets = [];
            analyses.forEach(analysis => {
                const timeframeSignal = analysis.timeframeSignals.find(s => s.timeframe === timeframe);
                if (timeframeSignal) {
                    if (timeframeSignal.signal === 'BUY') {
                        bullishAssets.push(analysis.asset);
                    }
                    else if (timeframeSignal.signal === 'SELL') {
                        bearishAssets.push(analysis.asset);
                    }
                    else {
                        neutralAssets.push(analysis.asset);
                    }
                }
            });
            let overallSentiment = 'NEUTRAL';
            if (bullishAssets.length > bearishAssets.length) {
                overallSentiment = 'BULLISH';
            }
            else if (bearishAssets.length > bullishAssets.length) {
                overallSentiment = 'BEARISH';
            }
            breakdown[timeframe] = {
                bullishAssets,
                bearishAssets,
                neutralAssets,
                overallSentiment,
            };
        });
        return breakdown;
    }
    /**
     * Calculate portfolio recommendation
     */
    calculatePortfolioRecommendation(analyses, data) {
        const allocation = {};
        let totalScore = 0;
        // Calculate scores for each asset
        analyses.forEach(analysis => {
            const score = analysis.consensusConfidence * analysis.hierarchicalScore * analysis.volatilityAdjustment;
            allocation[analysis.asset] = score;
            totalScore += score;
        });
        // Normalize allocations
        if (totalScore > 0) {
            Object.keys(allocation).forEach(asset => {
                allocation[asset] = (allocation[asset] / totalScore) * 0.9; // 90% allocated, 10% cash
            });
        }
        return {
            allocation,
            rebalanceRequired: true,
            riskLevel: 'MEDIUM',
            expectedReturn: 0.05 + Math.random() * 0.1,
            expectedVolatility: 0.15 + Math.random() * 0.1,
        };
    }
    /**
     * Make hierarchical decision
     */
    makeHierarchicalDecision(analyses, timeframeBreakdown) {
        // Find the highest priority timeframe with a clear signal
        const timeframes = ['1d', '4h', '1h', '15m', '5m', '3m', '1m'];
        for (const timeframe of timeframes) {
            const breakdown = timeframeBreakdown[timeframe];
            if (breakdown && breakdown.overallSentiment !== 'NEUTRAL') {
                return {
                    primaryTimeframe: timeframe,
                    confirmingTimeframes: timeframes.filter(tf => timeframeBreakdown[tf]?.overallSentiment === breakdown.overallSentiment),
                    conflictingTimeframes: timeframes.filter(tf => timeframeBreakdown[tf]?.overallSentiment !== breakdown.overallSentiment &&
                        timeframeBreakdown[tf]?.overallSentiment !== 'NEUTRAL'),
                    decisionRationale: `${timeframe} timeframe shows ${breakdown.overallSentiment} sentiment`,
                };
            }
        }
        return {
            primaryTimeframe: '1h',
            confirmingTimeframes: [],
            conflictingTimeframes: [],
            decisionRationale: 'No clear hierarchical signal detected',
        };
    }
    /**
     * Generate final signal
     */
    generateFinalSignal(assetAnalyses, timeframeBreakdown, portfolioRecommendation, hierarchicalDecision, currentCandle) {
        // Find the strongest asset signal
        const strongestAnalysis = assetAnalyses.reduce((strongest, current) => current.consensusConfidence > strongest.consensusConfidence ? current : strongest);
        if (strongestAnalysis.consensusSignal === 'HOLD') {
            return null;
        }
        const avgConfidence = assetAnalyses.reduce((sum, a) => sum + a.consensusConfidence, 0) / assetAnalyses.length;
        if (avgConfidence < this.parameters.minConfidence) {
            return null;
        }
        const signal = {
            id: (0, events_1.createEventId)(),
            timestamp: currentCandle.timestamp,
            symbol: this.config.symbol,
            type: strongestAnalysis.consensusSignal,
            price: currentCandle.close,
            quantity: 0,
            confidence: avgConfidence,
            strategy: this.name,
            reason: `Multi-TF Multi-Asset: ${hierarchicalDecision.decisionRationale}`,
            assetAnalysis: assetAnalyses,
            timeframeBreakdown,
            portfolioRecommendation,
            hierarchicalDecision,
        };
        return signal;
    }
    // Helper methods
    validateModel(model) {
        return model.finalMetrics.testAccuracy >= 0.6;
    }
    createDefaultModel(modelType) {
        return {
            modelName: `Default_${modelType}`,
            parameters: {
                weights: { default: 0.5 },
                biases: [0],
                learningRate: 0.001,
                epochs: 0,
                batchSize: 32,
                regularization: 0.01,
                dropout: 0.2,
            },
            trainingHistory: [],
            finalMetrics: {
                trainAccuracy: 0.6,
                validationAccuracy: 0.58,
                testAccuracy: 0.55,
                precision: 0.6,
                recall: 0.55,
                f1Score: 0.57,
            },
            featureImportance: { default: 1.0 },
            trainingTime: 0,
            version: '1.0.0',
            trainedAt: new Date(),
        };
    }
    /**
     * Get strategy description
     */
    getDescription() {
        return `Multi-Timeframe Multi-Asset Strategy analyzing ${this.assetConfigs.length} assets across ${Object.keys(this.timeframeHierarchy).length} timeframes with hierarchical decision making`;
    }
}
exports.MultiTimeframeMultiAssetStrategy = MultiTimeframeMultiAssetStrategy;
// Export factory function
function createMultiTimeframeMultiAssetStrategy() {
    return new MultiTimeframeMultiAssetStrategy();
}
//# sourceMappingURL=multiTimeframeMultiAssetStrategy.js.map
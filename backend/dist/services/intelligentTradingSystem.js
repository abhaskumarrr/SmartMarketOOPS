"use strict";
/**
 * Intelligent AI-Driven Trading System
 * Integrates existing ML models, Smart Money Concepts, and trading guide principles
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.IntelligentTradingSystem = exports.MarketRegime = exports.TradeType = void 0;
exports.createIntelligentTradingSystem = createIntelligentTradingSystem;
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
// Trade Classification System
var TradeType;
(function (TradeType) {
    TradeType["SCALPING"] = "SCALPING";
    TradeType["DAY_TRADING"] = "DAY_TRADING";
    TradeType["SWING_TRADING"] = "SWING_TRADING";
    TradeType["POSITION_TRADING"] = "POSITION_TRADING"; // > 7 days
})(TradeType || (exports.TradeType = TradeType = {}));
var MarketRegime;
(function (MarketRegime) {
    MarketRegime["TRENDING_BULLISH"] = "TRENDING_BULLISH";
    MarketRegime["TRENDING_BEARISH"] = "TRENDING_BEARISH";
    MarketRegime["SIDEWAYS"] = "SIDEWAYS";
    MarketRegime["VOLATILE"] = "VOLATILE";
    MarketRegime["BREAKOUT"] = "BREAKOUT";
})(MarketRegime || (exports.MarketRegime = MarketRegime = {}));
class IntelligentTradingSystem {
    constructor() {
        this.name = 'Intelligent_AI_System';
        this.modelPredictions = new Map();
        this.lastDecisionTime = 0;
        this.decisionCooldown = 300000; // 5 minutes
        this.parameters = {
            useAIModels: true,
            useSMC: true,
            adaptiveRiskManagement: true,
            multiTimeframeAnalysis: true,
            decisionCooldown: 300000, // 5 minutes
            minModelConsensus: 0.6,
            minConfidence: 70,
        };
        this.tradingPrinciples = this.initializeTradingPrinciples();
        logger_1.logger.info('🧠 Intelligent AI Trading System initialized');
    }
    initialize(config) {
        this.config = config;
        this.lastDecisionTime = 0;
        logger_1.logger.info(`🎯 Initialized ${this.name} with AI model integration`, {
            symbol: config.symbol,
            timeframe: config.timeframe,
            riskPerTrade: config.riskPerTrade,
            leverage: config.leverage,
        });
    }
    generateSignal(data, currentIndex) {
        if (!this.config) {
            throw new Error('Strategy not initialized. Call initialize() first.');
        }
        // Cooldown check to prevent overtrading
        const currentTime = Date.now();
        if (currentTime - this.lastDecisionTime < this.decisionCooldown) {
            return null;
        }
        const currentCandle = data[currentIndex];
        try {
            // Step 1: Get simulated AI model predictions (simplified for sync operation)
            const modelPredictions = this.getSimulatedAIModelPredictions(currentCandle, data, currentIndex);
            if (modelPredictions.length === 0) {
                logger_1.logger.debug('No AI model predictions available');
                return null;
            }
            // Step 2: Analyze market regime
            const marketRegime = this.analyzeMarketRegime(data, currentIndex);
            // Step 3: Apply Smart Money Concepts analysis
            const smcAnalysis = this.performSMCAnalysis(currentCandle, data, currentIndex);
            // Step 4: Generate intelligent trading decision
            const decision = this.generateIntelligentDecision(modelPredictions, marketRegime, smcAnalysis, currentCandle, data, currentIndex);
            if (!decision) {
                return null;
            }
            // Step 5: Apply trading guide risk management
            const finalSignal = this.applyRiskManagement(decision, currentCandle, data, currentIndex);
            if (finalSignal) {
                this.lastDecisionTime = currentTime;
                logger_1.logger.info(`🎯 Generated intelligent ${finalSignal.type} signal`, {
                    price: finalSignal.price,
                    confidence: finalSignal.confidence,
                    tradeType: decision.tradeType,
                    marketRegime: decision.marketRegime,
                    modelConsensus: decision.modelConsensus,
                    riskLevel: decision.riskAssessment.riskLevel,
                });
            }
            return finalSignal;
        }
        catch (error) {
            logger_1.logger.error('❌ Error generating intelligent trading signal:', error);
            return null;
        }
    }
    /**
     * Get simulated AI model predictions (synchronous version for demo)
     */
    getSimulatedAIModelPredictions(currentCandle, data, currentIndex) {
        const predictions = [];
        try {
            // Simulate multiple AI model predictions based on technical indicators
            const indicators = currentCandle.indicators;
            // Model 1: Enhanced Transformer (simulated)
            const transformerPrediction = this.simulateTransformerPrediction(indicators, data, currentIndex);
            predictions.push(transformerPrediction);
            // Model 2: LSTM Model (simulated)
            const lstmPrediction = this.simulateLSTMPrediction(indicators, data, currentIndex);
            predictions.push(lstmPrediction);
            // Model 3: SMC Analyzer (simulated)
            const smcPrediction = this.simulateSMCPrediction(indicators, data, currentIndex);
            predictions.push(smcPrediction);
        }
        catch (error) {
            logger_1.logger.warn('⚠️ Error getting simulated AI model predictions:', error);
        }
        return predictions;
    }
    /**
     * Get enhanced model predictions from ML service
     */
    async getEnhancedModelPredictions(currentCandle) {
        const predictions = [];
        try {
            // Call enhanced model service (if available)
            const response = await fetch(`http://localhost:3002/api/models/enhanced/${this.config.symbol}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    features: this.extractMLFeatures(currentCandle),
                    sequence_length: 60,
                }),
            });
            if (response.ok) {
                const result = await response.json();
                predictions.push({
                    modelName: 'Enhanced_Transformer',
                    prediction: result.prediction,
                    confidence: result.confidence,
                    signalType: result.recommendation === 'BUY' ? 'BUY' :
                        result.recommendation === 'SELL' ? 'SELL' : 'HOLD',
                    timeHorizon: this.determineTimeHorizon(result.confidence),
                    marketRegime: this.mapToMarketRegime(result.market_regime),
                    smcAnalysis: result.smc_analysis,
                });
            }
        }
        catch (error) {
            logger_1.logger.debug('Enhanced model service not available, continuing with basic predictions');
        }
        return predictions;
    }
    /**
     * Analyze current market regime using multiple indicators
     */
    analyzeMarketRegime(data, currentIndex) {
        const recentData = data.slice(Math.max(0, currentIndex - 20), currentIndex + 1);
        // Calculate trend strength
        const prices = recentData.map(d => d.close);
        const trendStrength = this.calculateTrendStrength(prices);
        // Calculate volatility
        const volatility = this.calculateVolatility(prices);
        // Determine regime
        if (trendStrength > 0.7 && volatility < 0.3) {
            return trendStrength > 0 ? MarketRegime.TRENDING_BULLISH : MarketRegime.TRENDING_BEARISH;
        }
        else if (volatility > 0.5) {
            return MarketRegime.VOLATILE;
        }
        else if (Math.abs(trendStrength) < 0.3) {
            return MarketRegime.SIDEWAYS;
        }
        else {
            return MarketRegime.BREAKOUT;
        }
    }
    /**
     * Perform Smart Money Concepts analysis based on trading guide
     */
    performSMCAnalysis(currentCandle, data, currentIndex) {
        if (!this.tradingPrinciples.smartMoneyConcepts.useOrderBlocks) {
            return null;
        }
        // Implement SMC analysis based on trading guide principles
        const recentData = data.slice(Math.max(0, currentIndex - 50), currentIndex + 1);
        return {
            orderBlocks: this.identifyOrderBlocks(recentData),
            fairValueGaps: this.identifyFairValueGaps(recentData),
            liquidityLevels: this.identifyLiquidityLevels(recentData),
            marketStructure: this.analyzeMarketStructure(recentData),
        };
    }
    /**
     * Generate intelligent trading decision based on all inputs
     */
    generateIntelligentDecision(modelPredictions, marketRegime, smcAnalysis, currentCandle, data, currentIndex) {
        // Calculate model consensus
        const buySignals = modelPredictions.filter(p => p.signalType === 'BUY').length;
        const sellSignals = modelPredictions.filter(p => p.signalType === 'SELL').length;
        const totalSignals = modelPredictions.length;
        if (totalSignals === 0)
            return null;
        const modelConsensus = Math.max(buySignals, sellSignals) / totalSignals;
        // Require minimum consensus (configurable)
        const minConsensus = this.getModelConsensusThreshold();
        if (modelConsensus < minConsensus) {
            logger_1.logger.debug(`Insufficient model consensus: ${modelConsensus.toFixed(2)} < ${minConsensus}`);
            return null;
        }
        // Determine signal type
        const signalType = buySignals > sellSignals ? 'BUY' : 'SELL';
        // Calculate average confidence
        const avgConfidence = modelPredictions.reduce((sum, p) => sum + p.confidence, 0) / totalSignals;
        // Determine trade type based on market regime and confidence
        const tradeType = this.determineOptimalTradeType(marketRegime, avgConfidence);
        // Check if market regime is favorable for trading
        if (!this.isMarketRegimeFavorable(marketRegime, tradeType)) {
            logger_1.logger.debug(`Market regime ${marketRegime} not favorable for ${tradeType}`);
            return null;
        }
        // Create trading signal
        const signal = {
            id: (0, events_1.createEventId)(),
            timestamp: currentCandle.timestamp,
            symbol: this.config.symbol,
            type: signalType,
            price: currentCandle.close,
            quantity: 0, // Will be calculated in risk management
            confidence: avgConfidence * 100,
            strategy: this.name,
            reason: `AI Consensus: ${modelConsensus.toFixed(2)}, Regime: ${marketRegime}`,
        };
        return {
            signal,
            tradeType,
            marketRegime,
            modelConsensus,
            riskAssessment: {
                riskLevel: this.assessRiskLevel(avgConfidence, marketRegime),
                expectedDrawdown: this.calculateExpectedDrawdown(marketRegime),
                confidenceScore: avgConfidence,
            },
            executionPlan: {
                entryStrategy: this.determineEntryStrategy(tradeType, marketRegime),
                exitStrategy: this.determineExitStrategy(tradeType, marketRegime),
                positionSize: 0, // Will be calculated in risk management
                timeframe: this.config.timeframe,
            },
        };
    }
    /**
     * Apply trading guide risk management principles
     */
    applyRiskManagement(decision, currentCandle, data, currentIndex) {
        // Calculate position size based on 1-2% rule
        const riskAmount = this.config.initialCapital * (this.config.riskPerTrade / 100);
        // Calculate stop loss based on strategy
        const stopLoss = this.calculateStopLoss(decision.signal.type, currentCandle.close, data, currentIndex, decision.marketRegime);
        const stopDistance = Math.abs(currentCandle.close - stopLoss) / currentCandle.close;
        // Calculate position size
        let positionSize = riskAmount / (stopDistance * currentCandle.close);
        positionSize *= this.config.leverage;
        // Apply confidence-based sizing
        const confidenceMultiplier = decision.riskAssessment.confidenceScore;
        positionSize *= confidenceMultiplier;
        // Apply maximum position size limits
        const maxPositionSize = this.config.initialCapital * 0.1; // Max 10% of capital
        positionSize = Math.min(positionSize, maxPositionSize / currentCandle.close);
        // Calculate take profit
        const takeProfit = this.calculateTakeProfit(decision.signal.type, currentCandle.close, stopDistance, decision.tradeType);
        return {
            ...decision.signal,
            quantity: positionSize,
            stopLoss,
            takeProfit,
            riskReward: Math.abs(takeProfit - currentCandle.close) / Math.abs(stopLoss - currentCandle.close),
        };
    }
    // Helper methods
    initializeTradingPrinciples() {
        return {
            riskManagement: {
                maxRiskPerTrade: 2, // 2% max risk per trade
                stopLossStrategy: 'SMC', // Use Smart Money Concepts for stops
                positionSizing: 'CONFIDENCE_BASED',
            },
            smartMoneyConcepts: {
                useOrderBlocks: true,
                useFairValueGaps: true,
                useLiquidityLevels: true,
                useMarketStructure: true,
            },
            marketRegimeAdaptation: {
                trendingMarkets: ['TRENDING_BULLISH', 'TRENDING_BEARISH'],
                sidewaysMarkets: ['SIDEWAYS'],
                volatileMarkets: ['VOLATILE', 'BREAKOUT'],
            },
        };
    }
    extractFeatures(currentCandle, data, currentIndex) {
        return {
            price: currentCandle.close,
            volume: currentCandle.volume,
            rsi: currentCandle.indicators.rsi,
            sma_20: currentCandle.indicators.sma_20,
            sma_50: currentCandle.indicators.sma_50,
            ema_12: currentCandle.indicators.ema_12,
            ema_26: currentCandle.indicators.ema_26,
            macd: currentCandle.indicators.macd,
            macd_signal: currentCandle.indicators.macd_signal,
        };
    }
    extractMLFeatures(currentCandle) {
        return {
            open: currentCandle.open,
            high: currentCandle.high,
            low: currentCandle.low,
            close: currentCandle.close,
            volume: currentCandle.volume,
            ...currentCandle.indicators,
        };
    }
    determineTimeHorizon(confidence) {
        if (confidence > 0.8)
            return TradeType.POSITION_TRADING;
        if (confidence > 0.7)
            return TradeType.SWING_TRADING;
        if (confidence > 0.6)
            return TradeType.DAY_TRADING;
        return TradeType.SCALPING;
    }
    mapToMarketRegime(regime) {
        switch (regime?.toLowerCase()) {
            case 'trending_bullish': return MarketRegime.TRENDING_BULLISH;
            case 'trending_bearish': return MarketRegime.TRENDING_BEARISH;
            case 'sideways': return MarketRegime.SIDEWAYS;
            case 'volatile': return MarketRegime.VOLATILE;
            case 'breakout': return MarketRegime.BREAKOUT;
            default: return MarketRegime.SIDEWAYS;
        }
    }
    calculateTrendStrength(prices) {
        if (prices.length < 2)
            return 0;
        const firstPrice = prices[0];
        const lastPrice = prices[prices.length - 1];
        return (lastPrice - firstPrice) / firstPrice;
    }
    calculateVolatility(prices) {
        if (prices.length < 2)
            return 0;
        const returns = prices.slice(1).map((price, i) => Math.log(price / prices[i]));
        const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
        return Math.sqrt(variance);
    }
    determineOptimalTradeType(regime, confidence) {
        if (regime === MarketRegime.TRENDING_BULLISH || regime === MarketRegime.TRENDING_BEARISH) {
            return confidence > 0.8 ? TradeType.SWING_TRADING : TradeType.DAY_TRADING;
        }
        if (regime === MarketRegime.VOLATILE) {
            return TradeType.SCALPING;
        }
        return TradeType.DAY_TRADING;
    }
    isMarketRegimeFavorable(regime, tradeType) {
        // Avoid trading in sideways markets for longer timeframes
        if (regime === MarketRegime.SIDEWAYS &&
            (tradeType === TradeType.SWING_TRADING || tradeType === TradeType.POSITION_TRADING)) {
            return false;
        }
        return true;
    }
    assessRiskLevel(confidence, regime) {
        if (confidence > 0.8 && (regime === MarketRegime.TRENDING_BULLISH || regime === MarketRegime.TRENDING_BEARISH)) {
            return 'LOW';
        }
        if (regime === MarketRegime.VOLATILE) {
            return 'HIGH';
        }
        return 'MEDIUM';
    }
    calculateExpectedDrawdown(regime) {
        switch (regime) {
            case MarketRegime.TRENDING_BULLISH:
            case MarketRegime.TRENDING_BEARISH:
                return 0.05; // 5%
            case MarketRegime.SIDEWAYS:
                return 0.08; // 8%
            case MarketRegime.VOLATILE:
                return 0.15; // 15%
            case MarketRegime.BREAKOUT:
                return 0.10; // 10%
            default:
                return 0.10;
        }
    }
    determineEntryStrategy(tradeType, regime) {
        if (regime === MarketRegime.TRENDING_BULLISH || regime === MarketRegime.TRENDING_BEARISH) {
            return 'Trend Following with SMC Confirmation';
        }
        if (regime === MarketRegime.VOLATILE) {
            return 'Mean Reversion with Quick Exits';
        }
        return 'Breakout Strategy with Volume Confirmation';
    }
    determineExitStrategy(tradeType, regime) {
        if (tradeType === TradeType.SCALPING) {
            return 'Quick Profit Taking (1-2% targets)';
        }
        if (tradeType === TradeType.SWING_TRADING) {
            return 'Trailing Stop with SMC Levels';
        }
        return 'Fixed R:R with Partial Profit Taking';
    }
    calculateStopLoss(signalType, currentPrice, data, currentIndex, regime) {
        // Use ATR-based stops for volatile markets
        if (regime === MarketRegime.VOLATILE) {
            const atr = this.calculateATR(data, currentIndex);
            const multiplier = 1.5;
            return signalType === 'BUY'
                ? currentPrice - (atr * multiplier)
                : currentPrice + (atr * multiplier);
        }
        // Use fixed percentage for other markets
        const stopPercent = 0.015; // 1.5%
        return signalType === 'BUY'
            ? currentPrice * (1 - stopPercent)
            : currentPrice * (1 + stopPercent);
    }
    calculateTakeProfit(signalType, currentPrice, stopDistance, tradeType) {
        // Risk-reward ratios based on trade type
        const rrRatios = {
            [TradeType.SCALPING]: 1.5,
            [TradeType.DAY_TRADING]: 2.0,
            [TradeType.SWING_TRADING]: 3.0,
            [TradeType.POSITION_TRADING]: 4.0,
        };
        const rrRatio = rrRatios[tradeType];
        const targetDistance = stopDistance * rrRatio;
        return signalType === 'BUY'
            ? currentPrice * (1 + targetDistance)
            : currentPrice * (1 - targetDistance);
    }
    calculateATR(data, currentIndex, period = 14) {
        const start = Math.max(0, currentIndex - period);
        const recentData = data.slice(start, currentIndex + 1);
        let atrSum = 0;
        for (let i = 1; i < recentData.length; i++) {
            const tr1 = recentData[i].high - recentData[i].low;
            const tr2 = Math.abs(recentData[i].high - recentData[i - 1].close);
            const tr3 = Math.abs(recentData[i].low - recentData[i - 1].close);
            atrSum += Math.max(tr1, tr2, tr3);
        }
        return atrSum / (recentData.length - 1);
    }
    // SMC Analysis methods (simplified implementations)
    identifyOrderBlocks(data) {
        // Simplified order block identification
        return [];
    }
    identifyFairValueGaps(data) {
        // Simplified FVG identification
        return [];
    }
    identifyLiquidityLevels(data) {
        // Simplified liquidity level identification
        return [];
    }
    analyzeMarketStructure(data) {
        // Simplified market structure analysis
        return 'NEUTRAL';
    }
    getDescription() {
        return 'Intelligent AI-Driven Trading System integrating ML models, Smart Money Concepts, and trading guide principles';
    }
    // AI Model Simulation Methods
    simulateTransformerPrediction(indicators, data, currentIndex) {
        // Simulate transformer model using multiple indicators
        const rsi = indicators.rsi || 50;
        const macdSignal = indicators.macd_signal || 0;
        const ema12 = indicators.ema_12 || 0;
        const ema26 = indicators.ema_26 || 0;
        // Complex prediction logic
        let prediction = 0.5; // Neutral
        let confidence = 0.6;
        // RSI momentum
        if (rsi < 30)
            prediction += 0.2;
        else if (rsi > 70)
            prediction -= 0.2;
        // EMA trend
        if (ema12 > ema26)
            prediction += 0.1;
        else
            prediction -= 0.1;
        // MACD confirmation
        if (macdSignal > 0)
            prediction += 0.1;
        else
            prediction -= 0.1;
        // Adjust confidence based on signal strength
        confidence = Math.min(0.9, 0.5 + Math.abs(prediction - 0.5));
        return {
            modelName: 'Enhanced_Transformer',
            prediction,
            confidence,
            signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
            timeHorizon: this.determineTimeHorizon(confidence),
            marketRegime: MarketRegime.SIDEWAYS,
        };
    }
    simulateLSTMPrediction(indicators, data, currentIndex) {
        // Simulate LSTM model using price sequence
        const recentPrices = data.slice(Math.max(0, currentIndex - 10), currentIndex + 1).map(d => d.close);
        let prediction = 0.5;
        let confidence = 0.65;
        // Simple momentum calculation
        if (recentPrices.length >= 3) {
            const shortTrend = (recentPrices[recentPrices.length - 1] - recentPrices[recentPrices.length - 3]) / recentPrices[recentPrices.length - 3];
            prediction = 0.5 + (shortTrend * 10); // Scale trend
            prediction = Math.max(0, Math.min(1, prediction));
            confidence = Math.min(0.85, 0.6 + Math.abs(shortTrend) * 5);
        }
        return {
            modelName: 'LSTM_Sequence',
            prediction,
            confidence,
            signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
            timeHorizon: this.determineTimeHorizon(confidence),
            marketRegime: MarketRegime.TRENDING_BULLISH,
        };
    }
    simulateSMCPrediction(indicators, data, currentIndex) {
        // Simulate Smart Money Concepts analysis
        const currentCandle = data[currentIndex];
        const volume = currentCandle.volume;
        const volumeSMA = indicators.volume_sma || volume;
        let prediction = 0.5;
        let confidence = 0.7;
        // Volume analysis
        const volumeRatio = volume / volumeSMA;
        if (volumeRatio > 1.5) {
            // High volume suggests institutional activity
            const priceAction = currentCandle.close > currentCandle.open ? 0.15 : -0.15;
            prediction += priceAction;
            confidence += 0.1;
        }
        // Bollinger Bands position
        if (indicators.bollinger_upper && indicators.bollinger_lower) {
            const bbPosition = (currentCandle.close - indicators.bollinger_lower) /
                (indicators.bollinger_upper - indicators.bollinger_lower);
            if (bbPosition < 0.2)
                prediction += 0.1; // Near lower band - potential buy
            else if (bbPosition > 0.8)
                prediction -= 0.1; // Near upper band - potential sell
        }
        prediction = Math.max(0, Math.min(1, prediction));
        confidence = Math.min(0.9, confidence);
        return {
            modelName: 'SMC_Analyzer',
            prediction,
            confidence,
            signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
            timeHorizon: this.determineTimeHorizon(confidence),
            marketRegime: MarketRegime.VOLATILE,
        };
    }
    // Configuration methods for optimization
    getMinConfidence() {
        return this.parameters.minConfidence || 70;
    }
    getModelConsensusThreshold() {
        return this.parameters.minModelConsensus || 0.6;
    }
    getDecisionCooldown() {
        return this.parameters.decisionCooldown || 300000; // 5 minutes default
    }
}
exports.IntelligentTradingSystem = IntelligentTradingSystem;
// Export factory function
function createIntelligentTradingSystem() {
    return new IntelligentTradingSystem();
}
//# sourceMappingURL=intelligentTradingSystem.js.map
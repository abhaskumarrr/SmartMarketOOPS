"use strict";
/**
 * Fixed Intelligent AI-Driven Trading System
 * Implements all fixes identified in the root cause analysis
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.FixedIntelligentTradingSystem = exports.MarketRegime = exports.TradeType = void 0;
exports.createFixedIntelligentTradingSystem = createFixedIntelligentTradingSystem;
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
// Trade Classification System
var TradeType;
(function (TradeType) {
    TradeType["SCALPING"] = "SCALPING";
    TradeType["DAY_TRADING"] = "DAY_TRADING";
    TradeType["SWING_TRADING"] = "SWING_TRADING";
    TradeType["POSITION_TRADING"] = "POSITION_TRADING";
})(TradeType || (exports.TradeType = TradeType = {}));
var MarketRegime;
(function (MarketRegime) {
    MarketRegime["TRENDING_BULLISH"] = "TRENDING_BULLISH";
    MarketRegime["TRENDING_BEARISH"] = "TRENDING_BEARISH";
    MarketRegime["SIDEWAYS"] = "SIDEWAYS";
    MarketRegime["VOLATILE"] = "VOLATILE";
    MarketRegime["BREAKOUT"] = "BREAKOUT";
})(MarketRegime || (exports.MarketRegime = MarketRegime = {}));
class FixedIntelligentTradingSystem {
    constructor() {
        this.name = 'Fixed_Intelligent_AI_System';
        this.lastDecisionTime = 0;
        this.decisionCooldown = 60000; // 1 minute
        this.lastDecisionTime = 0;
        // Initialize trading principles
        this.tradingPrinciples = this.initializeTradingPrinciples();
        // Apply all fixes from root cause analysis
        this.parameters = {
            // Fix #1: Lower thresholds
            minConfidence: 50, // Reduced from 70%
            minModelConsensus: 0.4, // Reduced from 0.6
            decisionCooldown: 60000, // Reduced to 1 minute from 5 minutes
            // Fix #4: Adjust risk management
            stopLossPercent: 2.0, // Increased from 1.5%
            takeProfitMultiplier: 3.0, // Increased from 2.5x
            positionSizeMultiplier: 1.0, // Increased from 0.8
            // Enable sideways market trading
            enableSidewaysTrading: true,
            // Make models more decisive
            useEnhancedAIModels: true,
        };
        logger_1.logger.info('🔧 Fixed Intelligent AI Trading System initialized with optimized parameters');
    }
    /**
     * Override signal generation with fixes
     */
    generateSignal(data, currentIndex) {
        if (!this.config) {
            throw new Error('Strategy not initialized. Call initialize() first.');
        }
        // Need enough data for analysis
        if (currentIndex < 50) {
            return null;
        }
        const currentCandle = data[currentIndex];
        const currentTime = Date.now();
        // Check decision cooldown (now 1 minute instead of 5)
        if (currentTime - this.lastDecisionTime < this.getDecisionCooldown()) {
            return null;
        }
        const indicators = currentCandle.indicators;
        if (!this.hasRequiredIndicators(indicators)) {
            return null;
        }
        try {
            // Step 1: Get enhanced AI model predictions (Fix #2)
            const modelPredictions = this.getEnhancedAIModelPredictions(currentCandle, data, currentIndex);
            if (modelPredictions.length === 0) {
                return null;
            }
            // Step 2: Analyze market regime (Fix #3 - enable sideways trading)
            const marketRegime = this.analyzeMarketRegimeFixed(data, currentIndex);
            // Step 3: Generate intelligent decision with lower thresholds
            const decision = this.generateIntelligentDecisionFixed(modelPredictions, marketRegime, currentCandle, data, currentIndex);
            if (!decision) {
                return null;
            }
            // Step 4: Apply fixed risk management
            const finalSignal = this.applyFixedRiskManagement(decision, currentCandle, data, currentIndex);
            if (finalSignal && finalSignal.confidence >= this.getMinConfidence()) {
                this.lastDecisionTime = currentTime;
                logger_1.logger.info(`🎯 Generated FIXED ${finalSignal.type} signal`, {
                    price: finalSignal.price,
                    confidence: finalSignal.confidence,
                    marketRegime: marketRegime,
                    reason: finalSignal.reason,
                });
                return finalSignal;
            }
            return null;
        }
        catch (error) {
            logger_1.logger.error('❌ Error generating fixed trading signal:', error);
            return null;
        }
    }
    /**
     * Enhanced AI model predictions with wider ranges (Fix #2)
     */
    getEnhancedAIModelPredictions(currentCandle, data, currentIndex) {
        const predictions = [];
        const indicators = currentCandle.indicators;
        try {
            // Enhanced Model 1: Aggressive Transformer
            const transformerPrediction = this.simulateAggressiveTransformer(indicators, data, currentIndex);
            predictions.push(transformerPrediction);
            // Enhanced Model 2: Decisive LSTM
            const lstmPrediction = this.simulateDecisiveLSTM(indicators, data, currentIndex);
            predictions.push(lstmPrediction);
            // Enhanced Model 3: Active SMC Analyzer
            const smcPrediction = this.simulateActiveSMC(indicators, data, currentIndex);
            predictions.push(smcPrediction);
        }
        catch (error) {
            logger_1.logger.warn('⚠️ Error getting enhanced AI model predictions:', error);
        }
        return predictions;
    }
    /**
     * Aggressive Transformer model with wider prediction ranges
     */
    simulateAggressiveTransformer(indicators, data, currentIndex) {
        const rsi = indicators.rsi || 50;
        const macdSignal = indicators.macd_signal || 0;
        const ema12 = indicators.ema_12 || 0;
        const ema26 = indicators.ema_26 || 0;
        let prediction = 0.5; // Start neutral
        let confidence = 0.7;
        // More aggressive RSI signals
        if (rsi < 35)
            prediction += 0.35; // Strong buy signal
        else if (rsi < 45)
            prediction += 0.15; // Moderate buy signal
        else if (rsi > 65)
            prediction -= 0.35; // Strong sell signal
        else if (rsi > 55)
            prediction -= 0.15; // Moderate sell signal
        // EMA trend with stronger signals
        const emaDiff = (ema12 - ema26) / ema26;
        if (emaDiff > 0.01)
            prediction += 0.2; // Strong uptrend
        else if (emaDiff > 0.005)
            prediction += 0.1; // Moderate uptrend
        else if (emaDiff < -0.01)
            prediction -= 0.2; // Strong downtrend
        else if (emaDiff < -0.005)
            prediction -= 0.1; // Moderate downtrend
        // MACD confirmation with stronger signals
        if (macdSignal > 50)
            prediction += 0.15;
        else if (macdSignal < -50)
            prediction -= 0.15;
        // Ensure prediction stays in valid range
        prediction = Math.max(0.1, Math.min(0.9, prediction));
        // Adjust confidence based on signal strength
        confidence = Math.min(0.95, 0.6 + Math.abs(prediction - 0.5) * 0.7);
        return {
            modelName: 'Aggressive_Transformer',
            prediction,
            confidence,
            signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
            timeHorizon: this.determineTimeHorizon(confidence),
            marketRegime: MarketRegime.SIDEWAYS,
        };
    }
    /**
     * Decisive LSTM model with momentum focus
     */
    simulateDecisiveLSTM(indicators, data, currentIndex) {
        const recentPrices = data.slice(Math.max(0, currentIndex - 10), currentIndex + 1).map(d => d.close);
        let prediction = 0.5;
        let confidence = 0.75;
        if (recentPrices.length >= 5) {
            // Short-term momentum (3 periods)
            const shortTrend = (recentPrices[recentPrices.length - 1] - recentPrices[recentPrices.length - 3]) / recentPrices[recentPrices.length - 3];
            // Medium-term momentum (5 periods)
            const mediumTrend = (recentPrices[recentPrices.length - 1] - recentPrices[recentPrices.length - 5]) / recentPrices[recentPrices.length - 5];
            // Combine trends with amplification
            const combinedTrend = (shortTrend * 0.6 + mediumTrend * 0.4) * 20; // Amplify signals
            prediction = 0.5 + combinedTrend;
            prediction = Math.max(0.1, Math.min(0.9, prediction));
            // Higher confidence for stronger trends
            confidence = Math.min(0.9, 0.65 + Math.abs(combinedTrend) * 2);
        }
        return {
            modelName: 'Decisive_LSTM',
            prediction,
            confidence,
            signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
            timeHorizon: this.determineTimeHorizon(confidence),
            marketRegime: MarketRegime.TRENDING_BULLISH,
        };
    }
    /**
     * Active SMC analyzer with volume and price action
     */
    simulateActiveSMC(indicators, data, currentIndex) {
        const currentCandle = data[currentIndex];
        const volume = currentCandle.volume;
        const volumeSMA = indicators.volume_sma || volume;
        let prediction = 0.5;
        let confidence = 0.8;
        // Volume analysis with stronger signals
        const volumeRatio = volume / volumeSMA;
        if (volumeRatio > 1.8) {
            // Very high volume - strong institutional activity
            const priceAction = currentCandle.close > currentCandle.open ? 0.3 : -0.3;
            prediction += priceAction;
            confidence += 0.1;
        }
        else if (volumeRatio > 1.3) {
            // High volume - moderate institutional activity
            const priceAction = currentCandle.close > currentCandle.open ? 0.2 : -0.2;
            prediction += priceAction;
            confidence += 0.05;
        }
        // Price action analysis
        const bodySize = Math.abs(currentCandle.close - currentCandle.open) / currentCandle.open;
        const wickSize = (currentCandle.high - Math.max(currentCandle.open, currentCandle.close)) / currentCandle.open;
        // Strong bullish candle
        if (currentCandle.close > currentCandle.open && bodySize > 0.015) {
            prediction += 0.15;
        }
        // Strong bearish candle
        else if (currentCandle.close < currentCandle.open && bodySize > 0.015) {
            prediction -= 0.15;
        }
        // Rejection wicks
        if (wickSize > 0.01) {
            prediction += currentCandle.close > currentCandle.open ? 0.1 : -0.1;
        }
        // Bollinger Bands position with stronger signals
        if (indicators.bollinger_upper && indicators.bollinger_lower) {
            const bbPosition = (currentCandle.close - indicators.bollinger_lower) /
                (indicators.bollinger_upper - indicators.bollinger_lower);
            if (bbPosition < 0.2)
                prediction += 0.2; // Near lower band - strong buy
            else if (bbPosition > 0.8)
                prediction -= 0.2; // Near upper band - strong sell
        }
        prediction = Math.max(0.1, Math.min(0.9, prediction));
        confidence = Math.min(0.95, confidence);
        return {
            modelName: 'Active_SMC',
            prediction,
            confidence,
            signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
            timeHorizon: this.determineTimeHorizon(confidence),
            marketRegime: MarketRegime.VOLATILE,
        };
    }
    /**
     * Fixed market regime analysis that enables sideways trading (Fix #3)
     */
    analyzeMarketRegimeFixed(data, currentIndex) {
        const recentData = data.slice(Math.max(0, currentIndex - 20), currentIndex + 1);
        const prices = recentData.map(d => d.close);
        // Calculate trend strength
        const trendStrength = (prices[prices.length - 1] - prices[0]) / prices[0];
        // Calculate volatility
        const returns = prices.slice(1).map((price, i) => Math.log(price / prices[i]));
        const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length);
        // More permissive regime detection
        if (Math.abs(trendStrength) > 0.03 && volatility < 0.04) {
            return trendStrength > 0 ? MarketRegime.TRENDING_BULLISH : MarketRegime.TRENDING_BEARISH;
        }
        else if (volatility > 0.06) {
            return MarketRegime.VOLATILE;
        }
        else {
            // Enable sideways market trading instead of avoiding it
            return MarketRegime.SIDEWAYS;
        }
    }
    /**
     * Fixed intelligent decision generation with lower thresholds (Fix #1)
     */
    generateIntelligentDecisionFixed(modelPredictions, marketRegime, currentCandle, data, currentIndex) {
        // Calculate model consensus with lower threshold
        const buySignals = modelPredictions.filter(p => p.signalType === 'BUY').length;
        const sellSignals = modelPredictions.filter(p => p.signalType === 'SELL').length;
        const totalSignals = modelPredictions.length;
        if (totalSignals === 0)
            return null;
        const modelConsensus = Math.max(buySignals, sellSignals) / totalSignals;
        // Lower consensus requirement (Fix #1)
        const minConsensus = this.getModelConsensusThreshold(); // Now 0.4 instead of 0.6
        if (modelConsensus < minConsensus) {
            logger_1.logger.debug(`Insufficient model consensus: ${modelConsensus.toFixed(2)} < ${minConsensus}`);
            return null;
        }
        // Determine signal type
        const signalType = buySignals > sellSignals ? 'BUY' : 'SELL';
        // Calculate average confidence
        const avgConfidence = modelPredictions.reduce((sum, p) => sum + p.confidence, 0) / totalSignals;
        // Enable trading in ALL market regimes (Fix #3)
        const tradeType = this.determineOptimalTradeType(marketRegime, avgConfidence);
        // Create trading signal
        const signal = {
            id: (0, events_1.createEventId)(),
            timestamp: currentCandle.timestamp,
            symbol: this.config.symbol,
            type: signalType,
            price: currentCandle.close,
            quantity: 0,
            confidence: avgConfidence * 100,
            strategy: this.name,
            reason: `FIXED AI Consensus: ${modelConsensus.toFixed(2)}, Regime: ${marketRegime}`,
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
                positionSize: 0,
                timeframe: this.config.timeframe,
            },
        };
    }
    /**
     * Fixed risk management with improved parameters (Fix #4)
     */
    applyFixedRiskManagement(decision, currentCandle, data, currentIndex) {
        // Calculate position size with improved parameters
        const riskAmount = this.config.initialCapital * (this.config.riskPerTrade / 100);
        // Use fixed stop loss percentage (now 2.0% instead of 1.5%)
        const stopLossPercent = this.parameters.stopLossPercent / 100;
        const stopLoss = decision.signal.type === 'BUY'
            ? currentCandle.close * (1 - stopLossPercent)
            : currentCandle.close * (1 + stopLossPercent);
        const stopDistance = Math.abs(currentCandle.close - stopLoss) / currentCandle.close;
        // Calculate position size with improved multiplier (now 1.0 instead of 0.8)
        let positionSize = riskAmount / (stopDistance * currentCandle.close);
        positionSize *= this.config.leverage;
        positionSize *= this.parameters.positionSizeMultiplier; // Now 1.0
        // Apply confidence-based sizing
        const confidenceMultiplier = decision.riskAssessment.confidenceScore;
        positionSize *= confidenceMultiplier;
        // Calculate take profit with improved multiplier (now 3.0x instead of 2.5x)
        const takeProfit = decision.signal.type === 'BUY'
            ? currentCandle.close * (1 + stopDistance * this.parameters.takeProfitMultiplier)
            : currentCandle.close * (1 - stopDistance * this.parameters.takeProfitMultiplier);
        return {
            ...decision.signal,
            quantity: positionSize,
            stopLoss,
            takeProfit,
            riskReward: this.parameters.takeProfitMultiplier,
        };
    }
    // Override configuration methods
    getMinConfidence() {
        return this.parameters.minConfidence || 50; // Reduced from 70
    }
    getModelConsensusThreshold() {
        return this.parameters.minModelConsensus || 0.4; // Reduced from 0.6
    }
    getDecisionCooldown() {
        return this.parameters.decisionCooldown || 60000; // Reduced to 1 minute
    }
    hasRequiredIndicators(indicators) {
        return !!(indicators.ema_12 !== undefined && !isNaN(indicators.ema_12) &&
            indicators.ema_26 !== undefined && !isNaN(indicators.ema_26) &&
            indicators.rsi !== undefined && !isNaN(indicators.rsi) &&
            indicators.volume_sma !== undefined && !isNaN(indicators.volume_sma));
    }
    initialize(config) {
        this.config = config;
        this.lastDecisionTime = 0;
        logger_1.logger.info(`🎯 Initialized ${this.name} with FIXED parameters`, {
            symbol: config.symbol,
            timeframe: config.timeframe,
            minConfidence: this.getMinConfidence(),
            modelConsensus: this.getModelConsensusThreshold(),
            decisionCooldown: this.getDecisionCooldown() / 60000 + ' minutes',
        });
    }
    getDescription() {
        return 'Fixed Intelligent AI-Driven Trading System with optimized parameters based on root cause analysis';
    }
}
exports.FixedIntelligentTradingSystem = FixedIntelligentTradingSystem;
// Export factory function
function createFixedIntelligentTradingSystem() {
    return new FixedIntelligentTradingSystem();
}
//# sourceMappingURL=fixedIntelligentTradingSystem.js.map
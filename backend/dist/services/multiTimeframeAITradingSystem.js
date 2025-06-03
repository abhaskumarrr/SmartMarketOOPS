"use strict";
/**
 * Multi-Timeframe AI Trading System
 * Implements hierarchical decision making across multiple timeframes
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiTimeframeAITradingSystem = exports.MarketRegime = exports.TradeType = void 0;
exports.createMultiTimeframeAITradingSystem = createMultiTimeframeAITradingSystem;
const multiTimeframeDataProvider_1 = require("./multiTimeframeDataProvider");
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
class MultiTimeframeAITradingSystem {
    constructor() {
        this.name = 'Multi_Timeframe_AI_System';
        this.lastDecisionTime = 0;
        this.supportedTimeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
        this.dataProvider = new multiTimeframeDataProvider_1.MultiTimeframeDataProvider();
        this.parameters = {
            // Multi-timeframe specific parameters
            primaryTimeframe: '1h',
            enabledTimeframes: ['5m', '15m', '1h', '4h', '1d'],
            // Hierarchical decision weights
            timeframeWeights: {
                '1m': 0.05,
                '3m': 0.08,
                '5m': 0.10,
                '15m': 0.15,
                '1h': 0.25,
                '4h': 0.20,
                '1d': 0.17,
            },
            // Consensus requirements
            minConsensusScore: 60, // Require 60% agreement across timeframes
            requireHigherTimeframeSupport: true,
            // Signal thresholds
            minConfidence: 65,
            minTimeframeAgreement: 3, // At least 3 timeframes must agree
            // Risk management
            riskPerTrade: 2,
            stopLossPercent: 1.5,
            takeProfitMultiplier: 2.5,
            // Decision cooldown
            decisionCooldown: 300000, // 5 minutes
        };
        logger_1.logger.info('🕐 Multi-Timeframe AI Trading System initialized', {
            supportedTimeframes: this.supportedTimeframes,
            primaryTimeframe: this.parameters.primaryTimeframe,
        });
    }
    /**
     * Initialize the trading system
     */
    initialize(config) {
        this.config = config;
        this.lastDecisionTime = 0;
        // Validate timeframe relationships
        this.dataProvider.validateTimeframeRelationships();
        logger_1.logger.info(`🎯 Initialized ${this.name} with multi-timeframe analysis`, {
            symbol: config.symbol,
            primaryTimeframe: this.parameters.primaryTimeframe,
            enabledTimeframes: this.parameters.enabledTimeframes,
        });
    }
    /**
     * Generate trading signal using multi-timeframe analysis
     */
    generateSignal(data, currentIndex) {
        if (!this.config) {
            throw new Error('Strategy not initialized. Call initialize() first.');
        }
        // Need enough data for analysis
        if (currentIndex < 100) {
            return null;
        }
        const currentTime = Date.now();
        // Check decision cooldown
        if (currentTime - this.lastDecisionTime < this.parameters.decisionCooldown) {
            return null;
        }
        try {
            // Convert single timeframe data to multi-timeframe format
            const multiTimeframeData = this.prepareMultiTimeframeData(data, currentIndex);
            if (!multiTimeframeData) {
                return null;
            }
            // Analyze each timeframe
            const timeframeAnalyses = this.analyzeAllTimeframes(multiTimeframeData);
            if (timeframeAnalyses.length === 0) {
                return null;
            }
            // Generate multi-timeframe signal
            const multiSignal = this.generateMultiTimeframeSignal(timeframeAnalyses);
            if (!multiSignal || multiSignal.primarySignal === 'HOLD') {
                return null;
            }
            // Apply hierarchical decision making
            const finalSignal = this.applyHierarchicalDecision(multiSignal, data[currentIndex]);
            if (finalSignal && finalSignal.confidence >= this.parameters.minConfidence) {
                this.lastDecisionTime = currentTime;
                logger_1.logger.info(`🎯 Generated multi-timeframe ${finalSignal.type} signal`, {
                    price: finalSignal.price,
                    confidence: finalSignal.confidence,
                    consensusScore: multiSignal.consensusScore,
                    dominantTimeframe: multiSignal.dominantTimeframe,
                    supportingTimeframes: multiSignal.supportingTimeframes,
                });
                return finalSignal;
            }
            return null;
        }
        catch (error) {
            logger_1.logger.error('❌ Error generating multi-timeframe signal:', error);
            return null;
        }
    }
    /**
     * Prepare multi-timeframe data from single timeframe input
     */
    prepareMultiTimeframeData(data, currentIndex) {
        // For this implementation, we'll simulate multi-timeframe data
        // In a real system, this would come from the data provider
        const currentCandle = data[currentIndex];
        const timeframes = {};
        // Add current timeframe data
        timeframes[this.config.timeframe] = currentCandle;
        // Simulate higher timeframe data by aggregating recent candles
        this.parameters.enabledTimeframes.forEach((tf) => {
            if (tf !== this.config.timeframe) {
                const aggregatedCandle = this.simulateHigherTimeframeCandle(data, currentIndex, tf);
                if (aggregatedCandle) {
                    timeframes[tf] = aggregatedCandle;
                }
            }
        });
        return {
            timestamp: currentCandle.timestamp,
            timeframes,
        };
    }
    /**
     * Simulate higher timeframe candle by aggregating recent data
     */
    simulateHigherTimeframeCandle(data, currentIndex, targetTimeframe) {
        const multiplier = this.dataProvider.getTimeframeMultiplier(targetTimeframe);
        const baseMultiplier = this.dataProvider.getTimeframeMultiplier(this.config.timeframe);
        const ratio = multiplier / baseMultiplier;
        if (ratio <= 1 || currentIndex < ratio) {
            return null;
        }
        // Take the last 'ratio' number of candles
        const startIndex = Math.max(0, currentIndex - ratio + 1);
        const candlesToAggregate = data.slice(startIndex, currentIndex + 1);
        if (candlesToAggregate.length === 0) {
            return null;
        }
        // Aggregate OHLCV
        const first = candlesToAggregate[0];
        const last = candlesToAggregate[candlesToAggregate.length - 1];
        const aggregated = {
            ...last,
            timeframe: targetTimeframe,
            open: first.open,
            high: Math.max(...candlesToAggregate.map(c => c.high)),
            low: Math.min(...candlesToAggregate.map(c => c.low)),
            close: last.close,
            volume: candlesToAggregate.reduce((sum, c) => sum + c.volume, 0),
            // Keep the last candle's indicators for simplicity
            indicators: last.indicators,
        };
        return aggregated;
    }
    /**
     * Analyze all enabled timeframes
     */
    analyzeAllTimeframes(multiData) {
        const analyses = [];
        this.parameters.enabledTimeframes.forEach((timeframe) => {
            const candle = multiData.timeframes[timeframe];
            if (candle) {
                const analysis = this.analyzeTimeframe(candle, timeframe);
                if (analysis) {
                    analyses.push(analysis);
                }
            }
        });
        return analyses.sort((a, b) => b.priority - a.priority); // Sort by priority (highest first)
    }
    /**
     * Analyze a specific timeframe
     */
    analyzeTimeframe(candle, timeframe) {
        const indicators = candle.indicators;
        if (!indicators.rsi || !indicators.ema_12 || !indicators.ema_26) {
            return null;
        }
        const priority = this.dataProvider.getTimeframePriority(timeframe);
        // Determine trend
        let trend = 'NEUTRAL';
        let strength = 50;
        const ema12 = indicators.ema_12;
        const ema26 = indicators.ema_26;
        const emaDiff = (ema12 - ema26) / ema26;
        if (emaDiff > 0.01) {
            trend = 'BULLISH';
            strength = Math.min(100, 50 + (emaDiff * 1000));
        }
        else if (emaDiff < -0.01) {
            trend = 'BEARISH';
            strength = Math.min(100, 50 + (Math.abs(emaDiff) * 1000));
        }
        // Determine signal
        let signal = 'HOLD';
        let confidence = 50;
        const rsi = indicators.rsi;
        // RSI-based signals with trend confirmation
        if (rsi < 35 && trend === 'BULLISH') {
            signal = 'BUY';
            confidence = 70 + (35 - rsi);
        }
        else if (rsi > 65 && trend === 'BEARISH') {
            signal = 'SELL';
            confidence = 70 + (rsi - 65);
        }
        else if (trend === 'BULLISH' && rsi < 50) {
            signal = 'BUY';
            confidence = 60;
        }
        else if (trend === 'BEARISH' && rsi > 50) {
            signal = 'SELL';
            confidence = 60;
        }
        // Volume confirmation
        const volumeConfirmation = indicators.volume_sma ?
            candle.volume > indicators.volume_sma * 1.2 : false;
        if (volumeConfirmation) {
            confidence += 10;
        }
        // Determine market regime
        let regime = MarketRegime.SIDEWAYS;
        if (Math.abs(emaDiff) > 0.02) {
            regime = trend === 'BULLISH' ? MarketRegime.TRENDING_BULLISH : MarketRegime.TRENDING_BEARISH;
        }
        return {
            timeframe,
            priority,
            signal,
            confidence: Math.min(100, confidence),
            trend,
            strength,
            regime,
            indicators: {
                rsi,
                macd: indicators.macd || 0,
                ema_trend: emaDiff > 0.005 ? 'UP' : emaDiff < -0.005 ? 'DOWN' : 'FLAT',
                volume_confirmation: volumeConfirmation,
            },
        };
    }
    /**
     * Generate multi-timeframe signal with consensus analysis
     */
    generateMultiTimeframeSignal(analyses) {
        if (analyses.length === 0) {
            return null;
        }
        // Calculate weighted consensus
        let buyScore = 0;
        let sellScore = 0;
        let totalWeight = 0;
        analyses.forEach(analysis => {
            const weight = this.parameters.timeframeWeights[analysis.timeframe] || 0.1;
            const confidenceWeight = analysis.confidence / 100;
            const finalWeight = weight * confidenceWeight;
            if (analysis.signal === 'BUY') {
                buyScore += finalWeight;
            }
            else if (analysis.signal === 'SELL') {
                sellScore += finalWeight;
            }
            totalWeight += weight;
        });
        // Determine primary signal
        let primarySignal = 'HOLD';
        if (buyScore > sellScore && buyScore > totalWeight * 0.3) {
            primarySignal = 'BUY';
        }
        else if (sellScore > buyScore && sellScore > totalWeight * 0.3) {
            primarySignal = 'SELL';
        }
        // Calculate consensus score
        const maxScore = Math.max(buyScore, sellScore);
        const consensusScore = (maxScore / totalWeight) * 100;
        // Find supporting and opposing timeframes
        const supportingTimeframes = [];
        const opposingTimeframes = [];
        analyses.forEach(analysis => {
            if (analysis.signal === primarySignal) {
                supportingTimeframes.push(analysis.timeframe);
            }
            else if (analysis.signal !== 'HOLD' && analysis.signal !== primarySignal) {
                opposingTimeframes.push(analysis.timeframe);
            }
        });
        // Find dominant timeframe (highest priority supporting timeframe)
        const dominantTimeframe = supportingTimeframes.reduce((highest, current) => {
            const currentPriority = this.dataProvider.getTimeframePriority(current);
            const highestPriority = this.dataProvider.getTimeframePriority(highest);
            return currentPriority > highestPriority ? current : highest;
        }, supportingTimeframes[0]);
        // Calculate overall confidence
        const avgConfidence = analyses.reduce((sum, a) => sum + a.confidence, 0) / analyses.length;
        const consensusBonus = consensusScore > 70 ? 15 : consensusScore > 50 ? 10 : 0;
        const finalConfidence = Math.min(100, avgConfidence + consensusBonus);
        return {
            primarySignal,
            confidence: finalConfidence,
            timeframeAnalyses: analyses,
            consensusScore,
            conflictResolution: this.generateConflictResolution(analyses, primarySignal),
            dominantTimeframe,
            supportingTimeframes,
            opposingTimeframes,
        };
    }
    /**
     * Generate conflict resolution explanation
     */
    generateConflictResolution(analyses, primarySignal) {
        const buyCount = analyses.filter(a => a.signal === 'BUY').length;
        const sellCount = analyses.filter(a => a.signal === 'SELL').length;
        const holdCount = analyses.filter(a => a.signal === 'HOLD').length;
        if (buyCount === sellCount && buyCount > 0) {
            return `Timeframe conflict: ${buyCount} BUY vs ${sellCount} SELL. Higher timeframes prioritized.`;
        }
        else if (primarySignal !== 'HOLD') {
            return `Consensus: ${primarySignal} supported by ${primarySignal === 'BUY' ? buyCount : sellCount} timeframes.`;
        }
        else {
            return `No clear consensus: ${buyCount} BUY, ${sellCount} SELL, ${holdCount} HOLD.`;
        }
    }
    /**
     * Apply hierarchical decision making
     */
    applyHierarchicalDecision(multiSignal, currentCandle) {
        // Check minimum consensus requirements
        if (multiSignal.consensusScore < this.parameters.minConsensusScore) {
            return null;
        }
        // Check minimum timeframe agreement
        if (multiSignal.supportingTimeframes.length < this.parameters.minTimeframeAgreement) {
            return null;
        }
        // Check higher timeframe support if required
        if (this.parameters.requireHigherTimeframeSupport) {
            const higherTimeframes = ['4h', '1d'];
            const hasHigherSupport = multiSignal.supportingTimeframes.some(tf => higherTimeframes.includes(tf));
            if (!hasHigherSupport) {
                return null;
            }
        }
        // Create final trading signal
        const signal = {
            id: (0, events_1.createEventId)(),
            timestamp: currentCandle.timestamp,
            symbol: this.config.symbol,
            type: multiSignal.primarySignal,
            price: currentCandle.close,
            quantity: 0,
            confidence: multiSignal.confidence,
            strategy: this.name,
            reason: `Multi-TF: ${multiSignal.conflictResolution} (${multiSignal.consensusScore.toFixed(1)}% consensus)`,
        };
        return signal;
    }
    /**
     * Get strategy description
     */
    getDescription() {
        return 'Multi-Timeframe AI Trading System with hierarchical decision making across multiple timeframes';
    }
}
exports.MultiTimeframeAITradingSystem = MultiTimeframeAITradingSystem;
// Export factory function
function createMultiTimeframeAITradingSystem() {
    return new MultiTimeframeAITradingSystem();
}
//# sourceMappingURL=multiTimeframeAITradingSystem.js.map
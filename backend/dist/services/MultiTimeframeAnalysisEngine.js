"use strict";
/**
 * Multi-Timeframe Analysis Engine
 * Analyzes market data across multiple timeframes for intelligent trading decisions
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiTimeframeAnalysisEngine = void 0;
const logger_1 = require("../utils/logger");
class MultiTimeframeAnalysisEngine {
    constructor(deltaService) {
        this.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
        this.deltaService = deltaService;
    }
    /**
     * Perform comprehensive multi-timeframe analysis
     */
    async analyzeSymbol(symbol) {
        try {
            logger_1.logger.info(`🔍 Starting multi-timeframe analysis for ${symbol}`);
            // Get multi-timeframe data
            const data = await this.deltaService.getMultiTimeframeData(symbol, this.timeframes);
            // Analyze each timeframe
            const trends = {};
            for (const timeframe of this.timeframes) {
                const timeframeData = data.timeframes[timeframe];
                if (timeframeData && timeframeData.candles.length > 0) {
                    trends[timeframe] = this.analyzeTrend(timeframeData.candles, timeframeData.indicators, timeframe);
                }
            }
            // Calculate overall trend and alignment
            const overallTrend = this.calculateOverallTrend(trends);
            // Generate trading signals
            const signals = this.generateTradingSignals(trends, overallTrend, data);
            // Calculate risk metrics
            const riskMetrics = this.calculateRiskMetrics(data);
            const analysis = {
                symbol,
                timestamp: Date.now(),
                trends,
                overallTrend,
                signals,
                riskMetrics
            };
            logger_1.logger.info(`✅ Multi-timeframe analysis completed for ${symbol}`);
            logger_1.logger.info(`📊 Overall trend: ${overallTrend.direction} (${(overallTrend.confidence * 100).toFixed(1)}% confidence)`);
            logger_1.logger.info(`🎯 Signal: ${signals.entry} (${(signals.confidence * 100).toFixed(1)}% confidence)`);
            return analysis;
        }
        catch (error) {
            logger_1.logger.error(`❌ Error analyzing ${symbol}:`, error);
            throw error;
        }
    }
    /**
     * Analyze trend for a specific timeframe
     */
    analyzeTrend(candles, indicators, timeframe) {
        if (candles.length < 10) {
            return {
                direction: 'sideways',
                strength: 0,
                confidence: 0,
                timeframe
            };
        }
        const recentCandles = candles.slice(-20); // Last 20 candles
        const firstPrice = recentCandles[0].close;
        const lastPrice = recentCandles[recentCandles.length - 1].close;
        const priceChange = (lastPrice - firstPrice) / firstPrice;
        // Determine trend direction
        let direction = 'sideways';
        if (priceChange > 0.02)
            direction = 'bullish';
        else if (priceChange < -0.02)
            direction = 'bearish';
        // Calculate trend strength based on price movement and RSI
        let strength = Math.abs(priceChange) * 10; // Base strength from price movement
        if (indicators.rsi) {
            if (direction === 'bullish' && indicators.rsi > 50) {
                strength *= 1.2; // RSI confirms bullish trend
            }
            else if (direction === 'bearish' && indicators.rsi < 50) {
                strength *= 1.2; // RSI confirms bearish trend
            }
            else if (direction !== 'sideways') {
                strength *= 0.8; // RSI diverges from trend
            }
        }
        // MACD confirmation
        if (indicators.macd) {
            if (direction === 'bullish' && indicators.macd.macd > indicators.macd.signal) {
                strength *= 1.1;
            }
            else if (direction === 'bearish' && indicators.macd.macd < indicators.macd.signal) {
                strength *= 1.1;
            }
        }
        strength = Math.min(1, strength); // Cap at 1
        // Calculate confidence based on consistency
        const consistency = this.calculateTrendConsistency(recentCandles);
        const confidence = Math.min(1, strength * consistency);
        return {
            direction,
            strength,
            confidence,
            timeframe
        };
    }
    /**
     * Calculate trend consistency
     */
    calculateTrendConsistency(candles) {
        if (candles.length < 5)
            return 0;
        let bullishCandles = 0;
        let bearishCandles = 0;
        for (const candle of candles) {
            if (candle.close > candle.open) {
                bullishCandles++;
            }
            else if (candle.close < candle.open) {
                bearishCandles++;
            }
        }
        const total = candles.length;
        const dominantCount = Math.max(bullishCandles, bearishCandles);
        return dominantCount / total;
    }
    /**
     * Calculate overall trend from multiple timeframes
     */
    calculateOverallTrend(trends) {
        const timeframeWeights = {
            '1m': 0.05,
            '5m': 0.10,
            '15m': 0.15,
            '1h': 0.25,
            '4h': 0.30,
            '1d': 0.15
        };
        let bullishScore = 0;
        let bearishScore = 0;
        let totalWeight = 0;
        for (const [timeframe, trend] of Object.entries(trends)) {
            const weight = timeframeWeights[timeframe] || 0.1;
            const score = trend.strength * trend.confidence * weight;
            if (trend.direction === 'bullish') {
                bullishScore += score;
            }
            else if (trend.direction === 'bearish') {
                bearishScore += score;
            }
            totalWeight += weight;
        }
        // Normalize scores
        bullishScore /= totalWeight;
        bearishScore /= totalWeight;
        // Determine overall direction
        let direction = 'sideways';
        let strength = 0;
        if (bullishScore > bearishScore && bullishScore > 0.3) {
            direction = 'bullish';
            strength = bullishScore;
        }
        else if (bearishScore > bullishScore && bearishScore > 0.3) {
            direction = 'bearish';
            strength = bearishScore;
        }
        else {
            strength = Math.max(bullishScore, bearishScore);
        }
        // Calculate alignment (how much timeframes agree)
        const alignment = this.calculateTimeframeAlignment(trends);
        // Confidence based on strength and alignment
        const confidence = Math.min(1, strength * (0.5 + alignment * 0.5));
        return {
            direction,
            strength,
            confidence,
            alignment
        };
    }
    /**
     * Calculate how aligned different timeframes are
     */
    calculateTimeframeAlignment(trends) {
        const directions = Object.values(trends).map(t => t.direction);
        const bullishCount = directions.filter(d => d === 'bullish').length;
        const bearishCount = directions.filter(d => d === 'bearish').length;
        const sidewaysCount = directions.filter(d => d === 'sideways').length;
        const total = directions.length;
        const maxCount = Math.max(bullishCount, bearishCount, sidewaysCount);
        // Return alignment score (0 = no alignment, 1 = perfect alignment)
        return maxCount / total;
    }
    /**
     * Generate trading signals based on analysis
     */
    generateTradingSignals(trends, overallTrend, data) {
        const reasoning = [];
        let entry = 'HOLD';
        let confidence = 0;
        // Strong trend signals
        if (overallTrend.confidence > 0.7 && overallTrend.strength > 0.6) {
            if (overallTrend.direction === 'bullish') {
                entry = 'BUY';
                confidence = overallTrend.confidence;
                reasoning.push(`Strong bullish trend across timeframes (${(overallTrend.confidence * 100).toFixed(1)}% confidence)`);
            }
            else if (overallTrend.direction === 'bearish') {
                entry = 'SELL';
                confidence = overallTrend.confidence;
                reasoning.push(`Strong bearish trend across timeframes (${(overallTrend.confidence * 100).toFixed(1)}% confidence)`);
            }
        }
        // High timeframe alignment bonus
        if (overallTrend.alignment > 0.8) {
            confidence *= 1.2;
            reasoning.push(`High timeframe alignment (${(overallTrend.alignment * 100).toFixed(1)}%)`);
        }
        // RSI confirmation
        const hourlyData = data.timeframes['1h'];
        if (hourlyData?.indicators.rsi) {
            const rsi = hourlyData.indicators.rsi;
            if (entry === 'BUY' && rsi < 70) {
                confidence *= 1.1;
                reasoning.push(`RSI not overbought (${rsi.toFixed(1)})`);
            }
            else if (entry === 'SELL' && rsi > 30) {
                confidence *= 1.1;
                reasoning.push(`RSI not oversold (${rsi.toFixed(1)})`);
            }
            else if ((entry === 'BUY' && rsi > 80) || (entry === 'SELL' && rsi < 20)) {
                confidence *= 0.7;
                reasoning.push(`RSI extreme levels warning`);
            }
        }
        confidence = Math.min(1, confidence);
        return {
            entry,
            confidence,
            reasoning
        };
    }
    /**
     * Calculate risk metrics
     */
    calculateRiskMetrics(data) {
        const hourlyData = data.timeframes['1h'];
        const dailyData = data.timeframes['1d'];
        let volatility = 0;
        let atrNormalized = 0;
        let rsiDivergence = false;
        if (hourlyData?.candles && hourlyData.candles.length > 20) {
            // Calculate volatility from recent price movements
            const recentCandles = hourlyData.candles.slice(-20);
            const returns = recentCandles.map((candle, i) => i > 0 ? (candle.close - recentCandles[i - 1].close) / recentCandles[i - 1].close : 0).slice(1);
            volatility = Math.sqrt(returns.reduce((sum, r) => sum + r * r, 0) / returns.length);
        }
        if (hourlyData?.indicators.atr && hourlyData.candles.length > 0) {
            const currentPrice = hourlyData.candles[hourlyData.candles.length - 1].close;
            atrNormalized = hourlyData.indicators.atr / currentPrice;
        }
        // Check for RSI divergence between timeframes
        if (hourlyData?.indicators.rsi && dailyData?.indicators.rsi) {
            const rsiDiff = Math.abs(hourlyData.indicators.rsi - dailyData.indicators.rsi);
            rsiDivergence = rsiDiff > 20; // Significant divergence
        }
        return {
            volatility,
            atrNormalized,
            rsiDivergence
        };
    }
    /**
     * Detect market regime
     */
    detectMarketRegime(analysis) {
        const { overallTrend, riskMetrics } = analysis;
        let type = 'sideways';
        let confidence = 0;
        let duration = 60; // Default 1 hour
        // High volatility regime
        if (riskMetrics.volatility > 0.05) {
            type = 'volatile';
            confidence = Math.min(1, riskMetrics.volatility * 10);
            duration = 30; // Volatile periods tend to be shorter
        }
        // Strong trending regime
        else if (overallTrend.strength > 0.7 && overallTrend.confidence > 0.7) {
            type = overallTrend.direction === 'bullish' ? 'trending_bullish' : 'trending_bearish';
            confidence = overallTrend.confidence;
            duration = 240; // Trends can last longer
        }
        // Breakout detection (high alignment + moderate strength)
        else if (overallTrend.alignment > 0.8 && overallTrend.strength > 0.5) {
            type = 'breakout';
            confidence = overallTrend.alignment;
            duration = 120;
        }
        // Default to sideways
        else {
            type = 'sideways';
            confidence = 1 - overallTrend.strength;
            duration = 180;
        }
        return {
            type,
            confidence,
            duration,
            characteristics: {
                volatility: riskMetrics.volatility,
                trendStrength: overallTrend.strength,
                momentum: overallTrend.confidence
            }
        };
    }
}
exports.MultiTimeframeAnalysisEngine = MultiTimeframeAnalysisEngine;
//# sourceMappingURL=MultiTimeframeAnalysisEngine.js.map
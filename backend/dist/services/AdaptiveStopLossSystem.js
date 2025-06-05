"use strict";
/**
 * Adaptive Stop Loss System
 * Dynamic stop loss calculations using ATR, volatility, and market regime
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AdaptiveStopLossSystem = void 0;
const EnhancedMarketRegimeDetector_1 = require("./EnhancedMarketRegimeDetector");
const MultiTimeframeAnalysisEngine_1 = require("./MultiTimeframeAnalysisEngine");
const logger_1 = require("../utils/logger");
class AdaptiveStopLossSystem {
    constructor(deltaService) {
        this.deltaService = deltaService;
        this.mtfAnalyzer = new MultiTimeframeAnalysisEngine_1.MultiTimeframeAnalysisEngine(deltaService);
        this.regimeDetector = new EnhancedMarketRegimeDetector_1.EnhancedMarketRegimeDetector(deltaService);
        this.defaultConfig = {
            base_atr_multiplier: 2.0,
            volatility_adjustment: true,
            regime_adjustment: true,
            trend_adjustment: true,
            max_stop_distance: 0.05, // 5%
            min_stop_distance: 0.005, // 0.5%
            trailing_enabled: true,
            trailing_step: 0.01 // 1%
        };
    }
    /**
     * Calculate adaptive stop loss for a position
     */
    async calculateStopLoss(position, config = {}) {
        try {
            const finalConfig = { ...this.defaultConfig, ...config };
            logger_1.logger.info(`🎯 Calculating adaptive stop loss for ${position.symbol}`);
            // Get market analysis
            const mtfAnalysis = await this.mtfAnalyzer.analyzeSymbol(position.symbol);
            const regimeAnalysis = await this.regimeDetector.detectRegime(position.symbol);
            const marketData = await this.deltaService.getMultiTimeframeData(position.symbol);
            // Calculate base ATR
            const baseATR = this.calculateBaseATR(marketData);
            // Calculate adjustment factors
            const volatilityFactor = finalConfig.volatility_adjustment
                ? this.calculateVolatilityFactor(regimeAnalysis)
                : 1.0;
            const regimeFactor = finalConfig.regime_adjustment
                ? this.calculateRegimeFactor(regimeAnalysis)
                : 1.0;
            const trendFactor = finalConfig.trend_adjustment
                ? this.calculateTrendFactor(position, mtfAnalysis)
                : 1.0;
            // Calculate final multiplier
            const finalMultiplier = finalConfig.base_atr_multiplier *
                volatilityFactor *
                regimeFactor *
                trendFactor;
            // Calculate stop distance
            const stopDistance = baseATR * finalMultiplier;
            const stopDistancePercent = stopDistance / position.current_price;
            // Apply min/max constraints
            const constrainedDistance = Math.max(finalConfig.min_stop_distance, Math.min(finalConfig.max_stop_distance, stopDistancePercent));
            // Calculate stop price
            const stopPrice = position.side === 'LONG'
                ? position.current_price * (1 - constrainedDistance)
                : position.current_price * (1 + constrainedDistance);
            // Calculate confidence
            const confidence = this.calculateStopLossConfidence(mtfAnalysis, regimeAnalysis, constrainedDistance, finalConfig);
            // Generate reasoning
            const reasoning = this.generateStopLossReasoning(finalConfig, volatilityFactor, regimeFactor, trendFactor, regimeAnalysis);
            // Setup trailing stop if enabled
            const trailingInfo = finalConfig.trailing_enabled
                ? this.setupTrailingStop(position, finalConfig, stopPrice)
                : undefined;
            const adaptiveStopLoss = {
                stop_price: stopPrice,
                distance_percent: constrainedDistance,
                atr_multiplier: finalMultiplier,
                confidence,
                reasoning,
                adjustments: {
                    base_atr: baseATR,
                    volatility_factor: volatilityFactor,
                    regime_factor: regimeFactor,
                    trend_factor: trendFactor,
                    final_multiplier: finalMultiplier
                },
                trailing_info: trailingInfo
            };
            logger_1.logger.info(`✅ Adaptive stop loss calculated: $${stopPrice.toFixed(2)} (${(constrainedDistance * 100).toFixed(2)}%)`);
            logger_1.logger.info(`📊 Confidence: ${(confidence * 100).toFixed(1)}%, Multiplier: ${finalMultiplier.toFixed(2)}x`);
            return adaptiveStopLoss;
        }
        catch (error) {
            logger_1.logger.error(`❌ Error calculating adaptive stop loss for ${position.symbol}:`, error);
            throw error;
        }
    }
    /**
     * Update trailing stop loss
     */
    updateTrailingStop(position, currentStopLoss) {
        if (!currentStopLoss.trailing_info?.enabled) {
            return null;
        }
        const trailingInfo = currentStopLoss.trailing_info;
        const priceMovement = position.current_price - position.entry_price;
        const profitPercent = priceMovement / position.entry_price;
        // Check if we should update trailing stop
        let shouldUpdate = false;
        let newStopPrice = currentStopLoss.stop_price;
        if (position.side === 'LONG' && profitPercent > trailingInfo.step_size) {
            // Long position in profit - trail stop up
            const newStop = position.current_price * (1 - currentStopLoss.distance_percent);
            if (newStop > currentStopLoss.stop_price) {
                newStopPrice = newStop;
                shouldUpdate = true;
            }
        }
        else if (position.side === 'SHORT' && profitPercent < -trailingInfo.step_size) {
            // Short position in profit - trail stop down
            const newStop = position.current_price * (1 + currentStopLoss.distance_percent);
            if (newStop < currentStopLoss.stop_price) {
                newStopPrice = newStop;
                shouldUpdate = true;
            }
        }
        if (shouldUpdate) {
            logger_1.logger.info(`🔄 Trailing stop updated: $${newStopPrice.toFixed(2)} (was $${currentStopLoss.stop_price.toFixed(2)})`);
            return {
                ...currentStopLoss,
                stop_price: newStopPrice,
                trailing_info: {
                    ...trailingInfo,
                    last_update: Date.now()
                }
            };
        }
        return null;
    }
    /**
     * Calculate base ATR from market data
     */
    calculateBaseATR(marketData) {
        const hourlyData = marketData.timeframes['1h'];
        if (hourlyData?.indicators.atr) {
            return hourlyData.indicators.atr;
        }
        // Fallback: calculate simple ATR from candles
        if (hourlyData?.candles && hourlyData.candles.length > 14) {
            const candles = hourlyData.candles.slice(-14);
            let atrSum = 0;
            for (let i = 1; i < candles.length; i++) {
                const high = candles[i].high;
                const low = candles[i].low;
                const prevClose = candles[i - 1].close;
                const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
                atrSum += tr;
            }
            return atrSum / (candles.length - 1);
        }
        // Final fallback: 1% of current price
        const currentPrice = hourlyData?.candles[hourlyData.candles.length - 1]?.close || 0;
        return currentPrice * 0.01;
    }
    /**
     * Calculate volatility adjustment factor
     */
    calculateVolatilityFactor(regimeAnalysis) {
        const volatility = regimeAnalysis.volatility_metrics.price_volatility;
        // High volatility = wider stops (factor > 1)
        // Low volatility = tighter stops (factor < 1)
        if (volatility > 0.05) {
            return 1.5; // High volatility
        }
        else if (volatility > 0.03) {
            return 1.2; // Medium volatility
        }
        else if (volatility > 0.01) {
            return 1.0; // Normal volatility
        }
        else {
            return 0.8; // Low volatility
        }
    }
    /**
     * Calculate regime adjustment factor
     */
    calculateRegimeFactor(regimeAnalysis) {
        const regime = regimeAnalysis.current_regime;
        switch (regime) {
            case 'trending_bullish':
            case 'trending_bearish':
                return 0.8; // Tighter stops in trending markets
            case 'volatile':
                return 1.8; // Much wider stops in volatile markets
            case 'breakout_bullish':
            case 'breakout_bearish':
                return 1.3; // Wider stops for breakouts
            case 'consolidation':
            case 'sideways':
                return 1.1; // Slightly wider stops in ranging markets
            default:
                return 1.0;
        }
    }
    /**
     * Calculate trend alignment factor
     */
    calculateTrendFactor(position, mtfAnalysis) {
        const trendAlignment = mtfAnalysis.overallTrend.alignment;
        const positionDirection = position.side === 'LONG' ? 1 : -1;
        // If position aligns with trend, use tighter stops
        // If position against trend, use wider stops
        const alignment = trendAlignment * positionDirection;
        if (alignment > 0.7) {
            return 0.7; // Strong alignment - tight stops
        }
        else if (alignment > 0.3) {
            return 0.9; // Moderate alignment
        }
        else if (alignment > -0.3) {
            return 1.1; // Neutral
        }
        else {
            return 1.4; // Against trend - wider stops
        }
    }
    /**
     * Calculate stop loss confidence
     */
    calculateStopLossConfidence(mtfAnalysis, regimeAnalysis, stopDistance, config) {
        let confidence = 0.5; // Base confidence
        // MTF analysis confidence
        confidence += mtfAnalysis.overallTrend.confidence * 0.3;
        // Regime confidence
        confidence += regimeAnalysis.confidence * 0.2;
        // Stop distance appropriateness
        const optimalDistance = 0.02; // 2% is often optimal
        const distanceScore = 1 - Math.abs(stopDistance - optimalDistance) / optimalDistance;
        confidence += Math.max(0, distanceScore) * 0.2;
        // Configuration completeness
        const configScore = ((config.volatility_adjustment ? 0.1 : 0) +
            (config.regime_adjustment ? 0.1 : 0) +
            (config.trend_adjustment ? 0.1 : 0));
        confidence += configScore;
        return Math.min(1, confidence);
    }
    /**
     * Generate reasoning for stop loss calculation
     */
    generateStopLossReasoning(config, volatilityFactor, regimeFactor, trendFactor, regimeAnalysis) {
        const reasoning = [];
        reasoning.push(`Base ATR multiplier: ${config.base_atr_multiplier}x`);
        if (config.volatility_adjustment) {
            reasoning.push(`Volatility adjustment: ${volatilityFactor.toFixed(2)}x (${regimeAnalysis.volatility_metrics.price_volatility > 0.03 ? 'high' : 'normal'} volatility)`);
        }
        if (config.regime_adjustment) {
            reasoning.push(`Regime adjustment: ${regimeFactor.toFixed(2)}x (${regimeAnalysis.current_regime} market)`);
        }
        if (config.trend_adjustment) {
            reasoning.push(`Trend adjustment: ${trendFactor.toFixed(2)}x (trend alignment factor)`);
        }
        if (config.trailing_enabled) {
            reasoning.push(`Trailing stop enabled with ${(config.trailing_step * 100).toFixed(1)}% step`);
        }
        return reasoning;
    }
    /**
     * Setup trailing stop configuration
     */
    setupTrailingStop(position, config, initialStopPrice) {
        return {
            enabled: true,
            trigger_price: position.current_price,
            step_size: config.trailing_step,
            last_update: Date.now()
        };
    }
}
exports.AdaptiveStopLossSystem = AdaptiveStopLossSystem;
//# sourceMappingURL=AdaptiveStopLossSystem.js.map
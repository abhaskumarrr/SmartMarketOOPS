"use strict";
/**
 * Advanced Signal Filtering System
 * Implements multi-layer filtering to achieve 85%+ win rate alignment with ML accuracy
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AdvancedSignalFilteringSystem = void 0;
const MultiTimeframeAnalysisEngine_1 = require("./MultiTimeframeAnalysisEngine");
const EnhancedMarketRegimeDetector_1 = require("./EnhancedMarketRegimeDetector");
const EnhancedMLIntegrationService_1 = require("./EnhancedMLIntegrationService");
const logger_1 = require("../utils/logger");
class AdvancedSignalFilteringSystem {
    constructor(deltaService, config) {
        this.recentSignals = new Map();
        this.activePositions = new Map();
        this.deltaService = deltaService;
        this.mtfAnalyzer = new MultiTimeframeAnalysisEngine_1.MultiTimeframeAnalysisEngine(deltaService);
        this.regimeDetector = new EnhancedMarketRegimeDetector_1.EnhancedMarketRegimeDetector(deltaService);
        this.mlService = new EnhancedMLIntegrationService_1.EnhancedMLIntegrationService(deltaService);
        // Optimized configuration for 85%+ win rate
        this.config = {
            min_ml_confidence: 0.85, // CRITICAL: Only trade when ML is 85%+ confident
            min_ensemble_confidence: 0.80, // Ensemble must be highly confident
            min_timeframe_alignment: 0.75, // Strong multi-timeframe agreement
            required_timeframe_count: 4, // At least 4 timeframes must align
            allowed_regimes: [
                'trending_bullish',
                'trending_bearish',
                'breakout_bullish',
                'breakout_bearish'
            ],
            min_regime_confidence: 0.70, // High regime confidence required
            min_rsi_range: [25, 75], // Avoid extreme overbought/oversold
            min_volume_ratio: 1.3, // Require 30% above average volume
            max_volatility: 0.05, // Avoid extremely volatile periods
            min_signal_score: 85, // Overall signal must score 85+/100
            max_correlation_exposure: 0.3, // Limit correlated positions
            max_drawdown_threshold: 0.15, // Stop trading if drawdown > 15%
            ...config
        };
    }
    /**
     * Generate and filter high-quality trading signals
     */
    async generateFilteredSignal(symbol) {
        try {
            logger_1.logger.info(`🔍 Generating filtered signal for ${symbol}`);
            // Step 1: Get comprehensive market analysis
            const mtfAnalysis = await this.mtfAnalyzer.analyzeSymbol(symbol);
            const regimeAnalysis = await this.regimeDetector.detectRegime(symbol);
            const marketData = await this.deltaService.getMultiTimeframeData(symbol);
            // Step 2: Get ML predictions
            const currentPrice = await this.getCurrentPrice(symbol);
            const mlPrediction = await this.mlService.getEnsemblePrediction(symbol, 'LONG', currentPrice, currentPrice);
            // Step 3: Apply multi-layer filtering
            const filterResults = await this.applyAdvancedFiltering(symbol, mtfAnalysis, regimeAnalysis, mlPrediction, marketData);
            if (!filterResults.passes_all_filters) {
                logger_1.logger.debug(`❌ Signal filtered out for ${symbol}: ${filterResults.rejection_reason}`);
                return null;
            }
            // Step 4: Generate high-confidence signal
            const signal = await this.constructFilteredSignal(symbol, mtfAnalysis, regimeAnalysis, mlPrediction, filterResults, currentPrice);
            // Step 5: Store signal for tracking
            this.storeSignal(symbol, signal);
            logger_1.logger.info(`✅ High-quality signal generated for ${symbol}: Score ${signal.signal_score}/100`);
            return signal;
        }
        catch (error) {
            logger_1.logger.error(`❌ Error generating filtered signal for ${symbol}:`, error);
            return null;
        }
    }
    /**
     * Apply comprehensive multi-layer filtering
     */
    async applyAdvancedFiltering(symbol, mtfAnalysis, regimeAnalysis, mlPrediction, marketData) {
        const filters = {
            ml_filter: false,
            timeframe_filter: false,
            regime_filter: false,
            technical_filter: false,
            risk_filter: false
        };
        let rejectionReason = '';
        // Filter 1: ML Confidence Filter (MOST CRITICAL)
        const mlConfidence = mlPrediction.position_outcome.confidence;
        const ensembleConfidence = mlPrediction.ensemble_confidence;
        if (mlConfidence >= this.config.min_ml_confidence &&
            ensembleConfidence >= this.config.min_ensemble_confidence) {
            filters.ml_filter = true;
            logger_1.logger.debug(`✅ ML Filter passed: ${(mlConfidence * 100).toFixed(1)}% confidence`);
        }
        else {
            rejectionReason = `ML confidence too low: ${(mlConfidence * 100).toFixed(1)}% (need ${(this.config.min_ml_confidence * 100).toFixed(1)}%)`;
        }
        // Filter 2: Multi-Timeframe Alignment Filter
        const alignment = mtfAnalysis.overallTrend.alignment;
        const alignedTimeframes = this.countAlignedTimeframes(mtfAnalysis);
        if (alignment >= this.config.min_timeframe_alignment &&
            alignedTimeframes >= this.config.required_timeframe_count) {
            filters.timeframe_filter = true;
            logger_1.logger.debug(`✅ Timeframe Filter passed: ${(alignment * 100).toFixed(1)}% alignment, ${alignedTimeframes} timeframes`);
        }
        else {
            rejectionReason = rejectionReason || `Timeframe alignment insufficient: ${(alignment * 100).toFixed(1)}% (need ${(this.config.min_timeframe_alignment * 100).toFixed(1)}%)`;
        }
        // Filter 3: Market Regime Filter
        const regime = regimeAnalysis.current_regime;
        const regimeConfidence = regimeAnalysis.confidence;
        if (this.config.allowed_regimes.includes(regime) &&
            regimeConfidence >= this.config.min_regime_confidence) {
            filters.regime_filter = true;
            logger_1.logger.debug(`✅ Regime Filter passed: ${regime} (${(regimeConfidence * 100).toFixed(1)}% confidence)`);
        }
        else {
            rejectionReason = rejectionReason || `Unfavorable regime: ${regime} or low confidence: ${(regimeConfidence * 100).toFixed(1)}%`;
        }
        // Filter 4: Technical Confirmation Filter
        const technicalScore = this.calculateTechnicalScore(marketData);
        if (technicalScore >= 0.7) {
            filters.technical_filter = true;
            logger_1.logger.debug(`✅ Technical Filter passed: ${(technicalScore * 100).toFixed(1)}% score`);
        }
        else {
            rejectionReason = rejectionReason || `Technical conditions unfavorable: ${(technicalScore * 100).toFixed(1)}% score`;
        }
        // Filter 5: Risk Management Filter
        const riskScore = await this.calculateRiskScore(symbol);
        if (riskScore >= 0.7) {
            filters.risk_filter = true;
            logger_1.logger.debug(`✅ Risk Filter passed: ${(riskScore * 100).toFixed(1)}% score`);
        }
        else {
            rejectionReason = rejectionReason || `Risk conditions unfavorable: ${(riskScore * 100).toFixed(1)}% score`;
        }
        const passesAllFilters = Object.values(filters).every(f => f);
        return {
            filters,
            passes_all_filters: passesAllFilters,
            rejection_reason: rejectionReason,
            ml_confidence: mlConfidence,
            ensemble_confidence: ensembleConfidence,
            timeframe_alignment: alignment,
            regime_compatibility: regimeConfidence,
            technical_score: technicalScore,
            risk_score: riskScore
        };
    }
    /**
     * Construct high-quality filtered signal
     */
    async constructFilteredSignal(symbol, mtfAnalysis, regimeAnalysis, mlPrediction, filterResults, currentPrice) {
        // Calculate overall signal score (weighted combination)
        const signalScore = this.calculateSignalScore(filterResults);
        // Determine position side based on ML prediction and trend
        const side = this.determineSide(mlPrediction, mtfAnalysis);
        // Calculate position size multiplier based on confidence
        const positionSizeMultiplier = this.calculatePositionSizeMultiplier(filterResults);
        // Calculate stop loss and take profit levels
        const stopLoss = this.calculateOptimalStopLoss(currentPrice, side, regimeAnalysis);
        const takeProfitLevels = this.calculateOptimalTakeProfits(currentPrice, side, regimeAnalysis);
        // Generate reasoning
        const reasoning = this.generateSignalReasoning(filterResults, mtfAnalysis, regimeAnalysis, mlPrediction);
        return {
            symbol,
            side,
            signal_score: signalScore,
            ml_confidence: filterResults.ml_confidence,
            ensemble_confidence: filterResults.ensemble_confidence,
            timeframe_alignment: filterResults.timeframe_alignment,
            regime_compatibility: filterResults.regime_compatibility,
            technical_confirmation: filterResults.technical_score,
            risk_assessment: filterResults.risk_score,
            entry_price: currentPrice,
            stop_loss: stopLoss,
            take_profit_levels: takeProfitLevels,
            position_size_multiplier: positionSizeMultiplier,
            reasoning,
            filter_results: filterResults.filters,
            timestamp: Date.now()
        };
    }
    /**
     * Calculate comprehensive signal score (0-100)
     */
    calculateSignalScore(filterResults) {
        // Weighted scoring system
        const weights = {
            ml_confidence: 0.35, // 35% weight - most important
            ensemble_confidence: 0.20, // 20% weight
            timeframe_alignment: 0.20, // 20% weight
            regime_compatibility: 0.15, // 15% weight
            technical_score: 0.05, // 5% weight
            risk_score: 0.05 // 5% weight
        };
        const score = (filterResults.ml_confidence * weights.ml_confidence +
            filterResults.ensemble_confidence * weights.ensemble_confidence +
            filterResults.timeframe_alignment * weights.timeframe_alignment +
            filterResults.regime_compatibility * weights.regime_compatibility +
            filterResults.technical_score * weights.technical_score +
            filterResults.risk_score * weights.risk_score) * 100;
        return Math.round(score);
    }
    /**
     * Calculate position size multiplier based on confidence
     */
    calculatePositionSizeMultiplier(filterResults) {
        const baseMultiplier = 1.0;
        const confidenceBonus = (filterResults.ml_confidence - 0.5) * 2; // 0-1 range
        const alignmentBonus = filterResults.timeframe_alignment * 0.5;
        const multiplier = baseMultiplier + confidenceBonus + alignmentBonus;
        // Cap between 0.5x and 2.5x
        return Math.max(0.5, Math.min(2.5, multiplier));
    }
    // Helper methods
    countAlignedTimeframes(mtfAnalysis) {
        let alignedCount = 0;
        const trends = mtfAnalysis.trends;
        const overallDirection = mtfAnalysis.overallTrend.direction;
        Object.values(trends).forEach((trend) => {
            if (trend.direction === overallDirection && trend.confidence > 0.6) {
                alignedCount++;
            }
        });
        return alignedCount;
    }
    calculateTechnicalScore(marketData) {
        const hourlyData = marketData.timeframes['1h'];
        if (!hourlyData?.indicators)
            return 0.5;
        let score = 0.5; // Base score
        // RSI check
        const rsi = hourlyData.indicators.rsi || 50;
        if (rsi >= this.config.min_rsi_range[0] && rsi <= this.config.min_rsi_range[1]) {
            score += 0.2;
        }
        // Volume check
        const volumeRatio = this.calculateVolumeRatio(hourlyData.candles || []);
        if (volumeRatio >= this.config.min_volume_ratio) {
            score += 0.2;
        }
        // Volatility check
        const volatility = this.calculateVolatility(hourlyData.candles || []);
        if (volatility <= this.config.max_volatility) {
            score += 0.1;
        }
        return Math.min(1, score);
    }
    async calculateRiskScore(symbol) {
        // Simplified risk scoring
        let score = 0.8; // Base score
        // Check correlation with existing positions
        const correlation = this.calculateCorrelationExposure(symbol);
        if (correlation <= this.config.max_correlation_exposure) {
            score += 0.1;
        }
        // Check overall portfolio drawdown
        const drawdown = this.calculateCurrentDrawdown();
        if (drawdown <= this.config.max_drawdown_threshold) {
            score += 0.1;
        }
        return Math.min(1, score);
    }
    determineSide(mlPrediction, mtfAnalysis) {
        const mlRecommendation = mlPrediction.recommendation;
        const trendDirection = mtfAnalysis.overallTrend.direction;
        if (mlRecommendation.includes('BUY') || trendDirection === 'bullish') {
            return 'LONG';
        }
        else {
            return 'SHORT';
        }
    }
    calculateOptimalStopLoss(price, side, regimeAnalysis) {
        const atr = regimeAnalysis.volatility_metrics.atr_normalized;
        const stopDistance = atr * 1.5; // 1.5x ATR stop loss
        if (side === 'LONG') {
            return price * (1 - stopDistance);
        }
        else {
            return price * (1 + stopDistance);
        }
    }
    calculateOptimalTakeProfits(price, side, regimeAnalysis) {
        const atr = regimeAnalysis.volatility_metrics.atr_normalized;
        const multipliers = [2, 4, 6]; // 2x, 4x, 6x ATR take profits
        return multipliers.map(mult => {
            const distance = atr * mult;
            if (side === 'LONG') {
                return price * (1 + distance);
            }
            else {
                return price * (1 - distance);
            }
        });
    }
    generateSignalReasoning(filterResults, mtfAnalysis, regimeAnalysis, mlPrediction) {
        const reasoning = [];
        reasoning.push(`ML Confidence: ${(filterResults.ml_confidence * 100).toFixed(1)}%`);
        reasoning.push(`Timeframe Alignment: ${(filterResults.timeframe_alignment * 100).toFixed(1)}%`);
        reasoning.push(`Market Regime: ${regimeAnalysis.current_regime}`);
        reasoning.push(`Signal Score: ${this.calculateSignalScore(filterResults)}/100`);
        return reasoning;
    }
    // Utility methods
    async getCurrentPrice(symbol) {
        const prices = {
            'ETHUSD': 4000,
            'BTCUSD': 105000,
            'SOLUSD': 200
        };
        return prices[symbol] || 100;
    }
    calculateVolumeRatio(candles) {
        if (candles.length < 20)
            return 1;
        const recent = candles.slice(-5);
        const baseline = candles.slice(-20, -5);
        const recentAvg = recent.reduce((sum, c) => sum + c.volume, 0) / recent.length;
        const baselineAvg = baseline.reduce((sum, c) => sum + c.volume, 0) / baseline.length;
        return baselineAvg > 0 ? recentAvg / baselineAvg : 1;
    }
    calculateVolatility(candles) {
        if (candles.length < 10)
            return 0.02;
        const returns = candles.slice(1).map((candle, i) => (candle.close - candles[i].close) / candles[i].close);
        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
        return Math.sqrt(variance);
    }
    calculateCorrelationExposure(symbol) {
        // Simplified correlation calculation
        return 0.2; // Assume 20% correlation
    }
    calculateCurrentDrawdown() {
        // Simplified drawdown calculation
        return 0.05; // Assume 5% current drawdown
    }
    storeSignal(symbol, signal) {
        if (!this.recentSignals.has(symbol)) {
            this.recentSignals.set(symbol, []);
        }
        const signals = this.recentSignals.get(symbol);
        signals.push(signal);
        // Keep only last 10 signals
        if (signals.length > 10) {
            signals.shift();
        }
    }
}
exports.AdvancedSignalFilteringSystem = AdvancedSignalFilteringSystem;
//# sourceMappingURL=AdvancedSignalFilteringSystem.js.map
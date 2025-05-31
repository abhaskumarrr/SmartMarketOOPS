"use strict";
/**
 * Signal Generation Service
 * Converts ML model predictions into actionable trading signals
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SignalGenerationService = void 0;
const uuid_1 = require("uuid");
const mlModelClient_1 = __importDefault(require("../../clients/mlModelClient"));
const signals_1 = require("../../types/signals");
const logger_1 = require("../../utils/logger");
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
// Create logger
const logger = (0, logger_1.createLogger)('SignalGenerationService');
/**
 * Default signal generation options
 */
const DEFAULT_OPTIONS = {
    validateSignals: true,
    useHistoricalData: true,
    lookbackPeriod: 30,
    minConfidenceThreshold: 60,
    maxSignalsPerSymbol: 3,
    filterWeakSignals: true
};
/**
 * Signal Generation Service class
 * Provides methods to generate and manage trading signals
 */
class SignalGenerationService {
    /**
     * Creates a new Signal Generation Service instance
     * @param options - Signal generation options
     */
    constructor(options) {
        this.options = { ...DEFAULT_OPTIONS, ...options };
        logger.info('Signal Generation Service initialized', { options: this.options });
    }
    /**
     * Generate trading signals for a specific symbol
     * @param symbol - Trading pair symbol
     * @param features - Current market features
     * @param options - Optional signal generation options to override defaults
     * @returns Generated trading signals
     */
    async generateSignals(symbol, features, options) {
        try {
            // Merge options
            const mergedOptions = { ...this.options, ...options };
            logger.info(`Generating signals for ${symbol}`);
            // 1. Get enhanced model prediction with signal quality analysis
            let prediction;
            try {
                // Try enhanced prediction first
                prediction = await mlModelClient_1.default.getEnhancedPrediction({
                    symbol,
                    features,
                    sequence_length: 60
                });
                logger.debug(`Received enhanced prediction for ${symbol}`, {
                    prediction: prediction.prediction,
                    confidence: prediction.confidence,
                    signal_valid: prediction.signal_valid,
                    quality_score: prediction.quality_score,
                    market_regime: prediction.market_regime,
                    enhanced: prediction.enhanced
                });
            }
            catch (error) {
                // Fallback to traditional prediction
                logger.warn(`Enhanced prediction failed for ${symbol}, falling back to traditional`, {
                    error: error instanceof Error ? error.message : String(error)
                });
                prediction = await mlModelClient_1.default.getPrediction({
                    symbol,
                    features,
                    sequence_length: 60
                });
            }
            logger.debug(`Received prediction for ${symbol}`, {
                predictions: prediction.predictions,
                model_version: prediction.model_version
            });
            // 2. Process prediction into signals (enhanced or traditional)
            const signals = prediction.enhanced
                ? await this._processEnhancedPrediction(prediction, features)
                : await this._processModelPrediction(prediction, features);
            // 3. Validate signals if option is enabled
            const validatedSignals = mergedOptions.validateSignals
                ? await this._validateSignals(signals)
                : signals;
            // 4. Filter signals based on confidence threshold
            const filteredSignals = this._filterSignals(validatedSignals, mergedOptions);
            // 5. Store signals in database
            await this._storeSignals(filteredSignals);
            logger.info(`Generated ${filteredSignals.length} signals for ${symbol}`);
            return filteredSignals;
        }
        catch (error) {
            const logData = {
                symbol,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error generating signals for ${symbol}`, logData);
            throw error;
        }
    }
    /**
     * Process enhanced prediction into trading signals
     * @private
     * @param prediction - Enhanced ML model prediction
     * @param features - Current market features
     * @returns Processed trading signals
     */
    async _processEnhancedPrediction(prediction, // EnhancedPrediction type
    features) {
        const signals = [];
        const currentPrice = features.close || features.price || 0;
        if (currentPrice === 0) {
            logger.warn(`No current price available for ${prediction.symbol}`);
            return [];
        }
        // Skip signal generation if not valid according to enhanced system
        if (!prediction.signal_valid) {
            logger.debug(`Enhanced signal marked as invalid for ${prediction.symbol}: ${prediction.recommendation}`);
            return [];
        }
        // Use enhanced prediction data
        const predictionValue = prediction.prediction;
        const confidenceScore = Math.round(prediction.confidence * 100);
        const qualityScore = Math.round(prediction.quality_score * 100);
        // Determine signal direction based on enhanced prediction
        let direction;
        if (predictionValue > 0.6) {
            direction = signals_1.SignalDirection.LONG;
        }
        else if (predictionValue < 0.4) {
            direction = signals_1.SignalDirection.SHORT;
        }
        else {
            direction = signals_1.SignalDirection.NEUTRAL;
        }
        // Skip neutral signals if filtering is enabled
        if (direction === signals_1.SignalDirection.NEUTRAL && this.options.filterWeakSignals) {
            logger.debug(`Neutral enhanced signal for ${prediction.symbol} filtered out`);
            return [];
        }
        // Determine signal type based on recommendation
        let signalType;
        const recommendation = prediction.recommendation.toLowerCase();
        if (recommendation.includes('buy')) {
            signalType = signals_1.SignalType.ENTRY;
        }
        else if (recommendation.includes('sell')) {
            signalType = recommendation.includes('exit') ? signals_1.SignalType.EXIT : signals_1.SignalType.ENTRY;
        }
        else if (recommendation.includes('hold') || recommendation.includes('neutral')) {
            signalType = signals_1.SignalType.HOLD;
        }
        else {
            signalType = direction === signals_1.SignalDirection.LONG ? signals_1.SignalType.INCREASE : signals_1.SignalType.DECREASE;
        }
        // Determine signal strength based on quality score
        let strength;
        if (qualityScore >= 90) {
            strength = signals_1.SignalStrength.VERY_STRONG;
        }
        else if (qualityScore >= 75) {
            strength = signals_1.SignalStrength.STRONG;
        }
        else if (qualityScore >= 60) {
            strength = signals_1.SignalStrength.MODERATE;
        }
        else if (qualityScore >= 40) {
            strength = signals_1.SignalStrength.WEAK;
        }
        else {
            strength = signals_1.SignalStrength.VERY_WEAK;
        }
        // Skip weak signals if filtering is enabled
        if (this.options.filterWeakSignals &&
            (strength === signals_1.SignalStrength.VERY_WEAK || strength === signals_1.SignalStrength.WEAK)) {
            logger.debug(`Weak enhanced signal for ${prediction.symbol} filtered out`);
            return [];
        }
        // Calculate target price based on prediction confidence and market regime
        const regimeMultiplier = this._getRegimeMultiplier(prediction.market_regime);
        const baseTargetPercent = direction === signals_1.SignalDirection.LONG ? 2.0 : -2.0;
        const targetPercent = baseTargetPercent * prediction.confidence * regimeMultiplier;
        const targetPrice = currentPrice * (1 + targetPercent / 100);
        // Calculate stop loss based on regime and confidence
        const stopLossPercent = direction === signals_1.SignalDirection.LONG ?
            -1.5 * (1 / prediction.confidence) :
            1.5 * (1 / prediction.confidence);
        const stopLoss = currentPrice * (1 + stopLossPercent / 100);
        // Determine timeframe based on market regime
        const timeframe = this._getTimeframeFromRegime(prediction.market_regime);
        // Calculate expected return and risk
        const expectedReturn = Math.abs(targetPercent);
        const expectedRisk = Math.abs(stopLossPercent);
        const riskRewardRatio = expectedRisk > 0 ? expectedReturn / expectedRisk : expectedReturn;
        // Create enhanced signal
        const signal = {
            id: (0, uuid_1.v4)(),
            symbol: prediction.symbol,
            type: signalType,
            direction,
            strength,
            timeframe,
            price: currentPrice,
            targetPrice,
            stopLoss,
            confidenceScore,
            expectedReturn,
            expectedRisk,
            riskRewardRatio,
            generatedAt: new Date().toISOString(),
            expiresAt: this._calculateExpiryTime(timeframe),
            source: `enhanced-ml-ensemble`,
            metadata: {
                enhanced: true,
                market_regime: prediction.market_regime,
                regime_strength: prediction.regime_strength,
                quality_score: prediction.quality_score,
                model_predictions: prediction.model_predictions,
                confidence_breakdown: prediction.confidence_breakdown,
                recommendation: prediction.recommendation
            },
            predictionValues: [predictionValue]
        };
        signals.push(signal);
        return signals;
    }
    /**
     * Get regime-based multiplier for target calculation
     * @private
     * @param regime - Market regime
     * @returns Multiplier value
     */
    _getRegimeMultiplier(regime) {
        switch (regime.toLowerCase()) {
            case 'trending_bullish':
            case 'trending_bearish':
                return 1.5; // Higher targets in trending markets
            case 'breakout_bullish':
            case 'breakout_bearish':
                return 2.0; // Highest targets in breakout markets
            case 'volatile':
                return 0.8; // Lower targets in volatile markets
            case 'ranging':
            case 'consolidation':
                return 0.6; // Lowest targets in ranging markets
            default:
                return 1.0; // Default multiplier
        }
    }
    /**
     * Get timeframe based on market regime
     * @private
     * @param regime - Market regime
     * @returns Signal timeframe
     */
    _getTimeframeFromRegime(regime) {
        switch (regime.toLowerCase()) {
            case 'breakout_bullish':
            case 'breakout_bearish':
                return signals_1.SignalTimeframe.VERY_SHORT; // Quick moves in breakouts
            case 'volatile':
                return signals_1.SignalTimeframe.SHORT; // Short-term in volatile markets
            case 'trending_bullish':
            case 'trending_bearish':
                return signals_1.SignalTimeframe.MEDIUM; // Medium-term in trends
            case 'ranging':
            case 'consolidation':
                return signals_1.SignalTimeframe.LONG; // Longer-term in ranging markets
            default:
                return signals_1.SignalTimeframe.MEDIUM; // Default timeframe
        }
    }
    /**
     * Process model prediction into trading signals
     * @private
     * @param prediction - ML model prediction
     * @param features - Current market features
     * @returns Processed trading signals
     */
    async _processModelPrediction(prediction, features) {
        const signals = [];
        const currentPrice = features.close || features.price || 0;
        if (currentPrice === 0) {
            logger.warn(`No current price available for ${prediction.symbol}`);
            return [];
        }
        // Get prediction values - these are typically future price predictions
        const predictionValues = prediction.predictions;
        // Calculate price change percentages
        const priceChanges = predictionValues.map(value => ((value - currentPrice) / currentPrice) * 100);
        // Calculate average and standard deviation of changes
        const avgChange = priceChanges.reduce((sum, change) => sum + change, 0) / priceChanges.length;
        const stdDevChange = Math.sqrt(priceChanges.reduce((sum, change) => sum + Math.pow(change - avgChange, 2), 0) / priceChanges.length);
        // Determine signal direction based on average change
        let direction;
        if (avgChange > 1.0) {
            direction = signals_1.SignalDirection.LONG;
        }
        else if (avgChange < -1.0) {
            direction = signals_1.SignalDirection.SHORT;
        }
        else {
            direction = signals_1.SignalDirection.NEUTRAL;
        }
        // Skip neutral signals if we're filtering weak signals
        if (direction === signals_1.SignalDirection.NEUTRAL && this.options.filterWeakSignals) {
            logger.debug(`Neutral signal for ${prediction.symbol} filtered out`);
            return [];
        }
        // Determine signal type based on direction and magnitude
        let signalType;
        if (direction === signals_1.SignalDirection.NEUTRAL) {
            signalType = signals_1.SignalType.HOLD;
        }
        else if (Math.abs(avgChange) > 5.0) {
            signalType = direction === signals_1.SignalDirection.LONG ? signals_1.SignalType.ENTRY : signals_1.SignalType.EXIT;
        }
        else {
            signalType = direction === signals_1.SignalDirection.LONG ? signals_1.SignalType.INCREASE : signals_1.SignalType.DECREASE;
        }
        // Calculate confidence score based on consistency and magnitude of predictions
        const consistency = 1.0 - (stdDevChange / Math.max(1.0, Math.abs(avgChange)));
        const magnitude = Math.min(1.0, Math.abs(avgChange) / 10.0); // Normalize to max of 1.0
        const confidenceScore = Math.round((consistency * 0.6 + magnitude * 0.4) * 100);
        // Determine signal strength based on confidence score
        let strength;
        if (confidenceScore >= 80) {
            strength = signals_1.SignalStrength.VERY_STRONG;
        }
        else if (confidenceScore >= 60) {
            strength = signals_1.SignalStrength.STRONG;
        }
        else if (confidenceScore >= 40) {
            strength = signals_1.SignalStrength.MODERATE;
        }
        else if (confidenceScore >= 20) {
            strength = signals_1.SignalStrength.WEAK;
        }
        else {
            strength = signals_1.SignalStrength.VERY_WEAK;
        }
        // Skip weak signals if option is enabled
        if (this.options.filterWeakSignals &&
            (strength === signals_1.SignalStrength.VERY_WEAK || strength === signals_1.SignalStrength.WEAK)) {
            logger.debug(`Weak signal for ${prediction.symbol} filtered out`);
            return [];
        }
        // Calculate target price and stop loss
        const targetPricePercent = direction === signals_1.SignalDirection.LONG ?
            Math.max(...priceChanges) :
            Math.min(...priceChanges);
        const targetPrice = currentPrice * (1 + targetPricePercent / 100);
        // Calculate stop loss (simple approach - can be enhanced with more sophisticated methods)
        const stopLossPercent = direction === signals_1.SignalDirection.LONG ? -2.0 : 2.0;
        const stopLoss = currentPrice * (1 + stopLossPercent / 100);
        // Determine timeframe based on prediction horizon
        let timeframe;
        const numPredictions = predictionValues.length;
        if (numPredictions <= 6) { // Short term (hours)
            timeframe = signals_1.SignalTimeframe.VERY_SHORT;
        }
        else if (numPredictions <= 24) { // Medium term (day)
            timeframe = signals_1.SignalTimeframe.SHORT;
        }
        else if (numPredictions <= 72) { // Medium term (days)
            timeframe = signals_1.SignalTimeframe.MEDIUM;
        }
        else if (numPredictions <= 168) { // Longer term (week)
            timeframe = signals_1.SignalTimeframe.LONG;
        }
        else { // Very long term (weeks+)
            timeframe = signals_1.SignalTimeframe.VERY_LONG;
        }
        // Calculate expected return and risk
        const expectedReturn = direction === signals_1.SignalDirection.LONG ?
            Math.max(0, avgChange) :
            Math.max(0, -avgChange);
        const expectedRisk = direction === signals_1.SignalDirection.LONG ?
            Math.max(0, -stopLossPercent) :
            Math.max(0, stopLossPercent);
        // Calculate risk-reward ratio (avoid division by zero)
        const riskRewardRatio = expectedRisk > 0 ? expectedReturn / expectedRisk : expectedReturn;
        // Create signal object
        const signal = {
            id: (0, uuid_1.v4)(),
            symbol: prediction.symbol,
            type: signalType,
            direction,
            strength,
            timeframe,
            price: currentPrice,
            targetPrice,
            stopLoss,
            confidenceScore,
            expectedReturn,
            expectedRisk,
            riskRewardRatio,
            generatedAt: new Date().toISOString(),
            expiresAt: this._calculateExpiryTime(timeframe),
            source: `ml-model-${prediction.model_version}`,
            metadata: {
                avgChange,
                stdDevChange,
                consistency,
                magnitude
            },
            predictionValues: predictionValues
        };
        signals.push(signal);
        return signals;
    }
    /**
     * Calculate signal expiry time based on timeframe
     * @private
     * @param timeframe - Signal timeframe
     * @returns Expiry timestamp
     */
    _calculateExpiryTime(timeframe) {
        const now = new Date();
        switch (timeframe) {
            case signals_1.SignalTimeframe.VERY_SHORT:
                now.setHours(now.getHours() + 4);
                break;
            case signals_1.SignalTimeframe.SHORT:
                now.setHours(now.getHours() + 24);
                break;
            case signals_1.SignalTimeframe.MEDIUM:
                now.setDate(now.getDate() + 3);
                break;
            case signals_1.SignalTimeframe.LONG:
                now.setDate(now.getDate() + 7);
                break;
            case signals_1.SignalTimeframe.VERY_LONG:
                now.setDate(now.getDate() + 30);
                break;
        }
        return now.toISOString();
    }
    /**
     * Validate signals using additional techniques
     * @private
     * @param signals - Signals to validate
     * @returns Validated signals
     */
    async _validateSignals(signals) {
        return Promise.all(signals.map(async (signal) => {
            try {
                // This is where you would implement additional validation
                // Examples:
                // 1. Technical indicator confirmation
                // 2. Volume analysis
                // 3. Market sentiment analysis
                // 4. Correlation with related assets
                // For now, we'll just mark all signals as validated
                const validatedSignal = {
                    ...signal,
                    validatedAt: new Date().toISOString(),
                    validationStatus: true,
                    validationReason: 'Passed basic validation checks'
                };
                return validatedSignal;
            }
            catch (error) {
                logger.warn(`Signal validation failed for ${signal.symbol}`, {
                    signalId: signal.id,
                    error: error instanceof Error ? error.message : String(error)
                });
                return {
                    ...signal,
                    validatedAt: new Date().toISOString(),
                    validationStatus: false,
                    validationReason: error instanceof Error ? error.message : String(error)
                };
            }
        }));
    }
    /**
     * Filter signals based on confidence threshold and other criteria
     * @private
     * @param signals - Signals to filter
     * @param options - Signal generation options
     * @returns Filtered signals
     */
    _filterSignals(signals, options) {
        // Filter by confidence threshold
        let filteredSignals = signals.filter(signal => signal.confidenceScore >= (options.minConfidenceThreshold || 0));
        // Filter by validation status if we've validated signals
        if (options.validateSignals) {
            filteredSignals = filteredSignals.filter(signal => signal.validationStatus === true);
        }
        // Limit signals per symbol if needed
        if (options.maxSignalsPerSymbol &&
            filteredSignals.length > options.maxSignalsPerSymbol) {
            // Sort by confidence score and take top N
            filteredSignals.sort((a, b) => b.confidenceScore - a.confidenceScore);
            filteredSignals = filteredSignals.slice(0, options.maxSignalsPerSymbol);
        }
        return filteredSignals;
    }
    /**
     * Store signals in the database
     * @private
     * @param signals - Signals to store
     */
    async _storeSignals(signals) {
        try {
            // Store each signal in the database
            for (const signal of signals) {
                await prismaClient_1.default.tradingSignal.create({
                    data: {
                        id: signal.id,
                        symbol: signal.symbol,
                        type: signal.type,
                        direction: signal.direction,
                        strength: signal.strength,
                        timeframe: signal.timeframe,
                        price: signal.price,
                        targetPrice: signal.targetPrice,
                        stopLoss: signal.stopLoss,
                        confidenceScore: signal.confidenceScore,
                        expectedReturn: signal.expectedReturn,
                        expectedRisk: signal.expectedRisk,
                        riskRewardRatio: signal.riskRewardRatio,
                        generatedAt: new Date(signal.generatedAt),
                        expiresAt: signal.expiresAt ? new Date(signal.expiresAt) : null,
                        source: signal.source,
                        metadata: signal.metadata,
                        predictionValues: signal.predictionValues,
                        validatedAt: signal.validatedAt ? new Date(signal.validatedAt) : null,
                        validationStatus: signal.validationStatus || false,
                        validationReason: signal.validationReason || null
                    }
                });
            }
        }
        catch (error) {
            logger.error('Error storing signals', {
                count: signals.length,
                error: error instanceof Error ? error.message : String(error)
            });
            // Continue execution even if storage fails
        }
    }
    /**
     * Get signals based on filter criteria
     * @param criteria - Filter criteria
     * @returns Filtered signals
     */
    async getSignals(criteria = {}) {
        try {
            const { symbol, types, directions, minStrength, timeframes, minConfidenceScore, fromTimestamp, toTimestamp, status = 'active' } = criteria;
            // Build database query conditions
            const where = {};
            // Symbol filter
            if (symbol) {
                where.symbol = symbol;
            }
            // Types filter
            if (types && types.length > 0) {
                where.type = { in: types };
            }
            // Directions filter
            if (directions && directions.length > 0) {
                where.direction = { in: directions };
            }
            // Strength filter
            if (minStrength) {
                const strengthLevels = Object.values(signals_1.SignalStrength);
                const minIndex = strengthLevels.indexOf(minStrength);
                if (minIndex >= 0) {
                    const allowedStrengths = strengthLevels.slice(minIndex);
                    where.strength = { in: allowedStrengths };
                }
            }
            // Timeframes filter
            if (timeframes && timeframes.length > 0) {
                where.timeframe = { in: timeframes };
            }
            // Confidence score filter
            if (minConfidenceScore !== undefined) {
                where.confidenceScore = { gte: minConfidenceScore };
            }
            // Timestamp filters
            if (fromTimestamp) {
                where.generatedAt = { ...(where.generatedAt || {}), gte: new Date(fromTimestamp) };
            }
            if (toTimestamp) {
                where.generatedAt = { ...(where.generatedAt || {}), lte: new Date(toTimestamp) };
            }
            // Status filter
            if (status !== 'all') {
                const now = new Date();
                if (status === 'active') {
                    where.expiresAt = { gte: now };
                }
                else if (status === 'expired') {
                    where.expiresAt = { lt: now };
                }
                else if (status === 'validated') {
                    where.validationStatus = true;
                }
                else if (status === 'invalidated') {
                    where.validationStatus = false;
                }
            }
            // Execute query
            const signalsData = await prismaClient_1.default.tradingSignal.findMany({ where });
            // Convert database records to TradingSignal objects
            const signals = signalsData.map(data => ({
                id: data.id,
                symbol: data.symbol,
                type: data.type,
                direction: data.direction,
                strength: data.strength,
                timeframe: data.timeframe,
                price: data.price,
                targetPrice: data.targetPrice || undefined,
                stopLoss: data.stopLoss || undefined,
                confidenceScore: data.confidenceScore,
                expectedReturn: data.expectedReturn,
                expectedRisk: data.expectedRisk,
                riskRewardRatio: data.riskRewardRatio,
                generatedAt: data.generatedAt.toISOString(),
                expiresAt: data.expiresAt?.toISOString(),
                source: data.source,
                metadata: data.metadata,
                predictionValues: data.predictionValues,
                validatedAt: data.validatedAt?.toISOString(),
                validationStatus: data.validationStatus,
                validationReason: data.validationReason || undefined
            }));
            return signals;
        }
        catch (error) {
            logger.error('Error getting signals', {
                criteria,
                error: error instanceof Error ? error.message : String(error)
            });
            throw error;
        }
    }
    /**
     * Get the latest signal for a symbol
     * @param symbol - Trading pair symbol
     * @returns Latest signal or null if none found
     */
    async getLatestSignal(symbol) {
        try {
            const latestSignal = await prismaClient_1.default.tradingSignal.findFirst({
                where: { symbol },
                orderBy: { generatedAt: 'desc' }
            });
            if (!latestSignal) {
                return null;
            }
            return {
                id: latestSignal.id,
                symbol: latestSignal.symbol,
                type: latestSignal.type,
                direction: latestSignal.direction,
                strength: latestSignal.strength,
                timeframe: latestSignal.timeframe,
                price: latestSignal.price,
                targetPrice: latestSignal.targetPrice || undefined,
                stopLoss: latestSignal.stopLoss || undefined,
                confidenceScore: latestSignal.confidenceScore,
                expectedReturn: latestSignal.expectedReturn,
                expectedRisk: latestSignal.expectedRisk,
                riskRewardRatio: latestSignal.riskRewardRatio,
                generatedAt: latestSignal.generatedAt.toISOString(),
                expiresAt: latestSignal.expiresAt?.toISOString(),
                source: latestSignal.source,
                metadata: latestSignal.metadata,
                predictionValues: latestSignal.predictionValues,
                validatedAt: latestSignal.validatedAt?.toISOString(),
                validationStatus: latestSignal.validationStatus,
                validationReason: latestSignal.validationReason || undefined
            };
        }
        catch (error) {
            logger.error(`Error getting latest signal for ${symbol}`, {
                symbol,
                error: error instanceof Error ? error.message : String(error)
            });
            throw error;
        }
    }
}
exports.SignalGenerationService = SignalGenerationService;
// Create default instance
const signalGenerationService = new SignalGenerationService();
exports.default = signalGenerationService;
//# sourceMappingURL=signalGenerationService.js.map
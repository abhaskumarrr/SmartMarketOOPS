"use strict";
/**
 * Model Training Data Processor
 * Processes real market data for AI model training with proper feature engineering
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ModelTrainingDataProcessor = void 0;
exports.createModelTrainingDataProcessor = createModelTrainingDataProcessor;
const technicalAnalysis_1 = require("../utils/technicalAnalysis");
const logger_1 = require("../utils/logger");
class ModelTrainingDataProcessor {
    /**
     * Process raw market data into training features
     */
    processTrainingData(data, symbol, trainSplit = 0.7, validationSplit = 0.15, testSplit = 0.15) {
        logger_1.logger.info('🔄 Processing training data...', {
            symbol,
            dataPoints: data.length,
            trainSplit,
            validationSplit,
            testSplit,
        });
        // Sort data chronologically
        const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp);
        // Calculate technical indicators
        const enhancedData = this.calculateTechnicalIndicators(sortedData);
        // Generate features
        const features = this.generateFeatures(enhancedData);
        // Calculate future returns and signals
        const featuresWithTargets = this.calculateTargetVariables(features);
        // Remove samples with incomplete data
        const cleanFeatures = this.cleanData(featuresWithTargets);
        logger_1.logger.info('✅ Training data processed', {
            originalSamples: data.length,
            processedSamples: cleanFeatures.length,
            featureCount: Object.keys(cleanFeatures[0] || {}).length,
        });
        return {
            features: cleanFeatures,
            metadata: {
                symbol,
                startDate: new Date(sortedData[0].timestamp),
                endDate: new Date(sortedData[sortedData.length - 1].timestamp),
                totalSamples: cleanFeatures.length,
                featureCount: Object.keys(cleanFeatures[0] || {}).length,
                trainSplit,
                validationSplit,
                testSplit,
            },
        };
    }
    /**
     * Calculate comprehensive technical indicators
     */
    calculateTechnicalIndicators(data) {
        const closes = data.map(d => d.close);
        const highs = data.map(d => d.high);
        const lows = data.map(d => d.low);
        const volumes = data.map(d => d.volume);
        const opens = data.map(d => d.open);
        // RSI with multiple periods
        const rsi_7 = technicalAnalysis_1.technicalAnalysis.calculateRSI(closes, 7);
        const rsi_14 = technicalAnalysis_1.technicalAnalysis.calculateRSI(closes, 14);
        const rsi_21 = technicalAnalysis_1.technicalAnalysis.calculateRSI(closes, 21);
        // Moving averages
        const sma_10 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 10);
        const sma_20 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 20);
        const sma_50 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 50);
        const ema_12 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 12);
        const ema_26 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 26);
        const ema_50 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 50);
        // MACD
        const macd = technicalAnalysis_1.technicalAnalysis.calculateMACD(closes, 12, 26, 9);
        // Bollinger Bands
        const bollinger = technicalAnalysis_1.technicalAnalysis.calculateBollingerBands(closes, 20, 2);
        // Volume indicators
        const volume_sma_20 = technicalAnalysis_1.technicalAnalysis.calculateSMA(volumes, 20);
        // ATR
        const atr_14 = technicalAnalysis_1.technicalAnalysis.calculateATR(highs, lows, closes, 14);
        // Stochastic
        const stochastic = technicalAnalysis_1.technicalAnalysis.calculateStochastic(highs, lows, closes, 14, 3);
        return data.map((point, index) => ({
            ...point,
            indicators: {
                rsi_7: rsi_7[index],
                rsi_14: rsi_14[index],
                rsi_21: rsi_21[index],
                sma_10: sma_10[index],
                sma_20: sma_20[index],
                sma_50: sma_50[index],
                ema_12: ema_12[index],
                ema_26: ema_26[index],
                ema_50: ema_50[index],
                macd: macd.macd[index],
                macd_signal: macd.signal[index],
                macd_histogram: macd.histogram[index],
                bollinger_upper: bollinger.upper[index],
                bollinger_middle: bollinger.middle[index],
                bollinger_lower: bollinger.lower[index],
                volume_sma_20: volume_sma_20[index],
                atr_14: atr_14[index],
                stochastic_k: stochastic.k[index],
                stochastic_d: stochastic.d[index],
            },
        }));
    }
    /**
     * Generate comprehensive features for model training
     */
    generateFeatures(data) {
        return data.map((point, index) => {
            const indicators = point.indicators;
            // Price action features
            const bodySize = (point.close - point.open) / point.open;
            const upperWick = (point.high - Math.max(point.open, point.close)) / point.open;
            const lowerWick = (Math.min(point.open, point.close) - point.low) / point.open;
            // Bollinger Bands position
            const bbWidth = indicators.bollinger_upper && indicators.bollinger_lower
                ? (indicators.bollinger_upper - indicators.bollinger_lower) / indicators.bollinger_middle
                : 0;
            const bbPosition = indicators.bollinger_upper && indicators.bollinger_lower
                ? (point.close - indicators.bollinger_lower) / (indicators.bollinger_upper - indicators.bollinger_lower)
                : 0.5;
            // Volume ratio
            const volumeRatio = indicators.volume_sma ? point.volume / indicators.volume_sma : 1;
            // Volatility calculation
            const volatility10 = this.calculateVolatility(data, index, 10);
            // Momentum calculations
            const momentum5 = this.calculateMomentum(data, index, 5);
            const momentum10 = this.calculateMomentum(data, index, 10);
            // Multi-timeframe context (simplified)
            const higherTfTrend = this.calculateTrendDirection(data, index, 24); // 24-hour trend
            const lowerTfMomentum = this.calculateMomentum(data, index, 3); // 3-hour momentum
            // Support/Resistance levels (simplified)
            const supportLevel = this.findSupportLevel(data, index, 50);
            const resistanceLevel = this.findResistanceLevel(data, index, 50);
            // Trend strength
            const trendStrength = this.calculateTrendStrength(data, index, 20);
            // Time features
            const date = new Date(point.timestamp);
            const hourOfDay = date.getUTCHours() / 23; // Normalized 0-1
            const dayOfWeek = date.getUTCDay() / 6; // Normalized 0-1
            return {
                // Basic OHLCV
                open: point.open,
                high: point.high,
                low: point.low,
                close: point.close,
                volume: point.volume,
                // Technical indicators
                rsi_14: indicators.rsi || 50,
                rsi_7: indicators.rsi || 50, // Use same RSI for now
                rsi_21: indicators.rsi || 50, // Use same RSI for now
                sma_10: indicators.sma_20 || point.close, // Use sma_20 as proxy
                sma_20: indicators.sma_20 || point.close,
                sma_50: indicators.sma_50 || point.close,
                ema_12: indicators.ema_12 || point.close,
                ema_26: indicators.ema_26 || point.close,
                ema_50: indicators.ema_26 || point.close, // Use ema_26 as proxy
                macd: indicators.macd || 0,
                macd_signal: indicators.macd_signal || 0,
                macd_histogram: indicators.macd_histogram || 0,
                bb_upper: indicators.bollinger_upper || point.close * 1.02,
                bb_middle: indicators.bollinger_middle || point.close,
                bb_lower: indicators.bollinger_lower || point.close * 0.98,
                bb_width: bbWidth,
                bb_position: bbPosition,
                volume_sma_20: indicators.volume_sma || point.volume,
                volume_ratio: volumeRatio,
                // Price action
                body_size: bodySize,
                upper_wick: upperWick,
                lower_wick: lowerWick,
                // Volatility and momentum
                atr_14: 0, // ATR not available in current interface
                volatility_10: volatility10,
                momentum_5: momentum5,
                momentum_10: momentum10,
                // Multi-timeframe
                higher_tf_trend: higherTfTrend,
                lower_tf_momentum: lowerTfMomentum,
                // Market structure
                support_level: supportLevel,
                resistance_level: resistanceLevel,
                trend_strength: trendStrength,
                // Time features
                hour_of_day: hourOfDay,
                day_of_week: dayOfWeek,
            };
        });
    }
    /**
     * Calculate target variables for supervised learning
     */
    calculateTargetVariables(features) {
        return features.map((feature, index) => {
            const currentPrice = feature.close || 0;
            // Calculate future returns
            const future1h = this.getFuturePrice(features, index, 1);
            const future4h = this.getFuturePrice(features, index, 4);
            const future24h = this.getFuturePrice(features, index, 24);
            const futureReturn1h = future1h ? (future1h - currentPrice) / currentPrice : 0;
            const futureReturn4h = future4h ? (future4h - currentPrice) / currentPrice : 0;
            const futureReturn24h = future24h ? (future24h - currentPrice) / currentPrice : 0;
            // Generate classification signals based on returns
            const signal1h = this.returnToSignal(futureReturn1h);
            const signal4h = this.returnToSignal(futureReturn4h);
            const signal24h = this.returnToSignal(futureReturn24h);
            return {
                ...feature,
                future_return_1h: futureReturn1h,
                future_return_4h: futureReturn4h,
                future_return_24h: futureReturn24h,
                signal_1h: signal1h,
                signal_4h: signal4h,
                signal_24h: signal24h,
            };
        });
    }
    // Helper methods
    calculateVolatility(data, index, period) {
        if (index < period)
            return 0;
        const prices = data.slice(index - period + 1, index + 1).map(d => d.close);
        const returns = prices.slice(1).map((price, i) => Math.log(price / prices[i]));
        const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
        return Math.sqrt(variance);
    }
    calculateMomentum(data, index, period) {
        if (index < period)
            return 0;
        const currentPrice = data[index].close;
        const pastPrice = data[index - period].close;
        return (currentPrice - pastPrice) / pastPrice;
    }
    calculateTrendDirection(data, index, period) {
        if (index < period)
            return 0;
        const prices = data.slice(index - period + 1, index + 1).map(d => d.close);
        const firstPrice = prices[0];
        const lastPrice = prices[prices.length - 1];
        const change = (lastPrice - firstPrice) / firstPrice;
        if (change > 0.02)
            return 1; // Strong uptrend
        if (change < -0.02)
            return -1; // Strong downtrend
        return 0; // Sideways
    }
    findSupportLevel(data, index, period) {
        if (index < period)
            return data[index].low;
        const lows = data.slice(index - period + 1, index + 1).map(d => d.low);
        return Math.min(...lows);
    }
    findResistanceLevel(data, index, period) {
        if (index < period)
            return data[index].high;
        const highs = data.slice(index - period + 1, index + 1).map(d => d.high);
        return Math.max(...highs);
    }
    calculateTrendStrength(data, index, period) {
        if (index < period)
            return 0;
        const prices = data.slice(index - period + 1, index + 1).map(d => d.close);
        const sma = prices.reduce((sum, price) => sum + price, 0) / prices.length;
        let aboveCount = 0;
        prices.forEach(price => {
            if (price > sma)
                aboveCount++;
        });
        return (aboveCount / prices.length) * 2 - 1; // -1 to 1 scale
    }
    getFuturePrice(features, index, hoursAhead) {
        const futureIndex = index + hoursAhead;
        return futureIndex < features.length ? features[futureIndex].close || 0 : null;
    }
    returnToSignal(returnValue) {
        if (returnValue > 0.005)
            return 1; // Buy signal for >0.5% return
        if (returnValue < -0.005)
            return -1; // Sell signal for <-0.5% return
        return 0; // Hold signal
    }
    cleanData(features) {
        return features.filter((feature, index) => {
            // Remove samples with NaN or undefined values
            const values = Object.values(feature);
            const hasInvalidValues = values.some(value => value === null || value === undefined || isNaN(value));
            // Remove samples too close to the end (no future data)
            const tooCloseToEnd = index >= features.length - 24;
            return !hasInvalidValues && !tooCloseToEnd;
        });
    }
    /**
     * Split dataset into train/validation/test sets
     */
    splitDataset(dataset) {
        const { features, metadata } = dataset;
        const totalSamples = features.length;
        const trainSize = Math.floor(totalSamples * metadata.trainSplit);
        const validationSize = Math.floor(totalSamples * metadata.validationSplit);
        const train = features.slice(0, trainSize);
        const validation = features.slice(trainSize, trainSize + validationSize);
        const test = features.slice(trainSize + validationSize);
        logger_1.logger.info('📊 Dataset split completed', {
            total: totalSamples,
            train: train.length,
            validation: validation.length,
            test: test.length,
        });
        return { train, validation, test };
    }
}
exports.ModelTrainingDataProcessor = ModelTrainingDataProcessor;
// Export factory function
function createModelTrainingDataProcessor() {
    return new ModelTrainingDataProcessor();
}
//# sourceMappingURL=modelTrainingDataProcessor.js.map
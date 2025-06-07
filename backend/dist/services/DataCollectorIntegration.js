"use strict";
/**
 * Data Collector Integration Service
 * Bridges Multi-Timeframe Data Collector with ML Trading Decision Engine
 * Provides seamless data flow for ML feature engineering
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DataCollectorIntegration = void 0;
const MultiTimeframeDataCollector_1 = require("./MultiTimeframeDataCollector");
const logger_1 = require("../utils/logger");
class DataCollectorIntegration {
    constructor() {
        this.mlEngine = null;
        // Feature extraction configuration
        this.config = {
            fibonacciLevels: [0.236, 0.382, 0.5, 0.618, 0.786],
            confluenceWeights: {
                fibonacci: 0.3,
                smc: 0.25,
                candles: 0.2,
                volume: 0.15,
                timeframe: 0.1
            },
            candleFormationPeriods: [5, 10, 20],
            volumeAnalysisPeriods: [10, 20, 50]
        };
        this.dataCollector = new MultiTimeframeDataCollector_1.MultiTimeframeDataCollector();
    }
    /**
     * Initialize the integration service
     */
    async initialize() {
        try {
            logger_1.logger.info('🔗 Initializing Data Collector Integration...');
            // Initialize data collector
            await this.dataCollector.initialize();
            logger_1.logger.info('✅ Data Collector Integration initialized successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Data Collector Integration:', error.message);
            throw error;
        }
    }
    /**
     * Set ML Trading Decision Engine reference
     */
    setMLEngine(mlEngine) {
        this.mlEngine = mlEngine;
        logger_1.logger.info('🤖 ML Trading Decision Engine connected to Data Collector Integration');
    }
    /**
     * Start data collection for trading symbols
     */
    async startDataCollection(symbols) {
        try {
            logger_1.logger.info(`🔄 Starting integrated data collection for: ${symbols.join(', ')}`);
            await this.dataCollector.startCollection(symbols);
            logger_1.logger.info('✅ Integrated data collection started successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start integrated data collection:', error.message);
            throw error;
        }
    }
    /**
     * Stop data collection
     */
    async stopDataCollection() {
        try {
            logger_1.logger.info('🛑 Stopping integrated data collection...');
            await this.dataCollector.stopCollection();
            logger_1.logger.info('✅ Integrated data collection stopped');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to stop integrated data collection:', error.message);
            throw error;
        }
    }
    /**
     * Extract ML features from multi-timeframe data
     */
    async extractMLFeatures(symbol) {
        try {
            // Get multi-timeframe data
            const mtfData = await this.dataCollector.getMultiTimeframeData(symbol);
            if (!mtfData) {
                logger_1.logger.warn(`⚠️ No multi-timeframe data available for ${symbol}`);
                return null;
            }
            // Validate data quality
            const validation = await this.dataCollector.validateData(symbol);
            if (!validation.isValid || validation.dataQuality < 0.7) {
                logger_1.logger.warn(`⚠️ Data quality too low for ${symbol}: ${(validation.dataQuality * 100).toFixed(1)}%`);
                return null;
            }
            // Extract features
            const features = await this.performFeatureExtraction(mtfData, validation.dataQuality);
            logger_1.logger.debug(`🧠 Extracted ML features for ${symbol} (quality: ${(features.dataQuality * 100).toFixed(1)}%)`);
            return features;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to extract ML features for ${symbol}:`, error.message);
            return null;
        }
    }
    /**
     * Get real-time trading features for ML decision making
     */
    async getRealTimeTradingFeatures(symbol) {
        try {
            const features = await this.extractMLFeatures(symbol);
            if (!features) {
                return null;
            }
            // Add real-time context
            features.timestamp = Date.now();
            features.timeOfDay = this.getTimeOfDayFeature();
            features.marketSession = this.getMarketSessionFeature();
            logger_1.logger.debug(`⚡ Real-time trading features ready for ${symbol}`);
            return features;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to get real-time trading features for ${symbol}:`, error.message);
            return null;
        }
    }
    /**
     * Get data collection statistics
     */
    async getIntegrationStatistics() {
        try {
            const dataStats = await this.dataCollector.getDataStatistics();
            return {
                dataCollector: dataStats,
                integration: {
                    mlEngineConnected: this.mlEngine !== null,
                    featureExtractionConfig: this.config,
                    lastUpdate: Date.now()
                }
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to get integration statistics:', error.message);
            return { error: error.message };
        }
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger_1.logger.info('🧹 Cleaning up Data Collector Integration...');
            await this.dataCollector.cleanup();
            logger_1.logger.info('✅ Data Collector Integration cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during integration cleanup:', error.message);
        }
    }
    // Private feature extraction methods
    /**
     * Perform comprehensive feature extraction
     */
    async performFeatureExtraction(mtfData, dataQuality) {
        const currentData = {
            '5m': mtfData.timeframes['5m'][mtfData.timeframes['5m'].length - 1],
            '15m': mtfData.timeframes['15m'][mtfData.timeframes['15m'].length - 1],
            '1h': mtfData.timeframes['1h'][mtfData.timeframes['1h'].length - 1],
            '4h': mtfData.timeframes['4h'][mtfData.timeframes['4h'].length - 1]
        };
        // Extract Fibonacci features
        const fibFeatures = this.extractFibonacciFeatures(mtfData.timeframes['4h']);
        // Extract multi-timeframe bias features
        const biasFeatures = this.extractBiasFeatures(mtfData.timeframes);
        // Extract candle formation features
        const candleFeatures = this.extractCandleFeatures(currentData['5m'], mtfData.timeframes['5m']);
        // Extract market context features
        const contextFeatures = this.extractMarketContextFeatures(currentData, mtfData.timeframes);
        return {
            symbol: mtfData.symbol,
            timestamp: mtfData.timestamp,
            // Fibonacci features (7)
            fibonacciProximity: fibFeatures.proximity,
            nearestFibLevel: fibFeatures.nearest,
            fibStrength: fibFeatures.strength,
            // Multi-timeframe bias features (6)
            bias4h: biasFeatures.bias4h,
            bias1h: biasFeatures.bias1h,
            bias15m: biasFeatures.bias15m,
            bias5m: biasFeatures.bias5m,
            overallBias: biasFeatures.overall,
            biasAlignment: biasFeatures.alignment,
            // Candle formation features (7)
            bodyPercentage: candleFeatures.bodyPercentage,
            wickPercentage: candleFeatures.wickPercentage,
            buyingPressure: candleFeatures.buyingPressure,
            sellingPressure: candleFeatures.sellingPressure,
            candleType: candleFeatures.candleType,
            momentum: candleFeatures.momentum,
            volatility: candleFeatures.volatility,
            // Market context features (5)
            volume: contextFeatures.volume,
            volumeRatio: contextFeatures.volumeRatio,
            timeOfDay: this.getTimeOfDayFeature(),
            marketSession: this.getMarketSessionFeature(),
            pricePosition: contextFeatures.pricePosition,
            // Quality indicators
            dataQuality,
            synchronized: mtfData.synchronized
        };
    }
    /**
     * Extract Fibonacci retracement features
     */
    extractFibonacciFeatures(data4h) {
        if (data4h.length < 50) {
            return { proximity: [0, 0, 0, 0, 0], nearest: 0, strength: 0 };
        }
        // Find swing high and low over last 30 candles
        const recentData = data4h.slice(-30);
        const high = Math.max(...recentData.map(d => d.high));
        const low = Math.min(...recentData.map(d => d.low));
        const currentPrice = data4h[data4h.length - 1].close;
        // Calculate Fibonacci levels
        const range = high - low;
        const fibLevels = this.config.fibonacciLevels.map(level => high - (range * level));
        // Calculate proximity to each level (0-1, closer = higher)
        const proximity = fibLevels.map(level => {
            const distance = Math.abs(currentPrice - level) / range;
            return Math.max(0, 1 - distance * 10); // Normalize to 0-1
        });
        // Find nearest level
        const distances = fibLevels.map(level => Math.abs(currentPrice - level));
        const nearestIndex = distances.indexOf(Math.min(...distances));
        const nearest = this.config.fibonacciLevels[nearestIndex];
        // Calculate strength (how close to nearest level)
        const strength = proximity[nearestIndex];
        return { proximity, nearest, strength };
    }
    /**
     * Extract multi-timeframe bias features
     */
    extractBiasFeatures(timeframes) {
        const calculateBias = (data, period = 20) => {
            if (data.length < period)
                return 0;
            const recent = data.slice(-period);
            const sma = recent.reduce((sum, d) => sum + d.close, 0) / recent.length;
            const currentPrice = data[data.length - 1].close;
            return (currentPrice - sma) / sma; // Normalized bias (-1 to 1)
        };
        const bias4h = calculateBias(timeframes['4h'], 20);
        const bias1h = calculateBias(timeframes['1h'], 20);
        const bias15m = calculateBias(timeframes['15m'], 20);
        const bias5m = calculateBias(timeframes['5m'], 20);
        const overall = (bias4h * 0.4 + bias1h * 0.3 + bias15m * 0.2 + bias5m * 0.1);
        // Calculate alignment (how much timeframes agree)
        const biases = [bias4h, bias1h, bias15m, bias5m];
        const avgBias = biases.reduce((sum, b) => sum + b, 0) / biases.length;
        const variance = biases.reduce((sum, b) => sum + Math.pow(b - avgBias, 2), 0) / biases.length;
        const alignment = Math.max(0, 1 - variance); // Higher = more aligned
        return { bias4h, bias1h, bias15m, bias5m, overall, alignment };
    }
    /**
     * Extract candle formation features
     */
    extractCandleFeatures(currentCandle, data5m) {
        if (!currentCandle || data5m.length < 20) {
            return {
                bodyPercentage: 0,
                wickPercentage: 0,
                buyingPressure: 0,
                sellingPressure: 0,
                candleType: 0,
                momentum: 0,
                volatility: 0
            };
        }
        const { open, high, low, close } = currentCandle;
        const range = high - low;
        // Body and wick analysis
        const bodySize = Math.abs(close - open);
        const bodyPercentage = range > 0 ? bodySize / range : 0;
        const wickPercentage = 1 - bodyPercentage;
        // Buying/selling pressure
        const upperWick = high - Math.max(open, close);
        const lowerWick = Math.min(open, close) - low;
        const buyingPressure = range > 0 ? (close - low) / range : 0.5;
        const sellingPressure = range > 0 ? (high - close) / range : 0.5;
        // Candle type encoding
        let candleType = 0;
        if (close > open)
            candleType = 1; // Bullish
        else if (close < open)
            candleType = -1; // Bearish
        // Doji = 0
        // Momentum calculation
        const recent = data5m.slice(-10);
        const momentum = recent.length > 1 ?
            (recent[recent.length - 1].close - recent[0].close) / recent[0].close : 0;
        // Volatility calculation (ATR-like)
        const volatility = recent.length > 1 ?
            recent.reduce((sum, candle) => sum + (candle.high - candle.low), 0) / recent.length / close : 0;
        return {
            bodyPercentage,
            wickPercentage,
            buyingPressure,
            sellingPressure,
            candleType,
            momentum,
            volatility
        };
    }
    /**
     * Extract market context features
     */
    extractMarketContextFeatures(currentData, timeframes) {
        const current5m = currentData['5m'];
        const data5m = timeframes['5m'];
        if (!current5m || data5m.length < 50) {
            return {
                volume: 0,
                volumeRatio: 1,
                pricePosition: 0.5
            };
        }
        // Volume analysis
        const volume = current5m.volume;
        const avgVolume = data5m.slice(-20).reduce((sum, d) => sum + d.volume, 0) / 20;
        const volumeRatio = avgVolume > 0 ? volume / avgVolume : 1;
        // Price position in recent range
        const recent = data5m.slice(-50);
        const recentHigh = Math.max(...recent.map(d => d.high));
        const recentLow = Math.min(...recent.map(d => d.low));
        const pricePosition = recentHigh > recentLow ?
            (current5m.close - recentLow) / (recentHigh - recentLow) : 0.5;
        return {
            volume,
            volumeRatio,
            pricePosition
        };
    }
    /**
     * Get time of day feature (0-1)
     */
    getTimeOfDayFeature() {
        const now = new Date();
        const hours = now.getUTCHours();
        const minutes = now.getUTCMinutes();
        const totalMinutes = hours * 60 + minutes;
        return totalMinutes / (24 * 60); // Normalize to 0-1
    }
    /**
     * Get market session feature
     */
    getMarketSessionFeature() {
        const now = new Date();
        const utcHours = now.getUTCHours();
        // Market sessions (UTC):
        // Asian: 0-9 (0)
        // European: 7-16 (1)
        // American: 13-22 (2)
        // Overlap periods get higher values
        if (utcHours >= 0 && utcHours < 7)
            return 0; // Asian
        if (utcHours >= 7 && utcHours < 13)
            return 1; // European
        if (utcHours >= 13 && utcHours < 16)
            return 2; // Euro-American overlap
        if (utcHours >= 16 && utcHours < 22)
            return 1.5; // American
        return 0.5; // Late American/Early Asian
    }
}
exports.DataCollectorIntegration = DataCollectorIntegration;
//# sourceMappingURL=DataCollectorIntegration.js.map
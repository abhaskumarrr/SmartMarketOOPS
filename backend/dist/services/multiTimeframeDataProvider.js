"use strict";
/**
 * Multi-Timeframe Data Provider
 * Handles data aggregation and synchronization across multiple timeframes
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiTimeframeDataProvider = void 0;
exports.createMultiTimeframeDataProvider = createMultiTimeframeDataProvider;
const technicalAnalysis_1 = require("../utils/technicalAnalysis");
const logger_1 = require("../utils/logger");
class MultiTimeframeDataProvider {
    constructor() {
        this.timeframeConfigs = new Map();
        this.baseTimeframe = '1m';
        this.initializeTimeframeConfigs();
    }
    /**
     * Initialize timeframe configurations with proper relationships
     */
    initializeTimeframeConfigs() {
        const configs = [
            { timeframe: '1m', multiplier: 1, priority: 1 },
            { timeframe: '3m', multiplier: 3, priority: 2 },
            { timeframe: '5m', multiplier: 5, priority: 3 },
            { timeframe: '15m', multiplier: 15, priority: 4 },
            { timeframe: '1h', multiplier: 60, priority: 5 },
            { timeframe: '4h', multiplier: 240, priority: 6 },
            { timeframe: '1d', multiplier: 1440, priority: 7 },
        ];
        configs.forEach(config => {
            this.timeframeConfigs.set(config.timeframe, config);
        });
        logger_1.logger.info('📊 Multi-timeframe configurations initialized', {
            timeframes: configs.map(c => `${c.timeframe}(${c.multiplier}m, P${c.priority})`),
        });
    }
    /**
     * Generate multi-timeframe data from base 1-minute data
     */
    generateMultiTimeframeData(baseData, targetTimeframes) {
        logger_1.logger.info('🔄 Generating multi-timeframe data...', {
            baseDataPoints: baseData.length,
            targetTimeframes,
        });
        // Sort base data by timestamp
        const sortedBaseData = [...baseData].sort((a, b) => a.timestamp - b.timestamp);
        // Generate data for each timeframe
        const timeframeData = {};
        // Start with base timeframe (1m)
        if (targetTimeframes.includes('1m')) {
            timeframeData['1m'] = this.enhanceMarketData(sortedBaseData);
        }
        // Generate higher timeframes
        targetTimeframes.forEach(timeframe => {
            if (timeframe !== '1m') {
                const config = this.timeframeConfigs.get(timeframe);
                if (config) {
                    timeframeData[timeframe] = this.aggregateToTimeframe(sortedBaseData, config);
                }
            }
        });
        // Synchronize all timeframes
        const synchronizedData = this.synchronizeTimeframes(timeframeData, targetTimeframes);
        logger_1.logger.info('✅ Multi-timeframe data generated', {
            synchronizedDataPoints: synchronizedData.length,
            timeframes: Object.keys(timeframeData),
        });
        return synchronizedData;
    }
    /**
     * Aggregate base data to higher timeframe
     */
    aggregateToTimeframe(baseData, config) {
        const aggregatedData = [];
        const multiplierMs = config.multiplier * 60 * 1000; // Convert to milliseconds
        // Group data by timeframe periods
        const groups = [];
        let currentGroup = [];
        let currentPeriodStart = 0;
        baseData.forEach(point => {
            // Calculate which period this point belongs to
            const periodStart = Math.floor(point.timestamp / multiplierMs) * multiplierMs;
            if (currentPeriodStart === 0) {
                currentPeriodStart = periodStart;
            }
            if (periodStart === currentPeriodStart) {
                currentGroup.push(point);
            }
            else {
                if (currentGroup.length > 0) {
                    groups.push([...currentGroup]);
                }
                currentGroup = [point];
                currentPeriodStart = periodStart;
            }
        });
        // Add the last group
        if (currentGroup.length > 0) {
            groups.push(currentGroup);
        }
        // Aggregate each group into a single candle
        groups.forEach(group => {
            if (group.length > 0) {
                const aggregated = this.aggregateGroup(group, config.timeframe);
                aggregatedData.push(aggregated);
            }
        });
        // Enhance with technical indicators
        return this.enhanceMarketData(aggregatedData);
    }
    /**
     * Aggregate a group of candles into a single candle
     */
    aggregateGroup(group, timeframe) {
        const first = group[0];
        const last = group[group.length - 1];
        // OHLC aggregation
        const open = first.open;
        const close = last.close;
        const high = Math.max(...group.map(p => p.high));
        const low = Math.min(...group.map(p => p.low));
        const volume = group.reduce((sum, p) => sum + p.volume, 0);
        // Use the timestamp of the first candle in the group
        const timestamp = first.timestamp;
        return {
            timestamp,
            symbol: first.symbol,
            exchange: first.exchange,
            timeframe,
            open,
            high,
            low,
            close,
            volume,
        };
    }
    /**
     * Enhance market data with technical indicators
     */
    enhanceMarketData(data) {
        if (data.length === 0)
            return [];
        const closes = data.map(d => d.close);
        const volumes = data.map(d => d.volume);
        const highs = data.map(d => d.high);
        const lows = data.map(d => d.low);
        // Calculate technical indicators
        const sma20 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 20);
        const sma50 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 50);
        const ema12 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 12);
        const ema26 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 26);
        const rsi = technicalAnalysis_1.technicalAnalysis.calculateRSI(closes, 14);
        const macd = technicalAnalysis_1.technicalAnalysis.calculateMACD(closes, 12, 26, 9);
        const bollinger = technicalAnalysis_1.technicalAnalysis.calculateBollingerBands(closes, 20, 2);
        const volumeSMA = technicalAnalysis_1.technicalAnalysis.calculateSMA(volumes, 20);
        const stochastic = technicalAnalysis_1.technicalAnalysis.calculateStochastic(highs, lows, closes, 14, 3);
        const atr = technicalAnalysis_1.technicalAnalysis.calculateATR(highs, lows, closes, 14);
        return data.map((point, index) => ({
            ...point,
            indicators: {
                sma_20: sma20[index],
                sma_50: sma50[index],
                ema_12: ema12[index],
                ema_26: ema26[index],
                rsi: rsi[index],
                macd: macd.macd[index],
                macd_signal: macd.signal[index],
                macd_histogram: macd.histogram[index],
                bollinger_upper: bollinger.upper[index],
                bollinger_middle: bollinger.middle[index],
                bollinger_lower: bollinger.lower[index],
                volume_sma: volumeSMA[index],
                stochastic_k: stochastic.k[index],
                stochastic_d: stochastic.d[index],
                atr: atr[index],
            },
        }));
    }
    /**
     * Synchronize all timeframes to create aligned multi-timeframe data
     */
    synchronizeTimeframes(timeframeData, targetTimeframes) {
        const synchronizedData = [];
        // Use 1-minute data as the base for synchronization
        const baseData = timeframeData['1m'] || [];
        baseData.forEach(baseCandle => {
            const multiData = {
                timestamp: baseCandle.timestamp,
                timeframes: {},
            };
            // Add data for each timeframe
            targetTimeframes.forEach(timeframe => {
                const data = timeframeData[timeframe];
                if (data) {
                    // Find the corresponding candle for this timestamp
                    const correspondingCandle = this.findCorrespondingCandle(baseCandle.timestamp, data, timeframe);
                    if (correspondingCandle) {
                        multiData.timeframes[timeframe] = correspondingCandle;
                    }
                }
            });
            synchronizedData.push(multiData);
        });
        return synchronizedData;
    }
    /**
     * Find the corresponding candle for a given timestamp in a specific timeframe
     */
    findCorrespondingCandle(timestamp, data, timeframe) {
        const config = this.timeframeConfigs.get(timeframe);
        if (!config)
            return null;
        const multiplierMs = config.multiplier * 60 * 1000;
        // Calculate which period this timestamp belongs to
        const periodStart = Math.floor(timestamp / multiplierMs) * multiplierMs;
        // Find the candle that starts at this period
        return data.find(candle => {
            const candlePeriodStart = Math.floor(candle.timestamp / multiplierMs) * multiplierMs;
            return candlePeriodStart === periodStart;
        }) || null;
    }
    /**
     * Get timeframe priority for decision making
     */
    getTimeframePriority(timeframe) {
        return this.timeframeConfigs.get(timeframe)?.priority || 0;
    }
    /**
     * Get timeframe multiplier
     */
    getTimeframeMultiplier(timeframe) {
        return this.timeframeConfigs.get(timeframe)?.multiplier || 1;
    }
    /**
     * Validate timeframe relationships
     */
    validateTimeframeRelationships() {
        const relationships = [
            { from: '1m', to: '3m', expected: 3 },
            { from: '1m', to: '5m', expected: 5 },
            { from: '1m', to: '15m', expected: 15 },
            { from: '1m', to: '1h', expected: 60 },
            { from: '1m', to: '4h', expected: 240 },
            { from: '1m', to: '1d', expected: 1440 },
            { from: '3m', to: '15m', expected: 5 },
            { from: '5m', to: '1h', expected: 12 },
            { from: '15m', to: '1h', expected: 4 },
            { from: '1h', to: '4h', expected: 4 },
            { from: '4h', to: '1d', expected: 6 },
        ];
        let isValid = true;
        relationships.forEach(rel => {
            const fromMultiplier = this.getTimeframeMultiplier(rel.from);
            const toMultiplier = this.getTimeframeMultiplier(rel.to);
            const actualRatio = toMultiplier / fromMultiplier;
            if (actualRatio !== rel.expected) {
                logger_1.logger.error(`❌ Invalid timeframe relationship: ${rel.from} to ${rel.to}`, {
                    expected: rel.expected,
                    actual: actualRatio,
                });
                isValid = false;
            }
        });
        if (isValid) {
            logger_1.logger.info('✅ All timeframe relationships validated successfully');
        }
        return isValid;
    }
    /**
     * Get supported timeframes in priority order
     */
    getSupportedTimeframes() {
        return Array.from(this.timeframeConfigs.keys()).sort((a, b) => {
            const priorityA = this.getTimeframePriority(a);
            const priorityB = this.getTimeframePriority(b);
            return priorityA - priorityB;
        });
    }
    /**
     * Check if timeframe is supported
     */
    isTimeframeSupported(timeframe) {
        return this.timeframeConfigs.has(timeframe);
    }
}
exports.MultiTimeframeDataProvider = MultiTimeframeDataProvider;
// Export factory function
function createMultiTimeframeDataProvider() {
    return new MultiTimeframeDataProvider();
}
//# sourceMappingURL=multiTimeframeDataProvider.js.map
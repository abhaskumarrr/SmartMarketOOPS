"use strict";
/**
 * Multi-Timeframe Data Collector
 * Implements comprehensive data collection across 4H, 1H, 15M, and 5M timeframes
 * with intelligent caching, synchronization, and validation for ML feature engineering
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiTimeframeDataCollector = void 0;
const ccxt_1 = __importDefault(require("ccxt"));
const ioredis_1 = __importDefault(require("ioredis"));
const logger_1 = require("../utils/logger");
const deltaApiService_1 = __importDefault(require("./deltaApiService"));
class MultiTimeframeDataCollector {
    constructor() {
        // Timeframe configurations optimized for ML feature engineering
        this.timeframeConfigs = {
            '4h': {
                timeframe: '4h',
                limit: 100, // 400 hours = ~16.7 days of data
                cacheTTL: 3600, // 1 hour cache (4h candles don't change often)
                refreshInterval: 300000 // 5 minutes
            },
            '1h': {
                timeframe: '1h',
                limit: 168, // 168 hours = 7 days of data
                cacheTTL: 900, // 15 minutes cache
                refreshInterval: 120000 // 2 minutes
            },
            '15m': {
                timeframe: '15m',
                limit: 672, // 672 * 15min = 7 days of data
                cacheTTL: 300, // 5 minutes cache
                refreshInterval: 60000 // 1 minute
            },
            '5m': {
                timeframe: '5m',
                limit: 2016, // 2016 * 5min = 7 days of data
                cacheTTL: 120, // 2 minutes cache
                refreshInterval: 30000 // 30 seconds
            }
        };
        // Active data collection intervals
        this.intervals = new Map();
        this.isCollecting = false;
        this.deltaApi = new deltaApiService_1.default({ testnet: true });
        this.redis = new ioredis_1.default({
            host: process.env.REDIS_HOST || 'localhost',
            port: parseInt(process.env.REDIS_PORT || '6379'),
            retryDelayOnFailover: 100,
            maxRetriesPerRequest: 3
        });
        // Initialize CCXT exchange for backup data source
        this.ccxtExchange = new ccxt_1.default.binance({
            apiKey: process.env.BINANCE_API_KEY,
            secret: process.env.BINANCE_SECRET,
            sandbox: true,
            enableRateLimit: true
        });
    }
    /**
     * Initialize the data collector
     */
    async initialize() {
        try {
            logger_1.logger.info('🚀 Initializing Multi-Timeframe Data Collector...');
            // Initialize Delta Exchange API
            await this.deltaApi.initialize({
                key: process.env.DELTA_EXCHANGE_API_KEY || '',
                secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
            });
            // Test Redis connection
            await this.redis.ping();
            logger_1.logger.info('✅ Redis connection established');
            // Load CCXT markets
            await this.ccxtExchange.loadMarkets();
            logger_1.logger.info('✅ CCXT markets loaded');
            logger_1.logger.info('🎯 Multi-Timeframe Data Collector initialized successfully');
            logger_1.logger.info(`📊 Configured timeframes: ${Object.keys(this.timeframeConfigs).join(', ')}`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Multi-Timeframe Data Collector:', error.message);
            throw error;
        }
    }
    /**
     * Start data collection for specified symbols
     */
    async startCollection(symbols) {
        if (this.isCollecting) {
            logger_1.logger.warn('⚠️ Data collection already running');
            return;
        }
        this.isCollecting = true;
        logger_1.logger.info(`🔄 Starting multi-timeframe data collection for symbols: ${symbols.join(', ')}`);
        for (const symbol of symbols) {
            await this.startSymbolCollection(symbol);
        }
        logger_1.logger.info('✅ Multi-timeframe data collection started successfully');
    }
    /**
     * Stop data collection
     */
    async stopCollection() {
        if (!this.isCollecting) {
            logger_1.logger.warn('⚠️ Data collection not running');
            return;
        }
        logger_1.logger.info('🛑 Stopping multi-timeframe data collection...');
        // Clear all intervals
        for (const [key, interval] of this.intervals) {
            clearInterval(interval);
            logger_1.logger.debug(`Cleared interval for ${key}`);
        }
        this.intervals.clear();
        this.isCollecting = false;
        logger_1.logger.info('✅ Multi-timeframe data collection stopped');
    }
    /**
     * Get synchronized multi-timeframe data for a symbol
     */
    async getMultiTimeframeData(symbol) {
        try {
            const cacheKey = `mtf_data:${symbol}`;
            const cachedData = await this.redis.get(cacheKey);
            if (cachedData) {
                const data = JSON.parse(cachedData);
                // Check if data is recent enough (within 5 minutes)
                if (Date.now() - data.lastUpdate < 300000) {
                    logger_1.logger.debug(`📊 Retrieved cached multi-timeframe data for ${symbol}`);
                    return data;
                }
            }
            // Fetch fresh data if cache miss or stale
            return await this.fetchAndSynchronizeData(symbol);
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to get multi-timeframe data for ${symbol}:`, error.message);
            return null;
        }
    }
    /**
     * Validate multi-timeframe data quality
     */
    async validateData(symbol) {
        const result = {
            isValid: true,
            errors: [],
            warnings: [],
            dataQuality: 1.0
        };
        try {
            const data = await this.getMultiTimeframeData(symbol);
            if (!data) {
                result.isValid = false;
                result.errors.push('No data available');
                result.dataQuality = 0;
                return result;
            }
            // Validate each timeframe
            for (const [timeframe, ohlcvData] of Object.entries(data.timeframes)) {
                const validation = this.validateTimeframeData(timeframe, ohlcvData);
                if (!validation.isValid) {
                    result.isValid = false;
                    result.errors.push(...validation.errors.map(e => `${timeframe}: ${e}`));
                }
                result.warnings.push(...validation.warnings.map(w => `${timeframe}: ${w}`));
                result.dataQuality = Math.min(result.dataQuality, validation.dataQuality);
            }
            // Check synchronization
            if (!data.synchronized) {
                result.warnings.push('Data not synchronized across timeframes');
                result.dataQuality *= 0.9;
            }
            // Check data freshness
            const dataAge = Date.now() - data.lastUpdate;
            if (dataAge > 600000) { // 10 minutes
                result.warnings.push(`Data is ${Math.round(dataAge / 60000)} minutes old`);
                result.dataQuality *= 0.95;
            }
            logger_1.logger.debug(`📊 Data validation for ${symbol}: Quality ${(result.dataQuality * 100).toFixed(1)}%`);
        }
        catch (error) {
            result.isValid = false;
            result.errors.push(`Validation error: ${error.message}`);
            result.dataQuality = 0;
        }
        return result;
    }
    /**
     * Get data statistics for monitoring
     */
    async getDataStatistics() {
        try {
            const stats = {
                isCollecting: this.isCollecting,
                activeSymbols: this.intervals.size,
                cacheStats: {},
                timeframeConfigs: this.timeframeConfigs,
                lastUpdate: Date.now()
            };
            // Get cache statistics
            const cacheKeys = await this.redis.keys('mtf_data:*');
            stats.cacheStats = {
                totalCachedSymbols: cacheKeys.length,
                cacheKeys: cacheKeys.map(key => key.replace('mtf_data:', ''))
            };
            return stats;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to get data statistics:', error.message);
            return { error: error.message };
        }
    }
    // Private methods continue in next part...
    /**
     * Start data collection for a specific symbol
     */
    async startSymbolCollection(symbol) {
        logger_1.logger.info(`🔄 Starting data collection for ${symbol}`);
        // Initial data fetch
        await this.fetchAndSynchronizeData(symbol);
        // Set up intervals for each timeframe
        for (const [timeframe, config] of Object.entries(this.timeframeConfigs)) {
            const intervalKey = `${symbol}_${timeframe}`;
            const interval = setInterval(async () => {
                try {
                    await this.fetchTimeframeData(symbol, timeframe);
                }
                catch (error) {
                    logger_1.logger.error(`❌ Error fetching ${timeframe} data for ${symbol}:`, error.message);
                }
            }, config.refreshInterval);
            this.intervals.set(intervalKey, interval);
            logger_1.logger.debug(`⏰ Set up ${timeframe} interval for ${symbol} (${config.refreshInterval}ms)`);
        }
    }
    /**
     * Fetch and synchronize data across all timeframes
     */
    async fetchAndSynchronizeData(symbol) {
        logger_1.logger.debug(`📊 Fetching and synchronizing data for ${symbol}`);
        const data = {
            symbol,
            timestamp: Date.now(),
            timeframes: {
                '4h': [],
                '1h': [],
                '15m': [],
                '5m': []
            },
            synchronized: false,
            lastUpdate: Date.now()
        };
        // Fetch data for all timeframes
        const fetchPromises = Object.keys(this.timeframeConfigs).map(async (timeframe) => {
            const ohlcvData = await this.fetchTimeframeData(symbol, timeframe);
            data.timeframes[timeframe] = ohlcvData;
        });
        await Promise.all(fetchPromises);
        // Synchronize timestamps across timeframes
        data.synchronized = this.synchronizeTimeframes(data);
        // Cache the synchronized data
        await this.cacheMultiTimeframeData(data);
        logger_1.logger.debug(`✅ Synchronized data for ${symbol} (${data.synchronized ? 'synced' : 'not synced'})`);
        return data;
    }
    /**
     * Fetch OHLCV data for a specific timeframe
     */
    async fetchTimeframeData(symbol, timeframe) {
        const config = this.timeframeConfigs[timeframe];
        const cacheKey = `ohlcv:${symbol}:${timeframe}`;
        try {
            // Check cache first
            const cachedData = await this.redis.get(cacheKey);
            if (cachedData) {
                const data = JSON.parse(cachedData);
                logger_1.logger.debug(`📊 Retrieved cached ${timeframe} data for ${symbol} (${data.length} candles)`);
                return data;
            }
            // Fetch from Delta Exchange first
            let ohlcvData = [];
            try {
                const deltaData = await this.fetchFromDeltaExchange(symbol, timeframe, config.limit);
                ohlcvData = deltaData;
                logger_1.logger.debug(`📊 Fetched ${timeframe} data from Delta Exchange for ${symbol} (${ohlcvData.length} candles)`);
            }
            catch (deltaError) {
                logger_1.logger.warn(`⚠️ Delta Exchange failed for ${symbol} ${timeframe}: ${deltaError.message}`);
                // Fallback to CCXT/Binance
                try {
                    const ccxtData = await this.fetchFromCCXT(symbol, timeframe, config.limit);
                    ohlcvData = ccxtData;
                    logger_1.logger.debug(`📊 Fetched ${timeframe} data from CCXT fallback for ${symbol} (${ohlcvData.length} candles)`);
                }
                catch (ccxtError) {
                    logger_1.logger.error(`❌ Both Delta and CCXT failed for ${symbol} ${timeframe}: ${ccxtError.message}`);
                    throw new Error(`Failed to fetch data from all sources: ${deltaError.message}, ${ccxtError.message}`);
                }
            }
            // Cache the data
            await this.redis.setex(cacheKey, config.cacheTTL, JSON.stringify(ohlcvData));
            logger_1.logger.debug(`💾 Cached ${timeframe} data for ${symbol} (TTL: ${config.cacheTTL}s)`);
            return ohlcvData;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to fetch ${timeframe} data for ${symbol}:`, error.message);
            throw error;
        }
    }
    /**
     * Fetch data from Delta Exchange
     */
    async fetchFromDeltaExchange(symbol, timeframe, limit) {
        try {
            // Convert symbol format for Delta Exchange (BTCUSD -> BTC/USD)
            const deltaSymbol = symbol.includes('/') ? symbol : `${symbol.slice(0, 3)}/${symbol.slice(3)}`;
            // Delta Exchange uses different timeframe format
            const deltaTimeframe = this.convertTimeframeForDelta(timeframe);
            // Fetch OHLCV data
            const rawData = await this.deltaApi.getOHLCV(deltaSymbol, deltaTimeframe, limit);
            // Convert to our format
            return rawData.map((candle) => ({
                timestamp: candle[0],
                open: parseFloat(candle[1]),
                high: parseFloat(candle[2]),
                low: parseFloat(candle[3]),
                close: parseFloat(candle[4]),
                volume: parseFloat(candle[5])
            }));
        }
        catch (error) {
            throw new Error(`Delta Exchange fetch failed: ${error.message}`);
        }
    }
    /**
     * Fetch data from CCXT (Binance fallback)
     */
    async fetchFromCCXT(symbol, timeframe, limit) {
        try {
            // Convert symbol format for Binance (BTCUSD -> BTC/USDT)
            const binanceSymbol = this.convertSymbolForBinance(symbol);
            // Fetch OHLCV data
            const rawData = await this.ccxtExchange.fetchOHLCV(binanceSymbol, timeframe, undefined, limit);
            // Convert to our format
            return rawData.map((candle) => ({
                timestamp: candle[0],
                open: candle[1],
                high: candle[2],
                low: candle[3],
                close: candle[4],
                volume: candle[5]
            }));
        }
        catch (error) {
            throw new Error(`CCXT fetch failed: ${error.message}`);
        }
    }
    /**
     * Synchronize timestamps across timeframes
     */
    synchronizeTimeframes(data) {
        try {
            // Get the latest timestamp from 5m data (most frequent)
            const fiveMinData = data.timeframes['5m'];
            if (fiveMinData.length === 0)
                return false;
            const latestTimestamp = fiveMinData[fiveMinData.length - 1].timestamp;
            // Check if all timeframes have data within acceptable time windows
            const timeWindows = {
                '4h': 4 * 60 * 60 * 1000, // 4 hours
                '1h': 60 * 60 * 1000, // 1 hour
                '15m': 15 * 60 * 1000, // 15 minutes
                '5m': 5 * 60 * 1000 // 5 minutes
            };
            let synchronized = true;
            for (const [timeframe, ohlcvData] of Object.entries(data.timeframes)) {
                if (ohlcvData.length === 0) {
                    synchronized = false;
                    continue;
                }
                const lastCandle = ohlcvData[ohlcvData.length - 1];
                const timeDiff = Math.abs(latestTimestamp - lastCandle.timestamp);
                const maxDiff = timeWindows[timeframe];
                if (timeDiff > maxDiff) {
                    logger_1.logger.warn(`⚠️ ${timeframe} data not synchronized: ${timeDiff}ms difference`);
                    synchronized = false;
                }
            }
            return synchronized;
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to synchronize timeframes:', error.message);
            return false;
        }
    }
    /**
     * Cache multi-timeframe data
     */
    async cacheMultiTimeframeData(data) {
        try {
            const cacheKey = `mtf_data:${data.symbol}`;
            await this.redis.setex(cacheKey, 300, JSON.stringify(data)); // 5 minutes cache
            logger_1.logger.debug(`💾 Cached multi-timeframe data for ${data.symbol}`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to cache multi-timeframe data for ${data.symbol}:`, error.message);
        }
    }
    /**
     * Validate timeframe data quality
     */
    validateTimeframeData(timeframe, data) {
        const result = {
            isValid: true,
            errors: [],
            warnings: [],
            dataQuality: 1.0
        };
        if (data.length === 0) {
            result.isValid = false;
            result.errors.push('No data available');
            result.dataQuality = 0;
            return result;
        }
        const config = this.timeframeConfigs[timeframe];
        // Check data completeness
        if (data.length < config.limit * 0.8) {
            result.warnings.push(`Only ${data.length}/${config.limit} candles available`);
            result.dataQuality *= 0.9;
        }
        // Check for gaps in data
        let gapCount = 0;
        const timeframeMs = this.getTimeframeMilliseconds(timeframe);
        for (let i = 1; i < data.length; i++) {
            const expectedTime = data[i - 1].timestamp + timeframeMs;
            const actualTime = data[i].timestamp;
            const timeDiff = Math.abs(actualTime - expectedTime);
            if (timeDiff > timeframeMs * 0.1) { // Allow 10% tolerance
                gapCount++;
            }
        }
        if (gapCount > 0) {
            result.warnings.push(`${gapCount} gaps detected in data`);
            result.dataQuality *= Math.max(0.7, 1 - (gapCount / data.length));
        }
        // Check for invalid OHLCV values
        let invalidCount = 0;
        for (const candle of data) {
            if (candle.high < candle.low ||
                candle.open < 0 || candle.close < 0 ||
                candle.volume < 0) {
                invalidCount++;
            }
        }
        if (invalidCount > 0) {
            result.errors.push(`${invalidCount} invalid candles detected`);
            result.isValid = false;
            result.dataQuality *= Math.max(0.5, 1 - (invalidCount / data.length));
        }
        return result;
    }
    /**
     * Convert timeframe to milliseconds
     */
    getTimeframeMilliseconds(timeframe) {
        const timeframes = {
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000
        };
        return timeframes[timeframe] || 0;
    }
    /**
     * Convert timeframe format for Delta Exchange
     */
    convertTimeframeForDelta(timeframe) {
        const mapping = {
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h'
        };
        return mapping[timeframe] || timeframe;
    }
    /**
     * Convert symbol format for Binance
     */
    convertSymbolForBinance(symbol) {
        // Convert BTCUSD -> BTC/USDT, ETHUSD -> ETH/USDT
        if (symbol === 'BTCUSD')
            return 'BTC/USDT';
        if (symbol === 'ETHUSD')
            return 'ETH/USDT';
        if (symbol === 'SOLUSD')
            return 'SOL/USDT';
        // Default conversion
        if (symbol.includes('/'))
            return symbol;
        return `${symbol.slice(0, 3)}/USDT`;
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger_1.logger.info('🧹 Cleaning up Multi-Timeframe Data Collector...');
            await this.stopCollection();
            await this.redis.quit();
            await this.ccxtExchange.close();
            logger_1.logger.info('✅ Multi-Timeframe Data Collector cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Error during cleanup:', error.message);
        }
    }
}
exports.MultiTimeframeDataCollector = MultiTimeframeDataCollector;
//# sourceMappingURL=MultiTimeframeDataCollector.js.map
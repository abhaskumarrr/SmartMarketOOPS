"use strict";
/**
 * Binance Market Data Provider
 * Fetches real historical data from Binance public API
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.BinanceDataProvider = void 0;
exports.createBinanceDataProvider = createBinanceDataProvider;
const logger_1 = require("../utils/logger");
const axios_1 = __importDefault(require("axios"));
class BinanceDataProvider {
    constructor() {
        this.name = 'binance';
        this.baseUrl = 'https://api.binance.com/api/v3';
        this.rateLimitDelay = 100; // 100ms between requests to respect rate limits
    }
    /**
     * Check if Binance API is available
     */
    isAvailable() {
        return true; // Binance public API is generally available
    }
    /**
     * Fetch historical data from Binance
     */
    async fetchHistoricalData(request) {
        logger_1.logger.info(`📊 Fetching real data from Binance for ${request.symbol}`, {
            timeframe: request.timeframe,
            startDate: request.startDate.toISOString(),
            endDate: request.endDate.toISOString(),
        });
        try {
            // Convert symbol format (BTCUSD -> BTCUSDT for Binance)
            const binanceSymbol = this.convertSymbolToBinanceFormat(request.symbol);
            // Convert timeframe to Binance format
            const binanceInterval = this.convertTimeframeToBinanceInterval(request.timeframe);
            // Calculate start and end times in milliseconds
            const startTime = request.startDate.getTime();
            const endTime = request.endDate.getTime();
            // Fetch data from Binance
            const klineData = await this.fetchBinanceKlines(binanceSymbol, binanceInterval, startTime, endTime);
            // Convert to our format
            const marketData = this.convertBinanceDataToMarketData(klineData, request.symbol, request.timeframe);
            logger_1.logger.info(`✅ Successfully fetched ${marketData.length} real data points from Binance`);
            return {
                data: marketData,
                count: marketData.length,
                symbol: request.symbol,
                timeframe: request.timeframe,
                startDate: request.startDate,
                endDate: request.endDate,
                source: this.name,
            };
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to fetch data from Binance:', error);
            // Fallback to mock data if real data fails
            logger_1.logger.warn('🔄 Falling back to enhanced mock data...');
            return this.generateFallbackData(request);
        }
    }
    /**
     * Fetch kline data from Binance API
     */
    async fetchBinanceKlines(symbol, interval, startTime, endTime) {
        const maxLimit = 1000; // Binance API limit
        const allKlines = [];
        let currentStartTime = startTime;
        while (currentStartTime < endTime) {
            const url = `${this.baseUrl}/klines`;
            const params = {
                symbol,
                interval,
                startTime: currentStartTime,
                endTime,
                limit: maxLimit,
            };
            logger_1.logger.debug(`📡 Fetching Binance data:`, { symbol, interval, startTime: new Date(currentStartTime).toISOString() });
            const response = await axios_1.default.get(url, { params, timeout: 10000 });
            const klines = response.data;
            if (klines.length === 0) {
                break;
            }
            // Convert array format to object format
            const formattedKlines = klines.map(kline => ({
                openTime: kline[0],
                open: kline[1],
                high: kline[2],
                low: kline[3],
                close: kline[4],
                volume: kline[5],
                closeTime: kline[6],
                quoteAssetVolume: kline[7],
                numberOfTrades: kline[8],
                takerBuyBaseAssetVolume: kline[9],
                takerBuyQuoteAssetVolume: kline[10],
                ignore: kline[11],
            }));
            allKlines.push(...formattedKlines);
            // Update start time for next batch
            if (klines.length < maxLimit) {
                break; // No more data available
            }
            currentStartTime = formattedKlines[formattedKlines.length - 1].closeTime + 1;
            // Rate limiting
            await this.delay(this.rateLimitDelay);
        }
        return allKlines;
    }
    /**
     * Convert Binance data to our MarketDataPoint format
     */
    convertBinanceDataToMarketData(klineData, symbol, timeframe) {
        return klineData.map(kline => ({
            timestamp: kline.openTime,
            symbol,
            exchange: this.name,
            timeframe,
            open: parseFloat(kline.open),
            high: parseFloat(kline.high),
            low: parseFloat(kline.low),
            close: parseFloat(kline.close),
            volume: parseFloat(kline.volume),
        }));
    }
    /**
     * Convert our symbol format to Binance format
     */
    convertSymbolToBinanceFormat(symbol) {
        const symbolMap = {
            'BTCUSD': 'BTCUSDT',
            'ETHUSD': 'ETHUSDT',
            'SOLUSD': 'SOLUSDT',
            'ADAUSD': 'ADAUSDT',
            'DOTUSD': 'DOTUSDT',
            'LINKUSD': 'LINKUSDT',
            'AVAXUSD': 'AVAXUSDT',
            'MATICUSD': 'MATICUSDT',
        };
        return symbolMap[symbol] || symbol;
    }
    /**
     * Convert our timeframe to Binance interval format
     */
    convertTimeframeToBinanceInterval(timeframe) {
        const intervalMap = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M',
        };
        const binanceInterval = intervalMap[timeframe];
        if (!binanceInterval) {
            throw new Error(`Unsupported timeframe for Binance: ${timeframe}`);
        }
        return binanceInterval;
    }
    /**
     * Generate fallback mock data if real data fails
     */
    generateFallbackData(request) {
        logger_1.logger.info('📊 Generating fallback mock data...');
        const data = [];
        const startTime = request.startDate.getTime();
        const endTime = request.endDate.getTime();
        // Calculate interval based on timeframe
        const intervalMs = this.getIntervalMs(request.timeframe);
        // Generate realistic BTC price data
        let currentPrice = 45000; // Starting BTC price
        let currentTime = startTime;
        while (currentTime <= endTime) {
            // Generate realistic price movement
            const volatility = 0.02; // 2% volatility
            const trend = 0.0001; // Slight upward trend
            const randomChange = (Math.random() - 0.5) * volatility;
            const trendChange = trend;
            const priceChange = randomChange + trendChange;
            currentPrice *= (1 + priceChange);
            // Generate OHLC data
            const open = currentPrice;
            const close = currentPrice * (1 + (Math.random() - 0.5) * 0.01);
            const high = Math.max(open, close) * (1 + Math.random() * 0.005);
            const low = Math.min(open, close) * (1 - Math.random() * 0.005);
            const volume = 100 + Math.random() * 1000;
            data.push({
                timestamp: currentTime,
                symbol: request.symbol,
                exchange: 'mock-fallback',
                timeframe: request.timeframe,
                open,
                high,
                low,
                close,
                volume,
            });
            currentTime += intervalMs;
            currentPrice = close;
        }
        return {
            data,
            count: data.length,
            symbol: request.symbol,
            timeframe: request.timeframe,
            startDate: request.startDate,
            endDate: request.endDate,
            source: 'mock-fallback',
        };
    }
    /**
     * Get interval in milliseconds for a timeframe
     */
    getIntervalMs(timeframe) {
        const intervals = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        };
        return intervals[timeframe] || 60 * 60 * 1000; // Default to 1 hour
    }
    /**
     * Delay utility for rate limiting
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.BinanceDataProvider = BinanceDataProvider;
// Export factory function
function createBinanceDataProvider() {
    return new BinanceDataProvider();
}
//# sourceMappingURL=binanceDataProvider.js.map
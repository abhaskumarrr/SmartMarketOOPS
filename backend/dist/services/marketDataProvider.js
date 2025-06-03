"use strict";
/**
 * Market Data Provider Service
 * Provides historical market data for backtesting
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.marketDataService = exports.MarketDataService = exports.EnhancedMockMarketDataProvider = exports.MockMarketDataProvider = void 0;
const marketData_1 = require("../types/marketData");
const logger_1 = require("../utils/logger");
const binanceDataProvider_1 = require("./binanceDataProvider");
class MockMarketDataProvider {
    constructor() {
        this.name = 'MockProvider';
    }
    isAvailable() {
        return true;
    }
    /**
     * Generate realistic BTCUSD historical data for backtesting
     */
    async fetchHistoricalData(request) {
        logger_1.logger.info(`📊 Generating mock historical data for ${request.symbol}`, {
            timeframe: request.timeframe,
            startDate: request.startDate.toISOString(),
            endDate: request.endDate.toISOString(),
        });
        const timeframeMs = marketData_1.TIMEFRAMES[request.timeframe] || marketData_1.TIMEFRAMES['1h'];
        const startTime = request.startDate.getTime();
        const endTime = request.endDate.getTime();
        const data = [];
        let currentTime = startTime;
        let currentPrice = 45000; // Starting BTC price
        // Market parameters for realistic simulation
        const volatility = 0.02; // 2% volatility
        const trend = 0.0001; // Slight upward trend
        const meanReversion = 0.1; // Mean reversion strength
        while (currentTime <= endTime) {
            // Generate realistic price movement using random walk with mean reversion
            const randomFactor = (Math.random() - 0.5) * 2; // -1 to 1
            const trendFactor = trend;
            const meanReversionFactor = (45000 - currentPrice) * meanReversion * 0.0001;
            const priceChange = currentPrice * ((randomFactor * volatility) +
                trendFactor +
                meanReversionFactor);
            currentPrice += priceChange;
            // Ensure price doesn't go negative or too extreme
            currentPrice = Math.max(currentPrice, 1000);
            currentPrice = Math.min(currentPrice, 100000);
            // Generate OHLC data
            const open = currentPrice;
            const volatilityRange = currentPrice * volatility * 0.5;
            const high = open + (Math.random() * volatilityRange);
            const low = open - (Math.random() * volatilityRange);
            const close = low + (Math.random() * (high - low));
            // Generate volume (higher volume during price movements)
            const priceChangePercent = Math.abs(priceChange / currentPrice);
            const baseVolume = 100 + (Math.random() * 200); // 100-300 base volume
            const volumeMultiplier = 1 + (priceChangePercent * 10); // Higher volume on big moves
            const volume = baseVolume * volumeMultiplier;
            data.push({
                timestamp: currentTime,
                symbol: request.symbol,
                exchange: request.exchange || 'mock',
                timeframe: request.timeframe,
                open,
                high: Math.max(open, high, close),
                low: Math.min(open, low, close),
                close,
                volume,
            });
            currentTime += timeframeMs;
            currentPrice = close; // Next candle starts where this one ended
        }
        logger_1.logger.info(`✅ Generated ${data.length} data points for ${request.symbol}`);
        return {
            symbol: request.symbol,
            timeframe: request.timeframe,
            data,
            count: data.length,
            startDate: request.startDate,
            endDate: request.endDate,
            source: this.name,
        };
    }
}
exports.MockMarketDataProvider = MockMarketDataProvider;
/**
 * Enhanced Mock Provider with more realistic market patterns
 */
class EnhancedMockMarketDataProvider {
    constructor() {
        this.name = 'enhanced-mock';
        this.marketRegimes = [
            { name: 'bull', probability: 0.3, volatility: 0.015, trend: 0.0005 },
            { name: 'bear', probability: 0.2, volatility: 0.025, trend: -0.0003 },
            { name: 'sideways', probability: 0.4, volatility: 0.01, trend: 0.0001 },
            { name: 'volatile', probability: 0.1, volatility: 0.04, trend: 0 },
        ];
    }
    isAvailable() {
        return true;
    }
    async fetchHistoricalData(request) {
        logger_1.logger.info(`📊 Generating enhanced mock historical data for ${request.symbol}`, {
            timeframe: request.timeframe,
            startDate: request.startDate.toISOString(),
            endDate: request.endDate.toISOString(),
        });
        const timeframeMs = marketData_1.TIMEFRAMES[request.timeframe] || marketData_1.TIMEFRAMES['1h'];
        const startTime = request.startDate.getTime();
        const endTime = request.endDate.getTime();
        const data = [];
        let currentTime = startTime;
        let currentPrice = 45000;
        let currentRegime = this.selectMarketRegime();
        let regimeChangeCounter = 0;
        const regimeChangePeriod = 100; // Change regime every ~100 candles
        while (currentTime <= endTime) {
            // Change market regime periodically
            if (regimeChangeCounter >= regimeChangePeriod) {
                currentRegime = this.selectMarketRegime();
                regimeChangeCounter = 0;
                logger_1.logger.debug(`📈 Market regime changed to: ${currentRegime.name}`);
            }
            // Generate price movement based on current regime
            const randomFactor = this.generateRandomFactor();
            const trendFactor = currentRegime.trend;
            const volatilityFactor = currentRegime.volatility;
            // Add some mean reversion
            const meanPrice = 45000;
            const meanReversionFactor = (meanPrice - currentPrice) * 0.00001;
            const priceChange = currentPrice * ((randomFactor * volatilityFactor) +
                trendFactor +
                meanReversionFactor);
            currentPrice += priceChange;
            currentPrice = Math.max(currentPrice, 1000);
            currentPrice = Math.min(currentPrice, 150000);
            // Generate realistic OHLC
            const candle = this.generateCandle(currentPrice, volatilityFactor);
            // Generate volume based on price action and regime
            const volume = this.generateVolume(priceChange, currentPrice, currentRegime);
            data.push({
                timestamp: currentTime,
                symbol: request.symbol,
                exchange: request.exchange || 'enhanced-mock',
                timeframe: request.timeframe,
                ...candle,
                volume,
            });
            currentTime += timeframeMs;
            currentPrice = candle.close;
            regimeChangeCounter++;
        }
        logger_1.logger.info(`✅ Generated ${data.length} enhanced data points for ${request.symbol}`);
        return {
            symbol: request.symbol,
            timeframe: request.timeframe,
            data,
            count: data.length,
            startDate: request.startDate,
            endDate: request.endDate,
            source: this.name,
        };
    }
    selectMarketRegime() {
        const random = Math.random();
        let cumulative = 0;
        for (const regime of this.marketRegimes) {
            cumulative += regime.probability;
            if (random <= cumulative) {
                return regime;
            }
        }
        return this.marketRegimes[0]; // Fallback
    }
    generateRandomFactor() {
        // Use Box-Muller transform for normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return z0; // Normal distribution with mean 0, std 1
    }
    generateCandle(basePrice, volatility) {
        const open = basePrice;
        const range = basePrice * volatility * 0.5;
        // Generate high and low
        const high = open + (Math.random() * range);
        const low = open - (Math.random() * range);
        // Generate close within the range
        const close = low + (Math.random() * (high - low));
        return {
            open,
            high: Math.max(open, high, close),
            low: Math.min(open, low, close),
            close,
        };
    }
    generateVolume(priceChange, currentPrice, regime) {
        const baseVolume = 50 + (Math.random() * 100); // 50-150 base
        const priceChangePercent = Math.abs(priceChange / currentPrice);
        // Higher volume during volatile periods and trend changes
        const volatilityMultiplier = 1 + (priceChangePercent * 20);
        const regimeMultiplier = regime.name === 'volatile' ? 2 : 1;
        return baseVolume * volatilityMultiplier * regimeMultiplier;
    }
}
exports.EnhancedMockMarketDataProvider = EnhancedMockMarketDataProvider;
/**
 * Market Data Service that manages multiple providers
 */
class MarketDataService {
    constructor() {
        this.providers = new Map();
        this.defaultProvider = 'enhanced-mock';
        this.registerProvider(new MockMarketDataProvider());
        this.registerProvider(new EnhancedMockMarketDataProvider());
        this.registerProvider((0, binanceDataProvider_1.createBinanceDataProvider)());
    }
    registerProvider(provider) {
        this.providers.set(provider.name.toLowerCase(), provider);
        logger_1.logger.info(`📊 Registered market data provider: ${provider.name}`);
    }
    async fetchHistoricalData(request, providerName) {
        const provider = this.getProvider(providerName);
        if (!provider.isAvailable()) {
            throw new Error(`Market data provider ${provider.name} is not available`);
        }
        const startTime = Date.now();
        const response = await provider.fetchHistoricalData(request);
        const duration = Date.now() - startTime;
        logger_1.logger.info(`📊 Fetched ${response.count} data points in ${duration}ms`, {
            provider: provider.name,
            symbol: request.symbol,
            timeframe: request.timeframe,
        });
        return response;
    }
    getProvider(providerName) {
        const name = (providerName || this.defaultProvider).toLowerCase();
        const provider = this.providers.get(name);
        if (!provider) {
            throw new Error(`Market data provider '${name}' not found`);
        }
        return provider;
    }
    getAvailableProviders() {
        return Array.from(this.providers.keys());
    }
    setDefaultProvider(providerName) {
        if (!this.providers.has(providerName.toLowerCase())) {
            throw new Error(`Provider '${providerName}' not found`);
        }
        this.defaultProvider = providerName.toLowerCase();
        logger_1.logger.info(`📊 Default market data provider set to: ${providerName}`);
    }
}
exports.MarketDataService = MarketDataService;
// Export singleton instance
exports.marketDataService = new MarketDataService();
//# sourceMappingURL=marketDataProvider.js.map
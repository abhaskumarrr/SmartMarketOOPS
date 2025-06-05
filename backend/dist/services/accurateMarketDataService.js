"use strict";
/**
 * Accurate Market Data Service
 * Uses multiple reliable sources with proper validation
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.accurateMarketDataService = void 0;
const ccxt_1 = __importDefault(require("ccxt"));
const axios_1 = __importDefault(require("axios"));
// Simple console logger
const logger = {
    info: (message, ...args) => console.log(`[INFO] ${message}`, ...args),
    error: (message, ...args) => console.error(`[ERROR] ${message}`, ...args),
    warn: (message, ...args) => console.warn(`[WARN] ${message}`, ...args),
    debug: (message, ...args) => console.log(`[DEBUG] ${message}`, ...args)
};
class AccurateMarketDataService {
    constructor() {
        this.exchanges = {};
        this.isInitialized = false;
        this.supportedSymbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'];
        this.lastPrices = {};
        this.priceValidationThreshold = 0.15; // 15% deviation threshold
        this.initializeExchanges();
    }
    /**
     * Initialize multiple exchange connections for data validation
     */
    async initializeExchanges() {
        try {
            // Initialize multiple exchanges for cross-validation
            this.exchanges = {
                // Binance - most reliable for major pairs
                binance: new ccxt_1.default.binance({
                    enableRateLimit: true,
                    sandbox: false,
                }),
                // Coinbase - good for USD pairs (coinbasepro is deprecated)
                coinbase: new ccxt_1.default.coinbase({
                    enableRateLimit: true,
                    sandbox: false,
                }),
                // Kraken - reliable alternative
                kraken: new ccxt_1.default.kraken({
                    enableRateLimit: true,
                    sandbox: false,
                })
            };
            // Test connections
            for (const [name, exchange] of Object.entries(this.exchanges)) {
                try {
                    await exchange.loadMarkets();
                    logger.info(`✅ Connected to ${name} exchange`);
                }
                catch (error) {
                    logger.warn(`⚠️ Failed to connect to ${name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
                    delete this.exchanges[name];
                }
            }
            this.isInitialized = Object.keys(this.exchanges).length > 0;
            if (this.isInitialized) {
                logger.info(`✅ Accurate Market Data Service initialized with ${Object.keys(this.exchanges).length} exchanges`);
            }
            else {
                logger.error('❌ No exchanges available, falling back to external APIs');
            }
        }
        catch (error) {
            logger.error('❌ Failed to initialize exchanges:', error instanceof Error ? error.message : 'Unknown error');
            this.isInitialized = false;
        }
    }
    /**
     * Get market data from external APIs as fallback
     */
    async getExternalMarketData(symbol) {
        try {
            // Map our symbols to CoinGecko IDs
            const coinGeckoMap = {
                'BTCUSD': 'bitcoin',
                'ETHUSD': 'ethereum',
                'ADAUSD': 'cardano',
                'SOLUSD': 'solana',
                'DOTUSD': 'polkadot'
            };
            const coinId = coinGeckoMap[symbol];
            if (!coinId) {
                return null;
            }
            // Get current price and 24h data from CoinGecko
            const response = await axios_1.default.get(`https://api.coingecko.com/api/v3/simple/price?ids=${coinId}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true`, { timeout: 10000 });
            const data = response.data[coinId];
            if (!data) {
                return null;
            }
            const currentPrice = data.usd;
            const change24h = data.usd_24h_change || 0;
            const volume24h = data.usd_24h_vol || 0;
            return {
                symbol,
                price: currentPrice,
                change: (currentPrice * change24h) / 100,
                changePercent: change24h,
                volume: volume24h,
                high24h: currentPrice * 1.02, // Approximate
                low24h: currentPrice * 0.98, // Approximate
                timestamp: Date.now(),
                source: 'coingecko',
                isValidated: true
            };
        }
        catch (error) {
            logger.error(`Failed to get external data for ${symbol}:`, error instanceof Error ? error.message : 'Unknown error');
            return null;
        }
    }
    /**
     * Get market data from CCXT exchanges
     */
    async getExchangeMarketData(symbol) {
        const results = [];
        // Map our symbols to exchange symbols
        const symbolMap = {
            'BTCUSD': {
                'binance': 'BTC/USDT',
                'coinbase': 'BTC/USD',
                'kraken': 'BTC/USD'
            },
            'ETHUSD': {
                'binance': 'ETH/USDT',
                'coinbase': 'ETH/USD',
                'kraken': 'ETH/USD'
            },
            'ADAUSD': {
                'binance': 'ADA/USDT',
                'coinbase': 'ADA/USD',
                'kraken': 'ADA/USD'
            },
            'SOLUSD': {
                'binance': 'SOL/USDT',
                'coinbase': 'SOL/USD',
                'kraken': 'SOL/USD'
            },
            'DOTUSD': {
                'binance': 'DOT/USDT',
                'coinbase': 'DOT/USD',
                'kraken': 'DOT/USD'
            }
        };
        const exchangeSymbols = symbolMap[symbol];
        if (!exchangeSymbols) {
            return results;
        }
        for (const [exchangeName, exchangeSymbol] of Object.entries(exchangeSymbols)) {
            const exchange = this.exchanges[exchangeName];
            if (!exchange)
                continue;
            try {
                const ticker = await exchange.fetchTicker(exchangeSymbol);
                const currentPrice = ticker.last || ticker.close || 0;
                const openPrice = ticker.open || currentPrice;
                const change = currentPrice - openPrice;
                const changePercent = openPrice > 0 ? (change / openPrice) * 100 : 0;
                results.push({
                    symbol,
                    price: currentPrice,
                    change,
                    changePercent,
                    volume: ticker.baseVolume || 0,
                    high24h: ticker.high || currentPrice,
                    low24h: ticker.low || currentPrice,
                    timestamp: ticker.timestamp || Date.now(),
                    source: exchangeName,
                    isValidated: false // Will be validated later
                });
            }
            catch (error) {
                logger.debug(`Failed to get ${symbol} from ${exchangeName}:`, error instanceof Error ? error.message : 'Unknown error');
            }
        }
        return results;
    }
    /**
     * Validate price data across multiple sources
     */
    validatePrices(prices) {
        if (prices.length === 0) {
            return null;
        }
        if (prices.length === 1) {
            prices[0].isValidated = true;
            return prices[0];
        }
        // Calculate median price for validation
        const sortedPrices = prices.map(p => p.price).sort((a, b) => a - b);
        const median = sortedPrices[Math.floor(sortedPrices.length / 2)];
        // Filter out prices that deviate too much from median
        const validPrices = prices.filter(p => {
            const deviation = Math.abs(p.price - median) / median;
            return deviation <= this.priceValidationThreshold;
        });
        if (validPrices.length === 0) {
            logger.warn(`All prices for ${prices[0].symbol} failed validation`);
            return prices[0]; // Return first price as fallback
        }
        // Use the most reliable source (prefer coinbase, then binance, then others)
        const sourcePreference = ['coinbase', 'binance', 'kraken', 'coingecko'];
        for (const preferredSource of sourcePreference) {
            const preferredPrice = validPrices.find(p => p.source === preferredSource);
            if (preferredPrice) {
                preferredPrice.isValidated = true;
                logger.debug(`Using ${preferredSource} price for ${preferredPrice.symbol}: $${preferredPrice.price}`);
                return preferredPrice;
            }
        }
        // Fallback to first valid price
        validPrices[0].isValidated = true;
        return validPrices[0];
    }
    /**
     * Get accurate market data for a symbol
     */
    async getMarketData(symbol) {
        try {
            const allPrices = [];
            // Get data from exchanges
            if (this.isInitialized) {
                const exchangePrices = await this.getExchangeMarketData(symbol);
                allPrices.push(...exchangePrices);
            }
            // Get data from external API
            const externalPrice = await this.getExternalMarketData(symbol);
            if (externalPrice) {
                allPrices.push(externalPrice);
            }
            // Validate and return best price
            const validatedPrice = this.validatePrices(allPrices);
            if (validatedPrice) {
                this.lastPrices[symbol] = validatedPrice.price;
                logger.info(`📊 ${symbol}: $${validatedPrice.price.toFixed(2)} (${validatedPrice.source}${validatedPrice.isValidated ? ' ✓' : ''})`);
            }
            return validatedPrice;
        }
        catch (error) {
            logger.error(`Failed to get market data for ${symbol}:`, error instanceof Error ? error.message : 'Unknown error');
            return this.getMockMarketData(symbol);
        }
    }
    /**
     * Get market data for multiple symbols
     */
    async getMultipleMarketData(symbols) {
        const results = [];
        for (const symbol of symbols) {
            try {
                const data = await this.getMarketData(symbol);
                if (data) {
                    results.push(data);
                }
                // Add delay to respect rate limits
                await this.delay(200);
            }
            catch (error) {
                logger.error(`Failed to fetch data for ${symbol}:`, error);
            }
        }
        return results;
    }
    /**
     * Check if service is ready
     */
    isReady() {
        return true; // Always ready with fallback to external APIs
    }
    /**
     * Get supported symbols
     */
    getSupportedSymbols() {
        return this.supportedSymbols;
    }
    /**
     * Fallback mock data generator (only used as last resort)
     */
    getMockMarketData(symbol) {
        logger.warn(`Using mock data for ${symbol} - all real sources failed`);
        const basePrice = this.getBasePriceForSymbol(symbol);
        const lastPrice = this.lastPrices[symbol] || basePrice;
        const changePercent = (Math.random() - 0.5) * 1.0;
        const newPrice = lastPrice * (1 + changePercent / 100);
        const change = newPrice - lastPrice;
        this.lastPrices[symbol] = newPrice;
        return {
            symbol,
            price: Number(newPrice.toFixed(2)),
            change: Number(change.toFixed(2)),
            changePercent: Number(changePercent.toFixed(2)),
            volume: Math.floor(Math.random() * 1000000) + 100000,
            high24h: Number((newPrice * 1.05).toFixed(2)),
            low24h: Number((newPrice * 0.95).toFixed(2)),
            timestamp: Date.now(),
            source: 'mock',
            isValidated: false
        };
    }
    /**
     * Get realistic base price for symbol
     */
    getBasePriceForSymbol(symbol) {
        const basePrices = {
            'BTCUSD': 104000, // Updated to realistic current prices
            'ETHUSD': 2540,
            'ADAUSD': 0.89,
            'SOLUSD': 240,
            'DOTUSD': 7.5
        };
        return basePrices[symbol] || 100;
    }
    /**
     * Utility delay function
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        for (const exchange of Object.values(this.exchanges)) {
            try {
                if (exchange && typeof exchange.close === 'function') {
                    await exchange.close();
                }
            }
            catch (error) {
                logger.error('Error closing exchange connection:', error);
            }
        }
        this.isInitialized = false;
        logger.info('Accurate Market Data Service cleaned up');
    }
}
// Export singleton instance
exports.accurateMarketDataService = new AccurateMarketDataService();
exports.default = exports.accurateMarketDataService;
//# sourceMappingURL=accurateMarketDataService.js.map
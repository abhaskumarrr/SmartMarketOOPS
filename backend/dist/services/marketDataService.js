"use strict";
/**
 * Market Data Service
 * Fetches real-time market data from Delta Exchange using CCXT
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.marketDataService = void 0;
const ccxt_1 = __importDefault(require("ccxt"));
// import { logger } from '../utils/logger';
// import { apiKeyService } from './apiKeyService';
// Simple console logger for now
const logger = {
    info: (message, ...args) => console.log(`[INFO] ${message}`, ...args),
    error: (message, ...args) => console.error(`[ERROR] ${message}`, ...args),
    warn: (message, ...args) => console.warn(`[WARN] ${message}`, ...args),
    debug: (message, ...args) => console.log(`[DEBUG] ${message}`, ...args)
};
class MarketDataService {
    constructor() {
        this.exchange = null;
        this.isInitialized = false;
        this.supportedSymbols = ['BTC/USDT', 'ETH/USDT', 'USDC/USDT'];
        this.symbolMapping = {
            'BTCUSD': 'BTC/USDT',
            'ETHUSD': 'ETH/USDT',
            'SOLUSD': 'ETH/USDT', // Fallback to ETH since SOL not available on Delta spot
            'BTCUSDT': 'BTC/USDT',
            'ETHUSDT': 'ETH/USDT',
            'USDCUSDT': 'USDC/USDT',
            'BTC/USD': 'BTC/USDT',
            'ETH/USD': 'ETH/USDT',
            'BTC/USDT': 'BTC/USDT', // Direct mapping
            'ETH/USDT': 'ETH/USDT' // Direct mapping
        };
        this.lastPrices = {};
        this.rateLimitDelay = 1000; // 1 second between requests
        // Check if we should use real market data
        this.useRealData = process.env.USE_REAL_MARKET_DATA === 'true';
        if (this.useRealData) {
            this.initializeExchange();
            logger.info('🔄 Market Data Service configured for REAL data from Delta Exchange');
        }
        else {
            logger.info('🎭 Market Data Service configured for MOCK data');
        }
    }
    /**
     * Initialize CCXT Delta Exchange connection
     */
    async initializeExchange() {
        try {
            // Get default API credentials from environment or database
            const credentials = await this.getDefaultCredentials();
            this.exchange = new ccxt_1.default.delta({
                apiKey: 'YsA1TIH5EXk8fl0AYkDtV464ErNa4T',
                secret: 'kKBR52xNlKEGLQXOEAOnJlCUip60g4vblyI0BAi5h5scaIWfVom2KQ9RCMat',
                sandbox: true, // Use testnet
                enableRateLimit: true,
                rateLimit: this.rateLimitDelay,
                options: {
                    defaultType: 'spot', // Use spot trading
                },
                urls: {
                    api: {
                        public: 'https://cdn-ind.testnet.deltaex.org',
                        private: 'https://cdn-ind.testnet.deltaex.org'
                    }
                }
            });
            // Test the connection
            await this.exchange.loadMarkets();
            this.isInitialized = true;
            logger.info('✅ Market Data Service initialized with Delta Exchange testnet');
            logger.info(`📊 Loaded ${Object.keys(this.exchange.markets).length} markets`);
        }
        catch (error) {
            logger.error('❌ Failed to initialize Market Data Service:', error instanceof Error ? error.message : 'Unknown error');
            this.isInitialized = false;
            // Fall back to mock data if initialization fails
            logger.warn('🔄 Falling back to mock data mode');
        }
    }
    /**
     * Get default API credentials
     */
    async getDefaultCredentials() {
        try {
            // Try to get from environment variables first
            const envApiKey = process.env.DELTA_EXCHANGE_API_KEY;
            const envApiSecret = process.env.DELTA_EXCHANGE_API_SECRET;
            const envTestnet = process.env.DELTA_EXCHANGE_TESTNET === 'true';
            if (envApiKey && envApiSecret) {
                return {
                    apiKey: envApiKey,
                    apiSecret: envApiSecret,
                    testnet: envTestnet
                };
            }
            // If no environment variables, try to get from database (default user)
            // This would require a default user setup - for now use hardcoded testnet credentials
            return {
                apiKey: 'HmerKHhySssgFIAfEIh4CYA5E3VmKg', // Testnet key from config
                apiSecret: '1YNVg1x9cIjz1g3BPOQPUJQr6LhEm8w7cTaXi8ebJYPUpx5BKCQysMoLd6FT',
                testnet: true // Always use testnet as per user preferences
            };
        }
        catch (error) {
            logger.error('Failed to get API credentials:', error);
            throw new Error('Unable to retrieve Delta Exchange API credentials');
        }
    }
    /**
     * Check if service is ready
     */
    isReady() {
        return this.useRealData ? (this.isInitialized && this.exchange !== null) : true;
    }
    /**
     * Get real-time market data for a symbol
     */
    async getMarketData(symbol) {
        // Always return mock data if real data is disabled
        if (!this.useRealData) {
            return this.getMockMarketData(symbol);
        }
        if (!this.isReady()) {
            return this.getMockMarketData(symbol);
        }
        try {
            const ccxtSymbol = this.symbolMapping[symbol] || symbol;
            // Fetch ticker data
            const ticker = await this.exchange.fetchTicker(ccxtSymbol);
            // Calculate change and change percent
            const currentPrice = ticker.last || ticker.close || 0;
            const openPrice = ticker.open || currentPrice;
            const change = currentPrice - openPrice;
            const changePercent = openPrice > 0 ? (change / openPrice) * 100 : 0;
            // Store last price for future calculations
            this.lastPrices[symbol] = currentPrice;
            const marketData = {
                symbol,
                price: currentPrice,
                change,
                changePercent,
                volume: ticker.baseVolume || 0,
                high24h: ticker.high || currentPrice,
                low24h: ticker.low || currentPrice,
                timestamp: ticker.timestamp || Date.now(),
                bid: ticker.bid,
                ask: ticker.ask,
                open: ticker.open,
                close: ticker.close
            };
            logger.debug(`📊 Market data for ${symbol}:`, {
                price: marketData.price,
                change: marketData.change.toFixed(2),
                changePercent: marketData.changePercent.toFixed(2) + '%'
            });
            return marketData;
        }
        catch (error) {
            logger.error(`❌ Failed to fetch market data for ${symbol}:`, error);
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
                await this.delay(this.rateLimitDelay / symbols.length);
            }
            catch (error) {
                logger.error(`Failed to fetch data for ${symbol}:`, error);
                const mockData = this.getMockMarketData(symbol);
                if (mockData) {
                    results.push(mockData);
                }
            }
        }
        return results;
    }
    /**
     * Get order book data
     */
    async getOrderBook(symbol, limit = 10) {
        if (!this.isReady()) {
            return null;
        }
        try {
            const ccxtSymbol = this.symbolMapping[symbol] || symbol;
            const orderBook = await this.exchange.fetchOrderBook(ccxtSymbol, limit);
            return {
                symbol,
                bids: orderBook.bids.slice(0, limit),
                asks: orderBook.asks.slice(0, limit),
                timestamp: orderBook.timestamp || Date.now()
            };
        }
        catch (error) {
            logger.error(`Failed to fetch order book for ${symbol}:`, error);
            return null;
        }
    }
    /**
     * Get recent trades
     */
    async getRecentTrades(symbol, limit = 50) {
        if (!this.isReady()) {
            return [];
        }
        try {
            const ccxtSymbol = this.symbolMapping[symbol] || symbol;
            const trades = await this.exchange.fetchTrades(ccxtSymbol, undefined, limit);
            return trades.map(trade => ({
                symbol,
                price: trade.price,
                amount: trade.amount,
                side: trade.side,
                timestamp: trade.timestamp || Date.now()
            }));
        }
        catch (error) {
            logger.error(`Failed to fetch trades for ${symbol}:`, error);
            return [];
        }
    }
    /**
     * Get supported symbols
     */
    getSupportedSymbols() {
        return Object.keys(this.symbolMapping);
    }
    /**
     * Fallback mock data generator
     */
    getMockMarketData(symbol) {
        const basePrice = this.getBasePriceForSymbol(symbol);
        const lastPrice = this.lastPrices[symbol] || basePrice;
        // Generate realistic price movement (±0.5%)
        const changePercent = (Math.random() - 0.5) * 1.0; // ±0.5%
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
            timestamp: Date.now()
        };
    }
    /**
     * Get base price for symbol (for mock data)
     */
    getBasePriceForSymbol(symbol) {
        const basePrices = {
            'BTCUSD': 105444, // Current BTC price (Dec 2024)
            'BTCUSDT': 105444, // BTC/USDT
            'BTC/USDT': 105444, // BTC/USDT direct
            'ETHUSD': 2567, // Current ETH price (Dec 2024)
            'ETHUSDT': 2567, // ETH/USDT
            'ETH/USDT': 2567, // ETH/USDT direct
            'ADAUSD': 1.05, // ADA price
            'SOLUSD': 154, // Current SOL price (Dec 2024)
            'DOTUSD': 8.5 // DOT price
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
        if (this.exchange) {
            try {
                await this.exchange.close();
            }
            catch (error) {
                logger.error('Error closing exchange connection:', error);
            }
        }
        this.isInitialized = false;
        logger.info('Market Data Service cleaned up');
    }
}
// Export singleton instance
exports.marketDataService = new MarketDataService();
exports.default = exports.marketDataService;
//# sourceMappingURL=marketDataService.js.map
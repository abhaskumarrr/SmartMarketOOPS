"use strict";
/**
 * Delta Exchange India Trading Service
 * Comprehensive integration with Delta Exchange India API
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DeltaExchangeService = void 0;
const axios_1 = __importDefault(require("axios"));
const crypto_1 = __importDefault(require("crypto"));
// Simple console logger
const logger = {
    info: (message, ...args) => console.log(`[INFO] ${message}`, ...args),
    error: (message, ...args) => console.error(`[ERROR] ${message}`, ...args),
    warn: (message, ...args) => console.warn(`[WARN] ${message}`, ...args),
    debug: (message, ...args) => console.log(`[DEBUG] ${message}`, ...args)
};
class DeltaExchangeService {
    constructor(credentials) {
        this.isInitialized = false;
        this.productCache = new Map();
        this.symbolToProductId = new Map();
        this.credentials = credentials;
        this.baseUrl = credentials.testnet
            ? 'https://cdn-ind.testnet.deltaex.org'
            : 'https://api.india.delta.exchange';
        this.client = axios_1.default.create({
            baseURL: this.baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'SmartMarketOOPS-v1.0'
            }
        });
        // Initialize product ID mappings for perpetual futures (correct IDs from testnet)
        this.symbolToProductId.set('BTCUSD', 84);
        this.symbolToProductId.set('ETHUSD', 1699);
        // Start initialization but don't wait for it
        this.initializeService().catch(error => {
            logger.error('Failed to initialize Delta Exchange Service:', error instanceof Error ? error.message : 'Unknown error');
        });
    }
    /**
     * Initialize the service and load products
     */
    async initializeService() {
        try {
            await this.loadProducts();
            this.isInitialized = true;
            logger.info(`✅ Delta Exchange Service initialized (${this.credentials.testnet ? 'TESTNET' : 'PRODUCTION'})`);
            logger.info(`🔗 Base URL: ${this.baseUrl}`);
            logger.info(`📊 Loaded ${this.productCache.size} products`);
        }
        catch (error) {
            logger.error('❌ Failed to initialize Delta Exchange Service:', error instanceof Error ? error.message : 'Unknown error');
            this.isInitialized = false;
        }
    }
    /**
     * Load and cache all available products
     */
    async loadProducts() {
        try {
            const response = await this.client.get('/v2/products');
            if (response.data.success) {
                const products = response.data.result;
                for (const product of products) {
                    this.productCache.set(product.symbol, product);
                    this.symbolToProductId.set(product.symbol, product.id);
                }
                logger.info(`📦 Cached ${products.length} products`);
                // Log major trading pairs
                const majorPairs = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'];
                const availablePairs = majorPairs.filter(pair => this.symbolToProductId.has(pair));
                logger.info(`🎯 Available major pairs: ${availablePairs.join(', ')}`);
            }
            else {
                throw new Error(`API Error: ${response.data.error}`);
            }
        }
        catch (error) {
            logger.error('Failed to load products:', error instanceof Error ? error.message : 'Unknown error');
            throw error;
        }
    }
    /**
     * Generate HMAC-SHA256 signature for authentication
     */
    generateSignature(method, path, queryString, body, timestamp) {
        const message = method + timestamp + path + queryString + body;
        return crypto_1.default.createHmac('sha256', this.credentials.apiSecret).update(message).digest('hex');
    }
    /**
     * Make authenticated request to Delta Exchange API
     */
    async makeAuthenticatedRequest(method, path, params = {}, data = null) {
        const timestamp = Math.floor(Date.now() / 1000).toString();
        const queryString = Object.keys(params).length > 0 ? '?' + new URLSearchParams(params).toString() : '';
        const body = data ? JSON.stringify(data) : '';
        const signature = this.generateSignature(method, path, queryString, body, timestamp);
        const headers = {
            'api-key': this.credentials.apiKey,
            'signature': signature,
            'timestamp': timestamp,
            'User-Agent': 'SmartMarketOOPS-v1.0',
            'Content-Type': 'application/json'
        };
        // Debug logging
        console.log('🔍 Service signature generation:', {
            method,
            timestamp,
            path,
            queryString,
            body,
            message: method + timestamp + path + queryString + body,
            signature,
            apiKey: this.credentials.apiKey,
            apiSecret: this.credentials.apiSecret ? '***' + this.credentials.apiSecret.slice(-4) : 'undefined'
        });
        try {
            // Use axios directly instead of the client instance to avoid conflicts
            const fullUrl = this.baseUrl + path + queryString;
            console.log('🌐 Service URL construction:', {
                baseUrl: this.baseUrl,
                path,
                queryString,
                fullUrl,
                headers
            });
            const response = await axios_1.default.request({
                method,
                url: fullUrl,
                data: data,
                headers,
                timeout: 30000
            });
            return response.data;
        }
        catch (error) {
            if (error.response) {
                logger.error(`API Error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
                throw new Error(`Delta API Error: ${error.response.data.error || error.response.statusText}`);
            }
            else {
                logger.error(`Request Error: ${error.message}`);
                throw error;
            }
        }
    }
    /**
     * Check if service is ready
     */
    isReady() {
        return this.isInitialized;
    }
    /**
     * Get product ID from symbol
     */
    getProductId(symbol) {
        return this.symbolToProductId.get(symbol) || null;
    }
    /**
     * Get product information
     */
    getProduct(symbol) {
        return this.productCache.get(symbol) || null;
    }
    /**
     * Get all available products
     */
    getAllProducts() {
        return Array.from(this.productCache.values());
    }
    /**
     * Get supported symbols
     */
    getSupportedSymbols() {
        return Array.from(this.symbolToProductId.keys());
    }
    /**
     * Get real-time market data for a symbol
     */
    async getMarketData(symbol) {
        try {
            // Get product ID for the symbol
            const productId = this.symbolToProductId.get(symbol);
            if (!productId) {
                logger.warn(`Product ID not found for symbol: ${symbol}`);
                return this.getMockMarketData(symbol);
            }
            // Use the correct Delta Exchange API endpoint with symbol (not product ID)
            const response = await this.client.get(`/v2/tickers/${symbol}`);
            if (response.data.success) {
                const ticker = response.data.result;
                return {
                    symbol,
                    price: parseFloat(ticker.close || ticker.last_price || '0'),
                    change: parseFloat(ticker.change || '0'),
                    changePercent: parseFloat(ticker.change_percent || '0'),
                    volume: parseFloat(ticker.volume || '0'),
                    high24h: parseFloat(ticker.high || '0'),
                    low24h: parseFloat(ticker.low || '0'),
                    timestamp: Date.now(),
                    source: 'delta_exchange_india',
                    markPrice: parseFloat(ticker.mark_price || '0'),
                    indexPrice: parseFloat(ticker.spot_price || '0'),
                    openInterest: parseFloat(ticker.open_interest || '0')
                };
            }
            else {
                logger.error(`Failed to get market data for ${symbol}:`, response.data.error);
                return this.getMockMarketData(symbol);
            }
        }
        catch (error) {
            logger.error(`Error fetching market data for ${symbol}:`, error instanceof Error ? error.message : 'Unknown error');
            return this.getMockMarketData(symbol);
        }
    }
    /**
     * Get mock market data as fallback
     */
    getMockMarketData(symbol) {
        const mockPrices = {
            'BTCUSD': 105563.43,
            'ETHUSD': 2579.39
        };
        return {
            symbol,
            price: mockPrices[symbol] || 50000,
            change: 0,
            changePercent: 0,
            volume: 0,
            high24h: 0,
            low24h: 0,
            timestamp: Date.now(),
            source: 'mock_fallback',
            markPrice: 0,
            indexPrice: 0,
            openInterest: 0
        };
    }
    /**
     * Get market data for multiple symbols
     */
    async getMultipleMarketData(symbols) {
        const results = [];
        // Use batch ticker API if available, otherwise fetch individually
        try {
            const response = await this.client.get('/v2/tickers');
            if (response.data.success) {
                const tickers = response.data.result;
                for (const symbol of symbols) {
                    const ticker = tickers.find((t) => t.symbol === symbol);
                    if (ticker) {
                        results.push({
                            symbol,
                            price: parseFloat(ticker.close || ticker.last || '0'),
                            change: parseFloat(ticker.change || '0'),
                            changePercent: parseFloat(ticker.change_percent || '0'),
                            volume: parseFloat(ticker.volume || '0'),
                            high24h: parseFloat(ticker.high || '0'),
                            low24h: parseFloat(ticker.low || '0'),
                            timestamp: Date.now(),
                            source: 'delta_exchange_india',
                            markPrice: parseFloat(ticker.mark_price || '0'),
                            indexPrice: parseFloat(ticker.spot_price || '0'),
                            openInterest: parseFloat(ticker.open_interest || '0')
                        });
                    }
                }
            }
        }
        catch (error) {
            logger.error('Error fetching multiple market data:', error instanceof Error ? error.message : 'Unknown error');
            // Fallback to individual requests
            for (const symbol of symbols) {
                const data = await this.getMarketData(symbol);
                if (data) {
                    results.push(data);
                }
                // Add delay to respect rate limits
                await this.delay(100);
            }
        }
        return results;
    }
    /**
     * Place a new order
     */
    async placeOrder(orderRequest) {
        if (!this.isReady()) {
            throw new Error('Delta Exchange Service not initialized');
        }
        try {
            const response = await this.makeAuthenticatedRequest('POST', '/v2/orders', {}, orderRequest);
            if (response.success) {
                logger.info(`✅ Order placed: ${orderRequest.side} ${orderRequest.size} @ ${orderRequest.limit_price || 'market'}`);
                return response.result;
            }
            else {
                logger.error('Failed to place order:', response.error);
                throw new Error(`Order placement failed: ${response.error.code || 'Unknown error'}`);
            }
        }
        catch (error) {
            logger.error('Error placing order:', error instanceof Error ? error.message : 'Unknown error');
            throw error;
        }
    }
    /**
     * Cancel an order
     */
    async cancelOrder(productId, orderId) {
        if (!this.isReady()) {
            throw new Error('Delta Exchange Service not initialized');
        }
        try {
            const response = await this.makeAuthenticatedRequest('DELETE', `/v2/orders/${orderId}`, { product_id: productId });
            if (response.success) {
                logger.info(`✅ Order cancelled: ${orderId}`);
                return true;
            }
            else {
                logger.error('Failed to cancel order:', response.error);
                return false;
            }
        }
        catch (error) {
            logger.error('Error cancelling order:', error instanceof Error ? error.message : 'Unknown error');
            return false;
        }
    }
    /**
     * Get open orders
     */
    async getOpenOrders(productId) {
        if (!this.isReady()) {
            throw new Error('Delta Exchange Service not initialized');
        }
        try {
            const params = productId ? { product_id: productId, state: 'open' } : { state: 'open' };
            const response = await this.makeAuthenticatedRequest('GET', '/v2/orders', params);
            if (response.success) {
                return response.result;
            }
            else {
                logger.error('Failed to get open orders:', response.error);
                return [];
            }
        }
        catch (error) {
            logger.error('Error getting open orders:', error instanceof Error ? error.message : 'Unknown error');
            return [];
        }
    }
    /**
     * Get positions
     */
    async getPositions() {
        if (!this.isReady()) {
            throw new Error('Delta Exchange Service not initialized');
        }
        try {
            const response = await this.makeAuthenticatedRequest('GET', '/v2/positions');
            if (response.success) {
                return response.result;
            }
            else {
                logger.error('Failed to get positions:', response.error);
                return [];
            }
        }
        catch (error) {
            logger.error('Error getting positions:', error instanceof Error ? error.message : 'Unknown error');
            return [];
        }
    }
    /**
     * Get wallet balances using proper Delta Exchange API
     */
    async getBalances() {
        if (!this.isReady()) {
            throw new Error('Delta Exchange Service not initialized');
        }
        try {
            logger.info('🔍 Fetching balances from Delta Exchange...');
            // Use the simple balance endpoint that we know works
            const response = await this.makeAuthenticatedRequest('GET', '/v2/wallet/balances');
            logger.debug('Balance response:', JSON.stringify(response, null, 2));
            if (response && response.success && response.result) {
                const balances = Array.isArray(response.result) ? response.result : [response.result];
                const nonZeroBalances = balances.filter((balance) => balance.balance && parseFloat(balance.balance) > 0);
                logger.info(`✅ Successfully fetched ${nonZeroBalances.length} non-zero balances from Delta Exchange`);
                return nonZeroBalances;
            }
            else {
                logger.error('Failed to get balances - API response:', response);
                return [];
            }
        }
        catch (error) {
            logger.error('Error getting balances:', error instanceof Error ? error.message : 'Unknown error');
            return [];
        }
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
        this.productCache.clear();
        this.symbolToProductId.clear();
        this.isInitialized = false;
        logger.info('Delta Exchange Service cleaned up');
    }
}
exports.DeltaExchangeService = DeltaExchangeService;
exports.default = DeltaExchangeService;

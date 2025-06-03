/**
 * Working Delta Exchange Service (JavaScript)
 * Bypasses TypeScript compilation issues for immediate functionality
 */
const axios = require('axios');
const crypto = require('crypto');
class DeltaExchangeServiceWorking {
    constructor(credentials) {
        this.credentials = credentials;
        this.baseUrl = credentials.testnet
            ? 'https://cdn-ind.testnet.deltaex.org'
            : 'https://api.india.delta.exchange';
        this.client = axios.create({
            baseURL: this.baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'SmartMarketOOPS-v1.0'
            }
        });
        this.isInitialized = false;
        this.productCache = new Map();
        this.symbolToProductId = new Map();
        // Initialize service
        this.initializeService().catch(error => {
            console.error('Failed to initialize Delta Exchange Service:', error.message);
        });
    }
    /**
     * Initialize the service and load products
     */
    async initializeService() {
        try {
            await this.loadProducts();
            this.isInitialized = true;
            console.log(`✅ Delta Exchange Service initialized (${this.credentials.testnet ? 'TESTNET' : 'PRODUCTION'})`);
            console.log(`🔗 Base URL: ${this.baseUrl}`);
            console.log(`📊 Loaded ${this.productCache.size} products`);
        }
        catch (error) {
            console.error('❌ Failed to initialize Delta Exchange Service:', error.message);
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
                console.log(`📦 Cached ${products.length} products`);
                // Log major trading pairs
                const majorPairs = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'];
                const availablePairs = majorPairs.filter(pair => this.symbolToProductId.has(pair));
                console.log(`🎯 Available major pairs: ${availablePairs.join(', ')}`);
            }
            else {
                throw new Error(`API Error: ${response.data.error}`);
            }
        }
        catch (error) {
            console.error('Failed to load products:', error.message);
            throw error;
        }
    }
    /**
     * Generate HMAC-SHA256 signature for authentication
     */
    generateSignature(method, path, queryString, body, timestamp) {
        const message = method + timestamp + path + queryString + body;
        return crypto.createHmac('sha256', this.credentials.apiSecret).update(message).digest('hex');
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
        try {
            const response = await this.client.request({
                method,
                url: path + queryString,
                data: data,
                headers
            });
            return response.data;
        }
        catch (error) {
            if (error.response) {
                console.error(`API Error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
                throw new Error(`Delta API Error: ${error.response.data.error || error.response.statusText}`);
            }
            else {
                console.error(`Request Error: ${error.message}`);
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
            const response = await this.client.get(`/v2/tickers/${symbol}`);
            if (response.data.success) {
                const ticker = response.data.result;
                return {
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
                };
            }
            else {
                console.error(`Failed to get market data for ${symbol}:`, response.data.error);
                return null;
            }
        }
        catch (error) {
            console.error(`Error fetching market data for ${symbol}:`, error.message);
            return null;
        }
    }
    /**
     * Get market data for multiple symbols
     */
    async getMultipleMarketData(symbols) {
        const results = [];
        try {
            const response = await this.client.get('/v2/tickers');
            if (response.data.success) {
                const tickers = response.data.result;
                for (const symbol of symbols) {
                    const ticker = tickers.find(t => t.symbol === symbol);
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
            console.error('Error fetching multiple market data:', error.message);
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
                console.log(`✅ Order placed: ${orderRequest.side} ${orderRequest.size} @ ${orderRequest.limit_price || 'market'}`);
                return response.result;
            }
            else {
                console.error('Failed to place order:', response.error);
                throw new Error(`Order placement failed: ${response.error.code || 'Unknown error'}`);
            }
        }
        catch (error) {
            console.error('Error placing order:', error.message);
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
                console.log(`✅ Order cancelled: ${orderId}`);
                return true;
            }
            else {
                console.error('Failed to cancel order:', response.error);
                return false;
            }
        }
        catch (error) {
            console.error('Error cancelling order:', error.message);
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
                console.error('Failed to get open orders:', response.error);
                return [];
            }
        }
        catch (error) {
            console.error('Error getting open orders:', error.message);
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
                console.error('Failed to get positions:', response.error);
                return [];
            }
        }
        catch (error) {
            console.error('Error getting positions:', error.message);
            return [];
        }
    }
    /**
     * Get wallet balances
     */
    async getBalances() {
        if (!this.isReady()) {
            throw new Error('Delta Exchange Service not initialized');
        }
        try {
            const response = await this.makeAuthenticatedRequest('GET', '/v2/wallet/balances');
            if (response.success) {
                return response.result;
            }
            else {
                console.error('Failed to get balances:', response.error);
                return [];
            }
        }
        catch (error) {
            console.error('Error getting balances:', error.message);
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
        console.log('Delta Exchange Service cleaned up');
    }
}
module.exports = DeltaExchangeServiceWorking;
//# sourceMappingURL=deltaExchangeServiceWorking.js.map
"use strict";
/**
 * Unified Delta Exchange Service for India Testnet
 * Production-ready implementation with proper error handling, authentication, and WebSocket support
 * Based on official Delta Exchange India API documentation
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DeltaExchangeUnified = void 0;
const axios_1 = __importDefault(require("axios"));
const crypto_1 = __importDefault(require("crypto"));
const ws_1 = __importDefault(require("ws"));
const events_1 = require("events");
const logger_1 = require("../utils/logger");
class DeltaExchangeUnified extends events_1.EventEmitter {
    constructor(credentials) {
        super();
        this.isInitialized = false;
        this.productCache = new Map();
        this.symbolToProductId = new Map();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 5000;
        this.credentials = credentials;
        // Use India testnet URLs
        this.baseUrl = credentials.testnet
            ? 'https://cdn-ind.testnet.deltaex.org'
            : 'https://api.india.delta.exchange';
        this.wsUrl = credentials.testnet
            ? 'wss://testnet-ws.delta.exchange'
            : 'wss://ws.delta.exchange';
        this.client = axios_1.default.create({
            baseURL: this.baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'SmartMarketOOPS-DeltaBot-v1.0'
            }
        });
        // Initialize the service asynchronously
        this.initialize().catch(error => {
            logger_1.logger.error('Failed to initialize in constructor:', error);
            this.emit('error', error);
        });
    }
    /**
     * Initialize the Delta Exchange service
     */
    async initialize() {
        try {
            logger_1.logger.info('🚀 Initializing Delta Exchange Unified Service...');
            logger_1.logger.info(`🔑 Using API Key: ${this.credentials.apiKey.substring(0, 8)}...${this.credentials.apiKey.substring(this.credentials.apiKey.length - 4)}`);
            logger_1.logger.info(`🔒 Using API Secret: ${this.credentials.apiSecret.substring(0, 8)}...${this.credentials.apiSecret.substring(this.credentials.apiSecret.length - 4)}`);
            logger_1.logger.info(`🌐 Base URL: ${this.baseUrl}`);
            // Load products and build symbol mappings
            await this.loadProducts();
            // Test authentication
            await this.testAuthentication();
            this.isInitialized = true;
            logger_1.logger.info('✅ Delta Exchange Unified Service initialized successfully');
            this.emit('initialized');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Delta Exchange service:', error);
            this.emit('error', error);
            throw error;
        }
    }
    /**
     * Load all available products from Delta Exchange
     */
    async loadProducts() {
        try {
            const response = await this.client.get('/v2/products');
            if (response.data.success) {
                const products = response.data.result;
                // Cache products and build symbol mappings
                for (const product of products) {
                    this.productCache.set(product.symbol, product);
                    this.symbolToProductId.set(product.symbol, product.id);
                }
                logger_1.logger.info(`📦 Loaded ${products.length} products from Delta Exchange`);
                // Log important trading pairs for BTCUSD and ETHUSD perpetuals
                const btcProduct = this.getProductBySymbol('BTCUSD');
                const ethProduct = this.getProductBySymbol('ETHUSD');
                if (btcProduct) {
                    logger_1.logger.info(`🟡 BTC/USD Perpetual: ID ${btcProduct.id}, State: ${btcProduct.state}`);
                }
                if (ethProduct) {
                    logger_1.logger.info(`🔵 ETH/USD Perpetual: ID ${ethProduct.id}, State: ${ethProduct.state}`);
                }
            }
            else {
                throw new Error(`Failed to load products: ${response.data.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error loading products:', error);
            throw error;
        }
    }
    /**
     * Test authentication with Delta Exchange
     */
    async testAuthentication() {
        try {
            const response = await this.makeAuthenticatedRequest('GET', '/v2/profile');
            if (response.success) {
                logger_1.logger.info('✅ Delta Exchange authentication successful');
                logger_1.logger.info(`👤 User ID: ${response.result.user_id}`);
            }
            else {
                throw new Error(`Authentication failed: ${response.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Delta Exchange authentication failed:', error);
            throw error;
        }
    }
    /**
     * Generate signature for authenticated requests
     */
    generateSignature(method, path, queryString, body, timestamp) {
        const message = method + timestamp + path + queryString + body;
        return crypto_1.default
            .createHmac('sha256', this.credentials.apiSecret)
            .update(message)
            .digest('hex');
    }
    /**
     * Make authenticated request to Delta Exchange API
     */
    async makeAuthenticatedRequest(method, path, params = {}, data = null) {
        const timestamp = Math.floor(Date.now() / 1000).toString();
        const queryString = Object.keys(params).length > 0
            ? '?' + new URLSearchParams(params).toString()
            : '';
        const body = data ? JSON.stringify(data) : '';
        const signature = this.generateSignature(method, path, queryString, body, timestamp);
        const headers = {
            'api-key': this.credentials.apiKey,
            'signature': signature,
            'timestamp': timestamp,
            'Content-Type': 'application/json',
            'User-Agent': 'SmartMarketOOPS-DeltaBot-v1.0'
        };
        // Debug logging
        logger_1.logger.info(`🔍 Making request: ${method} ${path}${queryString}`);
        logger_1.logger.info(`📝 Signature message: "${method}${timestamp}${path}${queryString}${body}"`);
        logger_1.logger.info(`✍️ Generated signature: ${signature}`);
        logger_1.logger.info(`📤 Request headers: ${JSON.stringify(headers, null, 2)}`);
        try {
            const response = await this.client.request({
                method: method,
                url: path + queryString,
                data: data || undefined, // Ensure undefined for GET requests
                headers
            });
            return response.data;
        }
        catch (error) {
            logger_1.logger.error(`Delta Exchange API error: ${error.message}`);
            if (error.response) {
                logger_1.logger.error(`Response status: ${error.response.status}`);
                logger_1.logger.error(`Response data:`, error.response.data);
            }
            throw error;
        }
    }
    /**
     * Get product by symbol
     */
    getProductBySymbol(symbol) {
        return this.productCache.get(symbol);
    }
    /**
     * Get product ID by symbol
     */
    getProductId(symbol) {
        return this.symbolToProductId.get(symbol);
    }
    /**
     * Check if service is ready for trading
     */
    isReady() {
        return this.isInitialized && !!this.credentials.apiKey && !!this.credentials.apiSecret;
    }
    /**
     * Get account balance
     */
    async getBalance() {
        if (!this.isReady()) {
            throw new Error('Delta Exchange service not ready');
        }
        try {
            const response = await this.makeAuthenticatedRequest('GET', '/v2/wallet/balances');
            if (response.success) {
                return response.result;
            }
            else {
                throw new Error(`Failed to get balance: ${response.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error getting balance:', error);
            throw error;
        }
    }
    /**
     * Get current positions
     */
    async getPositions() {
        if (!this.isReady()) {
            throw new Error('Delta Exchange service not ready');
        }
        try {
            const response = await this.makeAuthenticatedRequest('GET', '/v2/positions');
            if (response.success) {
                return response.result;
            }
            else {
                throw new Error(`Failed to get positions: ${response.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error getting positions:', error);
            throw error;
        }
    }
    /**
     * Place a new order
     */
    async placeOrder(orderRequest) {
        if (!this.isReady()) {
            throw new Error('Delta Exchange service not ready');
        }
        try {
            // Validate order request
            this.validateOrderRequest(orderRequest);
            const response = await this.makeAuthenticatedRequest('POST', '/v2/orders', {}, orderRequest);
            if (response.success) {
                const order = response.result;
                logger_1.logger.info(`✅ Order placed successfully: ${order.side} ${order.size} ${order.product.symbol} @ ${order.limit_price || 'market'}`);
                this.emit('orderPlaced', order);
                return order;
            }
            else {
                throw new Error(`Order placement failed: ${response.error.code} - ${response.error.message}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error placing order:', error);
            this.emit('orderError', error);
            throw error;
        }
    }
    /**
     * Cancel an order
     */
    async cancelOrder(orderId) {
        if (!this.isReady()) {
            throw new Error('Delta Exchange service not ready');
        }
        try {
            const response = await this.makeAuthenticatedRequest('DELETE', `/v2/orders/${orderId}`);
            if (response.success) {
                const order = response.result;
                logger_1.logger.info(`✅ Order cancelled: ${order.id}`);
                this.emit('orderCancelled', order);
                return order;
            }
            else {
                throw new Error(`Order cancellation failed: ${response.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error cancelling order:', error);
            throw error;
        }
    }
    /**
     * Get order status
     */
    async getOrder(orderId) {
        if (!this.isReady()) {
            throw new Error('Delta Exchange service not ready');
        }
        try {
            const response = await this.makeAuthenticatedRequest('GET', `/v2/orders/${orderId}`);
            if (response.success) {
                return response.result;
            }
            else {
                throw new Error(`Failed to get order: ${response.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error getting order:', error);
            throw error;
        }
    }
    /**
     * Get all open orders
     */
    async getOpenOrders(productId) {
        if (!this.isReady()) {
            throw new Error('Delta Exchange service not ready');
        }
        try {
            const params = productId ? { product_id: productId } : {};
            const response = await this.makeAuthenticatedRequest('GET', '/v2/orders', params);
            if (response.success) {
                return response.result;
            }
            else {
                throw new Error(`Failed to get open orders: ${response.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error getting open orders:', error);
            throw error;
        }
    }
    /**
     * Validate order request
     */
    validateOrderRequest(orderRequest) {
        if (!orderRequest.product_id) {
            throw new Error('Product ID is required');
        }
        if (!['buy', 'sell'].includes(orderRequest.side)) {
            throw new Error('Side must be "buy" or "sell"');
        }
        if (!orderRequest.size || orderRequest.size <= 0) {
            throw new Error('Size must be greater than 0');
        }
        if (!['limit_order', 'market_order'].includes(orderRequest.order_type)) {
            throw new Error('Order type must be "limit_order" or "market_order"');
        }
        if (orderRequest.order_type === 'limit_order' && !orderRequest.limit_price) {
            throw new Error('Limit price is required for limit orders');
        }
    }
    /**
     * Get market data for a product
     */
    async getMarketData(symbol) {
        const productId = this.getProductId(symbol);
        if (!productId) {
            throw new Error(`Product not found for symbol: ${symbol}`);
        }
        try {
            const response = await this.client.get(`/v2/products/${productId}`);
            if (response.data.success) {
                return response.data.result;
            }
            else {
                throw new Error(`Failed to get market data: ${response.data.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error getting market data:', error);
            throw error;
        }
    }
    /**
     * Get order book for a product
     */
    async getOrderBook(symbol, depth = 20) {
        const productId = this.getProductId(symbol);
        if (!productId) {
            throw new Error(`Product not found for symbol: ${symbol}`);
        }
        try {
            const response = await this.client.get(`/v2/l2orderbook/${productId}`, {
                params: { depth }
            });
            if (response.data.success) {
                return response.data.result;
            }
            else {
                throw new Error(`Failed to get order book: ${response.data.error}`);
            }
        }
        catch (error) {
            logger_1.logger.error('Error getting order book:', error);
            throw error;
        }
    }
    /**
     * Initialize WebSocket connection for real-time data
     */
    connectWebSocket(symbols) {
        if (this.wsClient) {
            this.wsClient.close();
        }
        try {
            this.wsClient = new ws_1.default(this.wsUrl);
            this.wsClient.on('open', () => {
                logger_1.logger.info('✅ Delta Exchange WebSocket connected');
                this.reconnectAttempts = 0;
                // Subscribe to channels for each symbol
                symbols.forEach(symbol => {
                    const productId = this.getProductId(symbol);
                    if (productId) {
                        this.subscribeToChannel(productId, 'ticker');
                        this.subscribeToChannel(productId, 'l2_orderbook');
                    }
                });
                this.emit('wsConnected');
            });
            this.wsClient.on('message', (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    this.handleWebSocketMessage(message);
                }
                catch (error) {
                    logger_1.logger.error('Error parsing WebSocket message:', error);
                }
            });
            this.wsClient.on('close', () => {
                logger_1.logger.warn('🔌 Delta Exchange WebSocket disconnected');
                this.emit('wsDisconnected');
                this.handleReconnect(symbols);
            });
            this.wsClient.on('error', (error) => {
                logger_1.logger.error('❌ Delta Exchange WebSocket error:', error);
                this.emit('wsError', error);
            });
        }
        catch (error) {
            logger_1.logger.error('Error connecting to WebSocket:', error);
            throw error;
        }
    }
    /**
     * Subscribe to a WebSocket channel
     */
    subscribeToChannel(productId, channel) {
        if (this.wsClient && this.wsClient.readyState === ws_1.default.OPEN) {
            const subscribeMessage = {
                type: 'subscribe',
                payload: {
                    channels: [
                        {
                            name: channel,
                            symbols: [`${productId}`]
                        }
                    ]
                }
            };
            this.wsClient.send(JSON.stringify(subscribeMessage));
            logger_1.logger.info(`📡 Subscribed to ${channel} for product ${productId}`);
        }
    }
    /**
     * Handle WebSocket message
     */
    handleWebSocketMessage(message) {
        if (message.type === 'ticker') {
            this.emit('ticker', message);
        }
        else if (message.type === 'l2_orderbook') {
            this.emit('orderbook', message);
        }
        else if (message.type === 'trade') {
            this.emit('trade', message);
        }
    }
    /**
     * Handle WebSocket reconnection
     */
    handleReconnect(symbols) {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            logger_1.logger.info(`🔄 Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => {
                this.connectWebSocket(symbols);
            }, this.reconnectDelay * this.reconnectAttempts);
        }
        else {
            logger_1.logger.error('❌ Max WebSocket reconnection attempts reached');
            this.emit('wsReconnectFailed');
        }
    }
    /**
     * Disconnect WebSocket
     */
    disconnectWebSocket() {
        if (this.wsClient) {
            this.wsClient.close();
            this.wsClient = undefined;
            logger_1.logger.info('🔌 Delta Exchange WebSocket disconnected manually');
        }
    }
    /**
     * Get all available products
     */
    getAllProducts() {
        return Array.from(this.productCache.values());
    }
    /**
     * Get trading pairs suitable for perpetual futures
     */
    getPerpetualProducts() {
        return this.getAllProducts().filter(product => product.contract_type === 'perpetual_futures' &&
            product.state === 'live');
    }
    /**
     * Cleanup resources
     */
    cleanup() {
        this.disconnectWebSocket();
        this.removeAllListeners();
        logger_1.logger.info('🧹 Delta Exchange service cleaned up');
    }
}
exports.DeltaExchangeUnified = DeltaExchangeUnified;

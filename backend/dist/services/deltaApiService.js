"use strict";
/**
 * Delta Exchange API Service
 * Handles communication with Delta Exchange API (both testnet and mainnet)
 *
 * References:
 * - Official Delta Exchange Documentation: https://docs.delta.exchange
 * - CCXT Delta Exchange Documentation: https://docs.ccxt.com/#/exchanges/delta
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const axios_1 = __importDefault(require("axios"));
const crypto = __importStar(require("crypto"));
const querystring = __importStar(require("querystring"));
// Get API key service for accessing stored keys
const apiKeyService = __importStar(require("./apiKeyService"));
const logger_1 = require("../utils/logger");
// Create logger
const logger = (0, logger_1.createLogger)('DeltaExchangeAPI');
// Environment configuration - Updated with correct URLs from official docs
const MAINNET_BASE_URL = 'https://api.india.delta.exchange';
const TESTNET_BASE_URL = 'https://cdn-ind.testnet.deltaex.org';
// Enhanced rate limit settings based on Delta Exchange documentation
const DEFAULT_RATE_LIMIT = {
    maxRetries: 5, // Increased retries for better reliability
    initialDelay: 2000, // Increased initial delay (2s)
    maxDelay: 30000, // Increased max delay (30s)
    factor: 2.5, // More aggressive backoff
    requestsPerWindow: 8000, // Conservative limit (80% of 10,000)
    windowDuration: 300000, // 5 minutes in ms
    productRateLimit: 400 // Conservative product limit (80% of 500)
};
/**
 * DeltaExchangeAPI Service
 * Provides methods to interact with Delta Exchange API
 */
class DeltaExchangeAPI {
    /**
     * Creates a new instance of the Delta Exchange API client
     * @param {DeltaExchange.ApiOptions} options - Configuration options
     */
    constructor(options = {}) {
        // Enhanced rate limiting tracking
        this.requestCount = 0;
        this.windowStartTime = Date.now();
        this.lastRequestTime = 0;
        this.productRequestCounts = new Map();
        this.testnet = options.testnet || false;
        this.baseUrl = this.testnet ? TESTNET_BASE_URL : MAINNET_BASE_URL;
        this.rateLimit = { ...DEFAULT_RATE_LIMIT, ...(options.rateLimit || {}) };
        this.userId = options.userId;
        this.apiKeys = null;
        this.client = null;
        // Initialize rate limiting tracking
        this.requestCount = 0;
        this.windowStartTime = Date.now();
        this.lastRequestTime = 0;
        this.productRequestCounts = new Map();
        // Log initialization with enhanced details
        logger.info(`🚀 Initializing Enhanced Delta Exchange API client`);
        logger.info(`   Environment: ${this.testnet ? 'TESTNET' : 'PRODUCTION'}`);
        logger.info(`   Base URL: ${this.baseUrl}`);
        logger.info(`   Rate Limits: ${this.rateLimit.requestsPerWindow} req/5min, ${this.rateLimit.productRateLimit} req/sec/product`);
    }
    /**
     * Initializes the API client with credentials
     * @param {DeltaExchange.ApiCredentials} credentials - API credentials (optional, will use stored keys if not provided)
     */
    async initialize(credentials = null) {
        if (credentials) {
            this.apiKeys = credentials;
            logger.debug('Using provided API credentials');
        }
        else if (this.userId) {
            // Retrieve API keys from the secure storage
            logger.debug(`Retrieving API keys for user ${this.userId}`);
            const apiKeyRecord = await apiKeyService.getApiKey(this.userId);
            if (!apiKeyRecord || !apiKeyRecord.key || !apiKeyRecord.secret) {
                logger.error(`No API keys found for user ${this.userId}`);
                throw new Error('No API keys found for this user');
            }
            // Convert ApiKeyRecord to ApiCredentials
            this.apiKeys = {
                key: apiKeyRecord.key,
                secret: apiKeyRecord.secret
            };
        }
        else {
            logger.error('No credentials provided and no userId to retrieve keys');
            throw new Error('No credentials provided and no userId to retrieve keys');
        }
        // Set up axios instance with enhanced configuration
        this.client = axios_1.default.create({
            baseURL: this.baseUrl,
            timeout: 30000, // 30 second timeout
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'SmartMarketOOPS-v2.0', // Required by Delta Exchange
                'Accept': 'application/json'
            },
            // Enhanced retry configuration
            validateStatus: (status) => {
                // Accept 2xx and specific error codes for retry logic
                return (status >= 200 && status < 300) || status === 429 || status === 502 || status === 503;
            }
        });
        // Add response interceptor for logging
        this.client.interceptors.response.use((response) => {
            this._logResponse(response);
            return response;
        }, (error) => {
            this._logError(error);
            return Promise.reject(error);
        });
        logger.info('Delta Exchange API client initialized successfully');
        return this;
    }
    /**
     * Gets server time from Delta Exchange
     * NOTE: This endpoint doesn't exist on Delta Exchange - commented out
     * @returns {Promise<DeltaExchange.ServerTime>} Server time information
     */
    // async getServerTime(): Promise<DeltaExchange.ServerTime> {
    //   return this._makeRequest({
    //     method: 'GET',
    //     endpoint: '/v2/time'
    //   });
    // }
    /**
     * Enhanced market data retrieval with caching and error handling
     * @param {Record<string, any>} params - Query parameters
     * @returns {Promise<DeltaExchange.Market[]>} Available markets
     */
    async getMarkets(params = {}) {
        try {
            logger.info('📊 Fetching markets from Delta Exchange...');
            const markets = await this._makeRequest({
                method: 'GET',
                endpoint: '/v2/products',
                params
            });
            logger.info(`✅ Retrieved ${markets.length} markets from Delta Exchange`);
            // Log important perpetual contracts for debugging
            const perpetuals = markets.filter((m) => m.contract_type === 'perpetual_futures' && m.is_active);
            logger.info(`📈 Active perpetual contracts: ${perpetuals.length}`);
            // Log BTC and ETH contracts specifically
            const btcContract = perpetuals.find((m) => m.symbol === 'BTCUSD');
            const ethContract = perpetuals.find((m) => m.symbol === 'ETHUSD');
            if (btcContract) {
                logger.info(`🟠 BTC Contract: ID=${btcContract.id}, Symbol=${btcContract.symbol}`);
            }
            if (ethContract) {
                logger.info(`🔵 ETH Contract: ID=${ethContract.id}, Symbol=${ethContract.symbol}`);
            }
            return markets;
        }
        catch (error) {
            logger.error('❌ Failed to fetch markets:', error.message);
            throw error;
        }
    }
    /**
     * Enhanced symbol to product ID mapping
     * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
     * @returns {Promise<number>} Product ID
     */
    async getProductIdBySymbol(symbol) {
        try {
            const markets = await this.getMarkets();
            const market = markets.find((m) => m.symbol === symbol);
            if (!market) {
                throw new Error(`Product not found for symbol: ${symbol}`);
            }
            logger.debug(`🔍 Symbol ${symbol} mapped to Product ID: ${market.id}`);
            return market.id;
        }
        catch (error) {
            logger.error(`❌ Failed to get product ID for symbol ${symbol}:`, error.message);
            throw error;
        }
    }
    /**
     * Enhanced product ID to symbol mapping
     * @param {number} productId - Product ID
     * @returns {Promise<string>} Market symbol
     */
    async getSymbolByProductId(productId) {
        try {
            const markets = await this.getMarkets();
            const market = markets.find((m) => m.id === productId);
            if (!market) {
                throw new Error(`Product not found for ID: ${productId}`);
            }
            logger.debug(`🔍 Product ID ${productId} mapped to Symbol: ${market.symbol}`);
            return market.symbol;
        }
        catch (error) {
            logger.error(`❌ Failed to get symbol for product ID ${productId}:`, error.message);
            throw error;
        }
    }
    /**
     * Gets market data for a specific symbol
     * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
     * @returns {Promise<DeltaExchange.Market>} Market data
     */
    async getMarketData(symbol) {
        return this._makeRequest({
            method: 'GET',
            endpoint: `/v2/products/${symbol}`
        });
    }
    /**
     * Gets ticker information for a specific symbol
     * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
     * @returns {Promise<DeltaExchange.Ticker>} Ticker data
     */
    async getTicker(symbol) {
        return this._makeRequest({
            method: 'GET',
            endpoint: `/v2/tickers`,
            params: { symbol }
        });
    }
    /**
     * Gets orderbook for a specific symbol
     * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
     * @param {number} depth - Orderbook depth (default: 10)
     * @returns {Promise<DeltaExchange.Orderbook>} Orderbook data
     */
    async getOrderbook(symbol, depth = 10) {
        return this._makeRequest({
            method: 'GET',
            endpoint: `/v2/l2orderbook/${symbol}`,
            params: { depth }
        });
    }
    /**
     * Gets the user's account information
     * @returns {Promise<DeltaExchange.AccountInfo>} Account information
     */
    async getAccountInfo() {
        return this._makeRequest({
            method: 'GET',
            endpoint: '/v2/user',
            authenticated: true
        });
    }
    /**
     * Gets the user's wallet balances
     * @returns {Promise<DeltaExchange.WalletBalance[]>} Wallet balances
     */
    async getWalletBalances() {
        return this._makeRequest({
            method: 'GET',
            endpoint: '/v2/wallet/balances',
            authenticated: true
        });
    }
    /**
     * Gets the user's active positions
     * @returns {Promise<DeltaExchange.Position[]>} Active positions
     */
    async getPositions() {
        return this._makeRequest({
            method: 'GET',
            endpoint: '/v2/positions',
            authenticated: true
        });
    }
    /**
     * Gets the user's active orders
     * @param {Record<string, any>} params - Query parameters
     * @returns {Promise<DeltaExchange.Order[]>} Active orders
     */
    async getActiveOrders(params = {}) {
        return this._makeRequest({
            method: 'GET',
            endpoint: '/v2/orders',
            params,
            authenticated: true
        });
    }
    /**
     * Enhanced order placement with comprehensive validation and error handling
     * @param {DeltaExchange.OrderParams} order - Order details
     * @returns {Promise<DeltaExchange.Order>} Order information
     */
    async placeOrder(order) {
        try {
            // Enhanced validation
            if (!order.symbol)
                throw new Error('Symbol is required');
            if (!order.side)
                throw new Error('Side is required');
            if (!order.size || order.size <= 0)
                throw new Error('Size must be positive');
            if (order.type === 'limit' && (!order.price || order.price <= 0)) {
                throw new Error('Price is required and must be positive for limit orders');
            }
            // Validate side
            if (!['buy', 'sell'].includes(order.side.toLowerCase())) {
                throw new Error('Side must be "buy" or "sell"');
            }
            // Validate order type
            const validTypes = ['limit', 'market', 'stop_loss_order', 'take_profit_order'];
            if (order.type && !validTypes.includes(order.type)) {
                throw new Error(`Invalid order type. Must be one of: ${validTypes.join(', ')}`);
            }
            // Get product ID for the symbol (Delta Exchange uses product_id in orders)
            let productId;
            try {
                productId = await this.getProductIdBySymbol(order.symbol);
            }
            catch (error) {
                throw new Error(`Invalid symbol: ${order.symbol}. ${error.message}`);
            }
            // Build payload with proper Delta Exchange format
            const payload = {
                product_id: productId, // Delta Exchange uses product_id, not symbol
                side: order.side.toLowerCase(), // Delta Exchange expects lowercase
                size: order.size,
                order_type: order.type === 'limit' ? 'limit_order' :
                    order.type === 'market' ? 'market_order' : order.type,
                time_in_force: order.timeInForce || 'gtc'
            };
            // Add price for limit orders
            if (order.type === 'limit' || !order.type) {
                payload.limit_price = order.price?.toString(); // Delta Exchange expects string
            }
            // Add optional parameters
            if (order.reduceOnly)
                payload.reduce_only = order.reduceOnly;
            if (order.postOnly)
                payload.post_only = order.postOnly;
            if (order.clientOrderId)
                payload.client_order_id = order.clientOrderId;
            logger.info(`🚀 Placing enhanced ${order.side.toUpperCase()} order:`);
            logger.info(`   Symbol: ${order.symbol} (Product ID: ${productId})`);
            logger.info(`   Size: ${order.size} contracts`);
            logger.info(`   Type: ${payload.order_type}`);
            if (order.price)
                logger.info(`   Price: $${order.price}`);
            logger.info(`   Time in Force: ${payload.time_in_force}`);
            const result = await this._makeRequest({
                method: 'POST',
                endpoint: '/v2/orders',
                data: payload,
                authenticated: true
            });
            logger.info(`✅ Order placed successfully! Order ID: ${result.id}`);
            return result;
        }
        catch (error) {
            logger.error(`❌ Failed to place order for ${order.symbol}:`, error.message);
            // Enhanced error handling for common Delta Exchange order errors
            if (error.message.includes('insufficient')) {
                throw new Error(`Insufficient balance to place order: ${error.message}`);
            }
            if (error.message.includes('invalid_product')) {
                throw new Error(`Invalid product/symbol: ${order.symbol}`);
            }
            if (error.message.includes('invalid_size')) {
                throw new Error(`Invalid order size: ${order.size}`);
            }
            if (error.message.includes('invalid_price')) {
                throw new Error(`Invalid order price: ${order.price}`);
            }
            throw error;
        }
    }
    /**
     * Cancels an order
     * @param {string} orderId - Order ID to cancel
     * @returns {Promise<any>} Cancellation response
     */
    async cancelOrder(orderId) {
        logger.info(`Cancelling order ${orderId}`);
        return this._makeRequest({
            method: 'DELETE',
            endpoint: `/v2/orders/${orderId}`,
            authenticated: true
        });
    }
    /**
     * Cancels all active orders
     * @param {DeltaExchange.CancelAllOrdersParams} params - Filter parameters
     * @returns {Promise<any>} Cancellation response
     */
    async cancelAllOrders(params = {}) {
        logger.info('Cancelling all active orders', params);
        return this._makeRequest({
            method: 'DELETE',
            endpoint: '/v2/orders',
            params,
            authenticated: true
        });
    }
    /**
     * Gets order history for the user
     * @param {DeltaExchange.OrderHistoryParams} params - Query parameters
     * @returns {Promise<DeltaExchange.Order[]>} Order history
     */
    async getOrderHistory(params = {}) {
        return this._makeRequest({
            method: 'GET',
            endpoint: '/v2/orders/history',
            params,
            authenticated: true
        });
    }
    /**
     * Gets trade history for the user
     * @param {DeltaExchange.TradeHistoryParams} params - Query parameters
     * @returns {Promise<DeltaExchange.Trade[]>} Trade history
     */
    async getTradeHistory(params = {}) {
        return this._makeRequest({
            method: 'GET',
            endpoint: '/v2/fills',
            params,
            authenticated: true
        });
    }
    /**
     * Enhanced rate limiting check before making requests
     * @private
     * @param {string} productId - Product ID for product-level rate limiting
     */
    async _checkRateLimit(productId) {
        const now = Date.now();
        // Check global rate limit (10,000 requests per 5 minutes)
        if (now - this.windowStartTime >= this.rateLimit.windowDuration) {
            // Reset window
            this.requestCount = 0;
            this.windowStartTime = now;
        }
        if (this.requestCount >= this.rateLimit.requestsPerWindow) {
            const waitTime = this.rateLimit.windowDuration - (now - this.windowStartTime);
            logger.warn(`🚫 Global rate limit reached. Waiting ${waitTime}ms`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            // Reset after waiting
            this.requestCount = 0;
            this.windowStartTime = Date.now();
        }
        // Check product-level rate limit (500 operations per second per product)
        if (productId) {
            const productKey = productId;
            const productLimit = this.productRequestCounts.get(productKey);
            if (productLimit) {
                const timeSinceLastRequest = now - productLimit.windowStart;
                if (timeSinceLastRequest < 1000) { // Within 1 second
                    if (productLimit.count >= this.rateLimit.productRateLimit) {
                        const waitTime = 1000 - timeSinceLastRequest;
                        logger.warn(`🚫 Product rate limit reached for ${productId}. Waiting ${waitTime}ms`);
                        await new Promise(resolve => setTimeout(resolve, waitTime));
                        // Reset product counter
                        this.productRequestCounts.set(productKey, { count: 0, windowStart: Date.now() });
                    }
                }
                else {
                    // Reset if more than 1 second has passed
                    this.productRequestCounts.set(productKey, { count: 0, windowStart: now });
                }
            }
            else {
                this.productRequestCounts.set(productKey, { count: 0, windowStart: now });
            }
            // Increment product counter
            const current = this.productRequestCounts.get(productKey);
            current.count++;
        }
        // Increment global counter
        this.requestCount++;
        // Ensure minimum delay between requests (prevent signature expiration)
        const timeSinceLastRequest = now - this.lastRequestTime;
        if (timeSinceLastRequest < 100) { // Minimum 100ms between requests
            await new Promise(resolve => setTimeout(resolve, 100 - timeSinceLastRequest));
        }
        this.lastRequestTime = Date.now();
    }
    /**
     * Enhanced request method with comprehensive retry logic and rate limiting
     * @private
     * @param {RequestOptions} options - Request options
     * @param {number} retryCount - Current retry count
     * @returns {Promise<any>} API response
     */
    async _makeRequest(options, retryCount = 0) {
        const { method, endpoint, params, data, authenticated } = options;
        if (!this.client) {
            logger.error('❌ API client not initialized');
            throw new Error('API client not initialized. Call initialize() first.');
        }
        // Extract product ID for rate limiting (if available)
        const productId = params?.product_id || data?.product_id || params?.symbol || data?.symbol;
        // Check rate limits before making request
        await this._checkRateLimit(productId);
        try {
            // Prepare request config
            const requestConfig = {
                method,
                url: endpoint
            };
            // Add query parameters if provided
            if (params) {
                requestConfig.params = params;
            }
            // Add data if provided
            if (data) {
                requestConfig.data = data;
            }
            // Add authentication if required
            if (authenticated) {
                if (!this.apiKeys) {
                    logger.error('API keys not initialized');
                    throw new Error('API keys not initialized');
                }
                this._addAuthHeaders(requestConfig);
            }
            // Log the request
            this._logRequest(requestConfig);
            // Make the request
            const response = await this.client(requestConfig);
            // Delta API wraps responses in { result: data, success: boolean, meta: {...} }
            if (response.data && typeof response.data === 'object') {
                if (response.data.success === false) {
                    throw new Error(`Delta API Error: ${JSON.stringify(response.data.error || response.data)}`);
                }
                // Return the result if it exists, otherwise return the full response
                return response.data.result !== undefined ? response.data.result : response.data;
            }
            return response.data;
        }
        catch (error) {
            // Enhanced error handling with specific Delta Exchange error codes
            if (error.response) {
                const status = error.response.status;
                const errorData = error.response.data;
                // Handle rate limiting (429)
                if (status === 429) {
                    if (retryCount < (this.rateLimit.maxRetries || 5)) {
                        // Check for X-RATE-LIMIT-RESET header
                        const resetTime = error.response.headers['x-rate-limit-reset'];
                        let delay = resetTime ? parseInt(resetTime) :
                            Math.min((this.rateLimit.initialDelay || 2000) * Math.pow((this.rateLimit.factor || 2.5), retryCount), (this.rateLimit.maxDelay || 30000));
                        logger.warn(`🚫 Rate limit exceeded (429). Retrying in ${delay}ms (attempt ${retryCount + 1}/${this.rateLimit.maxRetries})`);
                        await new Promise(resolve => setTimeout(resolve, delay));
                        return this._makeRequest(options, retryCount + 1);
                    }
                }
                // Handle server errors (502, 503, 504)
                if ([502, 503, 504].includes(status) && retryCount < 3) {
                    const delay = 1000 * Math.pow(2, retryCount); // 1s, 2s, 4s
                    logger.warn(`🔄 Server error ${status}. Retrying in ${delay}ms (attempt ${retryCount + 1}/3)`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    return this._makeRequest(options, retryCount + 1);
                }
                // Handle specific Delta Exchange errors
                if (errorData && errorData.error) {
                    const errorCode = errorData.error.code || errorData.error;
                    // Handle signature expiration
                    if (errorCode === 'SignatureExpired' || errorCode === 'signature_expired') {
                        if (retryCount < 2) {
                            logger.warn(`🔐 Signature expired. Regenerating and retrying (attempt ${retryCount + 1}/2)`);
                            await new Promise(resolve => setTimeout(resolve, 1000));
                            return this._makeRequest(options, retryCount + 1);
                        }
                    }
                    // Handle IP whitelisting errors
                    if (errorCode === 'ip_not_whitelisted_for_api_key' || errorCode === 'ip_not_whitelisted') {
                        logger.error('🚫 IP not whitelisted for API key. Please whitelist your IP in Delta Exchange settings.');
                        throw new Error(`IP not whitelisted: ${JSON.stringify(errorData)}`);
                    }
                    // Handle invalid API key
                    if (errorCode === 'InvalidApiKey' || errorCode === 'invalid_api_key') {
                        logger.error('🔑 Invalid API key. Please check your credentials.');
                        throw new Error(`Invalid API key: ${JSON.stringify(errorData)}`);
                    }
                    // Handle unauthorized access
                    if (errorCode === 'UnauthorizedApiAccess' || errorCode === 'unauthorized') {
                        logger.error('🚫 Unauthorized API access. Check API key permissions.');
                        throw new Error(`Unauthorized access: ${JSON.stringify(errorData)}`);
                    }
                }
            }
            // Handle other errors
            if (error.response) {
                logger.error(`API Error: ${error.response.status}`, error.response.data);
                throw new Error(`Delta Exchange API Error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
            }
            else if (error.request) {
                logger.error('Request Error', error.message);
                throw new Error(`Delta Exchange API Request Error: ${error.message}`);
            }
            else {
                logger.error('Error', error);
                throw error;
            }
        }
    }
    /**
     * Enhanced authentication header generation with improved signature handling
     * @private
     * @param {AxiosRequestConfig} requestConfig - Axios request configuration
     */
    _addAuthHeaders(requestConfig) {
        if (!this.apiKeys) {
            throw new Error('🔑 API keys not initialized');
        }
        // Generate fresh timestamp (SECONDS, not milliseconds) - Critical for Delta Exchange
        const timestamp = Math.floor(Date.now() / 1000);
        const method = requestConfig.method ? requestConfig.method.toUpperCase() : 'GET';
        const path = requestConfig.url || '';
        // Enhanced signature generation following Delta Exchange official format
        // Format: method + timestamp + path + query_string + body
        let message = method + timestamp.toString() + path;
        // Handle query parameters with proper encoding
        let queryString = '';
        if (requestConfig.params && Object.keys(requestConfig.params).length > 0) {
            // Sort parameters for consistent signature generation
            const sortedParams = Object.keys(requestConfig.params)
                .sort()
                .reduce((result, key) => {
                result[key] = requestConfig.params[key];
                return result;
            }, {});
            queryString = querystring.stringify(sortedParams);
            if (queryString) {
                message += '?' + queryString;
            }
        }
        // Handle request body with proper JSON formatting
        let body = '';
        if (requestConfig.data) {
            // Ensure consistent JSON formatting (no extra spaces)
            if (typeof requestConfig.data === 'string') {
                body = requestConfig.data;
            }
            else {
                body = JSON.stringify(requestConfig.data, null, 0); // No formatting
            }
            message += body;
        }
        // Generate signature using HMAC SHA256
        const signature = crypto
            .createHmac('sha256', this.apiKeys.secret)
            .update(message, 'utf8')
            .digest('hex');
        // Set enhanced authentication headers
        if (!requestConfig.headers) {
            requestConfig.headers = {};
        }
        requestConfig.headers = {
            ...requestConfig.headers,
            'api-key': this.apiKeys.key,
            'timestamp': timestamp.toString(),
            'signature': signature,
            'User-Agent': 'SmartMarketOOPS-v2.0', // Enhanced user agent
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
        // Enhanced logging for debugging
        logger.debug(`🔐 Enhanced Delta auth headers generated:`);
        logger.debug(`   Method: ${method}`);
        logger.debug(`   Timestamp: ${timestamp}`);
        logger.debug(`   Path: ${path}`);
        logger.debug(`   Query: ${queryString || 'none'}`);
        logger.debug(`   Body length: ${body.length}`);
        logger.debug(`   Signature message: ${message}`);
        logger.debug(`   API Key: ${this.apiKeys.key.substring(0, 8)}...`);
    }
    /**
     * Logs a request
     * @private
     * @param {AxiosRequestConfig} request - Request configuration
     */
    _logRequest(request) {
        // Create a safe copy for logging (remove sensitive data)
        const safeRequest = JSON.parse(JSON.stringify(request));
        if (safeRequest.headers && safeRequest.headers['api-key']) {
            safeRequest.headers['api-key'] = '***';
            safeRequest.headers['signature'] = '***';
        }
        logger.info(`API Request: ${request.method} ${request.url}`);
        logger.debug('Request details', safeRequest);
    }
    /**
     * Logs a response
     * @private
     * @param {AxiosResponse} response - Axios response
     */
    _logResponse(response) {
        logger.info(`API Response (${response.status}): ${response.config.method} ${response.config.url}`);
        // Log response data in debug mode
        if (response.data) {
            logger.debug('Response data', {
                status: response.status,
                statusText: response.statusText,
                headers: response.headers,
                // Only log a sample of the data for large responses
                dataSample: typeof response.data === 'object' ?
                    JSON.stringify(response.data).substring(0, 200) + '...' :
                    response.data
            });
        }
    }
    /**
     * Logs an error
     * @private
     * @param {any} error - Axios error
     */
    _logError(error) {
        if (error.response) {
            // The request was made and the server responded with a status code outside of 2xx
            logger.error(`API Error (${error.response.status}): ${error.config.method} ${error.config.url}`, error.response.data);
        }
        else if (error.request) {
            // The request was made but no response was received
            logger.error('Request Error', { message: error.message, request: error.request });
        }
        else {
            // Something happened in setting up the request
            logger.error('Config Error', error.message);
        }
    }
}
exports.default = DeltaExchangeAPI;
//# sourceMappingURL=deltaApiService.js.map
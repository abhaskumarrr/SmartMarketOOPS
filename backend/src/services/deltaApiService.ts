/**
 * Delta Exchange API Service
 * Handles communication with Delta Exchange API (both testnet and mainnet)
 * 
 * References:
 * - Official Delta Exchange Documentation: https://docs.delta.exchange
 * - CCXT Delta Exchange Documentation: https://docs.ccxt.com/#/exchanges/delta
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import * as crypto from 'crypto';
import * as querystring from 'querystring';

// Import DeltaExchange types
import * as DeltaExchange from '../types/deltaExchange';

// Get API key service for accessing stored keys
import * as apiKeyService from './apiKeyService';
import { createLogger } from '../utils/logger';

// Create logger
const logger = createLogger('DeltaExchangeAPI');

// Environment configuration - Updated with correct URLs from official docs
const MAINNET_BASE_URL = 'https://api.india.delta.exchange';
const TESTNET_BASE_URL = 'https://cdn-ind.testnet.deltaex.org';

// Default rate limit settings
const DEFAULT_RATE_LIMIT: DeltaExchange.RateLimitSettings = {
  maxRetries: 3,
  initialDelay: 1000, // ms
  maxDelay: 10000, // ms
  factor: 2 // exponential backoff factor
};

/**
 * Interface for request options used in _makeRequest
 */
interface RequestOptions {
  method: string;
  endpoint: string;
  params?: Record<string, any>;
  data?: Record<string, any>;
  authenticated?: boolean;
}

/**
 * DeltaExchangeAPI Service
 * Provides methods to interact with Delta Exchange API
 */
class DeltaExchangeAPI {
  private testnet: boolean;
  private baseUrl: string;
  private rateLimit: DeltaExchange.RateLimitSettings;
  private userId?: string;
  private apiKeys: DeltaExchange.ApiCredentials | null;
  private client: AxiosInstance | null;

  /**
   * Creates a new instance of the Delta Exchange API client
   * @param {DeltaExchange.ApiOptions} options - Configuration options
   */
  constructor(options: DeltaExchange.ApiOptions = {}) {
    this.testnet = options.testnet || false;
    this.baseUrl = this.testnet ? TESTNET_BASE_URL : MAINNET_BASE_URL;
    this.rateLimit = { ...DEFAULT_RATE_LIMIT, ...(options.rateLimit || {}) };
    this.userId = options.userId;
    this.apiKeys = null;
    this.client = null;
    
    // Log initialization
    logger.info(`Initializing Delta Exchange API client with ${this.testnet ? 'testnet' : 'mainnet'} environment`);
  }

  /**
   * Initializes the API client with credentials
   * @param {DeltaExchange.ApiCredentials} credentials - API credentials (optional, will use stored keys if not provided)
   */
  async initialize(credentials: DeltaExchange.ApiCredentials | null = null): Promise<this> {
    if (credentials) {
      this.apiKeys = credentials;
      logger.debug('Using provided API credentials');
    } else if (this.userId) {
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
    } else {
      logger.error('No credentials provided and no userId to retrieve keys');
      throw new Error('No credentials provided and no userId to retrieve keys');
    }

    // Set up axios instance with default configuration
    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Add response interceptor for logging
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        this._logResponse(response);
        return response;
      },
      (error: any) => {
        this._logError(error);
        return Promise.reject(error);
      }
    );

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
   * Gets all available markets from Delta Exchange
   * @param {Record<string, any>} params - Query parameters
   * @returns {Promise<DeltaExchange.Market[]>} Available markets
   */
  async getMarkets(params: Record<string, any> = {}): Promise<DeltaExchange.Market[]> {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/products',
      params
    });
  }

  /**
   * Gets market data for a specific symbol
   * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
   * @returns {Promise<DeltaExchange.Market>} Market data
   */
  async getMarketData(symbol: string): Promise<DeltaExchange.Market> {
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
  async getTicker(symbol: string): Promise<DeltaExchange.Ticker> {
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
  async getOrderbook(symbol: string, depth: number = 10): Promise<DeltaExchange.Orderbook> {
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
  async getAccountInfo(): Promise<DeltaExchange.AccountInfo> {
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
  async getWalletBalances(): Promise<DeltaExchange.WalletBalance[]> {
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
  async getPositions(): Promise<DeltaExchange.Position[]> {
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
  async getActiveOrders(params: Record<string, any> = {}): Promise<DeltaExchange.Order[]> {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/orders',
      params,
      authenticated: true
    });
  }

  /**
   * Places a new order
   * @param {DeltaExchange.OrderParams} order - Order details
   * @returns {Promise<DeltaExchange.Order>} Order information
   */
  async placeOrder(order: DeltaExchange.OrderParams): Promise<DeltaExchange.Order> {
    // Basic validation
    if (!order.symbol) throw new Error('Symbol is required');
    if (!order.side) throw new Error('Side is required');
    if (!order.size) throw new Error('Size is required');
    if (order.type === 'limit' && !order.price) throw new Error('Price is required for limit orders');

    const payload: Record<string, any> = {
      symbol: order.symbol,
      side: order.side.toUpperCase(),
      size: order.size,
      order_type: order.type || 'limit',
      time_in_force: order.timeInForce || 'gtc'
    };

    if (order.price) payload.price = order.price;
    if (order.reduceOnly) payload.reduce_only = order.reduceOnly;
    if (order.postOnly) payload.post_only = order.postOnly;
    if (order.clientOrderId) payload.client_order_id = order.clientOrderId;

    logger.info(`Placing ${order.side} order for ${order.size} ${order.symbol}`);
    
    return this._makeRequest({
      method: 'POST',
      endpoint: '/v2/orders',
      data: payload,
      authenticated: true
    });
  }

  /**
   * Cancels an order
   * @param {string} orderId - Order ID to cancel
   * @returns {Promise<any>} Cancellation response
   */
  async cancelOrder(orderId: string): Promise<any> {
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
  async cancelAllOrders(params: DeltaExchange.CancelAllOrdersParams = {}): Promise<any> {
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
  async getOrderHistory(params: DeltaExchange.OrderHistoryParams = {}): Promise<DeltaExchange.Order[]> {
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
  async getTradeHistory(params: DeltaExchange.TradeHistoryParams = {}): Promise<DeltaExchange.Trade[]> {
    return this._makeRequest({
      method: 'GET',
      endpoint: '/v2/fills',
      params,
      authenticated: true
    });
  }

  /**
   * Makes a request to the Delta Exchange API with retries and rate limit handling
   * @private
   * @param {RequestOptions} options - Request options
   * @param {number} retryCount - Current retry count
   * @returns {Promise<any>} API response
   */
  private async _makeRequest(options: RequestOptions, retryCount: number = 0): Promise<any> {
    const { method, endpoint, params, data, authenticated } = options;
    
    if (!this.client) {
      logger.error('API client not initialized');
      throw new Error('API client not initialized. Call initialize() first.');
    }
    
    try {
      // Prepare request config
      const requestConfig: AxiosRequestConfig = {
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
    } catch (error: any) {
      // Handle rate limiting errors
      if (error.response && error.response.status === 429) {
        // Rate limit exceeded
        if (retryCount < (this.rateLimit.maxRetries || 3)) {
          // Calculate delay with exponential backoff
          const delay = Math.min(
            (this.rateLimit.initialDelay || 1000) * Math.pow((this.rateLimit.factor || 2), retryCount),
            (this.rateLimit.maxDelay || 10000)
          );
          
          // Log the retry
          logger.warn(`Rate limit exceeded. Retrying in ${delay}ms (attempt ${retryCount + 1}/${this.rateLimit.maxRetries})`);
          
          // Wait and retry
          await new Promise(resolve => setTimeout(resolve, delay));
          return this._makeRequest(options, retryCount + 1);
        }
      }
      
      // Handle other errors
      if (error.response) {
        logger.error(`API Error: ${error.response.status}`, error.response.data);
        throw new Error(`Delta Exchange API Error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
      } else if (error.request) {
        logger.error('Request Error', error.message);
        throw new Error(`Delta Exchange API Request Error: ${error.message}`);
      } else {
        logger.error('Error', error);
        throw error;
      }
    }
  }

  /**
   * Adds authentication headers to a request
   * @private
   * @param {AxiosRequestConfig} requestConfig - Axios request configuration
   */
  private _addAuthHeaders(requestConfig: AxiosRequestConfig): void {
    if (!this.apiKeys) {
      throw new Error('API keys not initialized');
    }

    // Delta Exchange uses SECONDS timestamp (not milliseconds) - per official docs
    const timestamp = Math.floor(Date.now() / 1000);
    const method = requestConfig.method ? requestConfig.method.toUpperCase() : 'GET';
    const path = requestConfig.url || '';

    // Prepare the message to sign according to official Delta Exchange format
    // Format: method + timestamp + path + query_string + body
    let message = method + timestamp.toString() + path;

    // Add query string if it exists
    let queryString = '';
    if (requestConfig.params) {
      queryString = querystring.stringify(requestConfig.params);
      if (queryString) {
        message += '?' + queryString;
      }
    }

    // Add body if it exists
    let body = '';
    if (requestConfig.data) {
      body = JSON.stringify(requestConfig.data);
      message += body;
    }

    // Create the signature using HMAC SHA256
    const signature = crypto
      .createHmac('sha256', this.apiKeys.secret)
      .update(message)
      .digest('hex');

    // Add authentication headers according to Delta Exchange official format
    if (!requestConfig.headers) {
      requestConfig.headers = {};
    }

    requestConfig.headers = {
      ...requestConfig.headers,
      'api-key': this.apiKeys.key,
      'timestamp': timestamp.toString(),
      'signature': signature,
      'User-Agent': 'nodejs-rest-client', // Required per official docs
      'Content-Type': 'application/json'
    };

    logger.debug(`Delta auth headers added - method: ${method}, timestamp: ${timestamp}, path: ${path}, query: ${queryString}, body length: ${body.length}`);
    logger.debug(`Signature message: ${message}`);
  }

  /**
   * Logs a request
   * @private
   * @param {AxiosRequestConfig} request - Request configuration
   */
  private _logRequest(request: AxiosRequestConfig): void {
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
  private _logResponse(response: AxiosResponse): void {
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
  private _logError(error: any): void {
    if (error.response) {
      // The request was made and the server responded with a status code outside of 2xx
      logger.error(
        `API Error (${error.response.status}): ${error.config.method} ${error.config.url}`,
        error.response.data
      );
    } else if (error.request) {
      // The request was made but no response was received
      logger.error('Request Error', { message: error.message, request: error.request });
    } else {
      // Something happened in setting up the request
      logger.error('Config Error', error.message);
    }
  }
}

export default DeltaExchangeAPI; 
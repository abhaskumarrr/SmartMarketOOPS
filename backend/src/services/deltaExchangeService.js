/**
 * Delta Exchange Service
 * Handles all interactions with Delta Exchange API based on CCXT documentation
 */

const axios = require('axios');
const crypto = require('crypto');
const { env } = require('../utils/env');

class DeltaExchangeService {
  constructor(apiKey, apiSecret, isTestnet = false) {
    this.apiKey = apiKey;
    this.apiSecret = apiSecret;
    this.baseUrl = isTestnet 
      ? 'https://testnet.delta.exchange/api/v2' 
      : 'https://api.delta.exchange/v2';
  }

  /**
   * Sign a request with HMAC signature
   * @param {string} method - HTTP method
   * @param {string} path - Endpoint path
   * @param {Object} params - Query params or body data
   * @returns {Object} - Headers with signature
   */
  createSignedHeaders(method, path, params = {}) {
    const timestamp = Math.floor(Date.now() / 1000);
    let message = timestamp + method + path;
    
    // Add query params or body params if present
    if (Object.keys(params).length > 0) {
      const sortedParams = Object.keys(params).sort().reduce((acc, key) => {
        acc[key] = params[key];
        return acc;
      }, {});
      const paramString = new URLSearchParams(sortedParams).toString();
      message += paramString;
    }
    
    // Create HMAC signature
    const signature = crypto
      .createHmac('sha256', this.apiSecret)
      .update(message)
      .digest('hex');
    
    return {
      'api-key': this.apiKey,
      'timestamp': timestamp.toString(),
      'signature': signature
    };
  }

  /**
   * Execute a request to Delta Exchange API
   * @param {string} method - HTTP method
   * @param {string} endpoint - API endpoint
   * @param {Object} params - Query params or body data
   * @returns {Promise<Object>} - API response
   */
  async request(method, endpoint, params = {}) {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      const headers = this.createSignedHeaders(method, endpoint, params);
      const config = { 
        method, 
        url, 
        headers
      };
      
      // Add params as query or body depending on method
      if (method === 'GET' && Object.keys(params).length > 0) {
        config.params = params;
      } else if (Object.keys(params).length > 0) {
        config.data = params;
      }
      
      const response = await axios(config);
      return response.data;
    } catch (error) {
      if (error.response) {
        throw new Error(
          JSON.stringify({
            status: error.response.status,
            message: error.response.data.message || 'API error',
            data: error.response.data
          })
        );
      }
      throw error;
    }
  }

  /**
   * Fetches all available markets
   * @param {Object} params - Optional parameters
   * @returns {Promise<Array>} - List of available markets
   * @see https://docs.delta.exchange/#get-list-of-products
   */
  async fetchMarkets(params = {}) {
    return this.request('GET', '/products', params);
  }

  /**
   * Fetches ticker information for a specific market
   * @param {string} symbol - Market symbol
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Ticker information
   * @see https://docs.delta.exchange/#get-ticker-for-a-product-by-symbol
   */
  async fetchTicker(symbol, params = {}) {
    return this.request('GET', `/products/${symbol}/ticker`, params);
  }

  /**
   * Fetches account balance information
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Balance information
   * @see https://docs.delta.exchange/#get-wallet-balances
   */
  async fetchBalance(params = {}) {
    return this.request('GET', '/wallet/balances', params);
  }

  /**
   * Fetches data on a single open position
   * @param {string} symbol - Market symbol
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Position information
   * @see https://docs.delta.exchange/#get-position
   */
  async fetchPosition(symbol, params = {}) {
    return this.request('GET', '/positions', { 
      product_id: symbol,
      ...params 
    });
  }

  /**
   * Fetches the status of the exchange API
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Status information
   */
  async fetchStatus(params = {}) {
    return this.request('GET', '/system-status', params);
  }

  /**
   * Fetches all available currencies
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Currencies information
   * @see https://docs.delta.exchange/#get-list-of-all-assets
   */
  async fetchCurrencies(params = {}) {
    return this.request('GET', '/assets', params);
  }

  /**
   * Closes all open positions for a market type
   * @param {Object} params - Optional parameters
   * @returns {Promise<Array>} - List of closed positions
   * @see https://docs.delta.exchange/#close-all-positions
   */
  async closeAllPositions(params = {}) {
    return this.request('POST', '/positions/close_all', params);
  }

  /**
   * Fetches the margin mode of a trading pair
   * @param {string} symbol - Market symbol
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Margin mode information
   * @see https://docs.delta.exchange/#get-user
   */
  async fetchMarginMode(symbol, params = {}) {
    // First get user info
    const user = await this.request('GET', '/users/me', params);
    // Then check margin mode for the specified symbol
    return {
      user,
      symbol,
      marginMode: user.isolated_margin_enabled ? 'isolated' : 'cross'
    };
  }

  /**
   * Fetches funding rates for multiple markets
   * @param {Array<string>} symbols - List of market symbols
   * @param {Object} params - Optional parameters
   * @returns {Promise<Array>} - List of funding rate information
   * @see https://docs.delta.exchange/#get-tickers-for-products
   */
  async fetchFundingRates(symbols, params = {}) {
    const query = symbols && symbols.length > 0 
      ? { symbols: symbols.join(','), ...params }
      : params;
    return this.request('GET', '/tickers', query);
  }

  /**
   * Adds margin to a position
   * @param {string} symbol - Market symbol
   * @param {number} amount - Amount of margin to add
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Updated position information
   * @see https://docs.delta.exchange/#add-remove-position-margin
   */
  async addMargin(symbol, amount, params = {}) {
    return this.request('POST', '/positions/add_margin', {
      product_id: symbol,
      margin: amount,
      ...params
    });
  }

  /**
   * Fetches the set leverage for a market
   * @param {string} symbol - Market symbol
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Leverage information
   * @see https://docs.delta.exchange/#get-order-leverage
   */
  async fetchLeverage(symbol, params = {}) {
    return this.request('GET', '/orders/leverage', {
      product_id: symbol,
      ...params
    });
  }

  /**
   * Sets the level of leverage for a market
   * @param {number} leverage - The rate of leverage
   * @param {string} symbol - Market symbol
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Response from the exchange
   * @see https://docs.delta.exchange/#change-order-leverage
   */
  async setLeverage(leverage, symbol, params = {}) {
    return this.request('POST', '/orders/leverage', {
      product_id: symbol,
      leverage,
      ...params
    });
  }

  /**
   * Creates a new order
   * @param {string} symbol - Market symbol
   * @param {string} type - Order type (limit, market, etc.)
   * @param {string} side - Order side (buy, sell)
   * @param {number} amount - Order amount
   * @param {number} price - Order price (for limit orders)
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - New order information
   * @see https://docs.delta.exchange/#place-order
   */
  async createOrder(symbol, type, side, amount, price, params = {}) {
    const orderParams = {
      product_id: symbol,
      side: side.toLowerCase(),
      size: amount.toString(),
      ...params
    };

    if (type.toLowerCase() === 'limit') {
      orderParams.limit_price = price.toString();
      orderParams.order_type = 'limit_order';
    } else if (type.toLowerCase() === 'market') {
      orderParams.order_type = 'market_order';
    }

    return this.request('POST', '/orders', orderParams);
  }

  /**
   * Cancels an existing order
   * @param {string} id - Order ID
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Cancelled order information
   * @see https://docs.delta.exchange/#cancel-order
   */
  async cancelOrder(id, params = {}) {
    return this.request('DELETE', `/orders/${id}`, params);
  }

  /**
   * Fetches an order by ID
   * @param {string} id - Order ID
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Order information
   * @see https://docs.delta.exchange/#get-order-by-id
   */
  async fetchOrder(id, params = {}) {
    return this.request('GET', `/orders/${id}`, params);
  }

  /**
   * Fetches open orders
   * @param {string} symbol - Market symbol (optional)
   * @param {Object} params - Optional parameters
   * @returns {Promise<Array>} - List of open orders
   * @see https://docs.delta.exchange/#get-open-orders
   */
  async fetchOpenOrders(symbol = undefined, params = {}) {
    const query = symbol ? { product_id: symbol, ...params } : params;
    return this.request('GET', '/orders', query);
  }

  /**
   * Fetches order history
   * @param {string} symbol - Market symbol (optional)
   * @param {Object} params - Optional parameters
   * @returns {Promise<Array>} - List of historical orders
   * @see https://docs.delta.exchange/#get-order-history
   */
  async fetchOrderHistory(symbol = undefined, params = {}) {
    const query = symbol ? { product_id: symbol, ...params } : params;
    return this.request('GET', '/orders/history', query);
  }

  /**
   * Fetches trade history
   * @param {string} symbol - Market symbol (optional)
   * @param {Object} params - Optional parameters
   * @returns {Promise<Array>} - List of trades
   * @see https://docs.delta.exchange/#get-trade-history
   */
  async fetchMyTrades(symbol = undefined, params = {}) {
    const query = symbol ? { product_id: symbol, ...params } : params;
    return this.request('GET', '/fills', query);
  }

  /**
   * Fetches the orderbook for a market
   * @param {string} symbol - Market symbol
   * @param {Object} params - Optional parameters
   * @returns {Promise<Object>} - Orderbook information
   * @see https://docs.delta.exchange/#get-l2-orderbook
   */
  async fetchOrderBook(symbol, params = {}) {
    return this.request('GET', `/orderbooks/${symbol}`, params);
  }

  /**
   * Fetches recent trades for a market
   * @param {string} symbol - Market symbol
   * @param {number} limit - Number of trades to fetch
   * @param {Object} params - Optional parameters
   * @returns {Promise<Array>} - List of recent trades
   * @see https://docs.delta.exchange/#get-trade-history-for-a-product
   */
  async fetchTrades(symbol, limit = 100, params = {}) {
    return this.request('GET', '/trades', { 
      product_id: symbol, 
      limit: limit.toString(),
      ...params 
    });
  }
}

/**
 * Creates a Delta Exchange service instance using environment credentials
 * @returns {DeltaExchangeService} - Service instance
 */
function createDefaultService() {
  return new DeltaExchangeService(
    env.DELTA_EXCHANGE_API_KEY,
    env.DELTA_EXCHANGE_API_SECRET,
    env.DELTA_EXCHANGE_TESTNET
  );
}

/**
 * Creates a Delta Exchange service instance using provided credentials
 * @param {string} apiKey - API Key
 * @param {string} apiSecret - API Secret
 * @param {boolean} isTestnet - Whether to use testnet
 * @returns {DeltaExchangeService} - Service instance
 */
function createService(apiKey, apiSecret, isTestnet = false) {
  return new DeltaExchangeService(apiKey, apiSecret, isTestnet);
}

module.exports = {
  DeltaExchangeService,
  createDefaultService,
  createService
}; 
import axios from 'axios';

/**
 * API Client for Delta Exchange
 * Provides methods to interact with the Delta Exchange API through our backend
 */
class DeltaExchangeApi {
  private baseUrl: string;
  
  constructor() {
    this.baseUrl = '/api/delta';
  }
  
  /**
   * Get all available trading products/markets
   */
  async getProducts() {
    try {
      const response = await axios.get(`${this.baseUrl}/products`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch products');
    }
  }
  
  /**
   * Get order book for a specific product
   * @param productId - Product ID or symbol
   */
  async getOrderBook(productId: string) {
    try {
      const response = await axios.get(`${this.baseUrl}/products/${productId}/orderbook`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch order book');
    }
  }
  
  /**
   * Get recent trades for a specific product
   * @param productId - Product ID or symbol
   * @param limit - Number of trades to return
   */
  async getRecentTrades(productId: string, limit = 100) {
    try {
      const response = await axios.get(`${this.baseUrl}/products/${productId}/trades`, {
        params: { limit }
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch recent trades');
    }
  }
  
  /**
   * Get account balance
   */
  async getBalance() {
    try {
      const response = await axios.get(`${this.baseUrl}/balance`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch account balance');
    }
  }
  
  /**
   * Get positions
   * @param symbol - Optional product symbol to filter positions
   */
  async getPositions(symbol?: string) {
    try {
      const params = symbol ? { symbol } : {};
      const response = await axios.get(`${this.baseUrl}/positions`, { params });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch positions');
    }
  }
  
  /**
   * Get open orders
   * @param symbol - Optional product symbol to filter orders
   * @param status - Order status filter (default: 'open')
   */
  async getOrders(symbol?: string, status = 'open') {
    try {
      const params = { 
        ...(symbol ? { symbol } : {}),
        status
      };
      const response = await axios.get(`${this.baseUrl}/orders`, { params });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch orders');
    }
  }
  
  /**
   * Create a new order
   * @param orderData - Order data
   */
  async createOrder(orderData: {
    product_id: string;
    side: 'buy' | 'sell';
    size: number | string;
    limit_price?: number | string;
    order_type?: 'limit_order' | 'market_order';
    [key: string]: any;
  }) {
    try {
      const response = await axios.post(`${this.baseUrl}/orders`, orderData);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to create order');
    }
  }
  
  /**
   * Cancel an order
   * @param orderId - Order ID to cancel
   */
  async cancelOrder(orderId: string) {
    try {
      const response = await axios.delete(`${this.baseUrl}/orders/${orderId}`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to cancel order');
    }
  }
  
  /**
   * Cancel all orders with optional filters
   * @param filters - Optional filters (product_id, side, etc.)
   */
  async cancelAllOrders(filters: Record<string, any> = {}) {
    try {
      const response = await axios.delete(`${this.baseUrl}/orders`, { data: filters });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to cancel all orders');
    }
  }
  
  /**
   * Get order history
   * @param symbol - Optional product symbol
   * @param limit - Maximum number of orders to return
   * @param offset - Pagination offset
   */
  async getOrderHistory(symbol?: string, limit?: number, offset?: number, status?: string) {
    try {
      const params: Record<string, any> = {};
      if (symbol) params.symbol = symbol;
      if (limit) params.limit = limit;
      if (offset) params.offset = offset;
      if (status) params.status = status;
      
      const response = await axios.get(`${this.baseUrl}/orders/history`, { params });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch order history');
    }
  }
  
  /**
   * Get trade history
   * @param symbol - Optional product symbol
   * @param limit - Maximum number of trades to return
   * @param offset - Pagination offset
   */
  async getTradeHistory(symbol?: string, limit?: number, offset?: number, orderId?: string) {
    try {
      const params: Record<string, any> = {};
      if (symbol) params.symbol = symbol;
      if (limit) params.limit = limit;
      if (offset) params.offset = offset;
      if (orderId) params.order_id = orderId;
      
      const response = await axios.get(`${this.baseUrl}/fills`, { params });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch trade history');
    }
  }
  
  /**
   * Get comprehensive market data for a symbol
   * @param symbol - Product symbol (default: 'BTCUSDT')
   */
  async getMarketData(symbol = 'BTCUSDT') {
    try {
      const response = await axios.get(`${this.baseUrl}/market-data`, {
        params: { symbol }
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch market data');
    }
  }
  
  /**
   * Get leverage settings for a symbol
   * @param symbol - Product symbol
   */
  async getLeverage(symbol: string) {
    try {
      const response = await axios.get(`${this.baseUrl}/leverage`, {
        params: { symbol }
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch leverage');
    }
  }
  
  /**
   * Set leverage for a symbol
   * @param symbol - Product symbol
   * @param leverage - Leverage value
   */
  async setLeverage(symbol: string, leverage: number) {
    try {
      const response = await axios.post(`${this.baseUrl}/leverage`, {
        symbol,
        leverage
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to set leverage');
    }
  }
  
  /**
   * Get available currencies
   */
  async getCurrencies() {
    try {
      const response = await axios.get(`${this.baseUrl}/currencies`);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to fetch currencies');
    }
  }
  
  /**
   * Close all positions
   * @param params - Optional parameters
   */
  async closeAllPositions(params: Record<string, any> = {}) {
    try {
      const response = await axios.post(`${this.baseUrl}/positions/close-all`, params);
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to close all positions');
    }
  }
  
  /**
   * Add margin to a position
   * @param symbol - Product symbol
   * @param amount - Amount of margin to add
   */
  async addMargin(symbol: string, amount: number) {
    try {
      const response = await axios.post(`${this.baseUrl}/positions/add-margin`, {
        symbol,
        amount
      });
      return response.data;
    } catch (error) {
      this.handleError(error, 'Failed to add margin');
    }
  }
  
  /**
   * Error handling helper
   */
  private handleError(error: any, defaultMessage: string) {
    const errorMsg = error.response?.data?.message || defaultMessage;
    console.error(errorMsg, error);
    throw new Error(errorMsg);
  }
}

// Create singleton instance
const deltaExchangeApi = new DeltaExchangeApi();

export default deltaExchangeApi; 
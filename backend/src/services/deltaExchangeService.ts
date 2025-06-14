/**
 * Delta Exchange India Trading Service
 * Comprehensive integration with Delta Exchange India API
 */

import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import crypto from 'crypto';
import { createLogger, LogData } from '../utils/logger';

// Use structured logger instead of simple console logger
const logger = createLogger('DeltaExchangeService');

export interface DeltaCredentials {
  apiKey: string;
  apiSecret: string;
  testnet: boolean;
}

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high24h: number;
  low24h: number;
  timestamp: number;
  source: string;
  markPrice?: number;
  indexPrice?: number;
  openInterest?: number;
}

export interface OrderRequest {
  product_id: number;
  size: number;
  side: 'buy' | 'sell';
  order_type: 'market_order' | 'limit_order' | 'stop_market_order' | 'stop_limit_order';
  limit_price?: string;
  stop_price?: string;
  time_in_force?: 'gtc' | 'ioc' | 'fok';
  post_only?: boolean;
  reduce_only?: boolean;
  client_order_id?: string;
}

export interface Order {
  id: number;
  product_id: number;
  symbol: string;
  size: number;
  unfilled_size: number;
  side: 'buy' | 'sell';
  order_type: string;
  state: 'open' | 'pending' | 'closed' | 'cancelled';
  limit_price?: string;
  stop_price?: string;
  average_fill_price?: string;
  created_at: string;
  updated_at: string;
}

export interface Position {
  product_id: number;
  symbol: string;
  size: number;
  entry_price: string;
  margin: string;
  liquidation_price: string;
  bankruptcy_price: string;
  realized_pnl: string;
  unrealized_pnl: string;
}

export interface Balance {
  asset_id: number;
  asset_symbol: string;
  balance: string;
  available_balance: string;
  blocked_margin: string;
}

class DeltaExchangeService {
  private client: AxiosInstance;
  private credentials: DeltaCredentials;
  private baseUrl: string;
  private isInitialized = false;
  private productCache: Map<string, any> = new Map();
  private symbolToProductId: Map<string, number> = new Map();
  // Add cache timeouts
  private lastProductUpdateTime = 0;
  private readonly CACHE_TTL_MS = 1000 * 60 * 15; // 15 minutes
  private retryCount = 0;
  private readonly MAX_RETRIES = 3;

  constructor(credentials: DeltaCredentials) {
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

    // Initialize product ID mappings for perpetual futures (VERIFIED IDs from API)
    if (credentials.testnet) {
      // Testnet product IDs (verified 2025-01-27)
      this.symbolToProductId.set('BTCUSD', 84);
      this.symbolToProductId.set('ETHUSD', 1699);
      this.symbolToProductId.set('SOLUSD', 92572);
      this.symbolToProductId.set('ADAUSD', 101760);
    } else {
      // Production product IDs (verified 2025-01-27)
      this.symbolToProductId.set('BTCUSD', 27);
      this.symbolToProductId.set('ETHUSD', 3136);
      this.symbolToProductId.set('SOLUSD', 14823);
      this.symbolToProductId.set('ADAUSD', 16614);
      this.symbolToProductId.set('DOTUSD', 15304);
    }

    // Start initialization but don't wait for it
    this.initializeService().catch(error => {
      logger.error('Failed to initialize Delta Exchange Service:', error instanceof Error ? error.message : 'Unknown error');
    });
  }

  /**
   * Initialize the service and load products
   */
  private async initializeService(): Promise<void> {
    try {
      await this.loadProducts();
      this.isInitialized = true;
      logger.info(`✅ Delta Exchange Service initialized (${this.credentials.testnet ? 'TESTNET' : 'PRODUCTION'})`);
      logger.info(`🔗 Base URL: ${this.baseUrl}`);
      logger.info(`📊 Loaded ${this.productCache.size} products`);

      // Verify API connection with a simple public endpoint call
      try {
        const response = await this.client.get('/v2/time');
        if (response.data && response.data.success) {
          logger.info(`✅ Delta Exchange API connection verified (server time: ${response.data.result.server_time})`);
        }
      } catch (error) {
        logger.warn('API connection test failed, but continuing with initialization', { error });
      }
    } catch (error) {
      const errorObj: LogData = { error: error instanceof Error ? error.message : 'Unknown error' };
      logger.error('❌ Failed to initialize Delta Exchange Service', errorObj);
      this.isInitialized = false;
      // Schedule retry
      if (this.retryCount < this.MAX_RETRIES) {
        this.retryCount++;
        const retryDelay = Math.pow(2, this.retryCount) * 1000; // Exponential backoff
        logger.info(`Will retry initialization in ${retryDelay/1000} seconds (attempt ${this.retryCount}/${this.MAX_RETRIES})`);
        setTimeout(() => this.initializeService(), retryDelay);
      }
    }
  }

  /**
   * Load and cache all available products
   */
  private async loadProducts(): Promise<void> {
    // Skip loading if cache is still fresh
    const now = Date.now();
    if (this.productCache.size > 0 && (now - this.lastProductUpdateTime) < this.CACHE_TTL_MS) {
      logger.info(`Using cached products (cache age: ${Math.round((now - this.lastProductUpdateTime)/1000)}s)`);
      return;
    }

    try {
      const response = await this.client.get('/v2/products');

      if (response.data.success) {
        const products = response.data.result;

        // Clear existing cache before updating
        this.productCache.clear();
        
        for (const product of products) {
          this.productCache.set(product.symbol, product);
          this.symbolToProductId.set(product.symbol, product.id);
        }

        this.lastProductUpdateTime = now;
        logger.info(`📦 Cached ${products.length} products`);

        // Log major trading pairs
        const majorPairs = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'];
        const availablePairs = majorPairs.filter(pair => this.symbolToProductId.has(pair));
        logger.info(`🎯 Available major pairs: ${availablePairs.join(', ')}`);

      } else {
        throw new Error(`API Error: ${response.data.error}`);
      }
    } catch (error) {
      const errorObj: LogData = { error: error instanceof Error ? error.message : 'Unknown error' };
      logger.error('Failed to load products', errorObj);
      
      // If cache is empty, this is a critical error
      if (this.productCache.size === 0) {
        throw error;
      } else {
        // Otherwise log warning but continue with cached data
        logger.warn('Using stale product cache due to API error');
      }
    }
  }

  /**
   * Generate HMAC-SHA256 signature for authentication
   */
  private generateSignature(method: string, path: string, queryString: string, body: string, timestamp: string): string {
    const message = method + timestamp + path + queryString + body;
    
    if (!this.credentials.apiSecret) {
      const errorObj: LogData = { error: 'Missing API Secret' };
      logger.error('API Secret is missing or empty', errorObj);
      throw new Error('API Secret is required for authentication');
    }
    
    return crypto.createHmac('sha256', this.credentials.apiSecret).update(message).digest('hex');
  }

  /**
   * Make authenticated request to Delta Exchange API
   */
  private async makeAuthenticatedRequest(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    path: string,
    params: any = {},
    data: any = null,
    retryCount = 0
  ): Promise<any> {
    if (!this.credentials.apiKey || !this.credentials.apiSecret) {
      throw new Error('API Key and Secret are required for authenticated requests');
    }

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
    if (process.env.DEBUG_API === 'true') {
      const debugData: LogData = {
        method,
        path,
        queryString,
        bodyLength: body ? body.length : 0,
        timestamp
      };
      logger.debug('🔍 API request details', debugData);
    }

    try {
      // Use axios directly instead of the client instance to avoid conflicts
      const fullUrl = this.baseUrl + path + queryString;

      const response = await axios.request({
        method,
        url: fullUrl,
        data: data,
        headers,
        timeout: 30000
      });

      return response.data;
    } catch (error) {
      const axiosError = error as AxiosError;
      
      // Handle rate limits with automatic retry
      if (axiosError.response?.status === 429 && retryCount < this.MAX_RETRIES) {
        const retryDelay = Math.pow(2, retryCount + 1) * 1000; // Exponential backoff
        logger.warn(`Rate limited, retrying in ${retryDelay/1000} seconds...`);
        await this.delay(retryDelay);
        return this.makeAuthenticatedRequest(method, path, params, data, retryCount + 1);
      }
      
      // Handle authentication errors specifically
      if (axiosError.response?.status === 401) {
        const errorData: LogData = { status: 401 };
        logger.error('Authentication failed. Please check API keys.', errorData);
        throw new Error('Delta Exchange authentication failed. Invalid API credentials.');
      }

      if (axiosError.response) {
        const responseData = axiosError.response.data as any;
        const errorData: LogData = { 
          status: axiosError.response.status,
          error: responseData?.error || axiosError.response.statusText
        };
        logger.error(`API Error: ${axiosError.response.status}`, errorData);
        throw new Error(`Delta API Error: ${responseData?.error || axiosError.response.statusText}`);
      } else {
        const errorObj: LogData = { error: axiosError.message };
        logger.error(`Request Error`, errorObj);
        throw error;
      }
    }
  }

  /**
   * Check if service is ready
   */
  public isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Get product ID from symbol
   */
  public getProductId(symbol: string): number | null {
    return this.symbolToProductId.get(symbol) || null;
  }

  /**
   * Get product information
   */
  public getProduct(symbol: string): any | null {
    return this.productCache.get(symbol) || null;
  }

  /**
   * Get all available products
   */
  public getAllProducts(): any[] {
    return Array.from(this.productCache.values());
  }

  /**
   * Get supported symbols
   */
  public getSupportedSymbols(): string[] {
    return Array.from(this.symbolToProductId.keys());
  }

  /**
   * Get real-time market data for a symbol
   */
  public async getMarketData(symbol: string): Promise<MarketData | null> {
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
      } else {
        logger.error(`Failed to get market data for ${symbol}:`, response.data.error);
        return this.getMockMarketData(symbol);
      }
    } catch (error) {
      logger.error(`Error fetching market data for ${symbol}:`, error instanceof Error ? error.message : 'Unknown error');
      return this.getMockMarketData(symbol);
    }
  }

  /**
   * Get mock market data as fallback
   */
  private getMockMarketData(symbol: string): MarketData {
    const mockPrices: { [key: string]: number } = {
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
  public async getMultipleMarketData(symbols: string[]): Promise<MarketData[]> {
    const results: MarketData[] = [];

    // Use batch ticker API if available, otherwise fetch individually
    try {
      const response = await this.client.get('/v2/tickers');

      if (response.data.success) {
        const tickers = response.data.result;

        for (const symbol of symbols) {
          const ticker = tickers.find((t: any) => t.symbol === symbol);
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
    } catch (error) {
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
  public async placeOrder(orderRequest: OrderRequest): Promise<Order | null> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange Service not initialized');
    }

    try {
      const response = await this.makeAuthenticatedRequest('POST', '/v2/orders', {}, orderRequest);

      if (response.success) {
        logger.info(`✅ Order placed: ${orderRequest.side} ${orderRequest.size} @ ${orderRequest.limit_price || 'market'}`);
        return response.result;
      } else {
        logger.error('Failed to place order:', response.error);
        throw new Error(`Order placement failed: ${response.error.code || 'Unknown error'}`);
      }
    } catch (error) {
      logger.error('Error placing order:', error instanceof Error ? error.message : 'Unknown error');
      throw error;
    }
  }

  /**
   * Cancel an order
   */
  public async cancelOrder(productId: number, orderId: number): Promise<boolean> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange Service not initialized');
    }

    try {
      const response = await this.makeAuthenticatedRequest('DELETE', `/v2/orders/${orderId}`, { product_id: productId });

      if (response.success) {
        logger.info(`✅ Order cancelled: ${orderId}`);
        return true;
      } else {
        logger.error('Failed to cancel order:', response.error);
        return false;
      }
    } catch (error) {
      logger.error('Error cancelling order:', error instanceof Error ? error.message : 'Unknown error');
      return false;
    }
  }

  /**
   * Get open orders
   */
  public async getOpenOrders(productId?: number): Promise<Order[]> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange Service not initialized');
    }

    try {
      const params = productId ? { product_id: productId, state: 'open' } : { state: 'open' };
      const response = await this.makeAuthenticatedRequest('GET', '/v2/orders', params);

      if (response.success) {
        return response.result;
      } else {
        logger.error('Failed to get open orders:', response.error);
        return [];
      }
    } catch (error) {
      logger.error('Error getting open orders:', error instanceof Error ? error.message : 'Unknown error');
      return [];
    }
  }

  /**
   * Get positions
   */
  public async getPositions(): Promise<Position[]> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange Service not initialized');
    }

    try {
      // Delta Exchange requires either product_id or underlying_asset_symbol
      // Get positions for BTC by default
      const response = await this.makeAuthenticatedRequest('GET', '/v2/positions', {
        underlying_asset_symbol: 'BTC'
      });

      if (response.success) {
        return response.result;
      } else {
        logger.error('Failed to get positions:', response.error);
        return [];
      }
    } catch (error) {
      logger.error('Error getting positions:', error instanceof Error ? error.message : 'Unknown error');
      return [];
    }
  }

  /**
   * Get wallet balances using proper Delta Exchange API
   */
  public async getBalances(): Promise<Balance[]> {
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
        const nonZeroBalances = balances.filter((balance: any) =>
          balance.balance && parseFloat(balance.balance) > 0
        );

        logger.info(`✅ Successfully fetched ${nonZeroBalances.length} non-zero balances from Delta Exchange`);
        return nonZeroBalances;
      } else {
        logger.error('Failed to get balances - API response:', response);
        return [];
      }

    } catch (error) {
      logger.error('❌ Error getting REAL balances:', error instanceof Error ? error.message : 'Unknown error');

      // Check if it's an IP whitelisting issue
      if (error instanceof Error && error.message.includes('ip_not_whitelisted')) {
        logger.error('🚫 IP NOT WHITELISTED: Please whitelist your IP address in Delta Exchange dashboard');
        logger.error('📍 Current IP needs to be whitelisted for API access');
        logger.error('🔗 Go to: https://testnet.delta.exchange/app/account/manageapikeys');

        throw new Error('IP_NOT_WHITELISTED: Please whitelist your IP in Delta Exchange dashboard');
      }

      throw error;
    }
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Cleanup resources
   */
  public async cleanup(): Promise<void> {
    this.productCache.clear();
    this.symbolToProductId.clear();
    this.isInitialized = false;
    logger.info('Delta Exchange Service cleaned up');
  }
}

export default DeltaExchangeService;

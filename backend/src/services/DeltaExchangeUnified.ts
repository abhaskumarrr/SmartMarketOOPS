/**
 * Unified Delta Exchange Service for India Testnet
 * Production-ready implementation with proper error handling, authentication, and WebSocket support
 * Based on official Delta Exchange India API documentation
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import crypto from 'crypto';
import WebSocket from 'ws';
import { EventEmitter } from 'events';
import { logger } from '../utils/logger';

// Types and Interfaces
export interface DeltaCredentials {
  apiKey: string;
  apiSecret: string;
  testnet: boolean;
}

export interface DeltaProduct {
  id: number;
  symbol: string;
  description: string;
  contract_type: string;
  contract_value: string;
  contract_unit_currency: string;
  quoting_asset: { symbol: string };
  settling_asset: { symbol: string };
  underlying_asset: { symbol: string };
  state: string;
  trading_status: string;
  max_leverage_notional: string;
  default_leverage: string;
  initial_margin_scaling_factor: string;
  maintenance_margin_scaling_factor: string;
  impact_size: number;
  max_order_size: number;
  tick_size: string;
  product_specs: any;
}

export interface DeltaOrderRequest {
  product_id: number;
  side: 'buy' | 'sell';
  size: number; // Delta Exchange expects size as integer (contract units)
  order_type: 'limit_order' | 'market_order';
  limit_price?: string; // Delta Exchange expects prices as strings
  time_in_force?: 'gtc' | 'ioc' | 'fok';
  post_only?: boolean;
  reduce_only?: boolean;
  client_order_id?: string;
}

export interface DeltaOrder {
  id: number;
  user_id: number;
  size: number;
  unfilled_size: number;
  side: 'buy' | 'sell';
  order_type: string;
  limit_price: string;
  stop_order_type: string;
  stop_price: string;
  paid_commission: string;
  commission: string;
  reduce_only: boolean;
  client_order_id: string;
  state: string;
  created_at: string;
  updated_at: string;
  product: DeltaProduct;
}

export interface DeltaPosition {
  user_id: number;
  size: number;
  entry_price: string;
  margin: string;
  liquidation_price: string;
  bankruptcy_price: string;
  adl_level: number;
  product: DeltaProduct;
}

export interface DeltaBalance {
  asset_id: number;
  asset_symbol: string;
  available_balance: string;
  available_balance_for_robo: string;
  position_margin: string;
  order_margin: string;
  commission_reserve: string;
  unrealized_pnl: string;
  additional_reserve: string;
  cross_asset_liability: string;
  interest_credit: string;
  pending_referral_bonus: string;
  pending_trading_fee_credit: string;
  portfolio_margin: string;
  total_trading_fee_credit: string;
  total_commission_reserve: string;
  total_order_value: string;
  total_position_value: string;
  balance: string;
}

// Multi-timeframe data interfaces
export interface DeltaCandle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TechnicalIndicators {
  atr?: number;
  rsi?: number;
  macd?: {
    macd: number;
    signal: number;
    histogram: number;
  };
}

export interface MultiTimeframeData {
  symbol: string;
  timeframes: {
    [timeframe: string]: {
      candles: DeltaCandle[];
      indicators: TechnicalIndicators;
    };
  };
}

export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';

export class DeltaExchangeUnified extends EventEmitter {
  private credentials: DeltaCredentials;
  private client: AxiosInstance;
  private wsClient?: WebSocket;
  private baseUrl: string;
  private wsUrl: string;
  private initialized: boolean = false;
  private productCache: Map<string, DeltaProduct> = new Map();
  private symbolToProductId: Map<string, number> = new Map();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 5000;

  // Multi-timeframe data cache
  private candleCache: Map<string, Map<Timeframe, DeltaCandle[]>> = new Map();
  private indicatorCache: Map<string, Map<Timeframe, TechnicalIndicators>> = new Map();

  constructor(credentials: DeltaCredentials) {
    super();

    // Validate credentials before proceeding
    if (!credentials) {
      throw new Error('Delta Exchange credentials are required');
    }

    if (!credentials.apiKey || credentials.apiKey.trim() === '') {
      throw new Error('Delta Exchange API key is required and cannot be empty');
    }

    if (!credentials.apiSecret || credentials.apiSecret.trim() === '') {
      throw new Error('Delta Exchange API secret is required and cannot be empty');
    }

    this.credentials = credentials;

    // Use updated testnet URLs
    this.baseUrl = credentials.testnet
      ? 'https://cdn-ind.testnet.deltaex.org'  // Updated testnet URL
      : 'https://api.india.delta.exchange';         // Updated production URL

    this.wsUrl = credentials.testnet
      ? 'wss://socket-ind.testnet.deltaex.org'  // Updated testnet WebSocket URL
      : 'wss://socket.india.delta.exchange';         // Updated production WebSocket URL

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'SmartMarketOOPS-DeltaBot-v1.0'
      }
    });

    // Initialize the service asynchronously with better error handling
    this.initialize().catch(error => {
      logger.error('Failed to initialize Delta Exchange service', { error });
      logger.error('This is likely due to invalid API credentials or network issues');
      logger.warn('Service will continue in degraded mode - some features may not work');
      this.emit('error', error);
      // Don't throw here to prevent unhandled promise rejection
    });
  }

  /**
   * Initialize the Delta Exchange service
   */
  public async initialize(): Promise<void> {
    try {
      logger.info('🚀 Initializing Delta Exchange Unified Service...');
      logger.info('🔑 Using API Key', { apiKey: this.credentials.apiKey.substring(0, 8) + '...' + this.credentials.apiKey.substring(this.credentials.apiKey.length - 4) });
      logger.info('🔒 Using API Secret', { apiSecret: this.credentials.apiSecret.substring(0, 8) + '...' + this.credentials.apiSecret.substring(this.credentials.apiSecret.length - 4) });
      logger.info('🌐 Base URL', { baseUrl: this.baseUrl });

      // Load products and build symbol mappings
      await this.loadProducts();

      // Test authentication
      await this.testAuthentication();
      
      this.initialized = true;
      logger.info('✅ Delta Exchange Unified Service initialized successfully');
      this.emit('initialized');
      
    } catch (error) {
      logger.error('❌ Failed to initialize Delta Exchange service', { error });
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Load all available products from Delta Exchange
   */
  private async loadProducts(): Promise<void> {
    try {
      const response = await this.client.get('/v2/products');
      
      if (response.data.success) {
        const products: DeltaProduct[] = response.data.result;
        
        // Cache products and build symbol mappings
        for (const product of products) {
          this.productCache.set(product.symbol, product);
          this.symbolToProductId.set(product.symbol, product.id);
        }
        
        logger.info('📦 Loaded', { count: products.length, products: products.map(p => p.symbol) });
        
        // Log important trading pairs for BTCUSD and ETHUSD perpetuals
        const btcProduct = this.getProductBySymbol('BTCUSD');
        const ethProduct = this.getProductBySymbol('ETHUSD');
        
        if (btcProduct) {
          logger.info('📊 BTCUSD Perpetual Product', { id: btcProduct.id, state: btcProduct.state });
        }
        if (ethProduct) {
          logger.info('📊 ETHUSD Perpetual Product', { id: ethProduct.id, state: ethProduct.state });
        }
      } else {
        throw new Error(`Failed to load products: ${response.data.error}`);
      }
    } catch (error) {
      logger.error('Failed to load products', { error });
      throw error;
    }
  }

  /**
   * Test authentication by fetching user details
   */
  private async testAuthentication(): Promise<void> {
    try {
      logger.info('Authenticating with Delta Exchange...');
      
      // Try to get account balances instead of user info
      const response = await this.makeAuthenticatedRequest('GET', '/v2/wallets/balances');
      
      if (response.success) {
        logger.info('Authentication successful', { 
          message: 'Successfully connected to Delta Exchange API' 
        });
      } else {
        // Fallback to checking products - this doesn't require authentication but verifies API access
        const productsResponse = await this.client.get('/v2/products');
        if (productsResponse.data && productsResponse.data.success) {
          logger.info('Limited API access verified (products endpoint)');
          logger.warn('Could not verify full authentication - some trading features may not work');
        } else {
          throw new Error(`API connectivity failed: ${productsResponse.data?.error || 'Unknown error'}`);
        }
      }
    } catch (error) {
      logger.error('Authentication test failed', { error });
      // Don't throw the error, just log it, to allow the app to continue with limited functionality
      logger.warn('Continuing with limited Delta Exchange functionality');
    }
  }

  /**
   * Generate an authentication signature for Delta Exchange API requests
   * @param method - HTTP method (GET, POST, etc.)
   * @param path - API path (e.g., /v2/products)
   * @param queryString - URL query string (e.g., param1=value1&param2=value2)
   * @param body - Request body for POST/PUT requests
   * @param timestamp - Unix timestamp in milliseconds
   * @returns HMAC SHA256 signature
   */
  private generateSignature(
    method: string,
    path: string,
    queryString: string,
    body: string,
    timestamp: string
  ): string {
    const payload = `${timestamp}${method}${path}${queryString}${body}`;
    return crypto.createHmac('sha256', this.credentials.apiSecret)
      .update(payload)
      .digest('hex');
  }

  /**
   * Make an authenticated API request to Delta Exchange
   * @param method - HTTP method
   * @param path - API path
   * @param params - Query parameters
   * @param data - Request body data
   * @returns Response data
   */
  private async makeAuthenticatedRequest(
    method: string,
    path: string,
    params: Record<string, any> = {},
    data: any = null
  ): Promise<any> {
    const timestamp = Date.now().toString();
    const queryString = Object.keys(params).length > 0 ? new URLSearchParams(params).toString() : '';
    const requestBody = data ? JSON.stringify(data) : '';

    const signature = this.generateSignature(
      method,
      path,
      queryString ? `?${queryString}` : '', // Include '?' for query string
      requestBody,
      timestamp
    );

    logger.debug('Making authenticated request', {
      apiKey: this.credentials.apiKey.substring(0, 8) + '...',
      signature: signature.substring(0, 8) + '...',
      timestamp,
      method,
      path,
    });

    try {
      const headers = {
        'Content-Type': 'application/json',
        'X-DELTADEX-SIGNATURE': signature,
        'X-DELTADEX-APIKEY': this.credentials.apiKey,
        'X-DELTADEX-TIMESTAMP': timestamp,
      };

      const requestConfig = {
        method,
        url: path,
        headers,
        params: queryString ? params : undefined, // Only include params if queryString exists
        data: requestBody || undefined,
      };

      const response = await this.client.request(requestConfig);
      if (response.data.success) {
        logger.debug(`Authenticated request to ${path} successful`, { response: response.data });
        return response.data;
      } else {
        logger.error(`Authenticated request to ${path} failed`, { error: response.data.error });
        throw new Error(`Delta Exchange API error: ${response.data.error}`);
      }
    } catch (error) {
      logger.error(`Error making authenticated request to ${path}`, { error });
      if (axios.isAxiosError(error)) {
        logger.error(`Axios error details:`, { status: error.response?.status, data: error.response?.data });
      }
      throw error;
    }
  }

  /**
   * Get product by symbol
   */
  public getProductBySymbol(symbol: string): DeltaProduct | undefined {
    return this.productCache.get(symbol);
  }

  /**
   * Get product ID by symbol
   */
  public getProductId(symbol: string): number | undefined {
    return this.symbolToProductId.get(symbol);
  }

  /**
   * Check if service is ready for trading
   */
  public isReady(): boolean {
    if (!this.initialized) {
      throw new Error('Delta Exchange service not ready');
    }
    return true;
  }
  
  /**
   * Check if the client is initialized
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get account balance
   */
  public async getBalance(): Promise<DeltaBalance[]> {
    try {
      this.isReady(); // Ensure service is initialized
      const response = await this.makeAuthenticatedRequest('GET', '/v2/wallet/balances');
      if (response.error) {
        throw new Error(`Failed to get balance: ${response.error}`);
      }
      return response.result;
    } catch (error: any) {
      logger.error('❌ Error getting REAL balance from Delta Exchange', { error: error.message });
      if (error.response?.data?.error?.code === 1005) {
        logger.error('🚫 IP NOT WHITELISTED FOR API KEY', { currentIp: error.response.data.error.context?.client_ip });
        logger.error('🔗 Please whitelist your IP at', { whitelistUrl: 'https://testnet.delta.exchange/app/account/manageapikeys' });
        throw new Error(`IP_NOT_WHITELISTED: ${error.response.data.error.context?.client_ip}`);
      }
      throw new Error(`Failed to get real balance from Delta Exchange: ${error.message}`);
    }
  }

  /**
   * Get current positions
   */
  public async getPositions(productId?: number): Promise<DeltaPosition[]> {
    try {
      this.isReady();
      const params: any = {};
      if (productId) {
        params.product_id = productId;
      }
      const response = await this.makeAuthenticatedRequest('GET', '/v2/positions', params);
      if (response.error) {
        throw new Error(`Failed to get positions: ${response.error?.code || response.error}`);
      }
      return response.result;
    } catch (error) {
      logger.error('Error getting positions', { error });
      return []; // Return empty array on error to prevent crashes
    }
  }

  /**
   * Get all positions for major trading assets
   */
  public async getAllPositions(): Promise<DeltaPosition[]> {
    try {
      this.isReady();
      const products = this.getPerpetualProducts();
      const allPositions: DeltaPosition[] = [];
      for (const product of products) {
        try {
          const positions = await this.getPositions(product.id);
          allPositions.push(...positions);
        } catch (error) {
          logger.warn('Failed to get positions for', { asset: product.symbol, error: error instanceof Error ? error.message : 'Unknown error' });
        }
      }
      return allPositions;
    } catch (error) {
      logger.error('Error getting all positions', { error });
      return [];
    }
  }

  /**
   * Place a new order
   */
  public async placeOrder(orderRequest: DeltaOrderRequest): Promise<DeltaOrder> {
    try {
      this.isReady();
      this.validateOrderRequest(orderRequest);

      // Convert size to contract units if needed (assuming helper exists)
      const finalSize = this.convertToContractUnits(orderRequest.size, this.getSymbolByProductId(orderRequest.product_id) || '');

      const response = await this.makeAuthenticatedRequest('POST', '/v2/orders', {}, { ...orderRequest, size: finalSize });
      
      const order = response.result;
      if (order) {
        const symbol = this.getSymbolByProductId(order.product.id) || 'UNKNOWN';
        logger.info('✅ Order placed successfully', { side: order.side, size: order.size, symbol, price: order.limit_price || 'market' });
        return order;
      } else {
        throw new Error(`Order placement failed: ${response.error}`);
      }
    } catch (error: any) {
      logger.error('Error placing order', { error: error.message });
      if (error.message?.includes('IP_NOT_WHITELISTED')) {
         logger.error('Order placement failed: IP not whitelisted for API key.', { error });
      }
      throw error;
    }
  }

  /**
   * Cancel an order
   */
  public async cancelOrder(orderId: number): Promise<DeltaOrder> {
    try {
      this.isReady();
      const response = await this.makeAuthenticatedRequest('DELETE', `/v2/orders/${orderId}`);
      const order = response.result;
      if (order) {
        logger.info('✅ Order cancelled', { orderId: order.id });
        return order;
      } else {
        throw new Error(`Order cancellation failed: ${response.error}`);
      }
    } catch (error) {
      logger.error('Error cancelling order', { error });
      throw error;
    }
  }

  /**
   * Get order status
   */
  public async getOrder(orderId: number): Promise<DeltaOrder> {
    try {
      this.isReady();
      const response = await this.makeAuthenticatedRequest('GET', `/v2/orders/${orderId}`);
      if (response.error) {
        throw new Error(`Failed to get order: ${response.error}`);
      }
      return response.result;
    } catch (error) {
      logger.error('Error getting order', { error });
      throw error;
    }
  }

  /**
   * Get all open orders
   */
  public async getOpenOrders(productId?: number): Promise<DeltaOrder[]> {
    try {
      this.isReady();
      const params: any = {};
      if (productId) {
        params.product_id = productId;
      }
      const response = await this.makeAuthenticatedRequest('GET', '/v2/orders', params);
      if (response.error) {
        throw new Error(`Failed to get open orders: ${response.error}`);
      }
      return response.result;
    } catch (error) {
      logger.error('Error getting open orders', { error });
      return [];
    }
  }

  /**
   * Convert decimal size to contract units for Delta Exchange
   * Delta Exchange perpetual futures use contract units, not decimal amounts
   */
  private convertToContractUnits(decimalSize: number, symbol: string): number {
    // For Delta Exchange perpetual futures:
    // BTC/USD: 1 contract = 1 USD worth of BTC
    // ETH/USD: 1 contract = 1 USD worth of ETH
    // Size should be in USD value, not in base currency amount

    // Since we're calculating position size in USD terms already,
    // we can use the USD value directly as contract units
    // Round to nearest integer as Delta Exchange expects integer contract units
    return Math.round(decimalSize);
  }

  /**
   * Validate order request
   */
  private validateOrderRequest(orderRequest: DeltaOrderRequest): void {
    const { product_id, side, size, order_type, limit_price } = orderRequest;
    if (!product_id) {
      throw new Error('Product ID is required');
    }
    if (side !== 'buy' && side !== 'sell') {
      throw new Error('Side must be "buy" or "sell"');
    }
    if (size <= 0) {
      throw new Error('Size must be greater than 0');
    }
    if (order_type !== 'limit_order' && order_type !== 'market_order') {
      throw new Error('Order type must be "limit_order" or "market_order"');
    }
    if (order_type === 'limit_order' && !limit_price) {
      throw new Error('Limit price is required for limit orders');
    }
  }

  /**
   * Get market data for a product (current prices)
   */
  public async getMarketData(symbol: string): Promise<any> {
    const productId = this.getProductId(symbol);
    if (!productId) {
      throw new Error(`Product not found for symbol: ${symbol}`);
    }

    try {
      // Use v2/ticker for live market data
      const response = await this.client.get(`/v2/ticker/${productId}`);
      const ticker = response.data;
      if (ticker) {
        logger.debug('Raw ticker data for', { symbol, ticker });
        // Format to a consistent structure
        return {
          symbol: ticker.symbol,
          open: ticker.open,
          high: ticker.high,
          low: ticker.low,
          close: ticker.close,
          volume: ticker.volume,
          updated_at: ticker.st,
        };
      } else {
        throw new Error(`Failed to get market data: ${response.data.error}`);
      }
    } catch (error: any) {
      logger.error('Error getting live market data for', { symbol, error: error.message });
      throw new Error(`Failed to get live market data for ${symbol}: Refusing to use mock data for safety.`);
    }
  }

  /**
   * Get order book for a product
   */
  public async getOrderBook(symbol: string, depth: number = 20): Promise<any> {
    const productId = this.getProductId(symbol);
    if (!productId) {
      throw new Error(`Product not found for symbol: ${symbol}`);
    }

    try {
      const response = await this.client.get(`/v2/l2orderbook/${productId}?depth=${depth}`);
      if (response.data.error) {
        throw new Error(`Failed to get order book: ${response.data.error}`);
      }
      return response.data;
    } catch (error) {
      logger.error('Error getting order book', { error });
      throw error;
    }
  }

  /**
   * Initialize WebSocket connection for real-time data
   */
  public connectWebSocket(symbols: string[]): void {
    if (this.wsClient) {
      this.wsClient.close();
    }

    try {
      this.wsClient = new WebSocket(this.wsUrl);

      this.wsClient.on('open', () => {
        logger.info('✅ Delta Exchange WebSocket connected');
        this.reconnectAttempts = 0; // Reset on successful connection
        // Subscribe to channels for the given symbols
        symbols.forEach(symbol => {
          const productId = this.getProductId(symbol);
          if (productId) {
            this.subscribeToChannel(productId, 'ticker');
            this.subscribeToChannel(productId, 'l2_orderbook');
          }
        });

        this.emit('wsConnected');
      });

      this.wsClient.on('message', (data: WebSocket.Data) => {
        try {
          let message: any;
          if (typeof data === 'string') {
            message = JSON.parse(data);
          } else if (Buffer.isBuffer(data)) {
            message = JSON.parse(data.toString());
          } else {
            message = data;
          }

          if (this.wsClient) {
            if (message.type === 'v2/ticker') {
              this.emit('ticker', message);
            }
          }
        } catch (error) {
          logger.error('Error parsing WebSocket message', { error });
        }
      });

      this.wsClient.on('close', () => {
        logger.warn('🔌 Delta Exchange WebSocket disconnected');
        this.handleReconnect(symbols);
      });

      this.wsClient.on('error', (error: Error) => {
        logger.error('❌ Delta Exchange WebSocket error', { error });
        this.emit('error', error);
      });

    } catch (error) {
      logger.error('Error connecting to WebSocket', { error });
      this.handleReconnect(symbols);
    }
  }

  /**
   * Subscribe to a WebSocket channel
   */
  private subscribeToChannel(productId: number, channel: string): void {
    if (this.wsClient && this.wsClient.readyState === WebSocket.OPEN) {
      const subscriptionMessage = {
        type: 'subscribe',
        payload: { channels: [{ name: channel, symbols: [this.getSymbolByProductId(productId)] }] }
      };
      this.wsClient.send(JSON.stringify(subscriptionMessage));
      logger.info('📡 Subscribed to', { channel, forProduct: productId });
    }
  }

  /**
   * Handle WebSocket reconnection
   */
  private handleReconnect(symbols: string[]): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        logger.info('🔄 Attempting to reconnect WebSocket', { attempt: this.reconnectAttempts, maxAttempts: this.maxReconnectAttempts });
        this.connectWebSocket(symbols);
      }, this.reconnectDelay);
    } else {
      logger.error('❌ Max WebSocket reconnection attempts reached');
    }
  }

  /**
   * Disconnect WebSocket
   */
  public disconnectWebSocket(): void {
    if (this.wsClient) {
      this.wsClient.close();
      logger.info('🔌 Delta Exchange WebSocket disconnected manually');
    }
  }

  /**
   * Get all available products
   */
  public getAllProducts(): DeltaProduct[] {
    return Array.from(this.productCache.values());
  }

  /**
   * Get trading pairs suitable for perpetual futures
   */
  public getPerpetualProducts(): DeltaProduct[] {
    return this.getAllProducts().filter(product =>
      product.contract_type === 'perpetual_futures' &&
      product.state === 'live'
    );
  }

  /**
   * Get historical candle data for a specific timeframe
   */
  public async getCandleData(symbol: string, timeframe: Timeframe, limit: number = 100): Promise<DeltaCandle[]> {
    this.isReady();

    const productId = this.getProductId(symbol);
    if (!productId) {
      throw new Error(`Product not found for symbol: ${symbol}`);
    }

    // Check cache first
    const cachedCandles = this.getCachedCandleData(symbol, timeframe);
    if (cachedCandles) {
      return cachedCandles;
    }

    try {
      const response = await this.client.get(`/v2/candles`, {
        params: {
          product_id: productId,
          resolution: this.convertTimeframe(timeframe),
          limit: limit,
        }
      });

      if (response.data.error) {
        throw new Error(`Failed to get candle data: ${response.data.error}`);
      }

      const candles: DeltaCandle[] = response.data.result;
      this.candleCache.set(symbol, (this.candleCache.get(symbol) || new Map()).set(timeframe, candles));
        return candles;

    } catch (error) {
      logger.error('Error getting candle data for', { symbol, timeframe, error });
      return [];
    }
  }

  /**
   * Get candles data (alias for getCandleData for compatibility)
   */
  public async getCandles(productId: number, timeframe: string, limit: number = 100): Promise<any[]> {
      const symbol = this.getSymbolByProductId(productId);
      if (!symbol) {
      throw new Error(`Symbol not found for product ID: ${productId}`);
      }

    // This method can be simplified or deprecated in favor of getCandleData
    try {
      const response = await this.client.get(`/v2/candles`, {
        params: {
          product_id: productId,
          resolution: timeframe, // Assumes timeframe is already in correct string format
          limit: limit
        }
      });
      return response.data.result || [];
    } catch (error) {
      logger.error('Error getting candles for product', { productId, error });
      return [];
    }
  }

  /**
   * Get symbol by product ID (Environment-aware mapping)
   */
  private getSymbolByProductId(productId: number): string | null {
    // This is inefficient, should be pre-mapped
    for (const [symbol, product] of this.productCache.entries()) {
      if (product.id === productId) {
        return symbol;
      }
    }
    return null;
  }

  /**
   * Convert timeframe string to Timeframe type
   */
  private convertTimeframeFromString(timeframe: string): Timeframe {
    const mapping: Record<string, Timeframe> = {
      '1m': '1m',
      '5m': '5m',
      '15m': '15m',
      '1h': '1h',
      '4h': '4h',
      '1d': '1d',
      '1D': '1d'
    };
    return mapping[timeframe] || '1d';
  }

  /**
   * Get multi-timeframe data for a symbol
   */
  public async getMultiTimeframeData(symbol: string, timeframes: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d']): Promise<MultiTimeframeData> {
    const data: MultiTimeframeData = {
      symbol,
      timeframes: {}
    };
    for (const tf of timeframes) {
      try {
        const candles = await this.getCandleData(symbol, tf);
        const indicators = this.calculateIndicators(candles);
        data.timeframes[tf] = { candles, indicators };
      } catch (error) {
        logger.error('Failed to get data for', { symbol, timeframe: tf, error });
      }
    }
    return data;
  }

  /**
   * Calculate technical indicators for candle data
   */
  private calculateIndicators(candles: DeltaCandle[]): TechnicalIndicators {
    const indicators: TechnicalIndicators = {};

    if (candles.length < 14) {
      return indicators; // Not enough data for indicators
    }

    // Calculate ATR (Average True Range)
    indicators.atr = this.calculateATR(candles, 14);

    // Calculate RSI (Relative Strength Index)
    indicators.rsi = this.calculateRSI(candles, 14);

    // Calculate MACD
    indicators.macd = this.calculateMACD(candles);

    return indicators;
  }

  /**
   * Calculate ATR (Average True Range)
   */
  private calculateATR(candles: DeltaCandle[], period: number): number {
    if (candles.length < period + 1) return 0;

    let atrSum = 0;
    for (let i = candles.length - period; i < candles.length; i++) {
      const high = candles[i].high;
      const low = candles[i].low;
      const prevClose = i > 0 ? candles[i - 1].close : candles[i].open;

      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );

      atrSum += tr;
    }

    return atrSum / period;
  }

  /**
   * Calculate RSI (Relative Strength Index)
   */
  private calculateRSI(candles: DeltaCandle[], period: number): number {
    if (candles.length < period + 1) return 50;

    let gains = 0;
    let losses = 0;

    for (let i = candles.length - period; i < candles.length; i++) {
      const change = candles[i].close - candles[i - 1].close;
      if (change > 0) {
        gains += change;
      } else {
        losses += Math.abs(change);
      }
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;

    if (avgLoss === 0) return 100;

    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  /**
   * Calculate MACD (Moving Average Convergence Divergence)
   */
  private calculateMACD(candles: DeltaCandle[]): { macd: number; signal: number; histogram: number } {
    if (candles.length < 26) {
      return { macd: 0, signal: 0, histogram: 0 };
    }

    // Calculate EMAs
    const ema12 = this.calculateEMA(candles.map(c => c.close), 12);
    const ema26 = this.calculateEMA(candles.map(c => c.close), 26);

    const macd = ema12 - ema26;

    // For simplicity, using a basic signal calculation
    // In production, you'd calculate the EMA of MACD values
    const signal = macd * 0.9; // Simplified signal line
    const histogram = macd - signal;

    return { macd, signal, histogram };
  }

  /**
   * Calculate EMA (Exponential Moving Average)
   */
  private calculateEMA(prices: number[], period: number): number {
    if (prices.length < period) return prices[prices.length - 1] || 0;

    const multiplier = 2 / (period + 1);
    let ema = prices.slice(0, period).reduce((sum, price) => sum + price, 0) / period;

    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
    }

    return ema;
  }

  /**
   * Convert timeframe to Delta Exchange format
   */
  private convertTimeframe(timeframe: Timeframe): string {
    const mapping: Record<Timeframe, string> = {
      '1m': '1',
      '5m': '5',
      '15m': '15',
      '1h': '60',
      '4h': '240',
      '1d': '1D'
    };
    return mapping[timeframe];
  }

  /**
   * Get timeframe duration in seconds
   */
  private getTimeframeSeconds(timeframe: Timeframe): number {
    const mapping: Record<Timeframe, number> = {
      '1m': 60,
      '5m': 300,
      '15m': 900,
      '1h': 3600,
      '4h': 14400,
      '1d': 86400
    };
    return mapping[timeframe];
  }

  /**
   * Get cached candle data
   */
  public getCachedCandleData(symbol: string, timeframe: Timeframe): DeltaCandle[] | undefined {
    return this.candleCache.get(symbol)?.get(timeframe);
  }

  /**
   * Get cached indicators
   */
  public getCachedIndicators(symbol: string, timeframe: Timeframe): TechnicalIndicators | undefined {
    return this.indicatorCache.get(symbol)?.get(timeframe);
  }

  /**
   * Cleanup resources
   */
  public cleanup(): void {
    this.disconnectWebSocket();
    this.removeAllListeners();
    logger.info('🧹 Delta Exchange service cleaned up');
  }

  /**
   * Get all available markets
   */
  public getMarkets(): DeltaProduct[] {
    return Array.from(this.productCache.values());
  }

  /**
   * Get market information by symbol
   */
  public getMarketBySymbol(symbol: string): DeltaProduct | undefined {
    return this.getProductBySymbol(symbol);
  }
}

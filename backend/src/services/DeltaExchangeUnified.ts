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
  private isInitialized: boolean = false;
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
    this.credentials = credentials;
    
    // Use India testnet URLs
    this.baseUrl = credentials.testnet 
      ? 'https://cdn-ind.testnet.deltaex.org'
      : 'https://api.india.delta.exchange';
    
    this.wsUrl = credentials.testnet
      ? 'wss://testnet-ws.delta.exchange'
      : 'wss://ws.india.delta.exchange';

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'SmartMarketOOPS-DeltaBot-v1.0'
      }
    });

    // Initialize the service asynchronously
    this.initialize().catch(error => {
      logger.error('Failed to initialize in constructor:', error);
      this.emit('error', error);
    });
  }

  /**
   * Initialize the Delta Exchange service
   */
  private async initialize(): Promise<void> {
    try {
      logger.info('🚀 Initializing Delta Exchange Unified Service...');
      logger.info(`🔑 Using API Key: ${this.credentials.apiKey.substring(0, 8)}...${this.credentials.apiKey.substring(this.credentials.apiKey.length - 4)}`);
      logger.info(`🔒 Using API Secret: ${this.credentials.apiSecret.substring(0, 8)}...${this.credentials.apiSecret.substring(this.credentials.apiSecret.length - 4)}`);
      logger.info(`🌐 Base URL: ${this.baseUrl}`);

      // Load products and build symbol mappings
      await this.loadProducts();

      // Test authentication
      await this.testAuthentication();
      
      this.isInitialized = true;
      logger.info('✅ Delta Exchange Unified Service initialized successfully');
      this.emit('initialized');
      
    } catch (error) {
      logger.error('❌ Failed to initialize Delta Exchange service:', error);
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
        
        logger.info(`📦 Loaded ${products.length} products from Delta Exchange`);
        
        // Log important trading pairs for BTCUSD and ETHUSD perpetuals
        const btcProduct = this.getProductBySymbol('BTCUSD');
        const ethProduct = this.getProductBySymbol('ETHUSD');
        
        if (btcProduct) {
          logger.info(`🟡 BTC/USD Perpetual: ID ${btcProduct.id}, State: ${btcProduct.state}`);
        }
        if (ethProduct) {
          logger.info(`🔵 ETH/USD Perpetual: ID ${ethProduct.id}, State: ${ethProduct.state}`);
        }
        
      } else {
        throw new Error(`Failed to load products: ${response.data.error}`);
      }
    } catch (error) {
      logger.error('Error loading products:', error);
      throw error;
    }
  }

  /**
   * Test authentication with Delta Exchange
   */
  private async testAuthentication(): Promise<void> {
    try {
      const response = await this.makeAuthenticatedRequest('GET', '/v2/profile');
      
      if (response.success) {
        logger.info('✅ Delta Exchange authentication successful');
        logger.info(`👤 User ID: ${response.result.user_id}`);
      } else {
        throw new Error(`Authentication failed: ${response.error}`);
      }
    } catch (error) {
      logger.error('❌ Delta Exchange authentication failed:', error);
      throw error;
    }
  }

  /**
   * Generate signature for authenticated requests
   */
  private generateSignature(
    method: string,
    path: string,
    queryString: string,
    body: string,
    timestamp: string
  ): string {
    const message = method + timestamp + path + queryString + body;
    return crypto
      .createHmac('sha256', this.credentials.apiSecret)
      .update(message)
      .digest('hex');
  }

  /**
   * Make authenticated request to Delta Exchange API
   */
  private async makeAuthenticatedRequest(
    method: string,
    path: string,
    params: Record<string, any> = {},
    data: any = null
  ): Promise<any> {
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
    logger.info(`🔍 Making request: ${method} ${path}${queryString}`);
    logger.info(`📝 Signature message: "${method}${timestamp}${path}${queryString}${body}"`);
    logger.info(`✍️ Generated signature: ${signature}`);
    logger.info(`📤 Request headers: ${JSON.stringify(headers, null, 2)}`);

    try {
      const response: AxiosResponse = await this.client.request({
        method: method as any,
        url: path + queryString,
        data: data || undefined, // Ensure undefined for GET requests
        headers
      });

      return response.data;
    } catch (error: any) {
      logger.error(`Delta Exchange API error: ${error.message}`);
      if (error.response) {
        logger.error(`Response status: ${error.response.status}`);
        logger.error(`Response data:`, error.response.data);

        // Log detailed schema errors if available
        if (error.response.data?.error?.context?.schema_errors) {
          logger.error('🔍 DETAILED SCHEMA ERRORS:');
          logger.error(JSON.stringify(error.response.data.error.context.schema_errors, null, 2));
        }
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
    return this.isInitialized && !!this.credentials.apiKey && !!this.credentials.apiSecret;
  }

  /**
   * Get account balance
   */
  public async getBalance(): Promise<DeltaBalance[]> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    try {
      const response = await this.makeAuthenticatedRequest('GET', '/v2/wallet/balances');

      if (response.success) {
        return response.result;
      } else {
        throw new Error(`Failed to get balance: ${response.error}`);
      }
    } catch (error) {
      logger.error('Error getting balance:', error);
      throw error;
    }
  }

  /**
   * Get current positions
   */
  public async getPositions(productId?: number): Promise<DeltaPosition[]> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    try {
      // Delta Exchange requires either product_id or underlying_asset_symbol
      // If no productId provided, get positions for all major assets
      let params: Record<string, any>;

      if (productId) {
        params = { product_id: productId };
      } else {
        // Get positions for BTC first, then we can call again for ETH if needed
        params = { underlying_asset_symbol: 'BTC' };
      }

      const response = await this.makeAuthenticatedRequest('GET', '/v2/positions', params);

      if (response.success) {
        return response.result;
      } else {
        throw new Error(`Failed to get positions: ${response.error?.code || response.error}`);
      }
    } catch (error) {
      logger.error('Error getting positions:', error);
      throw error;
    }
  }

  /**
   * Get all positions for major trading assets
   */
  public async getAllPositions(): Promise<DeltaPosition[]> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    try {
      const allPositions: DeltaPosition[] = [];
      const assets = ['BTC', 'ETH']; // Major assets we trade

      // Get positions for each asset
      for (const asset of assets) {
        try {
          const response = await this.makeAuthenticatedRequest('GET', '/v2/positions', {
            underlying_asset_symbol: asset
          });

          if (response.success && response.result) {
            allPositions.push(...response.result);
          }
        } catch (error) {
          logger.warn(`Failed to get positions for ${asset}:`, error instanceof Error ? error.message : 'Unknown error');
          // Continue with other assets even if one fails
        }
      }

      return allPositions;
    } catch (error) {
      logger.error('Error getting all positions:', error);
      throw error;
    }
  }

  /**
   * Place a new order
   */
  public async placeOrder(orderRequest: DeltaOrderRequest): Promise<DeltaOrder> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    try {
      // Validate order request
      this.validateOrderRequest(orderRequest);

      const response = await this.makeAuthenticatedRequest('POST', '/v2/orders', {}, orderRequest);

      if (response.success) {
        const order: DeltaOrder = response.result;
        const symbol = order.product?.symbol || `Product-${order.product?.id || 'Unknown'}`;
        logger.info(`✅ Order placed successfully: ${order.side} ${order.size} ${symbol} @ ${order.limit_price || 'market'}`);
        this.emit('orderPlaced', order);
        return order;
      } else {
        throw new Error(`Order placement failed: ${response.error.code} - ${response.error.message}`);
      }
    } catch (error) {
      logger.error('Error placing order:', error);
      this.emit('orderError', error);
      throw error;
    }
  }

  /**
   * Cancel an order
   */
  public async cancelOrder(orderId: number): Promise<DeltaOrder> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    try {
      const response = await this.makeAuthenticatedRequest('DELETE', `/v2/orders/${orderId}`);

      if (response.success) {
        const order: DeltaOrder = response.result;
        logger.info(`✅ Order cancelled: ${order.id}`);
        this.emit('orderCancelled', order);
        return order;
      } else {
        throw new Error(`Order cancellation failed: ${response.error}`);
      }
    } catch (error) {
      logger.error('Error cancelling order:', error);
      throw error;
    }
  }

  /**
   * Get order status
   */
  public async getOrder(orderId: number): Promise<DeltaOrder> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    try {
      const response = await this.makeAuthenticatedRequest('GET', `/v2/orders/${orderId}`);

      if (response.success) {
        return response.result;
      } else {
        throw new Error(`Failed to get order: ${response.error}`);
      }
    } catch (error) {
      logger.error('Error getting order:', error);
      throw error;
    }
  }

  /**
   * Get all open orders
   */
  public async getOpenOrders(productId?: number): Promise<DeltaOrder[]> {
    if (!this.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    try {
      const params = productId ? { product_id: productId } : {};
      const response = await this.makeAuthenticatedRequest('GET', '/v2/orders', params);

      if (response.success) {
        return response.result;
      } else {
        throw new Error(`Failed to get open orders: ${response.error}`);
      }
    } catch (error) {
      logger.error('Error getting open orders:', error);
      throw error;
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
   * Get market data for a product (current prices)
   */
  public async getMarketData(symbol: string): Promise<any> {
    const productId = this.getProductId(symbol);
    if (!productId) {
      throw new Error(`Product not found for symbol: ${symbol}`);
    }

    try {
      // Use the correct ticker endpoint from Delta Exchange API docs
      const response = await this.makeAuthenticatedRequest('GET', `/v2/tickers/${symbol}`);

      if (response.success) {
        const ticker = response.result;

        logger.debug(`Raw ticker data for ${symbol}:`, ticker);

        // Return standardized market data with current prices
        return {
          symbol: symbol,
          product_id: productId,
          last_price: ticker.close || ticker.last_price || ticker.price,
          mark_price: ticker.mark_price || ticker.close || ticker.price,
          bid: ticker.bid,
          ask: ticker.ask,
          high: ticker.high,
          low: ticker.low,
          volume: ticker.volume,
          timestamp: ticker.timestamp || Date.now()
        };
      } else {
        throw new Error(`Failed to get market data: ${response.error}`);
      }
    } catch (error) {
      logger.error(`Error getting live market data for ${symbol}:`, error);

      // ONLY USE LIVE DATA - No fallback to mock data
      throw new Error(`Failed to get live market data for ${symbol}. Refusing to use mock data for safety.`);
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
      const response = await this.client.get(`/v2/l2orderbook/${productId}`, {
        params: { depth }
      });

      if (response.data.success) {
        return response.data.result;
      } else {
        throw new Error(`Failed to get order book: ${response.data.error}`);
      }
    } catch (error) {
      logger.error('Error getting order book:', error);
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

      this.wsClient.on('message', (data: WebSocket.Data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleWebSocketMessage(message);
        } catch (error) {
          logger.error('Error parsing WebSocket message:', error);
        }
      });

      this.wsClient.on('close', () => {
        logger.warn('🔌 Delta Exchange WebSocket disconnected');
        this.emit('wsDisconnected');
        this.handleReconnect(symbols);
      });

      this.wsClient.on('error', (error) => {
        logger.error('❌ Delta Exchange WebSocket error:', error);
        this.emit('wsError', error);
      });

    } catch (error) {
      logger.error('Error connecting to WebSocket:', error);
      throw error;
    }
  }

  /**
   * Subscribe to a WebSocket channel
   */
  private subscribeToChannel(productId: number, channel: string): void {
    if (this.wsClient && this.wsClient.readyState === WebSocket.OPEN) {
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
      logger.info(`📡 Subscribed to ${channel} for product ${productId}`);
    }
  }

  /**
   * Handle WebSocket message
   */
  private handleWebSocketMessage(message: any): void {
    if (message.type === 'ticker') {
      this.emit('ticker', message);
    } else if (message.type === 'l2_orderbook') {
      this.emit('orderbook', message);
    } else if (message.type === 'trade') {
      this.emit('trade', message);
    }
  }

  /**
   * Handle WebSocket reconnection
   */
  private handleReconnect(symbols: string[]): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      logger.info(`🔄 Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

      setTimeout(() => {
        this.connectWebSocket(symbols);
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      logger.error('❌ Max WebSocket reconnection attempts reached');
      this.emit('wsReconnectFailed');
    }
  }

  /**
   * Disconnect WebSocket
   */
  public disconnectWebSocket(): void {
    if (this.wsClient) {
      this.wsClient.close();
      this.wsClient = undefined;
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
    if (!this.isReady()) {
      throw new Error('Delta Exchange service not ready');
    }

    const productId = this.getProductId(symbol);
    if (!productId) {
      throw new Error(`Product not found for symbol: ${symbol}`);
    }

    try {
      // Convert timeframe to Delta Exchange format
      const deltaTimeframe = this.convertTimeframe(timeframe);

      const params = {
        resolution: deltaTimeframe,
        symbol: symbol,
        start: Math.floor(Date.now() / 1000) - (limit * this.getTimeframeSeconds(timeframe)),
        end: Math.floor(Date.now() / 1000)
      };

      const response = await this.client.get('/v2/history', { params });

      if (response.data.success) {
        const candles: DeltaCandle[] = response.data.result.map((candle: any) => ({
          timestamp: candle.time * 1000, // Convert to milliseconds
          open: parseFloat(candle.open),
          high: parseFloat(candle.high),
          low: parseFloat(candle.low),
          close: parseFloat(candle.close),
          volume: parseFloat(candle.volume || '0')
        }));

        // Cache the data
        if (!this.candleCache.has(symbol)) {
          this.candleCache.set(symbol, new Map());
        }
        this.candleCache.get(symbol)!.set(timeframe, candles);

        return candles;
      } else {
        throw new Error(`Failed to get candle data: ${response.data.error}`);
      }
    } catch (error) {
      logger.error(`Error getting candle data for ${symbol} ${timeframe}:`, error);
      throw error;
    }
  }

  /**
   * Get multi-timeframe data for a symbol
   */
  public async getMultiTimeframeData(symbol: string, timeframes: Timeframe[] = ['1m', '5m', '15m', '1h', '4h', '1d']): Promise<MultiTimeframeData> {
    const result: MultiTimeframeData = {
      symbol,
      timeframes: {}
    };

    // Fetch data for each timeframe
    for (const timeframe of timeframes) {
      try {
        const candles = await this.getCandleData(symbol, timeframe);
        const indicators = this.calculateIndicators(candles);

        result.timeframes[timeframe] = {
          candles,
          indicators
        };

        // Cache indicators
        if (!this.indicatorCache.has(symbol)) {
          this.indicatorCache.set(symbol, new Map());
        }
        this.indicatorCache.get(symbol)!.set(timeframe, indicators);

      } catch (error) {
        logger.error(`Failed to get data for ${symbol} ${timeframe}:`, error);
        // Continue with other timeframes
      }
    }

    return result;
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
    this.candleCache.clear();
    this.indicatorCache.clear();
    logger.info('🧹 Delta Exchange service cleaned up');
  }
}

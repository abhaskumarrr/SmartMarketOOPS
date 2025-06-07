/**
 * TypeScript interfaces for Delta Exchange API
 * Based on Delta Exchange API documentation: https://docs.delta.exchange
 */

declare namespace DeltaExchange {
  /**
   * API credentials interface
   */
  export interface ApiCredentials {
    key: string;
    secret: string;
  }

  /**
   * Configuration options for DeltaExchangeAPI
   */
  export interface ApiOptions {
    testnet?: boolean;
    userId?: string;
    rateLimit?: RateLimitSettings;
  }

  /**
   * Enhanced rate limit settings for Delta Exchange API
   */
  export interface RateLimitSettings {
    maxRetries?: number;
    initialDelay?: number; // milliseconds
    maxDelay?: number; // milliseconds
    factor?: number; // exponential backoff factor
    requestsPerWindow?: number; // requests per 5-minute window (default: 8000)
    windowDuration?: number; // window duration in milliseconds (default: 300000)
    productRateLimit?: number; // requests per second per product (default: 400)
  }

  /**
   * Server time response
   */
  export interface ServerTime {
    server_time: number;
    server_time_iso: string;
  }

  /**
   * Market/Product
   */
  export interface Market {
    id: number;
    symbol: string;
    description: string;
    underlying_asset: Asset;
    quote_asset: Asset;
    contract_type: string;
    contract_size: number;
    contract_unit_currency: string;
    contract_value: string;
    contract_value_currency: string;
    tick_size: string;
    quoting_precision: number;
    min_price: string;
    max_price: string;
    min_qty: string;
    max_qty: string;
    maintenance_margin: string;
    initial_margin: string;
    position_size_limit: number;
    is_active: boolean;
    created_at: string;
    updated_at: string;
  }

  /**
   * Asset
   */
  export interface Asset {
    id: number;
    symbol: string;
    precision: number;
    name: string;
    is_active: boolean;
    created_at: string;
    updated_at: string;
  }

  /**
   * Ticker
   */
  export interface Ticker {
    symbol: string;
    open: string;
    high: string;
    low: string;
    close: string;
    volume: string;
    timestamp: string;
  }

  /**
   * Orderbook
   */
  export interface Orderbook {
    symbol: string;
    bids: [string, string][];  // [price, size]
    asks: [string, string][];  // [price, size]
    timestamp: string;
  }

  /**
   * Trade
   */
  export interface Trade {
    id: number;
    symbol: string;
    price: string;
    size: string;
    side: 'buy' | 'sell';
    timestamp: string;
  }

  /**
   * Account information
   */
  export interface AccountInfo {
    id: number;
    email: string;
    name: string;
    is_verified: boolean;
    created_at: string;
    updated_at: string;
  }

  /**
   * Wallet balance
   */
  export interface WalletBalance {
    asset: string;
    available_balance: string;
    balance: string;
    last_updated_at: string;
  }

  /**
   * Position
   */
  export interface Position {
    id: number;
    symbol: string;
    size: string;
    entry_price: string;
    mark_price: string;
    liquidation_price: string;
    unrealized_pnl: string;
    realized_pnl: string;
    side: 'long' | 'short';
    created_at: string;
    updated_at: string;
  }

  /**
   * Order
   */
  export interface Order {
    id: number;
    client_order_id?: string;
    symbol: string;
    side: 'buy' | 'sell';
    order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
    time_in_force: 'gtc' | 'ioc' | 'fok';
    size: string;
    price?: string;
    stop_price?: string;
    limit_price?: string;
    status: 'open' | 'filled' | 'cancelled' | 'rejected' | 'partial_fill';
    created_at: string;
    updated_at: string;
  }

  /**
   * Order creation parameters
   */
  export interface OrderParams {
    symbol: string;
    side: 'buy' | 'sell';
    type?: 'market' | 'limit' | 'stop' | 'stop_limit';
    size: number;
    price?: number;
    timeInForce?: 'gtc' | 'ioc' | 'fok';
    clientOrderId?: string;
    reduceOnly?: boolean;
    postOnly?: boolean;
  }

  /**
   * Cancel all orders parameters
   */
  export interface CancelAllOrdersParams {
    symbol?: string;
    side?: 'buy' | 'sell';
    orderType?: 'market' | 'limit' | 'stop' | 'stop_limit';
  }

  /**
   * Order history parameters
   */
  export interface OrderHistoryParams {
    symbol?: string;
    status?: string;
    limit?: number;
    offset?: number;
    startTime?: number;
    endTime?: number;
  }

  /**
   * Trade history parameters
   */
  export interface TradeHistoryParams {
    symbol?: string;
    limit?: number;
    offset?: number;
    startTime?: number;
    endTime?: number;
  }

  /**
   * WebSocket event types
   */
  export type WebSocketEventType = 
    'ticker' | 
    'trade' | 
    'orderbook' | 
    'order' | 
    'position' | 
    'wallet';

  /**
   * WebSocket subscription request
   */
  export interface WebSocketSubscription {
    type: 'subscribe';
    channels: string[];
    symbols?: string[];
  }

  /**
   * WebSocket message
   */
  export interface WebSocketMessage {
    type: string;
    channel: string;
    symbol?: string;
    data: any;
    timestamp: string;
  }
}

export = DeltaExchange; 
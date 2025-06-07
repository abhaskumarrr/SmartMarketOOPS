/**
 * Unified Delta Exchange Service for India Testnet
 * Production-ready implementation with proper error handling, authentication, and WebSocket support
 * Based on official Delta Exchange India API documentation
 */
import { EventEmitter } from 'events';
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
    quoting_asset: {
        symbol: string;
    };
    settling_asset: {
        symbol: string;
    };
    underlying_asset: {
        symbol: string;
    };
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
    size: number;
    order_type: 'limit_order' | 'market_order';
    limit_price?: string;
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
export declare class DeltaExchangeUnified extends EventEmitter {
    private credentials;
    private client;
    private wsClient?;
    private baseUrl;
    private wsUrl;
    private isInitialized;
    private productCache;
    private symbolToProductId;
    private reconnectAttempts;
    private maxReconnectAttempts;
    private reconnectDelay;
    private candleCache;
    private indicatorCache;
    constructor(credentials: DeltaCredentials);
    /**
     * Initialize the Delta Exchange service
     */
    private initialize;
    /**
     * Load all available products from Delta Exchange
     */
    private loadProducts;
    /**
     * Test authentication with Delta Exchange
     */
    private testAuthentication;
    /**
     * Generate signature for authenticated requests
     */
    private generateSignature;
    /**
     * Make authenticated request to Delta Exchange API
     */
    private makeAuthenticatedRequest;
    /**
     * Get product by symbol
     */
    getProductBySymbol(symbol: string): DeltaProduct | undefined;
    /**
     * Get product ID by symbol
     */
    getProductId(symbol: string): number | undefined;
    /**
     * Check if service is ready for trading
     */
    isReady(): boolean;
    /**
     * Get account balance
     */
    getBalance(): Promise<DeltaBalance[]>;
    /**
     * Get current positions
     */
    getPositions(productId?: number): Promise<DeltaPosition[]>;
    /**
     * Get all positions for major trading assets
     */
    getAllPositions(): Promise<DeltaPosition[]>;
    /**
     * Place a new order
     */
    placeOrder(orderRequest: DeltaOrderRequest): Promise<DeltaOrder>;
    /**
     * Cancel an order
     */
    cancelOrder(orderId: number): Promise<DeltaOrder>;
    /**
     * Get order status
     */
    getOrder(orderId: number): Promise<DeltaOrder>;
    /**
     * Get all open orders
     */
    getOpenOrders(productId?: number): Promise<DeltaOrder[]>;
    /**
     * Convert decimal size to contract units for Delta Exchange
     * Delta Exchange perpetual futures use contract units, not decimal amounts
     */
    private convertToContractUnits;
    /**
     * Validate order request
     */
    private validateOrderRequest;
    /**
     * Get market data for a product (current prices)
     */
    getMarketData(symbol: string): Promise<any>;
    /**
     * Get order book for a product
     */
    getOrderBook(symbol: string, depth?: number): Promise<any>;
    /**
     * Initialize WebSocket connection for real-time data
     */
    connectWebSocket(symbols: string[]): void;
    /**
     * Subscribe to a WebSocket channel
     */
    private subscribeToChannel;
    /**
     * Handle WebSocket message
     */
    private handleWebSocketMessage;
    /**
     * Handle WebSocket reconnection
     */
    private handleReconnect;
    /**
     * Disconnect WebSocket
     */
    disconnectWebSocket(): void;
    /**
     * Get all available products
     */
    getAllProducts(): DeltaProduct[];
    /**
     * Get trading pairs suitable for perpetual futures
     */
    getPerpetualProducts(): DeltaProduct[];
    /**
     * Get historical candle data for a specific timeframe
     */
    getCandleData(symbol: string, timeframe: Timeframe, limit?: number): Promise<DeltaCandle[]>;
    /**
     * Get candles data (alias for getCandleData for compatibility)
     */
    getCandles(productId: number, timeframe: string, limit?: number): Promise<any[]>;
    /**
     * Get symbol by product ID (Environment-aware mapping)
     */
    private getSymbolByProductId;
    /**
     * Convert timeframe string to Timeframe type
     */
    private convertTimeframeFromString;
    /**
     * Get multi-timeframe data for a symbol
     */
    getMultiTimeframeData(symbol: string, timeframes?: Timeframe[]): Promise<MultiTimeframeData>;
    /**
     * Calculate technical indicators for candle data
     */
    private calculateIndicators;
    /**
     * Calculate ATR (Average True Range)
     */
    private calculateATR;
    /**
     * Calculate RSI (Relative Strength Index)
     */
    private calculateRSI;
    /**
     * Calculate MACD (Moving Average Convergence Divergence)
     */
    private calculateMACD;
    /**
     * Calculate EMA (Exponential Moving Average)
     */
    private calculateEMA;
    /**
     * Convert timeframe to Delta Exchange format
     */
    private convertTimeframe;
    /**
     * Get timeframe duration in seconds
     */
    private getTimeframeSeconds;
    /**
     * Get cached candle data
     */
    getCachedCandleData(symbol: string, timeframe: Timeframe): DeltaCandle[] | undefined;
    /**
     * Get cached indicators
     */
    getCachedIndicators(symbol: string, timeframe: Timeframe): TechnicalIndicators | undefined;
    /**
     * Cleanup resources
     */
    cleanup(): void;
}

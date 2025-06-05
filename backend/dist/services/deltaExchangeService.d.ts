/**
 * Delta Exchange India Trading Service
 * Comprehensive integration with Delta Exchange India API
 */
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
declare class DeltaExchangeService {
    private client;
    private credentials;
    private baseUrl;
    private isInitialized;
    private productCache;
    private symbolToProductId;
    constructor(credentials: DeltaCredentials);
    /**
     * Initialize the service and load products
     */
    private initializeService;
    /**
     * Load and cache all available products
     */
    private loadProducts;
    /**
     * Generate HMAC-SHA256 signature for authentication
     */
    private generateSignature;
    /**
     * Make authenticated request to Delta Exchange API
     */
    private makeAuthenticatedRequest;
    /**
     * Check if service is ready
     */
    isReady(): boolean;
    /**
     * Get product ID from symbol
     */
    getProductId(symbol: string): number | null;
    /**
     * Get product information
     */
    getProduct(symbol: string): any | null;
    /**
     * Get all available products
     */
    getAllProducts(): any[];
    /**
     * Get supported symbols
     */
    getSupportedSymbols(): string[];
    /**
     * Get real-time market data for a symbol
     */
    getMarketData(symbol: string): Promise<MarketData | null>;
    /**
     * Get mock market data as fallback
     */
    private getMockMarketData;
    /**
     * Get market data for multiple symbols
     */
    getMultipleMarketData(symbols: string[]): Promise<MarketData[]>;
    /**
     * Place a new order
     */
    placeOrder(orderRequest: OrderRequest): Promise<Order | null>;
    /**
     * Cancel an order
     */
    cancelOrder(productId: number, orderId: number): Promise<boolean>;
    /**
     * Get open orders
     */
    getOpenOrders(productId?: number): Promise<Order[]>;
    /**
     * Get positions
     */
    getPositions(): Promise<Position[]>;
    /**
     * Get wallet balances using proper Delta Exchange API
     */
    getBalances(): Promise<Balance[]>;
    /**
     * Utility delay function
     */
    private delay;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export { DeltaExchangeService };
export default DeltaExchangeService;

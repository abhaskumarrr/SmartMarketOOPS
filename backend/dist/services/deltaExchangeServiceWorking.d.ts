export = DeltaExchangeServiceWorking;
declare class DeltaExchangeServiceWorking {
    constructor(credentials: any);
    credentials: any;
    baseUrl: string;
    client: any;
    isInitialized: boolean;
    productCache: Map<any, any>;
    symbolToProductId: Map<any, any>;
    /**
     * Initialize the service and load products
     */
    initializeService(): Promise<void>;
    /**
     * Load and cache all available products
     */
    loadProducts(): Promise<void>;
    /**
     * Generate HMAC-SHA256 signature for authentication
     */
    generateSignature(method: any, path: any, queryString: any, body: any, timestamp: any): string;
    /**
     * Make authenticated request to Delta Exchange API
     */
    makeAuthenticatedRequest(method: any, path: any, params?: {}, data?: any): Promise<any>;
    /**
     * Check if service is ready
     */
    isReady(): boolean;
    /**
     * Get product ID from symbol
     */
    getProductId(symbol: any): any;
    /**
     * Get product information
     */
    getProduct(symbol: any): any;
    /**
     * Get all available products
     */
    getAllProducts(): any[];
    /**
     * Get supported symbols
     */
    getSupportedSymbols(): any[];
    /**
     * Get real-time market data for a symbol
     */
    getMarketData(symbol: any): Promise<{
        symbol: any;
        price: number;
        change: number;
        changePercent: number;
        volume: number;
        high24h: number;
        low24h: number;
        timestamp: number;
        source: string;
        markPrice: number;
        indexPrice: number;
        openInterest: number;
    }>;
    /**
     * Get market data for multiple symbols
     */
    getMultipleMarketData(symbols: any): Promise<{
        symbol: any;
        price: number;
        change: number;
        changePercent: number;
        volume: number;
        high24h: number;
        low24h: number;
        timestamp: number;
        source: string;
        markPrice: number;
        indexPrice: number;
        openInterest: number;
    }[]>;
    /**
     * Place a new order
     */
    placeOrder(orderRequest: any): Promise<any>;
    /**
     * Cancel an order
     */
    cancelOrder(productId: any, orderId: any): Promise<boolean>;
    /**
     * Get open orders
     */
    getOpenOrders(productId: any): Promise<any>;
    /**
     * Get positions
     */
    getPositions(): Promise<any>;
    /**
     * Get wallet balances
     */
    getBalances(): Promise<any>;
    /**
     * Utility delay function
     */
    delay(ms: any): Promise<any>;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}

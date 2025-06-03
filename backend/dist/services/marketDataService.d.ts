/**
 * Market Data Service
 * Fetches real-time market data from Delta Exchange using CCXT
 */
export interface MarketData {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    volume: number;
    high24h: number;
    low24h: number;
    timestamp: number;
    bid?: number;
    ask?: number;
    open?: number;
    close?: number;
}
export interface OrderBookData {
    symbol: string;
    bids: [number, number][];
    asks: [number, number][];
    timestamp: number;
}
export interface TradeData {
    symbol: string;
    price: number;
    amount: number;
    side: 'buy' | 'sell';
    timestamp: number;
}
declare class MarketDataService {
    private exchange;
    private isInitialized;
    private supportedSymbols;
    private symbolMapping;
    private lastPrices;
    private rateLimitDelay;
    private useRealData;
    constructor();
    /**
     * Initialize CCXT Delta Exchange connection
     */
    private initializeExchange;
    /**
     * Get default API credentials
     */
    private getDefaultCredentials;
    /**
     * Check if service is ready
     */
    isReady(): boolean;
    /**
     * Get real-time market data for a symbol
     */
    getMarketData(symbol: string): Promise<MarketData | null>;
    /**
     * Get market data for multiple symbols
     */
    getMultipleMarketData(symbols: string[]): Promise<MarketData[]>;
    /**
     * Get order book data
     */
    getOrderBook(symbol: string, limit?: number): Promise<OrderBookData | null>;
    /**
     * Get recent trades
     */
    getRecentTrades(symbol: string, limit?: number): Promise<TradeData[]>;
    /**
     * Get supported symbols
     */
    getSupportedSymbols(): string[];
    /**
     * Fallback mock data generator
     */
    private getMockMarketData;
    /**
     * Get base price for symbol (for mock data)
     */
    private getBasePriceForSymbol;
    /**
     * Utility delay function
     */
    private delay;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export declare const marketDataService: MarketDataService;
export default marketDataService;

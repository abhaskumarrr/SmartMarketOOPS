/**
 * Accurate Market Data Service
 * Uses multiple reliable sources with proper validation
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
    source: string;
    isValidated: boolean;
}
declare class AccurateMarketDataService {
    private exchanges;
    private isInitialized;
    private supportedSymbols;
    private lastPrices;
    private priceValidationThreshold;
    constructor();
    /**
     * Initialize multiple exchange connections for data validation
     */
    private initializeExchanges;
    /**
     * Get market data from external APIs as fallback
     */
    private getExternalMarketData;
    /**
     * Get market data from CCXT exchanges
     */
    private getExchangeMarketData;
    /**
     * Validate price data across multiple sources
     */
    private validatePrices;
    /**
     * Get accurate market data for a symbol
     */
    getMarketData(symbol: string): Promise<MarketData | null>;
    /**
     * Get market data for multiple symbols
     */
    getMultipleMarketData(symbols: string[]): Promise<MarketData[]>;
    /**
     * Check if service is ready
     */
    isReady(): boolean;
    /**
     * Get supported symbols
     */
    getSupportedSymbols(): string[];
    /**
     * Fallback mock data generator (only used as last resort)
     */
    private getMockMarketData;
    /**
     * Get realistic base price for symbol
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
export declare const accurateMarketDataService: AccurateMarketDataService;
export default accurateMarketDataService;

/**
 * Multi-Timeframe Data Collector
 * Implements comprehensive data collection across 4H, 1H, 15M, and 5M timeframes
 * with intelligent caching, synchronization, and validation for ML feature engineering
 */
export interface TimeframeConfig {
    timeframe: string;
    limit: number;
    cacheTTL: number;
    refreshInterval: number;
}
export interface OHLCVData {
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}
export interface MultiTimeframeData {
    symbol: string;
    timestamp: number;
    timeframes: {
        '4h': OHLCVData[];
        '1h': OHLCVData[];
        '15m': OHLCVData[];
        '5m': OHLCVData[];
    };
    synchronized: boolean;
    lastUpdate: number;
}
export interface ValidationResult {
    isValid: boolean;
    errors: string[];
    warnings: string[];
    dataQuality: number;
}
export declare class MultiTimeframeDataCollector {
    private deltaApi;
    private redis;
    private ccxtExchange;
    private timeframeConfigs;
    private intervals;
    private isCollecting;
    constructor();
    /**
     * Initialize the data collector
     */
    initialize(): Promise<void>;
    /**
     * Start data collection for specified symbols
     */
    startCollection(symbols: string[]): Promise<void>;
    /**
     * Stop data collection
     */
    stopCollection(): Promise<void>;
    /**
     * Get synchronized multi-timeframe data for a symbol
     */
    getMultiTimeframeData(symbol: string): Promise<MultiTimeframeData | null>;
    /**
     * Validate multi-timeframe data quality
     */
    validateData(symbol: string): Promise<ValidationResult>;
    /**
     * Get data statistics for monitoring
     */
    getDataStatistics(): Promise<Record<string, any>>;
    /**
     * Start data collection for a specific symbol
     */
    private startSymbolCollection;
    /**
     * Fetch and synchronize data across all timeframes
     */
    private fetchAndSynchronizeData;
    /**
     * Fetch OHLCV data for a specific timeframe
     */
    private fetchTimeframeData;
    /**
     * Fetch data from Delta Exchange
     */
    private fetchFromDeltaExchange;
    /**
     * Fetch data from CCXT (Binance fallback)
     */
    private fetchFromCCXT;
    /**
     * Synchronize timestamps across timeframes
     */
    private synchronizeTimeframes;
    /**
     * Cache multi-timeframe data
     */
    private cacheMultiTimeframeData;
    /**
     * Validate timeframe data quality
     */
    private validateTimeframeData;
    /**
     * Convert timeframe to milliseconds
     */
    private getTimeframeMilliseconds;
    /**
     * Convert timeframe format for Delta Exchange
     */
    private convertTimeframeForDelta;
    /**
     * Convert symbol format for Binance
     */
    private convertSymbolForBinance;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}

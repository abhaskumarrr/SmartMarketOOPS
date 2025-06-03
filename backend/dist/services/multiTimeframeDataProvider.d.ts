/**
 * Multi-Timeframe Data Provider
 * Handles data aggregation and synchronization across multiple timeframes
 */
import { MarketDataPoint, EnhancedMarketData } from '../types/marketData';
export type Timeframe = '1m' | '3m' | '5m' | '15m' | '1h' | '4h' | '1d';
export interface TimeframeConfig {
    timeframe: Timeframe;
    multiplier: number;
    priority: number;
}
export interface MultiTimeframeData {
    timestamp: number;
    timeframes: {
        [key in Timeframe]?: EnhancedMarketData;
    };
}
export declare class MultiTimeframeDataProvider {
    private timeframeConfigs;
    private baseTimeframe;
    constructor();
    /**
     * Initialize timeframe configurations with proper relationships
     */
    private initializeTimeframeConfigs;
    /**
     * Generate multi-timeframe data from base 1-minute data
     */
    generateMultiTimeframeData(baseData: MarketDataPoint[], targetTimeframes: Timeframe[]): MultiTimeframeData[];
    /**
     * Aggregate base data to higher timeframe
     */
    private aggregateToTimeframe;
    /**
     * Aggregate a group of candles into a single candle
     */
    private aggregateGroup;
    /**
     * Enhance market data with technical indicators
     */
    private enhanceMarketData;
    /**
     * Synchronize all timeframes to create aligned multi-timeframe data
     */
    private synchronizeTimeframes;
    /**
     * Find the corresponding candle for a given timestamp in a specific timeframe
     */
    private findCorrespondingCandle;
    /**
     * Get timeframe priority for decision making
     */
    getTimeframePriority(timeframe: Timeframe): number;
    /**
     * Get timeframe multiplier
     */
    getTimeframeMultiplier(timeframe: Timeframe): number;
    /**
     * Validate timeframe relationships
     */
    validateTimeframeRelationships(): boolean;
    /**
     * Get supported timeframes in priority order
     */
    getSupportedTimeframes(): Timeframe[];
    /**
     * Check if timeframe is supported
     */
    isTimeframeSupported(timeframe: string): timeframe is Timeframe;
}
export declare function createMultiTimeframeDataProvider(): MultiTimeframeDataProvider;

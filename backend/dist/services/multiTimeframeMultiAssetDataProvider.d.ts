/**
 * Multi-Timeframe Multi-Asset Data Provider
 * Combines multi-timeframe analysis with multi-asset support for comprehensive market data
 */
import { MarketDataPoint } from '../types/marketData';
import { CryptoPair } from './multiAssetDataProvider';
import { Timeframe } from './multiTimeframeDataProvider';
export interface TimeframeAssetConfig {
    asset: CryptoPair;
    timeframes: Timeframe[];
    priority: 'PRIMARY' | 'SECONDARY' | 'CONFIRMATION';
    weight: number;
}
export interface MultiTimeframeMultiAssetData {
    timestamp: number;
    assets: {
        [asset in CryptoPair]?: {
            [timeframe in Timeframe]?: MarketDataPoint;
        };
    };
    crossAssetAnalysis: {
        correlations: {
            [timeframePair: string]: {
                btc_eth: number;
                btc_sol: number;
                eth_sol: number;
            };
        };
        dominance: {
            [timeframe in Timeframe]?: {
                btc: number;
                eth: number;
                sol: number;
            };
        };
        volatilityRanking: {
            [timeframe in Timeframe]?: CryptoPair[];
        };
    };
    timeframeConsensus: {
        [asset in CryptoPair]?: {
            bullishTimeframes: Timeframe[];
            bearishTimeframes: Timeframe[];
            neutralTimeframes: Timeframe[];
            overallSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
            consensusStrength: number;
        };
    };
}
export declare class MultiTimeframeMultiAssetDataProvider {
    private multiAssetProvider;
    private multiTimeframeProvider;
    private supportedTimeframes;
    private supportedAssets;
    constructor();
    /**
     * Fetch comprehensive multi-timeframe multi-asset data
     */
    fetchComprehensiveData(startDate: Date, endDate: Date, assetConfigs: TimeframeAssetConfig[], primaryTimeframe?: Timeframe): Promise<MultiTimeframeMultiAssetData[]>;
    /**
     * Fetch base data for all asset-timeframe combinations
     */
    private fetchBaseData;
    /**
     * Fetch data for a single asset-timeframe combination
     */
    private fetchSingleAssetTimeframeData;
    /**
     * Generate fallback data when real data is unavailable
     */
    private generateFallbackData;
    /**
     * Align data across multiple timeframes and assets
     */
    private alignMultiTimeframeMultiAssetData;
    /**
     * Find the closest candle for a given timestamp and timeframe
     */
    private findClosestCandle;
    /**
     * Enhance data with cross-asset analysis
     */
    private enhanceWithCrossAssetAnalysis;
    /**
     * Add timeframe consensus analysis
     */
    private addTimeframeConsensus;
    /**
     * Calculate correlations for a specific timeframe
     */
    private calculateTimeframeCorrelations;
    /**
     * Calculate market dominance for a timeframe
     */
    private calculateMarketDominance;
    /**
     * Calculate volatility ranking for a timeframe
     */
    private calculateVolatilityRanking;
    /**
     * Calculate timeframe consensus for an asset
     */
    private calculateTimeframeConsensus;
    private createAssetTimeframeCombinations;
    private getUniqueTimeframes;
    private getTimeframeMinutes;
    private getAssetBasePrice;
    private getAssetVolatility;
    private calculateCorrelation;
    private calculateStandardDeviation;
    private determineCandleSentiment;
    /**
     * Get supported assets
     */
    getSupportedAssets(): CryptoPair[];
    /**
     * Get supported timeframes
     */
    getSupportedTimeframes(): Timeframe[];
}
export declare function createMultiTimeframeMultiAssetDataProvider(): MultiTimeframeMultiAssetDataProvider;

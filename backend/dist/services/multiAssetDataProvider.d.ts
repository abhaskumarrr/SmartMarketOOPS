/**
 * Multi-Asset Data Provider
 * Handles data fetching and processing for multiple cryptocurrency pairs
 */
import { MarketDataPoint } from '../types/marketData';
export type CryptoPair = 'BTCUSD' | 'ETHUSD' | 'SOLUSD';
export interface AssetConfig {
    symbol: CryptoPair;
    binanceSymbol: string;
    name: string;
    category: 'large-cap' | 'mid-cap' | 'alt-coin';
    volatilityProfile: 'low' | 'medium' | 'high';
    correlationGroup: 'bitcoin' | 'ethereum' | 'layer1';
}
export interface MultiAssetData {
    timestamp: number;
    assets: {
        [key in CryptoPair]?: MarketDataPoint;
    };
    correlations: {
        btc_eth: number;
        btc_sol: number;
        eth_sol: number;
    };
    marketDominance: {
        btc: number;
        eth: number;
        sol: number;
    };
}
export interface AssetSpecificFeatures {
    symbol: CryptoPair;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    volatilityRatio: number;
    volumeProfile: number;
    priceStability: number;
    supportStrength: number;
    resistanceStrength: number;
    trendConsistency: number;
    btcCorrelation: number;
    ethCorrelation: number;
    solCorrelation: number;
    largCapBehavior: number;
    altCoinBehavior: number;
    rsi_adjusted: number;
    macd_strength: number;
    volume_anomaly: number;
}
export declare class MultiAssetDataProvider {
    private assetConfigs;
    private correlationWindow;
    constructor();
    /**
     * Initialize asset configurations
     */
    private initializeAssetConfigs;
    /**
     * Fetch historical data for all supported assets
     */
    fetchMultiAssetData(timeframe: string, startDate: Date, endDate: Date, assets?: CryptoPair[]): Promise<{
        [key in CryptoPair]?: MarketDataPoint[];
    }>;
    /**
     * Fetch data for a specific asset
     */
    private fetchAssetData;
    /**
     * Generate asset-specific features for model training
     */
    generateAssetSpecificFeatures(assetData: {
        [key in CryptoPair]?: MarketDataPoint[];
    }, targetAsset: CryptoPair, index: number): AssetSpecificFeatures | null;
    /**
     * Calculate volatility ratio compared to BTC
     */
    private calculateVolatilityRatio;
    /**
     * Calculate recent volatility for an asset
     */
    private calculateRecentVolatility;
    /**
     * Calculate volume profile strength
     */
    private calculateVolumeProfile;
    /**
     * Calculate price stability index
     */
    private calculatePriceStability;
    /**
     * Calculate market structure strength
     */
    private calculateMarketStructure;
    /**
     * Calculate trend consistency
     */
    private calculateTrendConsistency;
    /**
     * Calculate cross-asset correlations
     */
    private calculateCrossAssetCorrelations;
    /**
     * Calculate Pearson correlation between two assets
     */
    private calculatePearsonCorrelation;
    /**
     * Calculate returns for a data slice
     */
    private calculateReturns;
    /**
     * Calculate asset category behaviors
     */
    private calculateCategoryBehaviors;
    /**
     * Calculate adjusted RSI for asset volatility
     */
    private calculateAdjustedRSI;
    /**
     * Calculate MACD strength relative to asset
     */
    private calculateMACDStrength;
    /**
     * Calculate volume anomaly detection
     */
    private calculateVolumeAnomaly;
    /**
     * Get asset configuration
     */
    getAssetConfig(asset: CryptoPair): AssetConfig | undefined;
    /**
     * Get all supported assets
     */
    getSupportedAssets(): CryptoPair[];
}
export declare function createMultiAssetDataProvider(): MultiAssetDataProvider;

/**
 * Multi-Asset Model Training Data Processor
 * Processes training data for multiple cryptocurrency pairs with cross-asset features
 */
import { TrainingFeatures } from './modelTrainingDataProcessor';
import { CryptoPair } from './multiAssetDataProvider';
export interface MultiAssetTrainingFeatures extends TrainingFeatures {
    asset_type: number;
    btc_price_normalized: number;
    eth_price_normalized: number;
    sol_price_normalized: number;
    btc_correlation: number;
    eth_correlation: number;
    sol_correlation: number;
    btc_dominance: number;
    eth_dominance: number;
    sol_dominance: number;
    volatility_ratio: number;
    volume_profile: number;
    price_stability: number;
    support_strength: number;
    resistance_strength: number;
    trend_consistency: number;
    large_cap_behavior: number;
    alt_coin_behavior: number;
    rsi_adjusted: number;
    macd_strength: number;
    volume_anomaly: number;
    cross_asset_momentum: number;
    relative_strength: number;
    btc_future_return_1h: number;
    eth_future_return_1h: number;
    sol_future_return_1h: number;
    portfolio_future_return_1h: number;
    best_asset_1h: number;
}
export interface MultiAssetTrainingDataset {
    features: MultiAssetTrainingFeatures[];
    assetBreakdown: {
        btc: number;
        eth: number;
        sol: number;
    };
    correlationStats: {
        btc_eth: number;
        btc_sol: number;
        eth_sol: number;
    };
    metadata: {
        symbols: CryptoPair[];
        startDate: Date;
        endDate: Date;
        totalSamples: number;
        featureCount: number;
        trainSplit: number;
        validationSplit: number;
        testSplit: number;
    };
}
export declare class MultiAssetModelTrainer {
    private dataProvider;
    /**
     * Process multi-asset training data
     */
    processMultiAssetTrainingData(timeframe: string, startDate: Date, endDate: Date, assets?: CryptoPair[], trainSplit?: number, validationSplit?: number, testSplit?: number): Promise<MultiAssetTrainingDataset>;
    /**
     * Validate asset data availability
     */
    private validateAssetData;
    /**
     * Generate unified features combining all assets
     */
    private generateUnifiedFeatures;
    /**
     * Generate features for a specific asset at a specific time point
     */
    private generateAssetFeatures;
    /**
     * Calculate cross-asset features
     */
    private calculateCrossAssetFeatures;
    /**
     * Normalize prices based on recent range
     */
    private normalizePrices;
    /**
     * Calculate cross-asset momentum
     */
    private calculateCrossAssetMomentum;
    /**
     * Calculate relative strength of target asset
     */
    private calculateRelativeStrength;
    /**
     * Calculate future returns for all assets
     */
    private calculateMultiAssetFutureReturns;
    /**
     * Determine best performing asset
     */
    private determineBestAsset;
    /**
     * Get asset type number
     */
    private getAssetTypeNumber;
    /**
     * Convert return to signal
     */
    private returnToSignal;
    /**
     * Calculate correlation statistics
     */
    private calculateCorrelationStats;
    /**
     * Calculate asset breakdown
     */
    private calculateAssetBreakdown;
    /**
     * Clean multi-asset data
     */
    private cleanMultiAssetData;
    /**
     * Split multi-asset dataset
     */
    splitMultiAssetDataset(dataset: MultiAssetTrainingDataset): {
        train: MultiAssetTrainingFeatures[];
        validation: MultiAssetTrainingFeatures[];
        test: MultiAssetTrainingFeatures[];
    };
}
export declare function createMultiAssetModelTrainer(): MultiAssetModelTrainer;

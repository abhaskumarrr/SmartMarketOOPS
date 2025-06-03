#!/usr/bin/env node
/**
 * Multi-Asset AI Model Retraining Script
 * Retrains all AI models using 6 months of real market data from BTC, ETH, and SOL
 */
declare class MultiAssetModelRetrainingRunner {
    private multiAssetTrainer;
    private modelTrainer;
    /**
     * Run comprehensive multi-asset model retraining
     */
    runMultiAssetRetraining(): Promise<void>;
    /**
     * Fetch and process multi-asset training data
     */
    private fetchMultiAssetTrainingData;
    /**
     * Train models with multi-asset data
     */
    private trainMultiAssetModels;
    /**
     * Convert multi-asset features to standard training format
     */
    private convertMultiAssetFeatures;
    /**
     * Save multi-asset trained models
     */
    private saveMultiAssetModels;
    /**
     * Generate comprehensive multi-asset training report
     */
    private generateMultiAssetTrainingReport;
    /**
     * Validate cross-asset performance
     */
    private validateCrossAssetPerformance;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export { MultiAssetModelRetrainingRunner };

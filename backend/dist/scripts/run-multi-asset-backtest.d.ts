#!/usr/bin/env node
/**
 * Multi-Asset Comprehensive Backtesting Script
 * Tests multi-asset AI trading strategies across BTC, ETH, and SOL with portfolio optimization
 */
declare class MultiAssetBacktestRunner {
    private dataProvider;
    private supportedAssets;
    /**
     * Run comprehensive multi-asset backtest
     */
    runMultiAssetBacktest(): Promise<void>;
    /**
     * Create test configurations for multi-asset backtesting
     */
    private createMultiAssetTestConfigurations;
    /**
     * Run a single multi-asset backtest
     */
    private runSingleMultiAssetBacktest;
    /**
     * Load multi-asset market data
     */
    private loadMultiAssetData;
    /**
     * Enhance market data with technical indicators
     */
    private enhanceMarketData;
    /**
     * Test individual asset performance
     */
    private testIndividualAssets;
    /**
     * Test portfolio strategy with multi-asset system
     */
    private testPortfolioStrategy;
    /**
     * Calculate average portfolio allocation
     */
    private calculateAverageAllocation;
    /**
     * Analyze cross-asset relationships
     */
    private analyzeCrossAssetRelationships;
    /**
     * Calculate correlation benefit
     */
    private calculateCorrelationBenefit;
    /**
     * Calculate diversification ratio
     */
    private calculateDiversificationRatio;
    /**
     * Analyze multi-asset backtest results
     */
    private analyzeMultiAssetResults;
    /**
     * Calculate average correlations across all results
     */
    private calculateAverageCorrelations;
    /**
     * Generate multi-asset recommendations
     */
    private generateMultiAssetRecommendations;
}
export { MultiAssetBacktestRunner };

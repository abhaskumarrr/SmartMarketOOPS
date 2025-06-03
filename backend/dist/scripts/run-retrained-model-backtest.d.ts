#!/usr/bin/env node
/**
 * Retrained Model Comprehensive Backtest
 * Tests the retrained AI models against the original models with multi-timeframe analysis
 */
declare class RetrainedModelBacktestRunner {
    private backtester;
    /**
     * Run comprehensive backtest comparing original vs retrained models
     */
    runComparativeBacktest(): Promise<void>;
    /**
     * Create test configurations for comparison
     */
    private createTestConfigurations;
    /**
     * Run model comparison for a specific configuration
     */
    private runModelComparison;
    /**
     * Load market data for testing
     */
    private loadMarketData;
    /**
     * Enhance market data with technical indicators
     */
    private enhanceMarketData;
    /**
     * Create and initialize retrained model
     */
    private createAndInitializeRetrainedModel;
    /**
     * Test a specific model
     */
    private testModel;
    /**
     * Calculate improvements between models
     */
    private calculateImprovements;
    /**
     * Analyze overall results across all configurations
     */
    private analyzeOverallResults;
    /**
     * Generate recommendations based on results
     */
    private generateRecommendations;
}
export { RetrainedModelBacktestRunner };

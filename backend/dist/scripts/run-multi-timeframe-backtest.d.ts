#!/usr/bin/env node
/**
 * Multi-Timeframe Backtesting Script
 * Comprehensive testing of the multi-timeframe AI trading system
 */
declare class MultiTimeframeBacktestRunner {
    private backtester;
    /**
     * Run comprehensive multi-timeframe backtest
     */
    runBacktest(): Promise<void>;
    /**
     * Initialize infrastructure
     */
    private initializeInfrastructure;
    /**
     * Create test configurations for different scenarios
     */
    private createTestConfigurations;
    /**
     * Run a single backtest configuration
     */
    private runSingleBacktest;
    /**
     * Analyze specific results for insights
     */
    private analyzeSpecificResults;
    /**
     * Run comparative analysis across all configurations
     */
    private runComparativeAnalysis;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export { MultiTimeframeBacktestRunner };

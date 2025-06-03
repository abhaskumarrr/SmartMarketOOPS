#!/usr/bin/env node
/**
 * Simple Backtesting Script
 * Focuses on core backtesting logic without database storage issues
 */
declare class SimpleBacktestRunner {
    /**
     * Run comprehensive backtesting simulation
     */
    runBacktest(): Promise<void>;
    /**
     * Create backtest configuration
     */
    private createBacktestConfig;
    /**
     * Create trading strategy
     */
    private createTradingStrategy;
    /**
     * Initialize infrastructure
     */
    private initializeInfrastructure;
    /**
     * Load and enhance market data
     */
    private loadMarketData;
    /**
     * Run the core backtesting logic
     */
    private runBacktestLogic;
    /**
     * Publish trading signal to Redis Streams
     */
    private publishTradingSignal;
    /**
     * Display comprehensive results
     */
    private displayResults;
    /**
     * Get performance rating
     */
    private getPerformanceRating;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export { SimpleBacktestRunner };

#!/usr/bin/env node
/**
 * Robust Backtesting Script
 * Focuses on core backtesting logic with graceful error handling
 */
declare class RobustBacktestRunner {
    /**
     * Run comprehensive backtesting simulation with robust error handling
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
     * Display comprehensive summary
     */
    private displaySummary;
    /**
     * Get performance rating
     */
    private getPerformanceRating;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export { RobustBacktestRunner };

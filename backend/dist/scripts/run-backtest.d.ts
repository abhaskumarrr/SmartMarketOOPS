#!/usr/bin/env node
/**
 * Comprehensive Backtesting Script
 * Executes a complete backtesting simulation with real infrastructure
 */
declare class BacktestRunner {
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
     * Display comprehensive summary
     */
    private displaySummary;
    /**
     * Get performance rating
     */
    private getPerformanceRating;
    /**
     * Verify data persistence in QuestDB
     */
    private verifyDataPersistence;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export { BacktestRunner };

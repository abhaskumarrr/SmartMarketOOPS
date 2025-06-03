/**
 * Performance Analytics for Backtesting
 * Calculates comprehensive trading performance metrics
 */
import { Trade, PortfolioSnapshot, PerformanceMetrics, BacktestConfig } from '../types/marketData';
export declare class PerformanceAnalytics {
    /**
     * Calculate comprehensive performance metrics
     */
    static calculateMetrics(trades: Trade[], portfolioHistory: PortfolioSnapshot[], config: BacktestConfig): PerformanceMetrics;
    /**
     * Calculate period returns from portfolio history
     */
    private static calculateReturns;
    /**
     * Calculate annualized volatility
     */
    private static calculateVolatility;
    /**
     * Calculate Sortino Ratio
     */
    private static calculateSortinoRatio;
    /**
     * Generate detailed performance report
     */
    static generateReport(metrics: PerformanceMetrics, trades: Trade[], config: BacktestConfig): string;
    /**
     * Get performance rating based on key metrics
     */
    private static getPerformanceRating;
    /**
     * Get top performing trades
     */
    private static getTopTrades;
}

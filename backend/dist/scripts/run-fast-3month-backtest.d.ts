#!/usr/bin/env node
/**
 * Fast 3-Month Backtest System
 * Streamlined version for quick execution
 * Mock Delta balance: $2500, 75% usage = $1875, 200x leverage
 */
declare class Fast3MonthBacktest {
    /**
     * Main execution function
     */
    runFast3MonthBacktest(): Promise<void>;
    /**
     * Run backtest for specific asset
     */
    private runAssetBacktest;
    /**
     * Generate fast trading signals
     */
    private generateFastSignal;
    /**
     * Generate fast report
     */
    private generateFastReport;
}
export { Fast3MonthBacktest };

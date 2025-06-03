#!/usr/bin/env node
/**
 * Real Delta Testnet 3-Month Backtest System
 * Fetches real balance from Delta testnet API, uses 75% with 200x leverage
 * Runs comprehensive 3-month backtest on BTC, ETH, SOL
 */
declare class RealDelta3MonthBacktest {
    private deltaApi;
    private strategy;
    constructor();
    /**
     * Main execution function
     */
    runRealDelta3MonthBacktest(): Promise<void>;
    /**
     * Fetch real balance from Delta testnet API
     */
    private fetchRealDeltaBalance;
    /**
     * Run 3-month backtest for a specific asset
     */
    private run3MonthAssetBacktest;
    /**
     * Calculate position size for real money trading
     */
    private calculateRealMoneyPosition;
    /**
     * Generate simple but effective trading signals (proven working strategy)
     */
    private generateSimpleSignal;
    /**
     * Generate comprehensive 3-month trading report
     */
    private generateComprehensive3MonthReport;
}
export { RealDelta3MonthBacktest };

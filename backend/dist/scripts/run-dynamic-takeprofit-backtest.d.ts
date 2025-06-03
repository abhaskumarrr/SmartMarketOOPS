#!/usr/bin/env node
/**
 * Dynamic Take Profit 3-Month Backtest System
 * Enhanced strategy with adaptive take profit levels
 * Target: Improve from +8.5% to +15-20% returns
 */
declare class DynamicTakeProfitBacktest {
    private takeProfitManager;
    constructor();
    /**
     * Main execution function
     */
    runDynamicTakeProfitBacktest(): Promise<void>;
    /**
     * Run enhanced backtest for specific asset
     */
    private runEnhancedAssetBacktest;
    /**
     * Generate enhanced trading signals with market analysis
     */
    private generateEnhancedSignal;
    /**
     * Detect market regime based on price action and volatility
     */
    private detectMarketRegime;
    /**
     * Calculate momentum indicator
     */
    private calculateMomentum;
    /**
     * Calculate volume strength
     */
    private calculateVolumeStrength;
    /**
     * Process dynamic take profit levels for existing positions
     */
    private processDynamicTakeProfits;
    /**
     * Calculate P&L for partial exit
     */
    private calculatePartialPnl;
    /**
     * Generate enhanced performance report
     */
    private generateEnhancedReport;
}
export { DynamicTakeProfitBacktest };

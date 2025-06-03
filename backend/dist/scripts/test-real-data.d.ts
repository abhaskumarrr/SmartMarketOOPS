#!/usr/bin/env node
/**
 * Test Real Data Fetching
 * Simple script to test Binance data provider and real data integration
 */
declare class RealDataTester {
    /**
     * Test real data fetching and basic backtesting
     */
    testRealData(): Promise<void>;
    /**
     * Test available data providers
     */
    private testAvailableProviders;
    /**
     * Test Binance data fetching
     */
    private testBinanceDataFetching;
    /**
     * Test real data backtesting
     */
    private testRealDataBacktest;
    /**
     * Validate data quality
     */
    private validateDataQuality;
    /**
     * Enhance data with basic technical indicators
     */
    private enhanceDataWithIndicators;
    /**
     * Display backtest results
     */
    private displayBacktestResults;
}
export { RealDataTester };

#!/usr/bin/env node
/**
 * Test Enhanced Strategy in Trending Market Conditions
 * Creates a trending market scenario to validate strategy performance
 */
declare class TrendingMarketTest {
    /**
     * Test strategy in trending market conditions
     */
    runTest(): Promise<void>;
    /**
     * Create synthetic trending market data
     */
    private createTrendingMarketData;
    /**
     * Create config optimized for trending markets
     */
    private createTrendingConfig;
    /**
     * Create strategy optimized for trending markets
     */
    private createOptimizedStrategy;
    /**
     * Test strategy with trending data
     */
    private testStrategy;
    /**
     * Display comprehensive results
     */
    private displayResults;
    private getPerformanceRating;
}
export { TrendingMarketTest };

#!/usr/bin/env node
/**
 * Multi-Timeframe Data Collector Test
 * Comprehensive testing of data collection, caching, synchronization, and validation
 */
declare class MultiTimeframeDataCollectorTest {
    private collector;
    private testSymbols;
    constructor();
    /**
     * Run comprehensive multi-timeframe data collection test
     */
    runTest(): Promise<void>;
    /**
     * Test collector initialization
     */
    private testInitialization;
    /**
     * Test data fetching for individual timeframes
     */
    private testTimeframeDataFetching;
    /**
     * Test multi-timeframe synchronization
     */
    private testMultiTimeframeSynchronization;
    /**
     * Test caching mechanisms
     */
    private testCachingMechanisms;
    /**
     * Test data validation
     */
    private testDataValidation;
    /**
     * Test real-time data collection
     */
    private testRealTimeDataCollection;
    /**
     * Test performance and statistics
     */
    private testPerformanceAndStatistics;
    /**
     * Sleep utility
     */
    private sleep;
}
export { MultiTimeframeDataCollectorTest };

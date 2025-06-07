#!/usr/bin/env node
/**
 * Enhanced Delta Exchange Integration Test
 * Tests all the improvements made to Delta Exchange API integration
 */
declare class EnhancedDeltaIntegrationTest {
    private deltaApi;
    constructor();
    /**
     * Run comprehensive enhanced integration test
     */
    runTest(): Promise<void>;
    /**
     * Test enhanced initialization
     */
    private testEnhancedInitialization;
    /**
     * Test enhanced rate limiting
     */
    private testEnhancedRateLimiting;
    /**
     * Test enhanced market data retrieval
     */
    private testEnhancedMarketData;
    /**
     * Test enhanced symbol/product ID mapping
     */
    private testEnhancedSymbolMapping;
    /**
     * Test enhanced authentication
     */
    private testEnhancedAuthentication;
    /**
     * Test enhanced order placement
     */
    private testEnhancedOrderPlacement;
    /**
     * Sleep utility
     */
    private sleep;
}
export { EnhancedDeltaIntegrationTest };

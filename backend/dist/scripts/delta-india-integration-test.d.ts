#!/usr/bin/env node
/**
 * Delta Exchange India Integration Test
 * Complete test of Delta India API with real order placement
 */
declare class DeltaIndiaIntegrationTest {
    private deltaApi;
    constructor();
    /**
     * Run complete Delta India integration test
     */
    runTest(): Promise<void>;
    /**
     * Initialize API with Delta India credentials
     */
    private initializeApi;
    /**
     * Test public endpoints
     */
    private testPublicEndpoints;
    /**
     * Test authentication endpoints
     */
    private testAuthentication;
    /**
     * Test perpetual contracts specific functionality
     */
    private testPerpetualContracts;
    /**
     * Place actual test order on Delta India
     */
    private placeTestOrder;
    /**
     * Sleep utility
     */
    private sleep;
}
export { DeltaIndiaIntegrationTest };

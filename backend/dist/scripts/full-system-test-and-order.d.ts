#!/usr/bin/env node
/**
 * Full System Test and Order Placement
 * Complete end-to-end test with real order placement on Delta Exchange
 */
declare class FullSystemTest {
    private deltaApi;
    private takeProfitManager;
    constructor();
    /**
     * Run complete system test
     */
    runFullTest(): Promise<void>;
    /**
     * Test connection and initialization
     */
    private testConnection;
    /**
     * Test dynamic take profit system
     */
    private testDynamicTakeProfit;
    /**
     * Test market data access
     */
    private testMarketData;
    /**
     * Test account access
     */
    private testAccountAccess;
    /**
     * Place actual test order on Delta Exchange
     */
    private placeTestOrder;
    /**
     * Sleep utility
     */
    private sleep;
}
export { FullSystemTest };

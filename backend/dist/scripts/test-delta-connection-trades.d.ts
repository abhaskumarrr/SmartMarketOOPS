#!/usr/bin/env node
/**
 * Delta Testnet Connection & Trade Execution Test
 * Comprehensive test to verify:
 * 1. Connection to Delta testnet
 * 2. Account balance retrieval
 * 3. Market data access
 * 4. Order placement (entry)
 * 5. Order cancellation (exit)
 * 6. Position management
 */
declare class DeltaTestnetTester {
    private deltaApi;
    private testResults;
    constructor();
    /**
     * Run comprehensive Delta testnet tests
     */
    runComprehensiveTest(): Promise<void>;
    /**
     * Test 1: Connection to Delta testnet
     */
    private testConnection;
    /**
     * Test 2: Account information retrieval
     */
    private testAccountInfo;
    /**
     * Test 3: Wallet balances retrieval
     */
    private testWalletBalances;
    /**
     * Test 4: Market data retrieval
     */
    private testMarketData;
    /**
     * Test 5: Current positions
     */
    private testPositions;
    /**
     * Test 6: Order placement (entry test)
     */
    private testOrderPlacement;
    /**
     * Test 7: Order cancellation (exit test)
     */
    private testOrderCancellation;
    /**
     * Test 8: Position management
     */
    private testPositionManagement;
    /**
     * Generate comprehensive test report
     */
    private generateTestReport;
}
export { DeltaTestnetTester };

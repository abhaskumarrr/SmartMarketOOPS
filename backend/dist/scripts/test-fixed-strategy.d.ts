#!/usr/bin/env node
/**
 * Test Fixed Strategy Implementation
 * Validates that all fixes from root cause analysis improve performance
 */
declare class FixedStrategyTester {
    private marketData;
    private baseConfig;
    constructor();
    /**
     * Run comprehensive comparison between original and fixed strategies
     */
    runComparison(): Promise<void>;
    /**
     * Test a specific strategy
     */
    private testStrategy;
    /**
     * Compare results between strategies
     */
    private compareResults;
    private loadMarketData;
    private createBaseConfig;
}
export { FixedStrategyTester };

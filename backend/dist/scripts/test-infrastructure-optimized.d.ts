#!/usr/bin/env node
/**
 * Optimized Infrastructure Test Script
 * Focuses on successful components with comprehensive performance metrics
 */
declare class OptimizedInfrastructureTester {
    private results;
    runAllTests(): Promise<void>;
    private testQuestDBConnection;
    private testRedisConnection;
    private testQuestDBDataInsertion;
    private testRedisStreamsOperations;
    private testQuestDBPerformance;
    private testRedisStreamsPerformance;
    private testBasicIntegration;
    private displayResults;
    cleanup(): Promise<void>;
}
export { OptimizedInfrastructureTester };

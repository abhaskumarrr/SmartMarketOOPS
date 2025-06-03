#!/usr/bin/env node
/**
 * Real Infrastructure Test Script
 * Comprehensive testing with actual Redis and QuestDB instances
 */
declare class RealInfrastructureTester {
    private results;
    runAllTests(): Promise<void>;
    private testQuestDBConnection;
    private testQuestDBOperations;
    private testQuestDBPerformance;
    private testRedisConnection;
    private testRedisStreams;
    private testRedisStreamsPerformance;
    private testEndToEndIntegration;
    private testEventProcessingPipeline;
    private displayResults;
    cleanup(): Promise<void>;
}
export { RealInfrastructureTester };

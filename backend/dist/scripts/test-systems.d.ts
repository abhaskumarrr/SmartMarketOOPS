#!/usr/bin/env node
/**
 * Test Systems Script
 * Comprehensive testing for QuestDB migration and Redis Streams
 */
declare class SystemTester {
    private results;
    runAllTests(): Promise<void>;
    private testQuestDBMock;
    private testRedisConnection;
    private testRedisStreamsMock;
    private testEventDrivenSystem;
    private testIntegration;
    private displayResults;
    cleanup(): Promise<void>;
}
export { SystemTester };

/**
 * Comprehensive Test Suite for Intelligent Trading Bot
 * Tests all components including multi-timeframe analysis, regime detection, and position management
 */
export interface TestResult {
    test_name: string;
    status: 'PASS' | 'FAIL' | 'SKIP';
    execution_time: number;
    details: string;
    metrics?: any;
}
export interface TestSuiteResults {
    total_tests: number;
    passed: number;
    failed: number;
    skipped: number;
    execution_time: number;
    results: TestResult[];
    overall_status: 'PASS' | 'FAIL';
}
export declare class IntelligentTradingBotTestSuite {
    private deltaService;
    private testResults;
    constructor();
    /**
     * Run complete test suite
     */
    runComprehensiveTests(): Promise<TestSuiteResults>;
    /**
     * Run unit tests for individual components
     */
    private runUnitTests;
    /**
     * Run integration tests
     */
    private runIntegrationTests;
    /**
     * Run performance tests
     */
    private runPerformanceTests;
    /**
     * Validate user stories
     */
    private runUserStoryValidation;
    /**
     * Test Multi-timeframe Analysis Engine
     */
    private testMultiTimeframeAnalysis;
    /**
     * Test Market Regime Detection
     */
    private testMarketRegimeDetection;
    /**
     * Test Adaptive Stop Loss System
     */
    private testAdaptiveStopLoss;
    /**
     * Test Smart Take Profit System
     */
    private testSmartTakeProfit;
    /**
     * Test ML Integration
     */
    private testMLIntegration;
    /**
     * Test Delta Exchange Integration
     */
    private testDeltaExchangeIntegration;
    /**
     * Test End-to-End Trading Flow
     */
    private testEndToEndTradingFlow;
    /**
     * Test Analysis Speed Performance
     */
    private testAnalysisSpeed;
    /**
     * Test Intelligent Position Management User Story
     */
    private testIntelligentPositionManagement;
    private addTestResult;
    private createMockMarketData;
    private validateAnalysisStructure;
    private testDataPipeline;
    private testMemoryUsage;
    private testConcurrentProcessing;
    private testAdaptiveRiskManagement;
    private testMultiTimeframeIntelligence;
    private logTestSummary;
}

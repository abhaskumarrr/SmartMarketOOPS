#!/usr/bin/env node
/**
 * Analysis-Execution Bridge Test
 * Comprehensive testing of real-time coordination layer with API and WebSocket functionality
 */
declare class AnalysisExecutionBridgeTest {
    private bridge;
    private testSymbols;
    private baseUrl;
    private wsUrl;
    constructor();
    /**
     * Run comprehensive analysis-execution bridge test
     */
    runTest(): Promise<void>;
    /**
     * Test bridge initialization and startup
     */
    private testBridgeInitialization;
    /**
     * Test REST API endpoints
     */
    private testRestApiEndpoints;
    /**
     * Test WebSocket functionality
     */
    private testWebSocketFunctionality;
    /**
     * Test trading signal flow
     */
    private testTradingSignalFlow;
    /**
     * Test error handling and failsafe mechanisms
     */
    private testErrorHandlingAndFailsafe;
    /**
     * Test real-time coordination
     */
    private testRealTimeCoordination;
    /**
     * Test performance and latency
     */
    private testPerformanceAndLatency;
    /**
     * Test emergency protocols
     */
    private testEmergencyProtocols;
    /**
     * Sleep utility
     */
    private sleep;
}
export { AnalysisExecutionBridgeTest };

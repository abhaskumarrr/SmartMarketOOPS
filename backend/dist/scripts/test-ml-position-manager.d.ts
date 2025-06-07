#!/usr/bin/env node
/**
 * ML Position Manager Test
 * Comprehensive testing of ML-powered position management, dynamic stops, and exit optimization
 */
declare class MLPositionManagerTest {
    private positionManager;
    private decisionEngine;
    private testSymbols;
    constructor();
    /**
     * Run comprehensive ML position manager test
     */
    runTest(): Promise<void>;
    /**
     * Test position manager initialization
     */
    private testInitialization;
    /**
     * Test position creation from trading decisions
     */
    private testPositionCreation;
    /**
     * Test position updates and ML predictions
     */
    private testPositionUpdates;
    /**
     * Test dynamic stop loss and take profit management
     */
    private testDynamicManagement;
    /**
     * Test exit signal detection
     */
    private testExitSignalDetection;
    /**
     * Test position closure and training data recording
     */
    private testPositionClosure;
    /**
     * Test performance metrics and analytics
     */
    private testPerformanceMetrics;
    /**
     * Test configuration management
     */
    private testConfigurationManagement;
    /**
     * Validate position structure
     */
    private validatePositionStructure;
    /**
     * Sleep utility
     */
    private sleep;
}
export { MLPositionManagerTest };

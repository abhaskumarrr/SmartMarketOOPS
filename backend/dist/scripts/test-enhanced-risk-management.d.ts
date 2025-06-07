#!/usr/bin/env node
/**
 * Enhanced Risk Management System Test
 * Comprehensive testing of risk controls, circuit breakers, and failsafe mechanisms
 */
declare class EnhancedRiskManagementTest {
    private riskManager;
    private decisionEngine;
    private testSymbols;
    constructor();
    /**
     * Run comprehensive enhanced risk management test
     */
    runTest(): Promise<void>;
    /**
     * Test risk management system initialization
     */
    private testInitialization;
    /**
     * Test risk assessment for trading decisions
     */
    private testRiskAssessment;
    /**
     * Test dynamic position sizing
     */
    private testDynamicPositionSizing;
    /**
     * Test circuit breakers and failsafe mechanisms
     */
    private testCircuitBreakers;
    /**
     * Test risk metrics calculation
     */
    private testRiskMetricsCalculation;
    /**
     * Test emergency actions and controls
     */
    private testEmergencyActions;
    /**
     * Test configuration management
     */
    private testConfigurationManagement;
    /**
     * Test performance monitoring and analytics
     */
    private testPerformanceMonitoring;
    /**
     * Validate risk assessment structure
     */
    private validateRiskAssessment;
    /**
     * Validate risk metrics structure
     */
    private validateRiskMetrics;
    /**
     * Get risk level description
     */
    private getRiskLevel;
    /**
     * Sleep utility
     */
    private sleep;
}
export { EnhancedRiskManagementTest };

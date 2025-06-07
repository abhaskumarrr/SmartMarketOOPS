#!/usr/bin/env node
/**
 * Enhanced Trading Decision Engine Test
 * Comprehensive testing of ML-driven trading decisions, ensemble voting, and risk management
 */
declare class EnhancedTradingDecisionEngineTest {
    private decisionEngine;
    private testSymbols;
    constructor();
    /**
     * Run comprehensive enhanced trading decision engine test
     */
    runTest(): Promise<void>;
    /**
     * Test decision engine initialization
     */
    private testInitialization;
    /**
     * Test ML feature processing
     */
    private testMLFeatureProcessing;
    /**
     * Test ensemble model voting
     */
    private testEnsembleVoting;
    /**
     * Test trading decision generation
     */
    private testTradingDecisionGeneration;
    /**
     * Test risk assessment and position sizing
     */
    private testRiskAssessmentAndPositionSizing;
    /**
     * Test confidence thresholds and filtering
     */
    private testConfidenceThresholds;
    /**
     * Test decision caching and history
     */
    private testDecisionCachingAndHistory;
    /**
     * Test configuration management
     */
    private testConfigurationManagement;
    /**
     * Validate decision structure
     */
    private validateDecisionStructure;
    /**
     * Sleep utility
     */
    private sleep;
}
export { EnhancedTradingDecisionEngineTest };

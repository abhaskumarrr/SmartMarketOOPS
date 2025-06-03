#!/usr/bin/env node
/**
 * Comprehensive Strategy Performance Analysis
 * Investigates why optimization shows poor trading returns despite technical success
 */
declare class StrategyPerformanceAnalyzer {
    private marketData;
    private baseConfig;
    constructor();
    /**
     * Run comprehensive strategy analysis
     */
    runAnalysis(): Promise<void>;
    /**
     * 1. Trade Execution Analysis
     */
    private analyzeTradeExecution;
    /**
     * Test a configuration with detailed logging
     */
    private testConfigurationDetailed;
    /**
     * 2. Signal Generation Investigation
     */
    private investigateSignalGeneration;
    /**
     * Test AI model predictions manually
     */
    private testAIModelPredictions;
    /**
     * Debug why signal generation failed
     */
    private debugSignalGeneration;
    /**
     * 3. Market Data Validation
     */
    private validateMarketData;
    /**
     * 4. Strategy Logic Review
     */
    private reviewStrategyLogic;
    /**
     * 5. Root Cause Analysis
     */
    private performRootCauseAnalysis;
    private loadMarketData;
    private createBaseConfig;
}
export { StrategyPerformanceAnalyzer };

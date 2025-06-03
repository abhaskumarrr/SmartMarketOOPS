#!/usr/bin/env node
/**
 * Multi-Timeframe Multi-Asset Comprehensive Backtesting Script
 * Executes comprehensive backtests combining multi-timeframe analysis with multi-asset portfolio optimization
 */
declare class MultiTimeframeMultiAssetBacktestRunner {
    private backtester;
    private configManager;
    /**
     * Run comprehensive multi-timeframe multi-asset backtesting suite
     */
    runComprehensiveBacktest(): Promise<void>;
    /**
     * Define test scenarios for comprehensive backtesting
     */
    private defineTestScenarios;
    /**
     * Run backtest for a single scenario
     */
    private runSingleScenario;
    /**
     * Generate comprehensive analysis of all results
     */
    private generateComprehensiveAnalysis;
    /**
     * Analyze timeframe performance across scenarios
     */
    private analyzeTimeframePerformance;
    /**
     * Analyze multi-asset performance
     */
    private analyzeMultiAssetPerformance;
    /**
     * Analyze hierarchical decision effectiveness
     */
    private analyzeHierarchicalDecisions;
    /**
     * Analyze cross-asset correlations
     */
    private analyzeCrossAssetCorrelations;
    /**
     * Analyze best performing configurations
     */
    private analyzeBestPerformers;
    /**
     * Generate strategic recommendations
     */
    private generateStrategicRecommendations;
}
export { MultiTimeframeMultiAssetBacktestRunner };

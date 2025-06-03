#!/usr/bin/env node
/**
 * Comprehensive Hyperparameter Optimization Script
 * Systematically tests parameter combinations to maximize trading performance
 */
declare class HyperparameterOptimizationRunner {
    /**
     * Run comprehensive hyperparameter optimization
     */
    runOptimization(): Promise<void>;
    /**
     * Initialize infrastructure
     */
    private initializeInfrastructure;
    /**
     * Get number of optimization iterations based on environment
     */
    private getOptimizationIterations;
    /**
     * Save optimization results to files
     */
    private saveOptimizationResults;
    /**
     * Save top 5 configurations as CSV
     */
    private saveTop5AsCSV;
    /**
     * Display execution summary
     */
    private displayExecutionSummary;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export { HyperparameterOptimizationRunner };

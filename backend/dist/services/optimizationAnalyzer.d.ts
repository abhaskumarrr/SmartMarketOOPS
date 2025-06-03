/**
 * Optimization Results Analyzer
 * Analyzes hyperparameter optimization results and provides insights
 */
import { OptimizationResult } from './hyperparameterOptimizer';
export interface ParameterImpactAnalysis {
    parameter: string;
    correlation: number;
    significance: number;
    optimalRange: [number, number];
    impact: 'HIGH' | 'MEDIUM' | 'LOW';
}
export interface OptimizationSummary {
    totalConfigurations: number;
    bestConfiguration: OptimizationResult;
    top5Configurations: OptimizationResult[];
    parameterImpacts: ParameterImpactAnalysis[];
    performanceDistribution: {
        sharpeRatio: {
            min: number;
            max: number;
            mean: number;
            std: number;
        };
        totalReturn: {
            min: number;
            max: number;
            mean: number;
            std: number;
        };
        maxDrawdown: {
            min: number;
            max: number;
            mean: number;
            std: number;
        };
    };
    insights: string[];
    recommendations: string[];
}
export declare class OptimizationAnalyzer {
    /**
     * Analyze optimization results and provide comprehensive insights
     */
    analyzeResults(results: OptimizationResult[]): OptimizationSummary;
    /**
     * Display comprehensive optimization report
     */
    displayOptimizationReport(summary: OptimizationSummary): void;
    private displayOverview;
    private displayTop5Configurations;
    private displayParameterImpacts;
    private displayPerformanceDistribution;
    private displayBestConfigurationDetails;
    private displayTradeAnalysis;
    private displayInsightsAndRecommendations;
    /**
     * Analyze parameter impacts on performance
     */
    private analyzeParameterImpacts;
    /**
     * Calculate correlation between parameter and performance metric
     */
    private calculateCorrelation;
    /**
     * Find optimal range for a parameter
     */
    private findOptimalRange;
    /**
     * Calculate performance distribution statistics
     */
    private calculatePerformanceDistribution;
    /**
     * Calculate standard deviation
     */
    private calculateStandardDeviation;
    /**
     * Generate insights from optimization results
     */
    private generateInsights;
    /**
     * Generate recommendations based on optimization results
     */
    private generateRecommendations;
}
export declare function createOptimizationAnalyzer(): OptimizationAnalyzer;

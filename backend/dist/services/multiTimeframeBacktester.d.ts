/**
 * Multi-Timeframe Backtesting Engine
 * Validates trading strategies across multiple timeframes with proper temporal consistency
 */
import { BacktestConfig } from '../types/marketData';
import { Timeframe } from './multiTimeframeDataProvider';
interface TimeframePerformance {
    timeframe: Timeframe;
    signalsGenerated: number;
    signalsAccurate: number;
    accuracy: number;
    avgConfidence: number;
    contribution: number;
}
interface MultiTimeframeBacktestResult {
    config: BacktestConfig;
    performance: any;
    trades: any[];
    portfolioHistory: any[];
    timeframePerformances: TimeframePerformance[];
    temporalConsistency: {
        lookAheadBiasDetected: boolean;
        timeframeAlignmentIssues: number;
        dataIntegrityScore: number;
    };
    hierarchicalDecisionStats: {
        totalDecisions: number;
        higherTimeframeOverrides: number;
        consensusDecisions: number;
        conflictResolutions: number;
    };
    executionMetrics: {
        avgExecutionDelay: number;
        timeframeProcessingTime: {
            [key in Timeframe]?: number;
        };
        totalProcessingTime: number;
    };
}
export declare class MultiTimeframeBacktester {
    private dataProvider;
    private supportedTimeframes;
    constructor();
    /**
     * Run comprehensive multi-timeframe backtest
     */
    runBacktest(config: BacktestConfig, targetTimeframes?: Timeframe[]): Promise<MultiTimeframeBacktestResult>;
    /**
     * Load and prepare multi-timeframe data
     */
    private loadMultiTimeframeData;
    /**
     * Validate temporal consistency across timeframes
     */
    private validateTemporalConsistency;
    /**
     * Execute multi-timeframe backtest
     */
    private executeMultiTimeframeBacktest;
    /**
     * Analyze timeframe-specific performance
     */
    private analyzeTimeframePerformances;
    /**
     * Calculate execution metrics
     */
    private calculateExecutionMetrics;
    /**
     * Display comprehensive multi-timeframe backtest results
     */
    displayResults(result: MultiTimeframeBacktestResult): void;
}
export declare function createMultiTimeframeBacktester(): MultiTimeframeBacktester;
export {};

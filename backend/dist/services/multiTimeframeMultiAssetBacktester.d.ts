/**
 * Multi-Timeframe Multi-Asset Backtester Core
 * Advanced backtesting engine for multi-timeframe multi-asset strategies
 */
import { BacktestConfig } from '../types/marketData';
import { MultiTimeframeMultiAssetSignal } from './multiTimeframeMultiAssetStrategy';
import { CryptoPair } from './multiAssetDataProvider';
import { Timeframe } from './multiTimeframeDataProvider';
export interface TimeframeAssetConfig {
    asset: CryptoPair;
    timeframes: Timeframe[];
    priority: 'PRIMARY' | 'SECONDARY' | 'CONFIRMATION';
    weight: number;
}
export interface MultiTimeframeMultiAssetBacktestConfig extends BacktestConfig {
    assetConfigs: TimeframeAssetConfig[];
    primaryTimeframe: Timeframe;
    timeframeWeights: {
        [timeframe in Timeframe]?: number;
    };
    portfolioMode: 'SINGLE_ASSET' | 'MULTI_ASSET' | 'DYNAMIC';
    rebalanceFrequency: 'SIGNAL_BASED' | 'HOURLY' | 'DAILY' | 'WEEKLY';
}
export interface TimeframePerformance {
    timeframe: Timeframe;
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    totalTrades: number;
    avgTradeReturn: number;
    signalAccuracy: number;
}
export interface AssetTimeframePerformance {
    asset: CryptoPair;
    timeframePerformance: TimeframePerformance[];
    overallPerformance: {
        totalReturn: number;
        sharpeRatio: number;
        maxDrawdown: number;
        winRate: number;
        bestTimeframe: Timeframe;
        worstTimeframe: Timeframe;
    };
}
export interface MultiTimeframeMultiAssetBacktestResult {
    config: MultiTimeframeMultiAssetBacktestConfig;
    assetTimeframePerformance: AssetTimeframePerformance[];
    overallPerformance: {
        totalReturn: number;
        sharpeRatio: number;
        maxDrawdown: number;
        winRate: number;
        totalTrades: number;
        portfolioValue: number;
    };
    hierarchicalAnalysis: {
        timeframeConsensusAccuracy: number;
        higherTimeframeWinRate: number;
        conflictResolutionSuccess: number;
        bestPerformingHierarchy: Timeframe[];
    };
    crossAssetAnalysis: {
        correlationBenefit: number;
        diversificationRatio: number;
        portfolioOptimizationGain: number;
        bestAssetCombination: CryptoPair[];
    };
    executionMetrics: {
        totalSignals: number;
        executedTrades: number;
        signalToTradeRatio: number;
        avgDecisionTime: number;
        dataQualityScore: number;
    };
    signalHistory: MultiTimeframeMultiAssetSignal[];
    portfolioHistory: any[];
    duration: number;
    startTime: Date;
    endTime: Date;
}
export declare class MultiTimeframeMultiAssetBacktester {
    private dataProvider;
    private strategy;
    /**
     * Run comprehensive multi-timeframe multi-asset backtest
     */
    runBacktest(config: MultiTimeframeMultiAssetBacktestConfig): Promise<MultiTimeframeMultiAssetBacktestResult>;
    /**
     * Load comprehensive multi-timeframe multi-asset data
     */
    private loadComprehensiveData;
    /**
     * Execute the main backtesting logic
     */
    private executeBacktest;
    /**
     * Convert multi-timeframe data to single-timeframe format
     */
    private convertToSingleTimeframeData;
    /**
     * Update positions for all assets
     */
    private updateAllAssetPositions;
    /**
     * Check stop-loss and take-profit for all assets
     */
    private checkAllAssetStopLoss;
    /**
     * Execute trades based on multi-timeframe multi-asset signal
     */
    private executeSignalTrades;
    /**
     * Execute portfolio rebalancing
     */
    private executePortfolioRebalancing;
    /**
     * Calculate data quality score
     */
    private calculateDataQualityScore;
    /**
     * Get snapshot frequency based on rebalance frequency
     */
    private getSnapshotFrequency;
    /**
     * Analyze backtest results
     */
    private analyzeResults;
    /**
     * Calculate asset-timeframe performance
     */
    private calculateAssetTimeframePerformance;
    /**
     * Calculate performance for a specific timeframe
     */
    private calculateTimeframePerformance;
    /**
     * Calculate overall asset performance
     */
    private calculateOverallAssetPerformance;
    private calculateSharpeRatio;
    private calculateMaxDrawdown;
    private calculateHierarchicalAnalysis;
    private calculateCrossAssetAnalysis;
    private generateFinalResult;
}
export declare function createMultiTimeframeMultiAssetBacktester(): MultiTimeframeMultiAssetBacktester;

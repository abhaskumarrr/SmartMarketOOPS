#!/usr/bin/env node
/**
 * Frequency Optimized Trading Backtest
 * Target: 3-5 PROFITABLE trades daily with 75%+ win rate
 * Solution: Optimized thresholds for increased frequency while maintaining quality
 */
interface FrequencyOptimizedConfig {
    symbol: string;
    startDate: string;
    endDate: string;
    initialCapital: number;
    leverage: number;
    riskPerTrade: number;
    targetTradesPerDay: number;
    targetWinRate: number;
    mlAccuracy: number;
}
interface FrequencyOptimizedTrade {
    id: string;
    side: 'LONG' | 'SHORT';
    entryPrice: number;
    exitPrice: number;
    size: number;
    pnl: number;
    exitReason: string;
    mlConfidence: number;
    signalScore: number;
    qualityScore: number;
    holdTimeMinutes: number;
    timestamp: number;
}
declare class FrequencyOptimizedBacktester {
    private config;
    private currentBalance;
    private trades;
    private maxDrawdown;
    private peakBalance;
    private dailyTrades;
    constructor(config: FrequencyOptimizedConfig);
    runBacktest(): Promise<void>;
    private generateFrequencyOptimizedETHData;
    private getOptimizedTrendFactor;
    private generateFrequencyOptimizedOpportunities;
    private simulateOptimizedMLPrediction;
    private generateOptimizedTradingSignal;
    private calculateOptimizedQualityScore;
    private passesOptimizedFiltering;
    private executeOptimizedTrade;
    private calculateOptimizedHoldTime;
    private exitOptimizedTrade;
    private displayOptimizedResults;
}

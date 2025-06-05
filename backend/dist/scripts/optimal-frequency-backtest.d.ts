#!/usr/bin/env node
/**
 * Optimal Frequency Trading Backtest
 * Target: 3-5 PROFITABLE trades daily with 75%+ win rate
 * Solution: Smart signal filtering with optimal frequency
 */
interface OptimalConfig {
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
interface OptimalTrade {
    id: string;
    side: 'LONG' | 'SHORT';
    entryPrice: number;
    exitPrice: number;
    size: number;
    pnl: number;
    exitReason: string;
    mlConfidence: number;
    signalScore: number;
    holdTimeMinutes: number;
    timestamp: number;
}
declare class OptimalFrequencyBacktester {
    private config;
    private currentBalance;
    private trades;
    private maxDrawdown;
    private peakBalance;
    private dailyTrades;
    constructor(config: OptimalConfig);
    runBacktest(): Promise<void>;
    private generateOptimalFrequencyETHData;
    private getPeriodTrendFactor;
    private generateOptimalOpportunities;
    private simulateAccurateMLPrediction;
    private generateQualityTradingSignal;
    private calculateQualityScore;
    private executeOptimalTrade;
    private calculateOptimalHoldTime;
    private exitOptimalTrade;
    private displayOptimalResults;
}

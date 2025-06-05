#!/usr/bin/env node
/**
 * Balanced 75% Win Rate Trading Backtest
 * Target: 3-5 PROFITABLE trades daily with 75%+ win rate
 * Solution: Balanced filtering - strict enough for quality, loose enough for frequency
 */
interface BalancedConfig {
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
interface BalancedTrade {
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
declare class Balanced75PercentBacktester {
    private config;
    private currentBalance;
    private trades;
    private maxDrawdown;
    private peakBalance;
    private dailyTrades;
    constructor(config: BalancedConfig);
    runBacktest(): Promise<void>;
    private generateBalancedFrequencyETHData;
    private getBalancedTrendFactor;
    private generateBalancedOpportunities;
    private simulateBalancedMLPrediction;
    private generateBalancedTradingSignal;
    private calculateBalancedQualityScore;
    private passesBalancedFiltering;
    private executeBalancedTrade;
    private calculateBalancedHoldTime;
    private exitBalancedTrade;
    private displayBalancedResults;
}

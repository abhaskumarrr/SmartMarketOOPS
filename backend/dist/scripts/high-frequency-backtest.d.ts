#!/usr/bin/env node
/**
 * High-Frequency Trading Backtest
 * Target: 3-5 profitable trades daily with 85% ML accuracy
 */
interface HFTConfig {
    symbol: string;
    startDate: string;
    endDate: string;
    initialCapital: number;
    leverage: number;
    riskPerTrade: number;
    targetTradesPerDay: number;
    mlAccuracy: number;
}
interface HFTTrade {
    id: string;
    side: 'LONG' | 'SHORT';
    entryPrice: number;
    exitPrice: number;
    size: number;
    pnl: number;
    exitReason: string;
    mlConfidence: number;
    holdTimeMinutes: number;
    timestamp: number;
}
declare class HighFrequencyBacktester {
    private config;
    private currentBalance;
    private trades;
    private maxDrawdown;
    private peakBalance;
    private dailyTrades;
    constructor(config: HFTConfig);
    runBacktest(): Promise<void>;
    private generateHighFrequencyETHData;
    private getHourlyTrendFactor;
    private generateHourlyOpportunities;
    private simulateMLPrediction;
    private generateTradingSignal;
    private executeHFTrade;
    private calculateOptimalHoldTime;
    private exitHFTrade;
    private displayHFTResults;
}

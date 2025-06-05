#!/usr/bin/env node
/**
 * Minimal Backtest Implementation
 * Direct implementation without complex dependencies
 */
interface BacktestConfig {
    symbol: string;
    startDate: string;
    endDate: string;
    initialCapital: number;
    leverage: number;
    riskPerTrade: number;
}
interface Trade {
    id: string;
    side: 'LONG' | 'SHORT';
    entryPrice: number;
    exitPrice: number;
    size: number;
    pnl: number;
    exitReason: string;
    signalScore: number;
}
declare class MinimalBacktester {
    private config;
    private currentBalance;
    private trades;
    private maxDrawdown;
    private peakBalance;
    constructor(config: BacktestConfig);
    runBacktest(): Promise<void>;
    private generateETHData;
    private getETHTrendFactor;
    private generateAdvancedFilteredSignal;
    private executeFilteredTrade;
    private exitTrade;
    private displayResults;
}

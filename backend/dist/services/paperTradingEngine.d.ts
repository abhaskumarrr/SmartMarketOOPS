/**
 * Enhanced Paper Trading Engine with 75% Balance Allocation
 * Simulates live trading with real Delta Exchange market data
 * Implements frequency-optimized trading strategy with 85% ML accuracy
 */
import { TakeProfitLevel } from '../types/marketData';
import { DeltaCredentials } from './deltaExchangeService';
export interface PaperTrade {
    id: string;
    symbol: string;
    side: 'BUY' | 'SELL';
    size: number;
    entryPrice: number;
    entryTime: number;
    exitPrice?: number;
    exitTime?: number;
    pnl?: number;
    status: 'OPEN' | 'CLOSED' | 'CANCELLED';
    reason?: string;
    takeProfitLevels: TakeProfitLevel[];
    partialExits: PartialExit[];
    stopLoss: number;
    currentPrice: number;
    unrealizedPnl: number;
    maxProfit: number;
    maxLoss: number;
}
export interface PartialExit {
    level: number;
    percentage: number;
    price: number;
    timestamp: number;
    pnl: number;
    reason: string;
}
export interface PaperPortfolio {
    initialBalance: number;
    currentBalance: number;
    allocatedBalance: number;
    totalBalance: number;
    totalPnl: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    winRate: number;
    maxDrawdown: number;
    currentDrawdown: number;
    leverage: number;
    riskPerTrade: number;
    dailyTrades: number;
    targetTradesPerDay: number;
    mlAccuracy: number;
    peakBalance: number;
}
export interface FrequencyOptimizedConfig {
    mlConfidenceThreshold: number;
    signalScoreThreshold: number;
    qualityScoreThreshold: number;
    targetTradesPerDay: number;
    targetWinRate: number;
    mlAccuracy: number;
    maxConcurrentTrades: number;
    balanceAllocationPercent: number;
}
export declare class PaperTradingEngine {
    private takeProfitManager;
    private activeTrades;
    private closedTrades;
    private portfolio;
    private isRunning;
    private tradingAssets;
    private deltaService;
    private config;
    private dailyTradeCount;
    private lastTradeDate;
    private sessionStartTime;
    constructor(deltaCredentials: DeltaCredentials, config?: Partial<FrequencyOptimizedConfig>);
    /**
     * Start enhanced paper trading system with 75% balance allocation
     */
    startPaperTrading(): Promise<void>;
    /**
     * Initialize balance from Delta Exchange (75% allocation)
     */
    private initializeBalanceFromDelta;
    /**
     * Stop paper trading system
     */
    stopPaperTrading(): void;
    /**
     * Enhanced trading loop with frequency optimization
     */
    private runTradingLoop;
    /**
     * Update daily trade tracking
     */
    private updateDailyTradeTracking;
    /**
     * Update dynamic risk management based on performance
     */
    private updateDynamicRiskManagement;
    /**
     * Process trading for a specific asset with frequency optimization
     */
    private processAssetWithFrequencyOptimization;
    /**
     * Get current market data from Delta Exchange - REAL DATA ONLY
     */
    private getCurrentMarketData;
    /**
     * Get current price for asset (legacy method for compatibility)
     */
    private getCurrentPrice;
    /**
     * Update existing trades with current price
     */
    private updateExistingTrades;
    /**
     * Check for partial exits based on dynamic take profit levels
     */
    private checkPartialExits;
    /**
     * Check for stop loss
     */
    private checkStopLoss;
    /**
     * Check for frequency-optimized trading opportunities
     */
    private checkFrequencyOptimizedTradingOpportunity;
    /**
     * Check if signal passes frequency-optimized filters
     */
    private passesFrequencyOptimizedFilters;
    /**
     * Generate frequency-optimized trading signal
     */
    private generateFrequencyOptimizedSignal;
    /**
     * Open a new paper trade
     */
    private openTrade;
    /**
     * Close a paper trade
     */
    private closeTrade;
    /**
     * Calculate partial P&L
     */
    private calculatePartialPnl;
    /**
     * Update portfolio metrics
     */
    private updatePortfolioMetrics;
    /**
     * Generate final report
     */
    private generateFinalReport;
    /**
     * Get current portfolio status
     */
    getPortfolioStatus(): PaperPortfolio;
    /**
     * Get active trades
     */
    getActiveTrades(): PaperTrade[];
    /**
     * Get closed trades
     */
    getClosedTrades(): PaperTrade[];
    /**
     * Generate progress report during trading
     */
    private generateProgressReport;
    /**
     * Delay utility
     */
    private delay;
    /**
     * Sleep utility (legacy compatibility)
     */
    private sleep;
}

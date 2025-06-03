/**
 * Paper Trading Engine
 * Simulates live trading with real market data but virtual money
 */
import { TakeProfitLevel } from '../types/marketData';
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
    totalPnl: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    winRate: number;
    maxDrawdown: number;
    currentDrawdown: number;
    leverage: number;
    riskPerTrade: number;
}
export declare class PaperTradingEngine {
    private takeProfitManager;
    private activeTrades;
    private closedTrades;
    private portfolio;
    private isRunning;
    private tradingAssets;
    constructor(initialBalance?: number, leverage?: number, riskPerTrade?: number);
    /**
     * Start paper trading system
     */
    startPaperTrading(): Promise<void>;
    /**
     * Stop paper trading system
     */
    stopPaperTrading(): void;
    /**
     * Main trading loop
     */
    private runTradingLoop;
    /**
     * Process trading for a specific asset
     */
    private processAsset;
    /**
     * Get current price for asset from Delta Exchange
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
     * Check for new trading opportunities
     */
    private checkNewTradingOpportunity;
    /**
     * Generate trading signal for paper trading
     */
    private generatePaperTradingSignal;
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
     * Sleep utility
     */
    private sleep;
}

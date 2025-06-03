/**
 * Portfolio Manager for Backtesting
 * Manages positions, trades, and portfolio performance
 */
import { Position, Trade, PortfolioSnapshot, TradingSignal, BacktestConfig } from '../types/marketData';
export declare class PortfolioManager {
    private config;
    private cash;
    private positions;
    private trades;
    private portfolioHistory;
    private highWaterMark;
    private maxDrawdown;
    private totalCommission;
    constructor(config: BacktestConfig);
    /**
     * Execute a trading signal
     */
    executeTrade(signal: TradingSignal, currentPrice: number, timestamp: number): Trade | null;
    /**
     * Open a long position
     */
    private openLongPosition;
    /**
     * Open a short position
     */
    private openShortPosition;
    /**
     * Close a position
     */
    closePosition(positionKey: string, exitPrice: number, timestamp: number, reason: string): Trade | null;
    /**
     * Update all positions with current market prices
     */
    updatePositions(symbol: string, currentPrice: number, timestamp: number): void;
    /**
     * Check for stop loss and take profit triggers
     */
    checkStopLossAndTakeProfit(symbol: string, currentPrice: number, timestamp: number, signal?: TradingSignal): Trade[];
    /**
     * Create portfolio snapshot
     */
    createSnapshot(timestamp: number): PortfolioSnapshot;
    /**
     * Calculate position size based on signal and risk management
     */
    private calculatePositionSize;
    /**
     * Calculate P&L for a position
     */
    private calculatePnL;
    /**
     * Get total portfolio value
     */
    private getTotalPortfolioValue;
    /**
     * Format duration in human readable format
     */
    private formatDuration;
    getCash(): number;
    getPositions(): Position[];
    getTrades(): Trade[];
    getPortfolioHistory(): PortfolioSnapshot[];
    getMaxDrawdown(): number;
    getTotalCommission(): number;
}

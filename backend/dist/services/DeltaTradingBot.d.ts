/**
 * Delta Exchange Trading Bot
 * Production-ready trading bot for Delta Exchange India testnet
 * Integrates with ML models and implements proper risk management
 */
import { EventEmitter } from 'events';
import { DeltaExchangeUnified, DeltaOrder, DeltaPosition } from './DeltaExchangeUnified';
export interface BotConfig {
    id: string;
    name: string;
    symbol: string;
    strategy: 'momentum' | 'mean_reversion' | 'ml_driven' | 'scalping';
    capital: number;
    leverage: number;
    riskPerTrade: number;
    maxPositions: number;
    stopLoss: number;
    takeProfit: number;
    enabled: boolean;
    testnet: boolean;
}
export interface BotStatus {
    id: string;
    status: 'running' | 'stopped' | 'paused' | 'error';
    uptime: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    totalPnL: number;
    currentPositions: number;
    lastTradeTime?: Date;
    errorMessage?: string;
}
export interface TradeSignal {
    symbol: string;
    action: 'buy' | 'sell' | 'close';
    confidence: number;
    price?: number;
    size?: number;
    reason: string;
    timestamp: Date;
}
export declare class DeltaTradingBot extends EventEmitter {
    private config;
    private deltaService;
    private status;
    private isRunning;
    private positions;
    private orders;
    private startTime;
    private lastHealthCheck;
    private healthCheckInterval;
    private tradingInterval;
    constructor(config: BotConfig, deltaService: DeltaExchangeUnified);
    /**
     * Set up event listeners for Delta Exchange service
     */
    private setupDeltaEventListeners;
    /**
     * Start the trading bot
     */
    start(): Promise<void>;
    /**
     * Stop the trading bot
     */
    stop(): Promise<void>;
    /**
     * Pause the trading bot
     */
    pause(): void;
    /**
     * Resume the trading bot
     */
    resume(): void;
    /**
     * Get current bot status
     */
    getStatus(): BotStatus;
    /**
     * Update bot configuration
     */
    updateConfig(newConfig: Partial<BotConfig>): void;
    /**
     * Validate bot configuration
     */
    private validateConfig;
    /**
     * Load initial state (positions and orders)
     */
    private loadInitialState;
    /**
     * Start health check interval
     */
    private startHealthCheck;
    /**
     * Start trading loop
     */
    private startTradingLoop;
    /**
     * Perform health check
     */
    private performHealthCheck;
    /**
     * Execute main trading logic
     */
    private executeTradingLogic;
    /**
     * Generate trading signal based on strategy
     */
    private generateTradingSignal;
    /**
     * Execute a trading signal
     */
    private executeTradeSignal;
    /**
     * Calculate position size based on risk management
     */
    private calculatePositionSize;
    /**
     * Check if we can place a trade (risk management)
     */
    private canPlaceTrade;
    /**
     * Manage existing positions (stop loss, take profit)
     * CRITICAL: Uses ONLY Delta Exchange live data for risk management calculations
     */
    private manageExistingPositions;
    /**
     * Close a position
     */
    private closePosition;
    /**
     * Cancel all open orders
     */
    private cancelAllOrders;
    /**
     * Update positions and orders from exchange
     */
    private updatePositionsAndOrders;
    /**
     * Check risk limits
     */
    private checkRiskLimits;
    /**
     * Handle trading error
     */
    private handleTradingError;
    /**
     * Handle order placed event
     */
    private handleOrderPlaced;
    /**
     * Handle order cancelled event
     */
    private handleOrderCancelled;
    /**
     * Handle ticker update
     */
    private handleTickerUpdate;
    /**
     * Get current positions
     */
    getCurrentPositions(): DeltaPosition[];
    /**
     * Get current orders
     */
    getCurrentOrders(): DeltaOrder[];
    /**
     * Get performance metrics
     */
    getPerformanceMetrics(): any;
    /**
     * Force close all positions (emergency)
     */
    emergencyCloseAll(): Promise<void>;
    /**
     * Cleanup bot resources
     */
    cleanup(): void;
}

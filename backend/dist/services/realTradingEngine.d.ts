import { DeltaCredentials } from './deltaExchangeService';
interface RealTradingConfig {
    balanceAllocationPercent: number;
    maxLeverage: number;
    riskPerTrade: number;
    targetTradesPerDay: number;
    targetWinRate: number;
    mlConfidenceThreshold: number;
    signalScoreThreshold: number;
    qualityScoreThreshold: number;
    maxDrawdownPercent: number;
    tradingAssets: string[];
    checkIntervalMs: number;
    progressReportIntervalMs: number;
}
export declare class RealTradingEngine {
    private deltaService;
    private takeProfitManager;
    private activeTrades;
    private closedTrades;
    private portfolio;
    private config;
    private isRunning;
    private tradingAssets;
    private dailyTradeCount;
    private lastTradeDate;
    private sessionStartTime;
    constructor(deltaCredentials: DeltaCredentials, config?: Partial<RealTradingConfig>);
    /**
     * Start real trading with actual Delta Exchange orders
     */
    startRealTrading(): Promise<void>;
    /**
     * Initialize real balance from Delta Exchange
     */
    private initializeRealBalance;
    /**
     * Wait for Delta Exchange service to be ready
     */
    private waitForDeltaService;
    /**
     * Main trading loop for real trading
     */
    private runTradingLoop;
    /**
     * Delay utility
     */
    private delay;
    /**
     * Stop real trading
     */
    stopRealTrading(): Promise<void>;
    /**
     * Update existing trades with real market data and manage exits
     */
    private updateExistingTrades;
    /**
     * Check and execute take profit exits with REAL orders
     */
    private checkTakeProfitExits;
    /**
     * Execute partial exit with REAL Delta Exchange order
     */
    private executePartialExit;
    /**
     * Check and execute stop loss with REAL order
     */
    private checkStopLoss;
    /**
     * Execute stop loss with REAL Delta Exchange order
     */
    private executeStopLoss;
    /**
     * Generate and execute trading signals with REAL orders
     */
    private generateAndExecuteSignals;
    /**
     * Generate trading signal (simplified for demo)
     */
    private generateTradingSignal;
    /**
     * Check if signal should be executed
     */
    private shouldExecuteSignal;
    /**
     * Execute real trade with actual Delta Exchange order
     */
    private executeRealTrade;
    /**
     * Create real trade record after successful order placement
     */
    private createRealTrade;
    /**
     * Get product ID for symbol
     */
    private getProductId;
    /**
     * Close a real trade
     */
    private closeTrade;
    /**
     * Check if trading should stop
     */
    private shouldStopTrading;
    /**
     * Generate progress report
     */
    private generateProgressReport;
    /**
     * Generate final report
     */
    private generateFinalReport;
}
export {};

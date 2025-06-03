#!/usr/bin/env node
/**
 * Paper Trading System Runner
 * Runs live paper trading with dynamic take profit system
 */
declare class PaperTradingRunner {
    private engine;
    private statusInterval;
    constructor();
    /**
     * Start paper trading system
     */
    startPaperTrading(): Promise<void>;
    /**
     * Stop paper trading system
     */
    stopPaperTrading(): void;
    /**
     * Start status monitoring
     */
    private startStatusMonitoring;
    /**
     * Stop status monitoring
     */
    private stopStatusMonitoring;
    /**
     * Display current status
     */
    private displayStatus;
    /**
     * Handle graceful shutdown
     */
    setupGracefulShutdown(): void;
}
export { PaperTradingRunner };

#!/usr/bin/env node
/**
 * Start ML Trading System
 *
 * This script starts the complete ML-driven trading system that integrates
 * all our trading analysis (Fibonacci, SMC, confluence, candle formation)
 * as features for ML models to make actual trading decisions.
 *
 * Usage:
 *   npm run ml-trading              # Start with default config
 *   npm run ml-trading -- --paper   # Start in paper trading mode
 *   npm run ml-trading -- --live    # Start with real money (DANGEROUS!)
 */
declare const ML_TRADING_CONFIG: {
    symbols: string[];
    refreshInterval: number;
    maxConcurrentTrades: number;
    minConfidenceThreshold: number;
    enablePaperTrading: boolean;
    riskManagement: {
        maxDailyLoss: number;
        maxPositionSize: number;
    };
};
declare class MLTradingSystemLauncher {
    private mlTrading;
    private isShuttingDown;
    constructor();
    /**
     * Start the ML Trading System
     */
    start(): Promise<void>;
    /**
     * Parse command line arguments
     */
    private parseArguments;
    /**
     * Display startup banner with system information
     */
    private displayStartupBanner;
    /**
     * Validate environment and configuration
     */
    private validateEnvironment;
    /**
     * Setup signal handlers for graceful shutdown
     */
    private setupSignalHandlers;
    /**
     * Emergency shutdown procedure
     */
    private emergencyShutdown;
    /**
     * Keep the process alive
     */
    private keepAlive;
}
export { MLTradingSystemLauncher, ML_TRADING_CONFIG };

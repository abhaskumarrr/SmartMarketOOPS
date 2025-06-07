/**
 * ML Trading Integration Script
 *
 * This script integrates the ML Trading Decision Engine with our existing
 * analysis and execution infrastructure to create a complete automated
 * trading system that uses ML models as primary decision makers.
 */
interface MLTradingConfig {
    symbols: string[];
    refreshInterval: number;
    maxConcurrentTrades: number;
    minConfidenceThreshold: number;
    enablePaperTrading: boolean;
    riskManagement: {
        maxDailyLoss: number;
        maxPositionSize: number;
    };
}
export declare class MLTradingIntegration {
    private mlEngine;
    private mtfAnalyzer;
    private mlService;
    private tradingBot;
    private config;
    private isRunning;
    private activePositions;
    private dailyStats;
    constructor(config: MLTradingConfig);
    /**
     * Initialize all components and start ML-driven trading
     */
    initialize(): Promise<void>;
    /**
     * Start automated ML-driven trading
     */
    startTrading(): Promise<void>;
    /**
     * Stop automated trading
     */
    stopTrading(): Promise<void>;
    /**
     * Main trading loop - analyzes markets and executes ML-driven trades
     */
    private runTradingLoop;
    /**
     * Process a single symbol for ML trading decisions
     */
    private processSymbol;
    /**
     * Monitor active positions and manage exits
     */
    private monitorPositions;
    /**
     * Check if a position should be exited
     */
    private checkPositionExit;
    /**
     * Exit a position
     */
    private exitPosition;
    private initializeComponents;
    private validateConfiguration;
    private checkRiskConstraints;
    private shouldExitBasedOnML;
    private closeAllPositions;
    private logFinalStats;
    private trackPerformance;
    private sleep;
}
export {};

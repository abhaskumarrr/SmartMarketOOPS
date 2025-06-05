/**
 * Intelligent Trading Bot V2.0
 * Complete implementation with multi-timeframe analysis, regime detection, and adaptive position management
 */
export declare class IntelligentTradingBotV2 {
    private config;
    private deltaService;
    private mtfAnalyzer;
    private regimeDetector;
    private stopLossSystem;
    private takeProfitSystem;
    private mlService;
    private testSuite;
    private activePositions;
    private isRunning;
    private scanInterval;
    private totalTrades;
    private winningTrades;
    private totalPnL;
    private maxDrawdown;
    private startingBalance;
    constructor();
    /**
     * Start the intelligent trading bot
     */
    start(): Promise<void>;
    /**
     * Stop the trading bot
     */
    stop(): Promise<void>;
    /**
     * Main trading cycle
     */
    private tradingCycle;
    /**
     * Scan for intelligent trading opportunities
     */
    private scanForIntelligentOpportunities;
    /**
     * Manage positions with intelligent algorithms
     */
    private managePositionsIntelligently;
    /**
     * Evaluate intelligent trading opportunity
     */
    private evaluateIntelligentOpportunity;
    /**
     * Execute intelligent trade
     */
    private executeIntelligentTrade;
    private getCurrentBalance;
    private getCurrentPrice;
    private calculateContractSize;
    private displayConfiguration;
    private updateActivePositions;
    private checkIntelligentExitConditions;
    private updatePerformanceMetrics;
    private displayPerformanceReport;
}

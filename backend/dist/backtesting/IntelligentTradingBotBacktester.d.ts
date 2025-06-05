/**
 * Intelligent Trading Bot Backtester
 * Comprehensive backtesting with $10 capital, 200x leverage on ETH using 1 year data
 */
export interface BacktestConfig {
    symbol: string;
    startDate: string;
    endDate: string;
    initialCapital: number;
    leverage: number;
    riskPerTrade: number;
    maxPositions: number;
    timeframe: string;
}
export interface BacktestPosition {
    id: string;
    symbol: string;
    side: 'LONG' | 'SHORT';
    entryPrice: number;
    exitPrice?: number;
    size: number;
    leverage: number;
    entryTime: number;
    exitTime?: number;
    stopLoss: number;
    takeProfitLevels: number[];
    pnl: number;
    pnlPercent: number;
    exitReason: string;
    healthScore: number;
    regimeAtEntry: string;
    signals: any;
}
export interface BacktestResults {
    config: BacktestConfig;
    summary: {
        totalTrades: number;
        winningTrades: number;
        losingTrades: number;
        winRate: number;
        totalReturn: number;
        totalReturnPercent: number;
        maxDrawdown: number;
        maxDrawdownPercent: number;
        sharpeRatio: number;
        profitFactor: number;
        averageWin: number;
        averageLoss: number;
        largestWin: number;
        largestLoss: number;
        averageHoldTime: number;
        finalBalance: number;
    };
    trades: BacktestPosition[];
    dailyReturns: number[];
    equityCurve: {
        date: string;
        balance: number;
        drawdown: number;
    }[];
    monthlyBreakdown: {
        month: string;
        trades: number;
        pnl: number;
        winRate: number;
    }[];
    regimePerformance: {
        regime: string;
        trades: number;
        winRate: number;
        avgReturn: number;
    }[];
}
export declare class IntelligentTradingBotBacktester {
    private config;
    private mtfAnalyzer;
    private regimeDetector;
    private stopLossSystem;
    private takeProfitSystem;
    private mlService;
    private signalFilter;
    private currentBalance;
    private peakBalance;
    private currentDrawdown;
    private maxDrawdown;
    private totalTrades;
    private positions;
    private activePositions;
    private dailyBalances;
    constructor(config: BacktestConfig);
    /**
     * Run comprehensive backtest
     */
    runBacktest(): Promise<BacktestResults>;
    /**
     * Generate ETH historical data for 1 year
     */
    private generateETHHistoricalData;
    /**
     * Get ETH trend factor based on historical patterns
     */
    private getETHTrendFactor;
    /**
     * Evaluate new trading opportunity with advanced signal filtering
     */
    private evaluateNewOpportunity;
    /**
     * Execute trade with advanced filtered signal
     */
    private executeAdvancedFilteredTrade;
    /**
     * Execute backtest trade with dynamic risk ladder strategy (legacy method)
     */
    private executeBacktestTrade;
    /**
     * Update active positions
     */
    private updateActivePositions;
    /**
     * Close position and calculate P&L
     */
    private closePosition;
    /**
     * Generate comprehensive backtest results
     */
    private generateBacktestResults;
    private simulateMultiTimeframeAnalysis;
    private simulateRegimeDetection;
    private simulateMLPrediction;
    private evaluateIntelligentOpportunity;
    private calculateVolatility;
    private updateDailyTracking;
    private calculateDailyReturns;
    private calculateMonthlyBreakdown;
    private calculateRegimePerformance;
    private closeAllPositions;
    private createMockDataService;
    private logBacktestSummary;
    /**
     * Calculate dynamic risk based on current balance
     * Risk Ladder Strategy: Start ultra-aggressive, become conservative as balance grows
     */
    private calculateDynamicRisk;
    /**
     * Calculate dynamic leverage based on current balance
     * Leverage Ladder: Start extreme, reduce as balance grows
     */
    private calculateDynamicLeverage;
}

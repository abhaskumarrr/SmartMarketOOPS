/**
 * Backtesting Engine for Bot Strategies
 * Simplified backtesting engine for bot strategy validation
 */
interface BacktestConfig {
    botId: string;
    strategy: string;
    parameters: Record<string, any>;
    symbol: string;
    timeframe: string;
    startDate: Date;
    endDate: Date;
    initialCapital: number;
    leverage: number;
    riskPerTrade: number;
    commission: number;
}
interface BacktestResult {
    performance: {
        totalReturn: number;
        totalReturnPercent: number;
        annualizedReturn: number;
        sharpeRatio: number;
        maxDrawdown: number;
        maxDrawdownPercent: number;
        winRate: number;
        profitFactor: number;
        totalTrades: number;
        winningTrades: number;
        losingTrades: number;
        averageWin: number;
        averageLoss: number;
        largestWin: number;
        largestLoss: number;
    };
    trades: any[];
    config: BacktestConfig;
    startTime: number;
    endTime: number;
    duration: number;
}
/**
 * Run backtest for a bot strategy
 */
export declare const runBacktest: (config: BacktestConfig) => Promise<BacktestResult>;
export {};

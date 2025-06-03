/**
 * Backtesting Engine
 * Orchestrates the complete backtesting process with event-driven architecture
 */
import { BacktestConfig, BacktestResult, TradingStrategy } from '../types/marketData';
export declare class BacktestingEngine {
    private config;
    private strategy;
    private portfolioManager;
    private marketData;
    private currentIndex;
    constructor(config: BacktestConfig, strategy: TradingStrategy);
    /**
     * Run the complete backtesting process
     */
    run(): Promise<BacktestResult>;
    /**
     * Initialize infrastructure services
     */
    private initializeInfrastructure;
    /**
     * Load and enhance market data
     */
    private loadMarketData;
    /**
     * Store market data in QuestDB
     */
    private storeMarketData;
    /**
     * Enhance market data with technical indicators
     */
    private enhanceMarketData;
    /**
     * Process market data chronologically
     */
    private processMarketData;
    /**
     * Publish trading signal to Redis Streams
     */
    private publishTradingSignal;
    /**
     * Store trade in QuestDB
     */
    private storeTrade;
    /**
     * Store portfolio snapshot in QuestDB
     */
    private storePortfolioSnapshot;
    /**
     * Calculate performance metrics
     */
    private calculatePerformance;
    /**
     * Store final results in QuestDB
     */
    private storeResults;
    /**
     * Generate final backtest result
     */
    private generateResult;
    /**
     * Generate and log performance report
     */
    generateReport(result: BacktestResult): string;
}

/**
 * Multi-Timeframe AI Trading System
 * Implements hierarchical decision making across multiple timeframes
 */
import { TradingStrategy, TradingSignal, EnhancedMarketData, BacktestConfig } from '../types/marketData';
export declare enum TradeType {
    SCALPING = "SCALPING",
    DAY_TRADING = "DAY_TRADING",
    SWING_TRADING = "SWING_TRADING",
    POSITION_TRADING = "POSITION_TRADING"
}
export declare enum MarketRegime {
    TRENDING_BULLISH = "TRENDING_BULLISH",
    TRENDING_BEARISH = "TRENDING_BEARISH",
    SIDEWAYS = "SIDEWAYS",
    VOLATILE = "VOLATILE",
    BREAKOUT = "BREAKOUT"
}
export declare class MultiTimeframeAITradingSystem implements TradingStrategy {
    readonly name = "Multi_Timeframe_AI_System";
    parameters: Record<string, any>;
    private config?;
    private dataProvider;
    private lastDecisionTime;
    private supportedTimeframes;
    constructor();
    /**
     * Initialize the trading system
     */
    initialize(config: BacktestConfig): void;
    /**
     * Generate trading signal using multi-timeframe analysis
     */
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    /**
     * Prepare multi-timeframe data from single timeframe input
     */
    private prepareMultiTimeframeData;
    /**
     * Simulate higher timeframe candle by aggregating recent data
     */
    private simulateHigherTimeframeCandle;
    /**
     * Analyze all enabled timeframes
     */
    private analyzeAllTimeframes;
    /**
     * Analyze a specific timeframe
     */
    private analyzeTimeframe;
    /**
     * Generate multi-timeframe signal with consensus analysis
     */
    private generateMultiTimeframeSignal;
    /**
     * Generate conflict resolution explanation
     */
    private generateConflictResolution;
    /**
     * Apply hierarchical decision making
     */
    private applyHierarchicalDecision;
    /**
     * Get strategy description
     */
    getDescription(): string;
}
export declare function createMultiTimeframeAITradingSystem(): MultiTimeframeAITradingSystem;

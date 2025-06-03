/**
 * Fixed Intelligent AI-Driven Trading System
 * Implements all fixes identified in the root cause analysis
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
export declare class FixedIntelligentTradingSystem implements TradingStrategy {
    readonly name = "Fixed_Intelligent_AI_System";
    parameters: Record<string, any>;
    private config?;
    private tradingPrinciples;
    private lastDecisionTime;
    private decisionCooldown;
    constructor();
    /**
     * Override signal generation with fixes
     */
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    /**
     * Enhanced AI model predictions with wider ranges (Fix #2)
     */
    private getEnhancedAIModelPredictions;
    /**
     * Aggressive Transformer model with wider prediction ranges
     */
    private simulateAggressiveTransformer;
    /**
     * Decisive LSTM model with momentum focus
     */
    private simulateDecisiveLSTM;
    /**
     * Active SMC analyzer with volume and price action
     */
    private simulateActiveSMC;
    /**
     * Fixed market regime analysis that enables sideways trading (Fix #3)
     */
    private analyzeMarketRegimeFixed;
    /**
     * Fixed intelligent decision generation with lower thresholds (Fix #1)
     */
    private generateIntelligentDecisionFixed;
    /**
     * Fixed risk management with improved parameters (Fix #4)
     */
    private applyFixedRiskManagement;
    protected getMinConfidence(): number;
    protected getModelConsensusThreshold(): number;
    protected getDecisionCooldown(): number;
    private hasRequiredIndicators;
    private lastDecisionTime;
    private config?;
    initialize(config: BacktestConfig): void;
    getDescription(): string;
}
export declare function createFixedIntelligentTradingSystem(): FixedIntelligentTradingSystem;

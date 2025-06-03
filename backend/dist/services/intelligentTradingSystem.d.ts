/**
 * Intelligent AI-Driven Trading System
 * Integrates existing ML models, Smart Money Concepts, and trading guide principles
 */
import { TradingStrategy, TradingSignal, EnhancedMarketData, BacktestConfig } from '../types/marketData';
export declare enum TradeType {
    SCALPING = "SCALPING",// < 15 minutes
    DAY_TRADING = "DAY_TRADING",// < 1 day
    SWING_TRADING = "SWING_TRADING",// 1-7 days
    POSITION_TRADING = "POSITION_TRADING"
}
export declare enum MarketRegime {
    TRENDING_BULLISH = "TRENDING_BULLISH",
    TRENDING_BEARISH = "TRENDING_BEARISH",
    SIDEWAYS = "SIDEWAYS",
    VOLATILE = "VOLATILE",
    BREAKOUT = "BREAKOUT"
}
export declare class IntelligentTradingSystem implements TradingStrategy {
    readonly name = "Intelligent_AI_System";
    parameters: Record<string, any>;
    private config?;
    private tradingPrinciples;
    private modelPredictions;
    private lastDecisionTime;
    private decisionCooldown;
    constructor();
    initialize(config: BacktestConfig): void;
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    /**
     * Get simulated AI model predictions (synchronous version for demo)
     */
    private getSimulatedAIModelPredictions;
    /**
     * Get enhanced model predictions from ML service
     */
    private getEnhancedModelPredictions;
    /**
     * Analyze current market regime using multiple indicators
     */
    private analyzeMarketRegime;
    /**
     * Perform Smart Money Concepts analysis based on trading guide
     */
    private performSMCAnalysis;
    /**
     * Generate intelligent trading decision based on all inputs
     */
    private generateIntelligentDecision;
    /**
     * Apply trading guide risk management principles
     */
    private applyRiskManagement;
    private initializeTradingPrinciples;
    private extractFeatures;
    private extractMLFeatures;
    private determineTimeHorizon;
    private mapToMarketRegime;
    private calculateTrendStrength;
    private calculateVolatility;
    private determineOptimalTradeType;
    private isMarketRegimeFavorable;
    private assessRiskLevel;
    private calculateExpectedDrawdown;
    private determineEntryStrategy;
    private determineExitStrategy;
    private calculateStopLoss;
    private calculateTakeProfit;
    private calculateATR;
    private identifyOrderBlocks;
    private identifyFairValueGaps;
    private identifyLiquidityLevels;
    private analyzeMarketStructure;
    getDescription(): string;
    private simulateTransformerPrediction;
    private simulateLSTMPrediction;
    private simulateSMCPrediction;
    protected getMinConfidence(): number;
    protected getModelConsensusThreshold(): number;
    protected getDecisionCooldown(): number;
}
export declare function createIntelligentTradingSystem(): IntelligentTradingSystem;

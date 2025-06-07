/**
 * Enhanced Trading Decision Engine
 * Core ML-driven trading logic with ensemble models, confidence scoring, and intelligent entry/exit decisions
 * Optimized for small capital + high leverage + pinpoint entry/exit precision
 */
export type TradingAction = 'buy' | 'sell' | 'hold' | 'close_long' | 'close_short';
export interface TradingDecision {
    action: TradingAction;
    confidence: number;
    symbol: string;
    timestamp: number;
    entryPrice?: number;
    exitPrice?: number;
    stopLoss: number;
    takeProfit: number;
    positionSize: number;
    leverage: number;
    modelVotes: {
        lstm: {
            action: TradingAction;
            confidence: number;
        };
        transformer: {
            action: TradingAction;
            confidence: number;
        };
        ensemble: {
            action: TradingAction;
            confidence: number;
        };
    };
    keyFeatures: {
        fibonacciSignal: number;
        biasAlignment: number;
        candleStrength: number;
        volumeConfirmation: number;
        marketTiming: number;
    };
    riskScore: number;
    maxDrawdown: number;
    winProbability: number;
    urgency: 'low' | 'medium' | 'high';
    timeToLive: number;
    reasoning: string[];
}
export interface DecisionEngineConfig {
    minConfidenceThreshold: number;
    highConfidenceThreshold: number;
    basePositionSize: number;
    maxPositionSize: number;
    confidenceMultiplier: number;
    baseLeverage: number;
    maxLeverage: number;
    stopLossBase: number;
    takeProfitBase: number;
    modelWeights: {
        lstm: number;
        transformer: number;
        ensemble: number;
    };
    featureWeights: {
        fibonacci: number;
        bias: number;
        candles: number;
        volume: number;
        timing: number;
    };
}
export declare class EnhancedTradingDecisionEngine {
    private dataIntegration;
    private mlEngine;
    private mtfAnalyzer;
    private mlService;
    private tradingBot;
    private config;
    private activeDecisions;
    private decisionHistory;
    constructor();
    /**
     * Initialize the enhanced trading decision engine
     */
    initialize(): Promise<void>;
    /**
     * Generate comprehensive trading decision for a symbol
     */
    generateTradingDecision(symbol: string): Promise<TradingDecision | null>;
    /**
     * Get the latest trading decision for a symbol
     */
    getLatestDecision(symbol: string): TradingDecision | null;
    /**
     * Get decision history
     */
    getDecisionHistory(limit?: number): TradingDecision[];
    /**
     * Update configuration
     */
    updateConfiguration(newConfig: Partial<DecisionEngineConfig>): void;
    /**
     * Get current configuration
     */
    getConfiguration(): DecisionEngineConfig;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
    /**
     * Get ML model predictions from all models
     */
    private getMLModelPredictions;
    /**
     * Calculate ensemble decision from model votes
     */
    private calculateEnsembleDecision;
    /**
     * Analyze key features for decision support
     */
    private analyzeKeyFeatures;
    /**
     * Calculate risk assessment
     */
    private calculateRiskAssessment;
    /**
     * Calculate position details based on confidence and risk
     */
    private calculatePositionDetails;
    /**
     * Calculate stop loss and take profit levels
     */
    private calculateStopLossAndTakeProfit;
    /**
     * Convert features to ML input format
     */
    private convertFeaturesToMLInput;
    /**
     * Convert ML prediction to trading action
     */
    private convertPredictionToAction;
    /**
     * Calculate Fibonacci signal strength
     */
    private calculateFibonacciSignal;
    /**
     * Calculate candle formation strength
     */
    private calculateCandleStrength;
    /**
     * Calculate market timing score
     */
    private calculateMarketTiming;
    /**
     * Determine trade urgency
     */
    private determineUrgency;
    /**
     * Calculate time to live for decision
     */
    private calculateTimeToLive;
    /**
     * Generate human-readable reasoning explanation
     */
    private generateReasoningExplanation;
}

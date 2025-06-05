export interface MarketData {
    symbol: string;
    price: number;
    volume: number;
    timestamp: number;
    high: number;
    low: number;
    open: number;
    close: number;
}
export interface Position {
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnl: number;
    entryTime: number;
    leverage: number;
}
export interface TechnicalAnalysis {
    atr: number;
    rsi: number;
    trend: 'bullish' | 'bearish' | 'sideways';
    support: number;
    resistance: number;
    volatility: number;
}
export interface MarketRegime {
    type: 'trending' | 'ranging' | 'volatile' | 'low_volatility';
    strength: number;
    duration: number;
}
export interface PositionHealth {
    score: number;
    trend_alignment: number;
    risk_level: 'low' | 'medium' | 'high';
    recommended_action: 'hold' | 'reduce' | 'close' | 'add';
    confidence: number;
}
export declare class IntelligentPositionManager {
    private marketData;
    private positionHistory;
    constructor();
    /**
     * Analyze position health across multiple timeframes
     */
    analyzePositionHealth(position: Position, marketData: MarketData[]): PositionHealth;
    /**
     * Calculate dynamic stop loss based on ATR and market conditions
     */
    calculateDynamicStopLoss(position: Position, marketData: MarketData[]): number;
    /**
     * Calculate dynamic take profit levels
     */
    calculateDynamicTakeProfit(position: Position, marketData: MarketData[]): number[];
    /**
     * Determine if position should be held, reduced, or closed
     */
    getPositionAction(position: Position, marketData: MarketData[]): {
        action: 'hold' | 'reduce' | 'close' | 'add';
        percentage?: number;
        reason: string;
    };
    private calculateATR;
    private calculateTechnicalIndicators;
    private calculateRSI;
    private analyzeTrend;
    private detectMarketRegime;
    private calculateTrendAlignment;
    private calculateAgeFactor;
    private calculatePnLMomentum;
    private calculateHealthScore;
    private assessRiskLevel;
    private getRecommendedAction;
    private calculateConfidence;
}

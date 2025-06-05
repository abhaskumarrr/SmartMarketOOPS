/**
 * Multi-Timeframe Analysis Engine
 * Analyzes market data across multiple timeframes for intelligent trading decisions
 */
import { DeltaExchangeUnified, Timeframe } from './DeltaExchangeUnified';
export interface TrendAnalysis {
    direction: 'bullish' | 'bearish' | 'sideways';
    strength: number;
    confidence: number;
    timeframe: Timeframe;
}
export interface CrossTimeframeAnalysis {
    symbol: string;
    timestamp: number;
    trends: {
        [timeframe: string]: TrendAnalysis;
    };
    overallTrend: {
        direction: 'bullish' | 'bearish' | 'sideways';
        strength: number;
        confidence: number;
        alignment: number;
    };
    signals: {
        entry: 'BUY' | 'SELL' | 'HOLD';
        confidence: number;
        reasoning: string[];
    };
    riskMetrics: {
        volatility: number;
        atrNormalized: number;
        rsiDivergence: boolean;
    };
}
export interface MarketRegime {
    type: 'trending_bullish' | 'trending_bearish' | 'sideways' | 'volatile' | 'breakout';
    confidence: number;
    duration: number;
    characteristics: {
        volatility: number;
        trendStrength: number;
        momentum: number;
    };
}
export declare class MultiTimeframeAnalysisEngine {
    private deltaService;
    private timeframes;
    constructor(deltaService: DeltaExchangeUnified);
    /**
     * Perform comprehensive multi-timeframe analysis
     */
    analyzeSymbol(symbol: string): Promise<CrossTimeframeAnalysis>;
    /**
     * Analyze trend for a specific timeframe
     */
    private analyzeTrend;
    /**
     * Calculate trend consistency
     */
    private calculateTrendConsistency;
    /**
     * Calculate overall trend from multiple timeframes
     */
    private calculateOverallTrend;
    /**
     * Calculate how aligned different timeframes are
     */
    private calculateTimeframeAlignment;
    /**
     * Generate trading signals based on analysis
     */
    private generateTradingSignals;
    /**
     * Calculate risk metrics
     */
    private calculateRiskMetrics;
    /**
     * Detect market regime
     */
    detectMarketRegime(analysis: CrossTimeframeAnalysis): MarketRegime;
}

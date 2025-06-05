/**
 * Enhanced Market Regime Detector
 * Integrates with existing SMC infrastructure and adds advanced regime detection
 */
import { DeltaExchangeUnified } from './DeltaExchangeUnified';
export declare enum EnhancedMarketRegime {
    TRENDING_BULLISH = "trending_bullish",
    TRENDING_BEARISH = "trending_bearish",
    SIDEWAYS = "sideways",
    VOLATILE = "volatile",
    BREAKOUT_BULLISH = "breakout_bullish",
    BREAKOUT_BEARISH = "breakout_bearish",
    CONSOLIDATION = "consolidation",
    ACCUMULATION = "accumulation",
    DISTRIBUTION = "distribution"
}
export interface VolatilityMetrics {
    atr_normalized: number;
    price_volatility: number;
    volume_volatility: number;
    garch_volatility?: number;
}
export interface TrendStrengthMetrics {
    adx: number;
    ma_slope: number;
    trend_consistency: number;
    momentum_strength: number;
}
export interface RegimeChangeDetection {
    change_detected: boolean;
    confidence: number;
    previous_regime: EnhancedMarketRegime;
    new_regime: EnhancedMarketRegime;
    change_timestamp: number;
    cusum_statistic?: number;
    bayesian_probability?: number;
}
export interface EnhancedRegimeAnalysis {
    current_regime: EnhancedMarketRegime;
    confidence: number;
    duration_minutes: number;
    volatility_metrics: VolatilityMetrics;
    trend_strength: TrendStrengthMetrics;
    regime_change: RegimeChangeDetection | null;
    trading_recommendations: {
        strategy_type: 'trend_following' | 'mean_reversion' | 'breakout' | 'scalping';
        risk_multiplier: number;
        optimal_timeframes: string[];
        position_sizing_factor: number;
    };
    smc_context: {
        order_block_strength: number;
        liquidity_levels: number;
        market_structure_quality: number;
    };
}
export declare class EnhancedMarketRegimeDetector {
    private deltaService;
    private mtfAnalyzer;
    private regimeHistory;
    private changeDetectionWindow;
    private cusumThreshold;
    constructor(deltaService: DeltaExchangeUnified);
    /**
     * Perform comprehensive regime detection with SMC integration
     */
    detectRegime(symbol: string): Promise<EnhancedRegimeAnalysis>;
    /**
     * Calculate comprehensive volatility metrics
     */
    private calculateVolatilityMetrics;
    /**
     * Calculate trend strength metrics including ADX-like calculations
     */
    private calculateTrendStrengthMetrics;
    /**
     * Calculate ADX-like trend strength indicator
     */
    private calculateADX;
    /**
     * Classify regime based on comprehensive metrics
     */
    private classifyRegime;
    /**
     * Calculate regime confidence score
     */
    private calculateRegimeConfidence;
    /**
     * Detect regime changes using CUSUM-like algorithm
     */
    private detectRegimeChange;
    /**
     * Estimate regime duration based on regime type and volatility
     */
    private estimateRegimeDuration;
    /**
     * Generate trading recommendations based on regime
     */
    private generateTradingRecommendations;
    /**
     * Get SMC context (placeholder for integration with existing SMC system)
     */
    private getSMCContext;
    /**
     * Update regime history for change detection
     */
    private updateRegimeHistory;
}

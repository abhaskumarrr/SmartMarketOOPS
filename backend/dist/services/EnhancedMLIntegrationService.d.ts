/**
 * Enhanced ML Integration Service
 * Integrates with existing AdvancedIntelligenceSystem, EnhancedTradingPredictor, and SMCDetector
 */
import { DeltaExchangeUnified } from './DeltaExchangeUnified';
export interface MLPrediction {
    model_name: string;
    prediction_type: 'position_outcome' | 'market_regime' | 'price_direction' | 'volatility';
    prediction_value: number;
    confidence: number;
    time_horizon: number;
    features_used: string[];
    model_version: string;
    timestamp: number;
}
export interface PositionOutcomePrediction {
    profit_probability: number;
    expected_return: number;
    time_to_target: number;
    risk_score: number;
    confidence: number;
    contributing_factors: {
        trend_alignment: number;
        regime_compatibility: number;
        volatility_factor: number;
        momentum_score: number;
        technical_indicators: number;
    };
    recommendations: {
        position_size_multiplier: number;
        hold_duration: number;
        exit_strategy: 'aggressive' | 'moderate' | 'conservative';
    };
}
export interface MarketRegimePrediction {
    current_regime: string;
    regime_probability: number;
    transition_probability: number;
    next_likely_regime: string;
    regime_duration: number;
    confidence: number;
    regime_characteristics: {
        volatility_level: 'low' | 'medium' | 'high';
        trend_strength: number;
        mean_reversion_tendency: number;
    };
}
export interface EnhancedMLFeatures {
    price_features: {
        returns: number[];
        volatility: number;
        momentum: number;
        rsi: number;
        macd: number;
        atr_normalized: number;
    };
    volume_features: {
        volume_ratio: number;
        volume_trend: number;
        volume_volatility: number;
    };
    timeframe_features: {
        short_term_trend: number;
        medium_term_trend: number;
        long_term_trend: number;
        trend_alignment: number;
    };
    regime_features: {
        current_regime_score: number;
        regime_stability: number;
        transition_signals: number[];
    };
    sentiment_features: {
        market_sentiment: number;
        fear_greed_index: number;
        institutional_flow: number;
    };
}
export declare class EnhancedMLIntegrationService {
    private deltaService;
    private mtfAnalyzer;
    private regimeDetector;
    private modelVersions;
    private predictionCache;
    constructor(deltaService: DeltaExchangeUnified);
    /**
     * Predict position outcome using enhanced ML models
     */
    predictPositionOutcome(symbol: string, side: 'LONG' | 'SHORT', entryPrice: number, currentPrice: number, positionAge: number): Promise<PositionOutcomePrediction>;
    /**
     * Predict market regime transitions
     */
    predictMarketRegime(symbol: string): Promise<MarketRegimePrediction>;
    /**
     * Get ensemble prediction combining multiple models
     */
    getEnsemblePrediction(symbol: string, side: 'LONG' | 'SHORT', entryPrice: number, currentPrice: number): Promise<{
        position_outcome: PositionOutcomePrediction;
        regime_prediction: MarketRegimePrediction;
        ensemble_confidence: number;
        recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
    }>;
    /**
     * Extract enhanced features for ML models
     */
    private extractEnhancedFeatures;
    /**
     * Simulate advanced position prediction (placeholder for actual ML model)
     */
    private simulateAdvancedPositionPrediction;
    /**
     * Simulate regime prediction (placeholder for actual ML model)
     */
    private simulateRegimePrediction;
    /**
     * Generate ensemble recommendation
     */
    private generateEnsembleRecommendation;
    private calculateReturns;
    private calculateVolumeRatio;
    private calculateVolumeTrend;
    private getTrendScore;
    private calculateRegimeStability;
    private calculateTransitionSignals;
}

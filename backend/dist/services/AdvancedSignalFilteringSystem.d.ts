/**
 * Advanced Signal Filtering System
 * Implements multi-layer filtering to achieve 85%+ win rate alignment with ML accuracy
 */
import { DeltaExchangeUnified } from './DeltaExchangeUnified';
export interface SignalFilterConfig {
    min_ml_confidence: number;
    min_ensemble_confidence: number;
    min_timeframe_alignment: number;
    required_timeframe_count: number;
    allowed_regimes: string[];
    min_regime_confidence: number;
    min_rsi_range: [number, number];
    min_volume_ratio: number;
    max_volatility: number;
    min_signal_score: number;
    max_correlation_exposure: number;
    max_drawdown_threshold: number;
}
export interface FilteredSignal {
    symbol: string;
    side: 'LONG' | 'SHORT';
    signal_score: number;
    ml_confidence: number;
    ensemble_confidence: number;
    timeframe_alignment: number;
    regime_compatibility: number;
    technical_confirmation: number;
    risk_assessment: number;
    entry_price: number;
    stop_loss: number;
    take_profit_levels: number[];
    position_size_multiplier: number;
    reasoning: string[];
    filter_results: {
        ml_filter: boolean;
        timeframe_filter: boolean;
        regime_filter: boolean;
        technical_filter: boolean;
        risk_filter: boolean;
    };
    timestamp: number;
}
export declare class AdvancedSignalFilteringSystem {
    private deltaService;
    private mtfAnalyzer;
    private regimeDetector;
    private mlService;
    private config;
    private recentSignals;
    private activePositions;
    constructor(deltaService: DeltaExchangeUnified, config?: Partial<SignalFilterConfig>);
    /**
     * Generate and filter high-quality trading signals
     */
    generateFilteredSignal(symbol: string): Promise<FilteredSignal | null>;
    /**
     * Apply comprehensive multi-layer filtering
     */
    private applyAdvancedFiltering;
    /**
     * Construct high-quality filtered signal
     */
    private constructFilteredSignal;
    /**
     * Calculate comprehensive signal score (0-100)
     */
    private calculateSignalScore;
    /**
     * Calculate position size multiplier based on confidence
     */
    private calculatePositionSizeMultiplier;
    private countAlignedTimeframes;
    private calculateTechnicalScore;
    private calculateRiskScore;
    private determineSide;
    private calculateOptimalStopLoss;
    private calculateOptimalTakeProfits;
    private generateSignalReasoning;
    private getCurrentPrice;
    private calculateVolumeRatio;
    private calculateVolatility;
    private calculateCorrelationExposure;
    private calculateCurrentDrawdown;
    private storeSignal;
}

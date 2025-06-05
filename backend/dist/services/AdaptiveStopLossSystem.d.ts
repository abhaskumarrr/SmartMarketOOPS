/**
 * Adaptive Stop Loss System
 * Dynamic stop loss calculations using ATR, volatility, and market regime
 */
import { DeltaExchangeUnified } from './DeltaExchangeUnified';
export interface StopLossConfig {
    base_atr_multiplier: number;
    volatility_adjustment: boolean;
    regime_adjustment: boolean;
    trend_adjustment: boolean;
    max_stop_distance: number;
    min_stop_distance: number;
    trailing_enabled: boolean;
    trailing_step: number;
}
export interface AdaptiveStopLoss {
    stop_price: number;
    distance_percent: number;
    atr_multiplier: number;
    confidence: number;
    reasoning: string[];
    adjustments: {
        base_atr: number;
        volatility_factor: number;
        regime_factor: number;
        trend_factor: number;
        final_multiplier: number;
    };
    trailing_info?: {
        enabled: boolean;
        trigger_price: number;
        step_size: number;
        last_update: number;
    };
}
export interface Position {
    symbol: string;
    side: 'LONG' | 'SHORT';
    entry_price: number;
    current_price: number;
    size: number;
    entry_time: number;
}
export declare class AdaptiveStopLossSystem {
    private deltaService;
    private mtfAnalyzer;
    private regimeDetector;
    private defaultConfig;
    constructor(deltaService: DeltaExchangeUnified);
    /**
     * Calculate adaptive stop loss for a position
     */
    calculateStopLoss(position: Position, config?: Partial<StopLossConfig>): Promise<AdaptiveStopLoss>;
    /**
     * Update trailing stop loss
     */
    updateTrailingStop(position: Position, currentStopLoss: AdaptiveStopLoss): AdaptiveStopLoss | null;
    /**
     * Calculate base ATR from market data
     */
    private calculateBaseATR;
    /**
     * Calculate volatility adjustment factor
     */
    private calculateVolatilityFactor;
    /**
     * Calculate regime adjustment factor
     */
    private calculateRegimeFactor;
    /**
     * Calculate trend alignment factor
     */
    private calculateTrendFactor;
    /**
     * Calculate stop loss confidence
     */
    private calculateStopLossConfidence;
    /**
     * Generate reasoning for stop loss calculation
     */
    private generateStopLossReasoning;
    /**
     * Setup trailing stop configuration
     */
    private setupTrailingStop;
}

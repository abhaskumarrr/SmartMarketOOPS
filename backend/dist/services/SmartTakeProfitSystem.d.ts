/**
 * Smart Take Profit System
 * Dynamic take profit levels based on market conditions, support/resistance, and trend strength
 */
import { DeltaExchangeUnified } from './DeltaExchangeUnified';
export interface TakeProfitLevel {
    level: number;
    target_price: number;
    percentage: number;
    distance_percent: number;
    confidence: number;
    reasoning: string;
    level_type: 'support_resistance' | 'fibonacci' | 'atr_based' | 'trend_target' | 'regime_based';
}
export interface SmartTakeProfitConfig {
    max_levels: number;
    partial_percentages: number[];
    base_targets: number[];
    use_support_resistance: boolean;
    use_fibonacci: boolean;
    use_trend_targets: boolean;
    use_regime_adjustment: boolean;
    min_target_distance: number;
    max_target_distance: number;
}
export interface SmartTakeProfit {
    levels: TakeProfitLevel[];
    total_confidence: number;
    strategy_type: 'aggressive' | 'moderate' | 'conservative';
    market_context: {
        regime: string;
        trend_strength: number;
        volatility: number;
        support_resistance_quality: number;
    };
    execution_plan: {
        immediate_targets: TakeProfitLevel[];
        conditional_targets: TakeProfitLevel[];
        trailing_activation: number;
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
export declare class SmartTakeProfitSystem {
    private deltaService;
    private mtfAnalyzer;
    private regimeDetector;
    private defaultConfig;
    constructor(deltaService: DeltaExchangeUnified);
    /**
     * Calculate smart take profit levels for a position
     */
    calculateTakeProfit(position: Position, config?: Partial<SmartTakeProfitConfig>): Promise<SmartTakeProfit>;
    /**
     * Calculate support/resistance levels
     */
    private calculateSupportResistanceLevels;
    /**
     * Calculate Fibonacci retracement levels
     */
    private calculateFibonacciLevels;
    /**
     * Calculate trend-based targets
     */
    private calculateTrendTargets;
    /**
     * Calculate regime-adjusted targets
     */
    private calculateRegimeTargets;
    /**
     * Optimize take profit levels
     */
    private optimizeTakeProfitLevels;
    /**
     * Determine strategy type based on market conditions
     */
    private determineStrategyType;
    /**
     * Create execution plan
     */
    private createExecutionPlan;
    /**
     * Calculate total confidence
     */
    private calculateTotalConfidence;
    private findResistanceLevels;
    private findSupportLevels;
    private assessSupportResistanceQuality;
}

/**
 * Enhanced Trend Following Strategy
 * Professional-grade strategy with trend analysis, market regime detection, and dynamic risk management
 */
import { TradingStrategy, TradingSignal, EnhancedMarketData, BacktestConfig } from '../types/marketData';
export interface EnhancedTrendParams {
    trendPeriod: number;
    trendThreshold: number;
    fastEMA: number;
    slowEMA: number;
    rsiPeriod: number;
    rsiOverbought: number;
    rsiOversold: number;
    volatilityPeriod: number;
    volumeConfirmation: number;
    baseStopLoss: number;
    dynamicStopLoss: boolean;
    takeProfitMultiplier: number;
    maxPositionSize: number;
    minTrendStrength: number;
    minConfidence: number;
    antiWhipsawPeriod: number;
}
export declare class EnhancedTrendStrategy implements TradingStrategy {
    readonly name = "Enhanced_Trend";
    parameters: EnhancedTrendParams;
    private config?;
    private lastSignalIndex;
    constructor(parameters?: Partial<EnhancedTrendParams>);
    initialize(config: BacktestConfig): void;
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    private hasRequiredIndicators;
    private analyzeMarketRegime;
    private analyzeTrend;
    private generateEntrySignal;
    private enhanceSignal;
    private calculateDynamicPositionSize;
    private calculateDynamicStopLoss;
    private calculateTrendConsistency;
    private calculateLinearRegression;
    private calculateATR;
    getDescription(): string;
    getParameters(): EnhancedTrendParams;
    updateParameters(newParams: Partial<EnhancedTrendParams>): void;
}
export declare function createEnhancedTrendStrategy(params?: Partial<EnhancedTrendParams>): EnhancedTrendStrategy;

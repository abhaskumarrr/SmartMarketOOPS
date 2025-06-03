/**
 * Moving Average Crossover Trading Strategy
 * Generates buy/sell signals based on moving average crossovers
 */
import { TradingStrategy, TradingSignal, EnhancedMarketData, BacktestConfig } from '../types/marketData';
export interface MACrossoverParams {
    fastPeriod: number;
    slowPeriod: number;
    rsiPeriod: number;
    rsiOverbought: number;
    rsiOversold: number;
    volumeThreshold: number;
    stopLossPercent: number;
    takeProfitPercent: number;
    minConfidence: number;
}
export declare class MovingAverageCrossoverStrategy implements TradingStrategy {
    readonly name = "MA_Crossover";
    parameters: MACrossoverParams;
    private config?;
    constructor(parameters?: Partial<MACrossoverParams>);
    initialize(config: BacktestConfig): void;
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    private hasRequiredIndicators;
    private detectCrossover;
    private applyFilters;
    private enhanceSignal;
    private calculatePositionSize;
    getDescription(): string;
    getParameters(): MACrossoverParams;
    updateParameters(newParams: Partial<MACrossoverParams>): void;
}
export declare function createMACrossoverStrategy(params?: Partial<MACrossoverParams>): MovingAverageCrossoverStrategy;

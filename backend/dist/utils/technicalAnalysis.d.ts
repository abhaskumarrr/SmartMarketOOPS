/**
 * Technical Analysis Utilities
 * Implements common technical indicators for trading strategies
 */
import { TechnicalAnalysis, TechnicalIndicators } from '../types/marketData';
export declare class TechnicalAnalysisService implements TechnicalAnalysis {
    /**
     * Calculate Simple Moving Average
     */
    calculateSMA(prices: number[], period: number): number[];
    /**
     * Calculate Exponential Moving Average
     */
    calculateEMA(prices: number[], period: number): number[];
    /**
     * Calculate Relative Strength Index
     */
    calculateRSI(prices: number[], period?: number): number[];
    /**
     * Calculate MACD (Moving Average Convergence Divergence)
     */
    calculateMACD(prices: number[], fastPeriod?: number, slowPeriod?: number, signalPeriod?: number): {
        macd: number[];
        signal: number[];
        histogram: number[];
    };
    /**
     * Calculate Bollinger Bands
     */
    calculateBollingerBands(prices: number[], period?: number, stdDev?: number): {
        upper: number[];
        middle: number[];
        lower: number[];
    };
    /**
     * Calculate all indicators for a price series
     */
    calculateAllIndicators(prices: number[], volumes?: number[]): TechnicalIndicators[];
    /**
     * Calculate price momentum
     */
    calculateMomentum(prices: number[], period?: number): number[];
    /**
     * Calculate Average True Range (ATR)
     */
    calculateATR(highs: number[], lows: number[], closes: number[], period?: number): number[];
    /**
     * Calculate Stochastic Oscillator
     */
    calculateStochastic(highs: number[], lows: number[], closes: number[], kPeriod?: number, dPeriod?: number): {
        k: number[];
        d: number[];
    };
    /**
     * Detect crossover signals
     */
    detectCrossover(series1: number[], series2: number[], lookback?: number): ('bullish' | 'bearish' | 'none')[];
    /**
     * Calculate support and resistance levels
     */
    calculateSupportResistance(highs: number[], lows: number[], period?: number): {
        support: number[];
        resistance: number[];
    };
}
export declare const technicalAnalysis: TechnicalAnalysisService;

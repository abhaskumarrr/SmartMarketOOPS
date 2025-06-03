#!/usr/bin/env node
/**
 * Optimized Trading Strategy - Enhanced Signal Generation
 * Based on performance analysis of current system
 */
import { TradingSignal, BacktestConfig } from '../types/marketData';
interface OptimizedTradingSignal extends TradingSignal {
    marketRegime?: string;
    volatility?: number;
    volumeStrength?: number;
}
declare class OptimizedTradingStrategy {
    /**
     * Enhanced signal generation with market regime awareness
     */
    generateOptimizedSignal(candle: any, config: BacktestConfig, index: number, marketData: any[]): OptimizedTradingSignal | null;
    /**
     * Detect market regime (trending vs ranging)
     */
    private detectMarketRegime;
    /**
     * Calculate comprehensive volatility metrics
     */
    private calculateVolatility;
    /**
     * Analyze volume profile and strength
     */
    private analyzeVolumeProfile;
    /**
     * Get comprehensive technical signals
     */
    private getTechnicalSignals;
    /**
     * Calculate momentum indicators
     */
    private calculateMomentum;
    /**
     * Find dynamic support and resistance levels
     */
    private findSupportResistance;
    /**
     * Get higher timeframe signal confirmation
     */
    private getHigherTimeframeSignal;
    private calculateTrendStrength;
    private calculateRangeVolatility;
    private calculateRSI;
    private calculateMACD;
    private calculateEMA;
    private calculateBollingerPosition;
}
export { OptimizedTradingStrategy };

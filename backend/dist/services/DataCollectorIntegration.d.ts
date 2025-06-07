/**
 * Data Collector Integration Service
 * Bridges Multi-Timeframe Data Collector with ML Trading Decision Engine
 * Provides seamless data flow for ML feature engineering
 */
import { MLTradingDecisionEngine } from './MLTradingDecisionEngine';
export interface FeatureExtractionConfig {
    fibonacciLevels: number[];
    confluenceWeights: Record<string, number>;
    candleFormationPeriods: number[];
    volumeAnalysisPeriods: number[];
}
export interface ExtractedFeatures {
    symbol: string;
    timestamp: number;
    fibonacciProximity: number[];
    nearestFibLevel: number;
    fibStrength: number;
    bias4h: number;
    bias1h: number;
    bias15m: number;
    bias5m: number;
    overallBias: number;
    biasAlignment: number;
    bodyPercentage: number;
    wickPercentage: number;
    buyingPressure: number;
    sellingPressure: number;
    candleType: number;
    momentum: number;
    volatility: number;
    volume: number;
    volumeRatio: number;
    timeOfDay: number;
    marketSession: number;
    pricePosition: number;
    dataQuality: number;
    synchronized: boolean;
}
export declare class DataCollectorIntegration {
    private dataCollector;
    private mlEngine;
    private config;
    constructor();
    /**
     * Initialize the integration service
     */
    initialize(): Promise<void>;
    /**
     * Set ML Trading Decision Engine reference
     */
    setMLEngine(mlEngine: MLTradingDecisionEngine): void;
    /**
     * Start data collection for trading symbols
     */
    startDataCollection(symbols: string[]): Promise<void>;
    /**
     * Stop data collection
     */
    stopDataCollection(): Promise<void>;
    /**
     * Extract ML features from multi-timeframe data
     */
    extractMLFeatures(symbol: string): Promise<ExtractedFeatures | null>;
    /**
     * Get real-time trading features for ML decision making
     */
    getRealTimeTradingFeatures(symbol: string): Promise<ExtractedFeatures | null>;
    /**
     * Get data collection statistics
     */
    getIntegrationStatistics(): Promise<Record<string, any>>;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
    /**
     * Perform comprehensive feature extraction
     */
    private performFeatureExtraction;
    /**
     * Extract Fibonacci retracement features
     */
    private extractFibonacciFeatures;
    /**
     * Extract multi-timeframe bias features
     */
    private extractBiasFeatures;
    /**
     * Extract candle formation features
     */
    private extractCandleFeatures;
    /**
     * Extract market context features
     */
    private extractMarketContextFeatures;
    /**
     * Get time of day feature (0-1)
     */
    private getTimeOfDayFeature;
    /**
     * Get market session feature
     */
    private getMarketSessionFeature;
}

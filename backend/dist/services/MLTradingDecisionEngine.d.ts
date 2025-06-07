/**
 * ML Trading Decision Engine
 *
 * Integrates all our trading analysis (Fibonacci, SMC, confluence, candle formation,
 * momentum trains) as features for ML models to make actual trading decisions.
 *
 * This replaces hard-coded rules with ML-driven intelligence that learns optimal
 * combinations of our comprehensive trading analysis.
 */
import { DeltaTradingBot } from '../scripts/delta-trading-bot';
import { MultiTimeframeAnalysisEngine } from './MultiTimeframeAnalysisEngine';
import { EnhancedMLIntegrationService } from './EnhancedMLIntegrationService';
export interface TradingFeatures {
    fibonacciProximity: {
        level236: number;
        level382: number;
        level500: number;
        level618: number;
        level786: number;
        nearestLevel: number;
        levelStrength: number;
    };
    timeframeBias: {
        bias4H: number;
        bias1H: number;
        bias15M: number;
        bias5M: number;
        alignment: number;
        strength: number;
    };
    candleFormation: {
        bodyPercent: number;
        upperWickPercent: number;
        lowerWickPercent: number;
        buyingPressure: number;
        sellingPressure: number;
        candleType: number;
        momentum: number;
    };
    smcAnalysis: {
        orderBlockStrength: number;
        fvgPresence: number;
        liquidityLevel: number;
        structureBreak: number;
        institutionalFlow: number;
    };
    confluence: {
        overallScore: number;
        fibWeight: number;
        biasWeight: number;
        smcWeight: number;
        momentumTrain: number;
        entryTiming: number;
    };
    marketContext: {
        volatility: number;
        volume: number;
        timeOfDay: number;
        marketRegime: number;
        sessionType: number;
    };
}
export interface MLTradingDecision {
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    positionSize: number;
    stopLoss: number;
    takeProfit: number;
    timeHorizon: 'SCALP' | 'DAY' | 'SWING';
    reasoning: {
        primaryFactors: string[];
        riskAssessment: string;
        modelContributions: {
            lstm: number;
            transformer: number;
            ensemble: number;
        };
    };
}
export declare class MLTradingDecisionEngine {
    private mtfAnalyzer;
    private mlService;
    private tradingBot;
    private isInitialized;
    private featureCache;
    private cacheTimeout;
    private modelWeights;
    private thresholds;
    constructor(mtfAnalyzer: MultiTimeframeAnalysisEngine, mlService: EnhancedMLIntegrationService, tradingBot: DeltaTradingBot);
    /**
     * Initialize the ML Trading Decision Engine
     */
    initialize(): Promise<void>;
    /**
     * Generate ML-driven trading decision
     */
    generateTradingDecision(symbol: string, currentPrice: number): Promise<MLTradingDecision>;
    /**
     * Execute trade based on ML decision
     */
    executeTrade(symbol: string, decision: MLTradingDecision, currentPrice: number): Promise<boolean>;
    /**
     * Extract comprehensive trading features from all our analysis
     */
    private extractTradingFeatures;
    /**
     * Validate that all ML models are loaded and ready
     */
    private validateMLModels;
    /**
     * Load optimized parameters from previous performance data
     */
    private loadOptimizedParameters;
    /**
     * Get predictions from all ML models
     */
    private getMLPredictions;
    /**
     * Combine ML predictions using weighted ensemble
     */
    private combineMLPredictions;
    /**
     * Apply risk management and position sizing to ML decision
     */
    private applyRiskManagement;
    /**
     * Cache decision for performance tracking
     */
    private cacheDecision;
    /**
     * Track trade performance for ML model optimization
     */
    private trackTradePerformance;
    /**
     * Extract Fibonacci analysis features
     */
    private extractFibonacciFeatures;
    /**
     * Extract multi-timeframe bias features
     */
    private extractTimeframeBiasFeatures;
    /**
     * Extract candle formation features
     */
    private extractCandleFormationFeatures;
    /**
     * Extract Smart Money Concepts features
     */
    private extractSMCFeatures;
    /**
     * Extract confluence features
     */
    private extractConfluenceFeatures;
    /**
     * Extract market context features
     */
    private extractMarketContextFeatures;
    private convertFeaturesToMLInput;
    private createDummyFeatures;
    private generateRiskAssessment;
    private encodeCandleType;
    private encodeSessionType;
    private getDefaultFibFeatures;
    private getDefaultBiasFeatures;
    private getDefaultCandleFeatures;
    private getDefaultSMCFeatures;
    private getDefaultConfluenceFeatures;
    private getDefaultMarketContextFeatures;
}

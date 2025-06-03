/**
 * Multi-Timeframe Multi-Asset Trading Strategy
 * Advanced strategy combining multi-timeframe analysis with multi-asset portfolio optimization
 */
import { TradingStrategy, TradingSignal, EnhancedMarketData, BacktestConfig } from '../types/marketData';
import { CryptoPair } from './multiAssetDataProvider';
import { Timeframe } from './multiTimeframeDataProvider';
export interface TimeframeSignal {
    timeframe: Timeframe;
    signal: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    strength: number;
    weight: number;
}
export interface AssetTimeframeAnalysis {
    asset: CryptoPair;
    timeframeSignals: TimeframeSignal[];
    consensusSignal: 'BUY' | 'SELL' | 'HOLD';
    consensusConfidence: number;
    hierarchicalScore: number;
    volatilityAdjustment: number;
}
export interface MultiTimeframeMultiAssetSignal extends TradingSignal {
    assetAnalysis: AssetTimeframeAnalysis[];
    timeframeBreakdown: {
        [timeframe in Timeframe]?: {
            bullishAssets: CryptoPair[];
            bearishAssets: CryptoPair[];
            neutralAssets: CryptoPair[];
            overallSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
        };
    };
    portfolioRecommendation: {
        allocation: {
            [asset in CryptoPair]?: number;
        };
        rebalanceRequired: boolean;
        riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
        expectedReturn: number;
        expectedVolatility: number;
    };
    hierarchicalDecision: {
        primaryTimeframe: Timeframe;
        confirmingTimeframes: Timeframe[];
        conflictingTimeframes: Timeframe[];
        decisionRationale: string;
    };
}
export declare class MultiTimeframeMultiAssetStrategy implements TradingStrategy {
    readonly name = "Multi_Timeframe_Multi_Asset_Strategy";
    parameters: Record<string, any>;
    private config?;
    private dataProvider;
    private trainedModels;
    private lastDecisionTime;
    private timeframeHierarchy;
    private assetConfigs;
    constructor();
    /**
     * Initialize the strategy
     */
    initialize(config: BacktestConfig): Promise<void>;
    /**
     * Load trained models for multi-timeframe multi-asset analysis
     */
    private loadTrainedModels;
    /**
     * Generate multi-timeframe multi-asset trading signal
     */
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    /**
     * Convert single-timeframe data to multi-timeframe format (simplified)
     */
    private convertToMultiTimeframeData;
    /**
     * Analyze all assets across all timeframes
     */
    private analyzeAllAssetsAllTimeframes;
    /**
     * Analyze a single asset across all its configured timeframes
     */
    private analyzeAssetAllTimeframes;
    /**
     * Analyze a specific asset-timeframe combination
     */
    private analyzeAssetTimeframe;
    /**
     * Get model predictions for asset-timeframe combination
     */
    private getModelPredictions;
    /**
     * Extract features for model prediction - ENHANCED FOR REAL SIGNALS
     */
    private extractFeatures;
    /**
     * Run model inference
     */
    private runModelInference;
    /**
     * Convert prediction to signal - LOWERED THRESHOLDS FOR MORE TRADES
     */
    private predictionToSignal;
    /**
     * Get asset-specific trading bias for more realistic signals
     */
    private getAssetTradingBias;
    /**
     * Calculate timeframe consensus
     */
    private calculateTimeframeConsensus;
    /**
     * Calculate hierarchical score
     */
    private calculateHierarchicalScore;
    /**
     * Calculate volatility adjustment
     */
    private calculateVolatilityAdjustment;
    /**
     * Generate timeframe breakdown
     */
    private generateTimeframeBreakdown;
    /**
     * Calculate portfolio recommendation
     */
    private calculatePortfolioRecommendation;
    /**
     * Make hierarchical decision
     */
    private makeHierarchicalDecision;
    /**
     * Generate final signal
     */
    private generateFinalSignal;
    private validateModel;
    private createDefaultModel;
    /**
     * Get strategy description
     */
    getDescription(): string;
}
export declare function createMultiTimeframeMultiAssetStrategy(): MultiTimeframeMultiAssetStrategy;

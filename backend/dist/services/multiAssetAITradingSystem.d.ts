/**
 * Multi-Asset AI Trading System
 * Advanced trading system supporting multiple cryptocurrency pairs with cross-asset analysis
 */
import { TradingStrategy, TradingSignal, EnhancedMarketData, BacktestConfig } from '../types/marketData';
import { CryptoPair } from './multiAssetDataProvider';
export interface MultiAssetPrediction {
    asset: CryptoPair;
    prediction: number;
    confidence: number;
    signalType: 'BUY' | 'SELL' | 'HOLD';
    modelConsensus: number;
    crossAssetScore: number;
    relativeStrength: number;
}
export interface PortfolioAllocation {
    btc: number;
    eth: number;
    sol: number;
    cash: number;
}
export interface MultiAssetSignal extends TradingSignal {
    targetAsset: CryptoPair;
    portfolioAllocation: PortfolioAllocation;
    crossAssetAnalysis: {
        correlations: {
            btc_eth: number;
            btc_sol: number;
            eth_sol: number;
        };
        relativeStrengths: {
            btc: number;
            eth: number;
            sol: number;
        };
        marketRegime: 'RISK_ON' | 'RISK_OFF' | 'NEUTRAL';
    };
}
export declare class MultiAssetAITradingSystem implements TradingStrategy {
    readonly name = "Multi_Asset_AI_Trading_System";
    parameters: Record<string, any>;
    private config?;
    private dataProvider;
    private trainedModels;
    private lastDecisionTime;
    private supportedAssets;
    private currentPortfolio;
    constructor();
    /**
     * Initialize the multi-asset trading system
     */
    initialize(config: BacktestConfig): Promise<void>;
    /**
     * Load trained models for multi-asset trading
     */
    private loadMultiAssetModels;
    /**
     * Initialize portfolio allocation
     */
    private initializePortfolio;
    /**
     * Generate multi-asset trading signal
     */
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    /**
     * Analyze all supported assets
     */
    private analyzeAllAssets;
    /**
     * Analyze a specific asset
     */
    private analyzeAsset;
    /**
     * Get model predictions for a specific asset
     */
    private getAssetModelPredictions;
    /**
     * Extract features for a specific asset
     */
    private extractAssetFeatures;
    /**
     * Run model inference
     */
    private runModelInference;
    /**
     * Convert prediction to signal
     */
    private predictionToSignal;
    /**
     * Calculate cross-asset score
     */
    private calculateCrossAssetScore;
    /**
     * Calculate asset relative strength
     */
    private calculateAssetRelativeStrength;
    /**
     * Calculate model consensus
     */
    private calculateModelConsensus;
    /**
     * Perform cross-asset analysis
     */
    private performCrossAssetAnalysis;
    /**
     * Calculate optimal portfolio allocation
     */
    private calculateOptimalAllocation;
    /**
     * Adjust allocation for correlation risk
     */
    private adjustForCorrelationRisk;
    /**
     * Generate portfolio rebalancing signal
     */
    private generatePortfolioSignal;
    /**
     * Check if portfolio rebalancing is needed
     */
    private isRebalanceNeeded;
    /**
     * Update portfolio allocation
     */
    private updatePortfolioAllocation;
    private validateModel;
    private createDefaultModel;
    /**
     * Get strategy description
     */
    getDescription(): string;
}
export declare function createMultiAssetAITradingSystem(): MultiAssetAITradingSystem;

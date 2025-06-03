/**
 * Retrained AI Trading System
 * Uses newly trained AI models with 6 months of real market data
 */
import { TradingStrategy, TradingSignal, EnhancedMarketData, BacktestConfig } from '../types/marketData';
export declare enum TradeType {
    SCALPING = "SCALPING",
    DAY_TRADING = "DAY_TRADING",
    SWING_TRADING = "SWING_TRADING",
    POSITION_TRADING = "POSITION_TRADING"
}
export declare enum MarketRegime {
    TRENDING_BULLISH = "TRENDING_BULLISH",
    TRENDING_BEARISH = "TRENDING_BEARISH",
    SIDEWAYS = "SIDEWAYS",
    VOLATILE = "VOLATILE",
    BREAKOUT = "BREAKOUT"
}
export declare class RetrainedAITradingSystem implements TradingStrategy {
    readonly name = "Retrained_AI_Trading_System";
    parameters: Record<string, any>;
    private config?;
    private trainedModels;
    private lastDecisionTime;
    private modelLoadTime;
    constructor();
    /**
     * Initialize the trading system and load retrained models
     */
    initialize(config: BacktestConfig): Promise<void>;
    /**
     * Load retrained models from disk
     */
    private loadRetrainedModels;
    /**
     * Validate loaded model
     */
    private validateModel;
    /**
     * Create default model if retrained model not available
     */
    private createDefaultModel;
    /**
     * Update system parameters based on loaded models
     */
    private updateParametersFromModels;
    /**
     * Generate trading signal using retrained models
     */
    generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
    /**
     * Get predictions from retrained models
     */
    private getRetrainedModelPredictions;
    /**
     * Run prediction using a specific retrained model
     */
    private runModelPrediction;
    /**
     * Extract features for model prediction
     */
    private extractFeatures;
    /**
     * Run model inference
     */
    private runModelInference;
    /**
     * Convert prediction to signal
     */
    private predictionToSignal;
    /**
     * Analyze consensus among model predictions
     */
    private analyzeModelConsensus;
    /**
     * Generate final trading signal
     */
    private generateFinalSignal;
    private getModelTypeFromName;
    private getModelWeight;
    /**
     * Get strategy description
     */
    getDescription(): string;
}
export declare function createRetrainedAITradingSystem(): RetrainedAITradingSystem;

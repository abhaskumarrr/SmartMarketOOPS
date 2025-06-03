/**
 * Model Training Data Processor
 * Processes real market data for AI model training with proper feature engineering
 */
import { MarketDataPoint } from '../types/marketData';
export interface TrainingFeatures {
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    rsi_14: number;
    rsi_7: number;
    rsi_21: number;
    sma_10: number;
    sma_20: number;
    sma_50: number;
    ema_12: number;
    ema_26: number;
    ema_50: number;
    macd: number;
    macd_signal: number;
    macd_histogram: number;
    bb_upper: number;
    bb_middle: number;
    bb_lower: number;
    bb_width: number;
    bb_position: number;
    volume_sma_20: number;
    volume_ratio: number;
    body_size: number;
    upper_wick: number;
    lower_wick: number;
    atr_14: number;
    volatility_10: number;
    momentum_5: number;
    momentum_10: number;
    higher_tf_trend: number;
    lower_tf_momentum: number;
    support_level: number;
    resistance_level: number;
    trend_strength: number;
    hour_of_day: number;
    day_of_week: number;
    future_return_1h: number;
    future_return_4h: number;
    future_return_24h: number;
    signal_1h: number;
    signal_4h: number;
    signal_24h: number;
}
export interface TrainingDataset {
    features: TrainingFeatures[];
    metadata: {
        symbol: string;
        startDate: Date;
        endDate: Date;
        totalSamples: number;
        featureCount: number;
        trainSplit: number;
        validationSplit: number;
        testSplit: number;
    };
}
export declare class ModelTrainingDataProcessor {
    /**
     * Process raw market data into training features
     */
    processTrainingData(data: MarketDataPoint[], symbol: string, trainSplit?: number, validationSplit?: number, testSplit?: number): TrainingDataset;
    /**
     * Calculate comprehensive technical indicators
     */
    private calculateTechnicalIndicators;
    /**
     * Generate comprehensive features for model training
     */
    private generateFeatures;
    /**
     * Calculate target variables for supervised learning
     */
    private calculateTargetVariables;
    private calculateVolatility;
    private calculateMomentum;
    private calculateTrendDirection;
    private findSupportLevel;
    private findResistanceLevel;
    private calculateTrendStrength;
    private getFuturePrice;
    private returnToSignal;
    private cleanData;
    /**
     * Split dataset into train/validation/test sets
     */
    splitDataset(dataset: TrainingDataset): {
        train: TrainingFeatures[];
        validation: TrainingFeatures[];
        test: TrainingFeatures[];
    };
}
export declare function createModelTrainingDataProcessor(): ModelTrainingDataProcessor;

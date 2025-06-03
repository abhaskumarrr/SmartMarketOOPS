/**
 * AI Model Trainer
 * Trains Enhanced Transformer, Decisive LSTM, and Active SMC models using real market data
 */
import { TrainingFeatures } from './modelTrainingDataProcessor';
export interface ModelWeights {
    [key: string]: number;
}
export interface ModelParameters {
    weights: ModelWeights;
    biases: number[];
    learningRate: number;
    epochs: number;
    batchSize: number;
    regularization: number;
    dropout: number;
}
export interface TrainingMetrics {
    epoch: number;
    trainLoss: number;
    validationLoss: number;
    trainAccuracy: number;
    validationAccuracy: number;
    learningRate: number;
}
export interface TrainedModel {
    modelName: string;
    parameters: ModelParameters;
    trainingHistory: TrainingMetrics[];
    finalMetrics: {
        trainAccuracy: number;
        validationAccuracy: number;
        testAccuracy: number;
        precision: number;
        recall: number;
        f1Score: number;
    };
    featureImportance: {
        [feature: string]: number;
    };
    trainingTime: number;
    version: string;
    trainedAt: Date;
}
export declare class AIModelTrainer {
    private readonly modelConfigs;
    /**
     * Train all AI models with the provided dataset
     */
    trainAllModels(trainData: TrainingFeatures[], validationData: TrainingFeatures[], testData: TrainingFeatures[]): Promise<{
        [modelName: string]: TrainedModel;
    }>;
    /**
     * Train Enhanced Transformer model
     */
    private trainTransformerModel;
    /**
     * Train Decisive LSTM model
     */
    private trainLSTMModel;
    /**
     * Train Active SMC model
     */
    private trainSMCModel;
    private getFeatureNames;
    private initializeWeights;
    private initializeLSTMWeights;
    private initializeBiases;
    private trainEpoch;
    private validateEpoch;
    private evaluateModel;
    private shouldEarlyStop;
    private prepareSequenceData;
    private extractSMCFeatures;
    private calculateFeatureImportance;
}
export declare function createAIModelTrainer(): AIModelTrainer;

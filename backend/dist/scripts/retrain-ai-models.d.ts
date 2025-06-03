#!/usr/bin/env node
/**
 * AI Model Retraining Script
 * Retrains all AI models using 6 months of real Binance data
 */
declare class AIModelRetrainingRunner {
    private dataProcessor;
    private modelTrainer;
    /**
     * Run comprehensive AI model retraining
     */
    runRetraining(): Promise<void>;
    /**
     * Fetch 6 months of real market data from Binance
     */
    private fetchTrainingData;
    /**
     * Save trained models to disk
     */
    private saveTrainedModels;
    /**
     * Generate comprehensive training report
     */
    private generateTrainingReport;
    /**
     * Validate model improvements
     */
    private validateModelImprovements;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
export { AIModelRetrainingRunner };

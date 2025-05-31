/**
 * ML Bridge Service
 * Provides an interface to interact with the ML system
 */
import { PredictionInput, PredictionOutput, ModelStatus, TrainingRequest, TrainingStatus, FeatureImportance } from '../../types/bridge';
/**
 * ML Bridge Service class
 * Handles communication with the ML service
 */
declare class MLBridgeService {
    private axios;
    private apiKey;
    /**
     * Creates a new ML Bridge Service instance
     */
    constructor();
    /**
     * Set authentication for requests
     * @param apiKey - API key for ML system
     */
    setAuth(apiKey: string): void;
    /**
     * Check connection to ML system
     * @returns Connection status
     */
    checkConnection(): Promise<{
        status: string;
        message: string;
        latency: number;
    }>;
    /**
     * Get prediction from ML model
     * @param input - Prediction input parameters
     * @returns Prediction output
     */
    getPrediction(input: PredictionInput): Promise<PredictionOutput>;
    /**
     * Get multiple predictions for a symbol
     * @param inputs - Array of prediction inputs
     * @returns Array of prediction outputs
     */
    getBatchPredictions(inputs: PredictionInput[]): Promise<PredictionOutput[]>;
    /**
     * Get prediction by ID
     * @param predictionId - Prediction ID
     * @returns Prediction output
     */
    getPredictionById(predictionId: string): Promise<PredictionOutput>;
    /**
     * Get ML model status
     * @param modelId - Model ID
     * @returns Model status
     */
    getModelStatus(modelId: string): Promise<ModelStatus>;
    /**
     * Get all available ML models
     * @returns Array of model status objects
     */
    getAllModels(): Promise<ModelStatus[]>;
    /**
     * Start model training
     * @param request - Training request
     * @returns Training status
     */
    startTraining(request: TrainingRequest): Promise<TrainingStatus>;
    /**
     * Get training status
     * @param trainingId - Training ID
     * @returns Training status
     */
    getTrainingStatus(trainingId: string): Promise<TrainingStatus>;
    /**
     * Cancel training
     * @param trainingId - Training ID
     * @returns Success status
     */
    cancelTraining(trainingId: string): Promise<boolean>;
    /**
     * Get feature importance for a model
     * @param modelId - Model ID
     * @returns Feature importance data
     */
    getFeatureImportance(modelId: string): Promise<FeatureImportance>;
    /**
     * Get ML system metrics
     * @returns System metrics
     */
    getSystemMetrics(): Promise<any>;
}
declare const mlBridgeService: MLBridgeService;
export default mlBridgeService;

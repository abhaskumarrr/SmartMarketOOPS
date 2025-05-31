/**
 * ML Model Client
 * Client for interacting with the ML prediction API
 */
import { ModelPrediction } from '../types/signals';
/**
 * Interface for ML model API configuration
 */
interface MLModelClientConfig {
    baseUrl: string;
    apiKey?: string;
    timeout?: number;
}
/**
 * Interface for prediction request payload
 */
interface PredictionRequest {
    symbol: string;
    features: Record<string, number>;
    sequence_length?: number;
}
/**
 * Interface for enhanced prediction response
 */
interface EnhancedPrediction {
    symbol: string;
    prediction: number;
    confidence: number;
    signal_valid: boolean;
    quality_score: number;
    recommendation: string;
    market_regime: string;
    regime_strength: number;
    model_predictions: Record<string, {
        prediction: number;
        confidence: number;
    }>;
    confidence_breakdown: Record<string, number>;
    prediction_time: string;
    enhanced: boolean;
}
/**
 * ML Model Client class
 * Provides methods to interact with the ML model API
 */
export declare class MLModelClient {
    private client;
    private baseUrl;
    /**
     * Creates a new ML Model Client instance
     * @param config - Configuration options
     */
    constructor(config: MLModelClientConfig);
    /**
     * Get a prediction from the ML model
     * @param request - Prediction request parameters
     * @returns Model prediction result
     */
    getPrediction(request: PredictionRequest): Promise<ModelPrediction>;
    /**
     * Get an enhanced prediction with signal quality analysis
     * @param request - Prediction request parameters
     * @returns Enhanced prediction result with signal quality metrics
     */
    getEnhancedPrediction(request: PredictionRequest): Promise<EnhancedPrediction>;
    /**
     * Get information about a specific model
     * @param symbol - Trading symbol
     * @returns Model information
     */
    getModelInfo(symbol: string): Promise<any>;
    /**
     * Load a specific model version
     * @param symbol - Trading symbol
     * @param modelVersion - Optional model version
     * @returns Load result
     */
    loadModel(symbol: string, modelVersion?: string): Promise<any>;
    /**
     * Load an enhanced model with signal quality system
     * @param symbol - Trading symbol
     * @param modelVersion - Optional model version
     * @returns Enhanced load result
     */
    loadEnhancedModel(symbol: string, modelVersion?: string): Promise<any>;
    /**
     * Get enhanced model status
     * @param symbol - Trading symbol
     * @returns Enhanced model status
     */
    getEnhancedModelStatus(symbol: string): Promise<any>;
    /**
     * Update model performance with actual trading results
     * @param symbol - Trading symbol
     * @param prediction - Model prediction value
     * @param actualOutcome - Actual trading outcome
     * @param confidence - Prediction confidence
     * @returns Update result
     */
    updateModelPerformance(symbol: string, prediction: number, actualOutcome: number, confidence: number): Promise<any>;
    /**
     * Log API response
     * @private
     * @param response - Axios response object
     */
    private _logResponse;
    /**
     * Log API error
     * @private
     * @param error - Error object
     */
    private _logError;
}
declare const defaultClient: MLModelClient;
export default defaultClient;

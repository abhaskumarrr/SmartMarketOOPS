/**
 * ML Model Client
 * Client for interacting with the ML prediction API
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { ModelPrediction } from '../types/signals';
import { createLogger, LogData } from '../utils/logger';

// Create logger
const logger = createLogger('MLModelClient');

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
  model_predictions: Record<string, { prediction: number; confidence: number }>;
  confidence_breakdown: Record<string, number>;
  prediction_time: string;
  enhanced: boolean;
}

/**
 * ML Model Client class
 * Provides methods to interact with the ML model API
 */
export class MLModelClient {
  private client: AxiosInstance;
  private baseUrl: string;

  /**
   * Creates a new ML Model Client instance
   * @param config - Configuration options
   */
  constructor(config: MLModelClientConfig) {
    this.baseUrl = config.baseUrl;

    // Set up axios instance with default configuration
    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
        ...(config.apiKey && { 'Authorization': `Bearer ${config.apiKey}` })
      }
    });

    // Add response interceptor for logging
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        this._logResponse(response);
        return response;
      },
      (error: any) => {
        this._logError(error);
        return Promise.reject(error);
      }
    );

    logger.info('ML Model Client initialized successfully');
  }

  /**
   * Get a prediction from the ML model
   * @param request - Prediction request parameters
   * @returns Model prediction result
   */
  async getPrediction(request: PredictionRequest): Promise<ModelPrediction> {
    try {
      logger.info(`Requesting prediction for ${request.symbol}`);

      const response = await this.client.post<ModelPrediction>('/predict', request);
      return response.data;
    } catch (error) {
      const logData: LogData = {
        symbol: request.symbol,
        error: error instanceof Error ? error.message : String(error)
      };
      logger.error(`Error getting prediction for ${request.symbol}`, logData);
      throw error;
    }
  }

  /**
   * Get an enhanced prediction with signal quality analysis
   * @param request - Prediction request parameters
   * @returns Enhanced prediction result with signal quality metrics
   */
  async getEnhancedPrediction(request: PredictionRequest): Promise<EnhancedPrediction> {
    try {
      logger.info(`Requesting enhanced prediction for ${request.symbol}`);

      const response = await this.client.post<EnhancedPrediction>('/enhanced/predict', request);
      return response.data;
    } catch (error) {
      const logData: LogData = {
        symbol: request.symbol,
        error: error instanceof Error ? error.message : String(error)
      };
      logger.error(`Error getting enhanced prediction for ${request.symbol}`, logData);
      throw error;
    }
  }

  /**
   * Get information about a specific model
   * @param symbol - Trading symbol
   * @returns Model information
   */
  async getModelInfo(symbol: string): Promise<any> {
    try {
      logger.info(`Getting model info for ${symbol}`);

      const response = await this.client.get(`/models/${symbol}`);
      return response.data;
    } catch (error) {
      const logData: LogData = {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      };
      logger.error(`Error getting model info for ${symbol}`, logData);
      throw error;
    }
  }

  /**
   * Load a specific model version
   * @param symbol - Trading symbol
   * @param modelVersion - Optional model version
   * @returns Load result
   */
  async loadModel(symbol: string, modelVersion?: string): Promise<any> {
    try {
      logger.info(`Loading model for ${symbol}${modelVersion ? ` (version: ${modelVersion})` : ''}`);

      const params = modelVersion ? { model_version: modelVersion } : {};
      const response = await this.client.post(`/models/${symbol}/load`, params);
      return response.data;
    } catch (error) {
      const logData: LogData = {
        symbol,
        modelVersion,
        error: error instanceof Error ? error.message : String(error)
      };
      logger.error(`Error loading model for ${symbol}`, logData);
      throw error;
    }
  }

  /**
   * Load an enhanced model with signal quality system
   * @param symbol - Trading symbol
   * @param modelVersion - Optional model version
   * @returns Enhanced load result
   */
  async loadEnhancedModel(symbol: string, modelVersion?: string): Promise<any> {
    try {
      logger.info(`Loading enhanced model for ${symbol}${modelVersion ? ` (version: ${modelVersion})` : ''}`);

      const params = modelVersion ? { model_version: modelVersion } : {};
      const response = await this.client.post(`/enhanced/models/${symbol}/load`, params);
      return response.data;
    } catch (error) {
      const logData: LogData = {
        symbol,
        modelVersion,
        error: error instanceof Error ? error.message : String(error)
      };
      logger.error(`Error loading enhanced model for ${symbol}`, logData);
      throw error;
    }
  }

  /**
   * Get enhanced model status
   * @param symbol - Trading symbol
   * @returns Enhanced model status
   */
  async getEnhancedModelStatus(symbol: string): Promise<any> {
    try {
      logger.info(`Getting enhanced model status for ${symbol}`);

      const response = await this.client.get(`/enhanced/models/${symbol}/status`);
      return response.data;
    } catch (error) {
      const logData: LogData = {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      };
      logger.error(`Error getting enhanced model status for ${symbol}`, logData);
      throw error;
    }
  }

  /**
   * Update model performance with actual trading results
   * @param symbol - Trading symbol
   * @param prediction - Model prediction value
   * @param actualOutcome - Actual trading outcome
   * @param confidence - Prediction confidence
   * @returns Update result
   */
  async updateModelPerformance(
    symbol: string,
    prediction: number,
    actualOutcome: number,
    confidence: number
  ): Promise<any> {
    try {
      logger.info(`Updating model performance for ${symbol}`);

      const response = await this.client.post(`/enhanced/models/${symbol}/performance`, {
        prediction,
        actual_outcome: actualOutcome,
        confidence
      });
      return response.data;
    } catch (error) {
      const logData: LogData = {
        symbol,
        prediction,
        actualOutcome,
        confidence,
        error: error instanceof Error ? error.message : String(error)
      };
      logger.error(`Error updating model performance for ${symbol}`, logData);
      throw error;
    }
  }

  /**
   * Log API response
   * @private
   * @param response - Axios response object
   */
  private _logResponse(response: AxiosResponse): void {
    const { config, status, statusText, headers } = response;
    const method = config.method?.toUpperCase() || 'UNKNOWN';
    const url = config.url || 'UNKNOWN';

    logger.debug(`${method} ${url} ${status} ${statusText}`);
  }

  /**
   * Log API error
   * @private
   * @param error - Error object
   */
  private _logError(error: any): void {
    if (error.response) {
      // Server responded with a status code outside of 2xx range
      const { config, status, statusText, data } = error.response;
      const method = config.method?.toUpperCase() || 'UNKNOWN';
      const url = config.url || 'UNKNOWN';

      const logData: LogData = {
        status,
        statusText,
        data,
        url
      };

      logger.error(`${method} ${url} ${status} ${statusText}`, logData);
    } else if (error.request) {
      // Request was made but no response received
      const logData: LogData = {
        request: error.request
      };
      logger.error('No response received', logData);
    } else {
      // Something happened in setting up the request
      logger.error('Error setting up request', error instanceof Error ? error : { message: String(error) });
    }
  }
}

// Create default client with environment variables
const defaultClient = new MLModelClient({
  baseUrl: process.env.ML_API_URL || 'http://localhost:8000',
  apiKey: process.env.ML_API_KEY,
  timeout: parseInt(process.env.ML_API_TIMEOUT || '30000', 10)
});

export default defaultClient;
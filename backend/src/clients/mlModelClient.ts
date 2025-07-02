/**
 * ML Model Client
 * Client for interacting with the ML prediction API
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { createLogger, LogData } from '../utils/logger';

// Create logger
const logger = createLogger('MLModelClient');

/**
 * Interface for ML model API configuration
 */
interface MLModelClientConfig {
  baseUrl: string;
  timeout?: number;
}

/**
 * Represents a single point of market data.
 */
export interface MarketData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Interface for the prediction request payload sent to the ML service.
 */
interface PredictionRequest {
  market_data: MarketData[];
}

/**
 * Interface for the prediction response received from the ML service.
 */
export interface PredictionResponse {
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reason: string;
  stop_loss?: number | null;
  target_price?: number | null;
}

/**
 * ML Model Client class
 * Provides methods to interact with the refactored ML model API
 */
export class MLModelClient {
  private client: AxiosInstance;

  /**
   * Creates a new ML Model Client instance
   * @param config - Configuration options
   */
  constructor(config: MLModelClientConfig) {
    // Set up axios instance with default configuration
    this.client = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
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
   * @param request - Prediction request parameters, containing market data
   * @returns Model prediction result
   */
  async getPrediction(request: PredictionRequest): Promise<PredictionResponse> {
    try {
      logger.info('Requesting prediction from core strategy service');

      const response = await this.client.post<PredictionResponse>('/predict', request);
      return response.data;
    } catch (error) {
      const logData: LogData = {
        error: error instanceof Error ? error.message : String(error)
      };
      logger.error('Error getting prediction from core strategy service', logData);
      throw error;
    }
  }

  /**
   * Logs successful API responses
   * @param response - The axios response object
   */
  private _logResponse(response: AxiosResponse): void {
    const logData: LogData = {
      status: response.status,
      statusText: response.statusText,
      url: response.config.url,
      method: response.config.method?.toUpperCase(),
    };
    logger.info(`Received API response for ${logData.method} ${logData.url}`, logData);
  }

  /**
   * Logs API errors
   * @param error - The error object
   */
  private _logError(error: any): void {
    if (error.response) {
      const logData: LogData = {
        status: error.response.status,
        statusText: error.response.statusText,
        url: error.config.url,
        method: error.config.method?.toUpperCase(),
        responseData: error.response.data,
      };
      logger.error(`API Error for ${logData.method} ${logData.url}`, logData);
    } else if (error.request) {
      logger.error('API Error: No response received', { url: error.config.url });
    } else {
      logger.error('API Error: Request setup failed', { message: error.message });
    }
  }
}

// Create default client with environment variables
const defaultClient = new MLModelClient({
  baseUrl: process.env.ML_API_URL || 'http://localhost:8000',
  timeout: parseInt(process.env.ML_API_TIMEOUT || '30000', 10)
});

export default defaultClient;
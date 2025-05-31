"use strict";
/**
 * ML Bridge Service
 * Provides an interface to interact with the ML system
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const axios_1 = __importDefault(require("axios"));
const logger_1 = require("../../utils/logger");
// Create logger
const logger = (0, logger_1.createLogger)('MLBridgeService');
// ML API base URL
const ML_API_URL = process.env.ML_API_URL || 'http://localhost:5000/api';
/**
 * ML Bridge Service class
 * Handles communication with the ML service
 */
class MLBridgeService {
    /**
     * Creates a new ML Bridge Service instance
     */
    constructor() {
        // Set up Axios instance with default configuration
        this.axios = axios_1.default.create({
            baseURL: ML_API_URL,
            timeout: 30000, // 30 seconds
            headers: {
                'Content-Type': 'application/json'
            }
        });
        // Get API key from environment
        this.apiKey = process.env.ML_API_KEY || '';
        logger.info('ML Bridge Service initialized');
    }
    /**
     * Set authentication for requests
     * @param apiKey - API key for ML system
     */
    setAuth(apiKey) {
        this.apiKey = apiKey;
        this.axios.defaults.headers.common['Authorization'] = `Bearer ${apiKey}`;
    }
    /**
     * Check connection to ML system
     * @returns Connection status
     */
    async checkConnection() {
        try {
            const startTime = Date.now();
            const response = await this.axios.get('/health');
            const latency = Date.now() - startTime;
            return {
                status: response.data.status,
                message: response.data.message,
                latency
            };
        }
        catch (error) {
            const logData = {
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error('Error checking ML system connection', logData);
            return {
                status: 'error',
                message: error instanceof Error ? error.message : String(error),
                latency: -1
            };
        }
    }
    /**
     * Get prediction from ML model
     * @param input - Prediction input parameters
     * @returns Prediction output
     */
    async getPrediction(input) {
        try {
            logger.info(`Getting prediction for ${input.symbol} on ${input.timeframe} timeframe`);
            const response = await this.axios.post('/predictions', input);
            return response.data;
        }
        catch (error) {
            const logData = {
                input,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting prediction for ${input.symbol}`, logData);
            throw error;
        }
    }
    /**
     * Get multiple predictions for a symbol
     * @param inputs - Array of prediction inputs
     * @returns Array of prediction outputs
     */
    async getBatchPredictions(inputs) {
        try {
            logger.info(`Getting batch predictions for ${inputs.length} inputs`);
            const response = await this.axios.post('/predictions/batch', { inputs });
            return response.data;
        }
        catch (error) {
            const logData = {
                inputCount: inputs.length,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error('Error getting batch predictions', logData);
            throw error;
        }
    }
    /**
     * Get prediction by ID
     * @param predictionId - Prediction ID
     * @returns Prediction output
     */
    async getPredictionById(predictionId) {
        try {
            logger.info(`Getting prediction by ID: ${predictionId}`);
            const response = await this.axios.get(`/predictions/${predictionId}`);
            return response.data;
        }
        catch (error) {
            const logData = {
                predictionId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting prediction by ID: ${predictionId}`, logData);
            throw error;
        }
    }
    /**
     * Get ML model status
     * @param modelId - Model ID
     * @returns Model status
     */
    async getModelStatus(modelId) {
        try {
            logger.info(`Getting status for model: ${modelId}`);
            const response = await this.axios.get(`/models/${modelId}`);
            return response.data;
        }
        catch (error) {
            const logData = {
                modelId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting status for model: ${modelId}`, logData);
            throw error;
        }
    }
    /**
     * Get all available ML models
     * @returns Array of model status objects
     */
    async getAllModels() {
        try {
            logger.info('Getting all ML models');
            const response = await this.axios.get('/models');
            return response.data;
        }
        catch (error) {
            const logData = {
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error('Error getting all ML models', logData);
            throw error;
        }
    }
    /**
     * Start model training
     * @param request - Training request
     * @returns Training status
     */
    async startTraining(request) {
        try {
            logger.info(`Starting training for model type: ${request.modelType} on ${request.symbol}`);
            const response = await this.axios.post('/training', request);
            return response.data;
        }
        catch (error) {
            const logData = {
                request,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error starting training for model type: ${request.modelType}`, logData);
            throw error;
        }
    }
    /**
     * Get training status
     * @param trainingId - Training ID
     * @returns Training status
     */
    async getTrainingStatus(trainingId) {
        try {
            logger.info(`Getting training status for: ${trainingId}`);
            const response = await this.axios.get(`/training/${trainingId}`);
            return response.data;
        }
        catch (error) {
            const logData = {
                trainingId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting training status for: ${trainingId}`, logData);
            throw error;
        }
    }
    /**
     * Cancel training
     * @param trainingId - Training ID
     * @returns Success status
     */
    async cancelTraining(trainingId) {
        try {
            logger.info(`Cancelling training: ${trainingId}`);
            await this.axios.delete(`/training/${trainingId}`);
            return true;
        }
        catch (error) {
            const logData = {
                trainingId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error cancelling training: ${trainingId}`, logData);
            throw error;
        }
    }
    /**
     * Get feature importance for a model
     * @param modelId - Model ID
     * @returns Feature importance data
     */
    async getFeatureImportance(modelId) {
        try {
            logger.info(`Getting feature importance for model: ${modelId}`);
            const response = await this.axios.get(`/models/${modelId}/features`);
            return response.data;
        }
        catch (error) {
            const logData = {
                modelId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting feature importance for model: ${modelId}`, logData);
            throw error;
        }
    }
    /**
     * Get ML system metrics
     * @returns System metrics
     */
    async getSystemMetrics() {
        try {
            logger.info('Getting ML system metrics');
            const response = await this.axios.get('/metrics');
            return response.data;
        }
        catch (error) {
            const logData = {
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error('Error getting ML system metrics', logData);
            throw error;
        }
    }
}
// Create singleton instance
const mlBridgeService = new MLBridgeService();
exports.default = mlBridgeService;
//# sourceMappingURL=mlBridgeService.js.map
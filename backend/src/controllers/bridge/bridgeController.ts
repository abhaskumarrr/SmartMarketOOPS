/**
 * Bridge Controller
 * Handles HTTP requests for the ML-Trading bridge API
 */

import { Response } from 'express';
import { AuthenticatedRequest } from '../../middleware/authMiddleware';
import { createLogger } from '../../utils/logger';
import bridgeService from '../../services/bridge/bridgeService';
import mlBridgeService from '../../services/bridge/mlBridgeService';
import { PredictionInput, TrainingRequest, BacktestRequest } from '../../types/bridge';

// Create logger
const logger = createLogger('BridgeController');

/**
 * Get ML prediction and generate trading signal
 * @route POST /api/bridge/predict-and-signal
 */
export const getPredictionAndGenerateSignal = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { symbol, timeframe, modelVersion, confidenceThreshold, signalExpiry } = req.body;
    
    if (!symbol || !timeframe) {
      res.status(400).json({ error: 'Symbol and timeframe are required' });
      return;
    }
    
    const options = {
      modelVersion,
      confidenceThreshold,
      signalExpiry
    };
    
    const signal = await bridgeService.getPredictionAndGenerateSignal(symbol, timeframe, options);
    
    res.status(200).json(signal);
  } catch (error) {
    logger.error('Error getting prediction and generating signal', {
      body: req.body,
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get prediction and generate signal',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get prediction
 * @route POST /api/bridge/predict
 */
export const getPrediction = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const predictionInput: PredictionInput = req.body;
    
    if (!predictionInput.symbol || !predictionInput.timeframe) {
      res.status(400).json({ error: 'Symbol and timeframe are required' });
      return;
    }
    
    const prediction = await mlBridgeService.getPrediction(predictionInput);
    
    res.status(200).json(prediction);
  } catch (error) {
    logger.error('Error getting prediction', {
      body: req.body,
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get prediction',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get batch predictions
 * @route POST /api/bridge/predict-batch
 */
export const getBatchPredictions = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { inputs } = req.body;
    
    if (!inputs || !Array.isArray(inputs) || inputs.length === 0) {
      res.status(400).json({ error: 'Valid inputs array is required' });
      return;
    }
    
    const predictions = await mlBridgeService.getBatchPredictions(inputs);
    
    res.status(200).json(predictions);
  } catch (error) {
    logger.error('Error getting batch predictions', {
      body: req.body,
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get batch predictions',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get prediction by ID
 * @route GET /api/bridge/predictions/:id
 */
export const getPredictionById = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    if (!id) {
      res.status(400).json({ error: 'Prediction ID is required' });
      return;
    }
    
    const prediction = await mlBridgeService.getPredictionById(id);
    
    res.status(200).json(prediction);
  } catch (error) {
    logger.error(`Error getting prediction by ID: ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get prediction',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get all ML models
 * @route GET /api/bridge/models
 */
export const getAllModels = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const models = await bridgeService.getAvailableModels();
    
    res.status(200).json(models);
  } catch (error) {
    logger.error('Error getting all models', {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get models',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get model by ID
 * @route GET /api/bridge/models/:id
 */
export const getModelById = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    if (!id) {
      res.status(400).json({ error: 'Model ID is required' });
      return;
    }
    
    const model = await mlBridgeService.getModelStatus(id);
    
    res.status(200).json(model);
  } catch (error) {
    logger.error(`Error getting model by ID: ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get model',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get feature importance for a model
 * @route GET /api/bridge/models/:id/features
 */
export const getFeatureImportance = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    if (!id) {
      res.status(400).json({ error: 'Model ID is required' });
      return;
    }
    
    const featureImportance = await mlBridgeService.getFeatureImportance(id);
    
    res.status(200).json(featureImportance);
  } catch (error) {
    logger.error(`Error getting feature importance for model: ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get feature importance',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Start model training
 * @route POST /api/bridge/training
 */
export const startTraining = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const trainingRequest: TrainingRequest = req.body;
    
    if (!trainingRequest.modelType || !trainingRequest.symbol || !trainingRequest.timeframe) {
      res.status(400).json({ error: 'Model type, symbol, and timeframe are required' });
      return;
    }
    
    const trainingStatus = await bridgeService.startModelTraining(trainingRequest);
    
    res.status(200).json(trainingStatus);
  } catch (error) {
    logger.error('Error starting model training', {
      body: req.body,
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to start model training',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get training status
 * @route GET /api/bridge/training/:id
 */
export const getTrainingStatus = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    if (!id) {
      res.status(400).json({ error: 'Training ID is required' });
      return;
    }
    
    const trainingStatus = await mlBridgeService.getTrainingStatus(id);
    
    res.status(200).json(trainingStatus);
  } catch (error) {
    logger.error(`Error getting training status: ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get training status',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Cancel training
 * @route DELETE /api/bridge/training/:id
 */
export const cancelTraining = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    
    if (!id) {
      res.status(400).json({ error: 'Training ID is required' });
      return;
    }
    
    const result = await mlBridgeService.cancelTraining(id);
    
    res.status(200).json({ success: result });
  } catch (error) {
    logger.error(`Error cancelling training: ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to cancel training',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Run backtest
 * @route POST /api/bridge/backtest
 */
export const runBacktest = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const backtestRequest: BacktestRequest = req.body;
    
    if (!backtestRequest.strategyId || !backtestRequest.symbol || !backtestRequest.timeframe) {
      res.status(400).json({ error: 'Strategy ID, symbol, and timeframe are required' });
      return;
    }
    
    const backtestResult = await bridgeService.runBacktest(backtestRequest);
    
    res.status(200).json(backtestResult);
  } catch (error) {
    logger.error('Error running backtest', {
      body: req.body,
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to run backtest',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get bridge health
 * @route GET /api/bridge/health
 */
export const getBridgeHealth = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const health = await bridgeService.getHealth();
    
    res.status(200).json(health);
  } catch (error) {
    logger.error('Error getting bridge health', {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get bridge health',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Check ML connection
 * @route GET /api/bridge/ml-health
 */
export const checkMLConnection = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const health = await mlBridgeService.checkConnection();
    
    res.status(200).json(health);
  } catch (error) {
    logger.error('Error checking ML connection', {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to check ML connection',
      details: error instanceof Error ? error.message : String(error)
    });
  }
}; 
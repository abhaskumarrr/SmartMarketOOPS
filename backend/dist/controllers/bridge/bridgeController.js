"use strict";
/**
 * Bridge Controller
 * Handles HTTP requests for the ML-Trading bridge API
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.checkMLConnection = exports.getBridgeHealth = exports.runBacktest = exports.cancelTraining = exports.getTrainingStatus = exports.startTraining = exports.getFeatureImportance = exports.getModelById = exports.getAllModels = exports.getPredictionById = exports.getBatchPredictions = exports.getPrediction = exports.getPredictionAndGenerateSignal = void 0;
const logger_1 = require("../../utils/logger");
const bridgeService_1 = __importDefault(require("../../services/bridge/bridgeService"));
const mlBridgeService_1 = __importDefault(require("../../services/bridge/mlBridgeService"));
// Create logger
const logger = (0, logger_1.createLogger)('BridgeController');
/**
 * Get ML prediction and generate trading signal
 * @route POST /api/bridge/predict-and-signal
 */
const getPredictionAndGenerateSignal = async (req, res) => {
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
        const signal = await bridgeService_1.default.getPredictionAndGenerateSignal(symbol, timeframe, options);
        res.status(200).json(signal);
    }
    catch (error) {
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
exports.getPredictionAndGenerateSignal = getPredictionAndGenerateSignal;
/**
 * Get prediction
 * @route POST /api/bridge/predict
 */
const getPrediction = async (req, res) => {
    try {
        const predictionInput = req.body;
        if (!predictionInput.symbol || !predictionInput.timeframe) {
            res.status(400).json({ error: 'Symbol and timeframe are required' });
            return;
        }
        const prediction = await mlBridgeService_1.default.getPrediction(predictionInput);
        res.status(200).json(prediction);
    }
    catch (error) {
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
exports.getPrediction = getPrediction;
/**
 * Get batch predictions
 * @route POST /api/bridge/predict-batch
 */
const getBatchPredictions = async (req, res) => {
    try {
        const { inputs } = req.body;
        if (!inputs || !Array.isArray(inputs) || inputs.length === 0) {
            res.status(400).json({ error: 'Valid inputs array is required' });
            return;
        }
        const predictions = await mlBridgeService_1.default.getBatchPredictions(inputs);
        res.status(200).json(predictions);
    }
    catch (error) {
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
exports.getBatchPredictions = getBatchPredictions;
/**
 * Get prediction by ID
 * @route GET /api/bridge/predictions/:id
 */
const getPredictionById = async (req, res) => {
    try {
        const { id } = req.params;
        if (!id) {
            res.status(400).json({ error: 'Prediction ID is required' });
            return;
        }
        const prediction = await mlBridgeService_1.default.getPredictionById(id);
        res.status(200).json(prediction);
    }
    catch (error) {
        logger.error(`Error getting prediction by ID: ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get prediction',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getPredictionById = getPredictionById;
/**
 * Get all ML models
 * @route GET /api/bridge/models
 */
const getAllModels = async (req, res) => {
    try {
        const models = await bridgeService_1.default.getAvailableModels();
        res.status(200).json(models);
    }
    catch (error) {
        logger.error('Error getting all models', {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get models',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getAllModels = getAllModels;
/**
 * Get model by ID
 * @route GET /api/bridge/models/:id
 */
const getModelById = async (req, res) => {
    try {
        const { id } = req.params;
        if (!id) {
            res.status(400).json({ error: 'Model ID is required' });
            return;
        }
        const model = await mlBridgeService_1.default.getModelStatus(id);
        res.status(200).json(model);
    }
    catch (error) {
        logger.error(`Error getting model by ID: ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get model',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getModelById = getModelById;
/**
 * Get feature importance for a model
 * @route GET /api/bridge/models/:id/features
 */
const getFeatureImportance = async (req, res) => {
    try {
        const { id } = req.params;
        if (!id) {
            res.status(400).json({ error: 'Model ID is required' });
            return;
        }
        const featureImportance = await mlBridgeService_1.default.getFeatureImportance(id);
        res.status(200).json(featureImportance);
    }
    catch (error) {
        logger.error(`Error getting feature importance for model: ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get feature importance',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getFeatureImportance = getFeatureImportance;
/**
 * Start model training
 * @route POST /api/bridge/training
 */
const startTraining = async (req, res) => {
    try {
        const trainingRequest = req.body;
        if (!trainingRequest.modelType || !trainingRequest.symbol || !trainingRequest.timeframe) {
            res.status(400).json({ error: 'Model type, symbol, and timeframe are required' });
            return;
        }
        const trainingStatus = await bridgeService_1.default.startModelTraining(trainingRequest);
        res.status(200).json(trainingStatus);
    }
    catch (error) {
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
exports.startTraining = startTraining;
/**
 * Get training status
 * @route GET /api/bridge/training/:id
 */
const getTrainingStatus = async (req, res) => {
    try {
        const { id } = req.params;
        if (!id) {
            res.status(400).json({ error: 'Training ID is required' });
            return;
        }
        const trainingStatus = await mlBridgeService_1.default.getTrainingStatus(id);
        res.status(200).json(trainingStatus);
    }
    catch (error) {
        logger.error(`Error getting training status: ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get training status',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getTrainingStatus = getTrainingStatus;
/**
 * Cancel training
 * @route DELETE /api/bridge/training/:id
 */
const cancelTraining = async (req, res) => {
    try {
        const { id } = req.params;
        if (!id) {
            res.status(400).json({ error: 'Training ID is required' });
            return;
        }
        const result = await mlBridgeService_1.default.cancelTraining(id);
        res.status(200).json({ success: result });
    }
    catch (error) {
        logger.error(`Error cancelling training: ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to cancel training',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.cancelTraining = cancelTraining;
/**
 * Run backtest
 * @route POST /api/bridge/backtest
 */
const runBacktest = async (req, res) => {
    try {
        const backtestRequest = req.body;
        if (!backtestRequest.strategyId || !backtestRequest.symbol || !backtestRequest.timeframe) {
            res.status(400).json({ error: 'Strategy ID, symbol, and timeframe are required' });
            return;
        }
        const backtestResult = await bridgeService_1.default.runBacktest(backtestRequest);
        res.status(200).json(backtestResult);
    }
    catch (error) {
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
exports.runBacktest = runBacktest;
/**
 * Get bridge health
 * @route GET /api/bridge/health
 */
const getBridgeHealth = async (req, res) => {
    try {
        const health = await bridgeService_1.default.getHealth();
        res.status(200).json(health);
    }
    catch (error) {
        logger.error('Error getting bridge health', {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get bridge health',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getBridgeHealth = getBridgeHealth;
/**
 * Check ML connection
 * @route GET /api/bridge/ml-health
 */
const checkMLConnection = async (req, res) => {
    try {
        const health = await mlBridgeService_1.default.checkConnection();
        res.status(200).json(health);
    }
    catch (error) {
        logger.error('Error checking ML connection', {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to check ML connection',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.checkMLConnection = checkMLConnection;
//# sourceMappingURL=bridgeController.js.map
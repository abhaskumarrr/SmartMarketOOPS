/**
 * Bridge Routes
 * Routes for the ML-Trading bridge API
 */

import express from 'express';
import { authenticateJWT } from '../../middleware/authMiddleware';
import {
  getPredictionAndGenerateSignal,
  getPrediction,
  getBatchPredictions,
  getPredictionById,
  getAllModels,
  getModelById,
  getFeatureImportance,
  startTraining,
  getTrainingStatus,
  cancelTraining,
  runBacktest,
  getBridgeHealth,
  checkMLConnection
} from '../../controllers/bridge/bridgeController';

const router = express.Router();

// Prediction routes
router.post('/predict-and-signal', authenticateJWT, getPredictionAndGenerateSignal);
router.post('/predict', authenticateJWT, getPrediction);
router.post('/predict-batch', authenticateJWT, getBatchPredictions);
router.get('/predictions/:id', authenticateJWT, getPredictionById);

// Model routes
router.get('/models', authenticateJWT, getAllModels);
router.get('/models/:id', authenticateJWT, getModelById);
router.get('/models/:id/features', authenticateJWT, getFeatureImportance);

// Training routes
router.post('/training', authenticateJWT, startTraining);
router.get('/training/:id', authenticateJWT, getTrainingStatus);
router.delete('/training/:id', authenticateJWT, cancelTraining);

// Backtest routes
router.post('/backtest', authenticateJWT, runBacktest);

// Health routes
router.get('/health', authenticateJWT, getBridgeHealth);
router.get('/ml-health', authenticateJWT, checkMLConnection);

export default router; 
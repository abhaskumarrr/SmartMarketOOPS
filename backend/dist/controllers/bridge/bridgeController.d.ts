/**
 * Bridge Controller
 * Handles HTTP requests for the ML-Trading bridge API
 */
import { Response } from 'express';
import { AuthenticatedRequest } from '../../middleware/authMiddleware';
/**
 * Get ML prediction and generate trading signal
 * @route POST /api/bridge/predict-and-signal
 */
export declare const getPredictionAndGenerateSignal: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get prediction
 * @route POST /api/bridge/predict
 */
export declare const getPrediction: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get batch predictions
 * @route POST /api/bridge/predict-batch
 */
export declare const getBatchPredictions: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get prediction by ID
 * @route GET /api/bridge/predictions/:id
 */
export declare const getPredictionById: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get all ML models
 * @route GET /api/bridge/models
 */
export declare const getAllModels: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get model by ID
 * @route GET /api/bridge/models/:id
 */
export declare const getModelById: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get feature importance for a model
 * @route GET /api/bridge/models/:id/features
 */
export declare const getFeatureImportance: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Start model training
 * @route POST /api/bridge/training
 */
export declare const startTraining: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get training status
 * @route GET /api/bridge/training/:id
 */
export declare const getTrainingStatus: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Cancel training
 * @route DELETE /api/bridge/training/:id
 */
export declare const cancelTraining: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Run backtest
 * @route POST /api/bridge/backtest
 */
export declare const runBacktest: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get bridge health
 * @route GET /api/bridge/health
 */
export declare const getBridgeHealth: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Check ML connection
 * @route GET /api/bridge/ml-health
 */
export declare const checkMLConnection: (req: AuthenticatedRequest, res: Response) => Promise<void>;

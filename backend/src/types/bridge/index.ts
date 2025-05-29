/**
 * Bridge API Types
 * Interfaces for connecting ML and Trading systems
 */

import { Timestamp } from '../common';

/**
 * ML Model Prediction Input
 */
export interface PredictionInput {
  symbol: string;
  timeframe: string;
  features?: Record<string, number[]>;
  timestamps?: string[];
  limit?: number;
  modelVersion?: string;
}

/**
 * ML Model Prediction Output
 */
export interface PredictionOutput {
  id: string;
  symbol: string;
  timeframe: string;
  predictionType: 'PRICE' | 'DIRECTION' | 'PROBABILITY';
  values: number[];
  timestamps: string[];
  confidenceScores: number[];
  metadata: {
    modelVersion: string;
    modelName: string;
    inputFeatures: string[];
    performance: {
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1Score?: number;
      mse?: number;
      mae?: number;
    };
  };
  generatedAt: string;
}

/**
 * Trading Signal Generation Request
 */
export interface SignalGenerationRequest {
  predictionId?: string;
  prediction?: PredictionOutput;
  symbol: string;
  timeframe: string;
  strategy?: string;
  options?: {
    confidenceThreshold?: number;
    signalExpiry?: number; // Minutes
    includeTechnicalOverlay?: boolean;
  };
}

/**
 * ML Model Status
 */
export interface ModelStatus {
  id: string;
  name: string;
  version: string;
  type: string;
  status: 'ACTIVE' | 'INACTIVE' | 'TRAINING' | 'ERROR';
  accuracy: number;
  supportedSymbols: string[];
  supportedTimeframes: string[];
  lastTrainedAt: string;
  lastUsedAt: string;
  errorRate: number;
  averageLatency: number; // ms
}

/**
 * ML Training Request
 */
export interface TrainingRequest {
  modelId?: string;
  modelType: string;
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  features: string[];
  hyperparameters?: Record<string, any>;
  validationSplit?: number;
  description?: string;
}

/**
 * ML Training Status
 */
export interface TrainingStatus {
  id: string;
  status: 'QUEUED' | 'TRAINING' | 'COMPLETED' | 'FAILED';
  progress: number; // 0-100
  metrics: {
    epoch: number;
    loss: number;
    valLoss: number;
    accuracy?: number;
    valAccuracy?: number;
    [key: string]: any;
  }[];
  startedAt: string;
  estimatedCompletionAt?: string;
  completedAt?: string;
  error?: string;
}

/**
 * Backtesting Request
 */
export interface BacktestRequest {
  strategyId: string;
  modelId?: string;
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  options?: {
    includeFees?: boolean;
    feePercentage?: number;
    includeSlippage?: boolean;
    slippagePercentage?: number;
    useHistoricalPredictions?: boolean;
  };
}

/**
 * Bridge System Health
 */
export interface BridgeHealth {
  status: 'HEALTHY' | 'DEGRADED' | 'UNAVAILABLE';
  mlSystemStatus: 'AVAILABLE' | 'DEGRADED' | 'UNAVAILABLE';
  tradingSystemStatus: 'AVAILABLE' | 'DEGRADED' | 'UNAVAILABLE';
  latency: {
    predictionAvg: number; // ms
    signalGenerationAvg: number; // ms
    endToEndAvg: number; // ms
  };
  lastSyncedAt: string;
  activeModels: number;
  activeStrategies: number;
  errors: {
    timestamp: string;
    component: string;
    message: string;
    count: number;
  }[];
  metrics: {
    predictionRequests1h: number;
    signalGenerationRequests1h: number;
    successRate: number;
  };
}

/**
 * Feature Importance
 */
export interface FeatureImportance {
  modelId: string;
  features: {
    name: string;
    importance: number; // 0-1
    correlation: number; // -1 to 1
  }[];
  timestamp: string;
}

/**
 * Real-time Update
 */
export interface RealTimeUpdate {
  type: 'PREDICTION' | 'SIGNAL' | 'EXECUTION' | 'MODEL_STATUS' | 'TRAINING_STATUS';
  data: any;
  timestamp: string;
} 
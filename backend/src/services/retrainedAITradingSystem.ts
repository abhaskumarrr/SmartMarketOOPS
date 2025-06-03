/**
 * Retrained AI Trading System
 * Uses newly trained AI models with 6 months of real market data
 */

import { 
  TradingStrategy,
  TradingSignal, 
  EnhancedMarketData, 
  BacktestConfig 
} from '../types/marketData';
import { TrainedModel } from './aiModelTrainer';
import { logger } from '../utils/logger';
import { createEventId } from '../types/events';
import fs from 'fs';
import path from 'path';

// Trade Classification System
export enum TradeType {
  SCALPING = 'SCALPING',
  DAY_TRADING = 'DAY_TRADING', 
  SWING_TRADING = 'SWING_TRADING',
  POSITION_TRADING = 'POSITION_TRADING'
}

export enum MarketRegime {
  TRENDING_BULLISH = 'TRENDING_BULLISH',
  TRENDING_BEARISH = 'TRENDING_BEARISH',
  SIDEWAYS = 'SIDEWAYS',
  VOLATILE = 'VOLATILE',
  BREAKOUT = 'BREAKOUT'
}

interface RetrainedModelPrediction {
  modelName: string;
  prediction: number;
  confidence: number;
  signalType: 'BUY' | 'SELL' | 'HOLD';
  modelVersion: string;
  featureImportance: { [feature: string]: number };
}

export class RetrainedAITradingSystem implements TradingStrategy {
  public readonly name = 'Retrained_AI_Trading_System';
  public parameters: Record<string, any>;
  
  private config?: BacktestConfig;
  private trainedModels: { [modelName: string]: TrainedModel } = {};
  private lastDecisionTime: number = 0;
  private modelLoadTime: Date = new Date();

  constructor() {
    this.parameters = {
      // Enhanced parameters based on retrained models
      minConfidence: 65, // Increased confidence threshold
      minModelConsensus: 0.6, // Require 60% model agreement
      decisionCooldown: 300000, // 5 minutes
      
      // Model-specific weights (will be updated after loading)
      modelWeights: {
        transformer: 0.4,
        lstm: 0.35,
        smc: 0.25,
      },
      
      // Feature importance thresholds
      minFeatureImportance: 0.1,
      
      // Risk management
      riskPerTrade: 2,
      stopLossPercent: 1.8,
      takeProfitMultiplier: 2.8,
      positionSizeMultiplier: 0.9,
      
      // Model performance requirements
      minModelAccuracy: 0.65,
      requireAllModelsLoaded: true,
    };

    logger.info('🧠 Retrained AI Trading System initialized');
  }

  /**
   * Initialize the trading system and load retrained models
   */
  public async initialize(config: BacktestConfig): Promise<void> {
    this.config = config;
    this.lastDecisionTime = 0;
    
    // Load retrained models
    await this.loadRetrainedModels();
    
    // Update parameters based on loaded models
    this.updateParametersFromModels();
    
    logger.info(`🎯 Initialized ${this.name} with retrained models`, {
      symbol: config.symbol,
      timeframe: config.timeframe,
      modelsLoaded: Object.keys(this.trainedModels).length,
      modelLoadTime: this.modelLoadTime.toISOString(),
    });
  }

  /**
   * Load retrained models from disk
   */
  private async loadRetrainedModels(): Promise<void> {
    logger.info('📂 Loading retrained AI models...');

    const modelsDir = path.join(process.cwd(), 'trained_models');
    const modelTypes = ['transformer', 'lstm', 'smc'];

    for (const modelType of modelTypes) {
      try {
        const latestModelPath = path.join(modelsDir, `${modelType}_model_latest.json`);
        
        if (fs.existsSync(latestModelPath)) {
          const modelData = fs.readFileSync(latestModelPath, 'utf8');
          const trainedModel: TrainedModel = JSON.parse(modelData);
          
          // Validate model
          if (this.validateModel(trainedModel)) {
            this.trainedModels[modelType] = trainedModel;
            logger.info(`✅ Loaded ${trainedModel.modelName} v${trainedModel.version}`, {
              testAccuracy: `${(trainedModel.finalMetrics.testAccuracy * 100).toFixed(1)}%`,
              trainedAt: trainedModel.trainedAt,
            });
          } else {
            logger.warn(`⚠️ Model validation failed for ${modelType}`);
          }
        } else {
          logger.warn(`⚠️ No retrained model found for ${modelType}, using default`);
          this.trainedModels[modelType] = this.createDefaultModel(modelType);
        }
      } catch (error) {
        logger.error(`❌ Failed to load ${modelType} model:`, error);
        this.trainedModels[modelType] = this.createDefaultModel(modelType);
      }
    }

    const loadedCount = Object.keys(this.trainedModels).length;
    logger.info(`📊 Model loading completed: ${loadedCount}/3 models loaded`);

    if (loadedCount === 0) {
      throw new Error('No AI models could be loaded');
    }
  }

  /**
   * Validate loaded model
   */
  private validateModel(model: TrainedModel): boolean {
    const requiredFields = ['modelName', 'parameters', 'finalMetrics', 'version'];
    const hasRequiredFields = requiredFields.every(field => model[field] !== undefined);
    
    const meetsAccuracy = model.finalMetrics.testAccuracy >= this.parameters.minModelAccuracy;
    const hasValidWeights = model.parameters.weights && Object.keys(model.parameters.weights).length > 0;
    
    return hasRequiredFields && meetsAccuracy && hasValidWeights;
  }

  /**
   * Create default model if retrained model not available
   */
  private createDefaultModel(modelType: string): TrainedModel {
    return {
      modelName: `Default_${modelType}`,
      parameters: {
        weights: { default: 0.5 },
        biases: [0],
        learningRate: 0.001,
        epochs: 0,
        batchSize: 32,
        regularization: 0.01,
        dropout: 0.2,
      },
      trainingHistory: [],
      finalMetrics: {
        trainAccuracy: 0.6,
        validationAccuracy: 0.58,
        testAccuracy: 0.55,
        precision: 0.6,
        recall: 0.55,
        f1Score: 0.57,
      },
      featureImportance: { default: 1.0 },
      trainingTime: 0,
      version: '1.0.0',
      trainedAt: new Date(),
    };
  }

  /**
   * Update system parameters based on loaded models
   */
  private updateParametersFromModels(): void {
    const models = Object.values(this.trainedModels);
    
    if (models.length === 0) return;

    // Calculate average model performance
    const avgAccuracy = models.reduce((sum, model) => sum + model.finalMetrics.testAccuracy, 0) / models.length;
    
    // Adjust confidence threshold based on model performance
    if (avgAccuracy > 0.8) {
      this.parameters.minConfidence = 70;
      this.parameters.minModelConsensus = 0.65;
    } else if (avgAccuracy > 0.7) {
      this.parameters.minConfidence = 65;
      this.parameters.minModelConsensus = 0.6;
    } else {
      this.parameters.minConfidence = 60;
      this.parameters.minModelConsensus = 0.55;
    }

    // Update model weights based on performance
    const totalPerformance = models.reduce((sum, model) => sum + model.finalMetrics.testAccuracy, 0);
    
    models.forEach(model => {
      const modelType = this.getModelTypeFromName(model.modelName);
      if (modelType) {
        this.parameters.modelWeights[modelType] = model.finalMetrics.testAccuracy / totalPerformance;
      }
    });

    logger.info('🔧 Parameters updated based on model performance', {
      avgAccuracy: `${(avgAccuracy * 100).toFixed(1)}%`,
      minConfidence: this.parameters.minConfidence,
      modelWeights: this.parameters.modelWeights,
    });
  }

  /**
   * Generate trading signal using retrained models
   */
  public generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null {
    if (!this.config) {
      throw new Error('Strategy not initialized. Call initialize() first.');
    }

    // Need enough data for analysis
    if (currentIndex < 50) {
      return null;
    }

    const currentTime = Date.now();

    // Check decision cooldown
    if (currentTime - this.lastDecisionTime < this.parameters.decisionCooldown) {
      return null;
    }

    const currentCandle = data[currentIndex];

    try {
      // Get predictions from all retrained models
      const modelPredictions = this.getRetrainedModelPredictions(currentCandle, data, currentIndex);
      
      if (modelPredictions.length === 0) {
        return null;
      }

      // Analyze model consensus
      const consensus = this.analyzeModelConsensus(modelPredictions);
      
      if (consensus.consensusScore < this.parameters.minModelConsensus) {
        return null;
      }

      // Generate final signal
      const signal = this.generateFinalSignal(consensus, currentCandle, data, currentIndex);
      
      if (signal && signal.confidence >= this.parameters.minConfidence) {
        this.lastDecisionTime = currentTime;
        
        logger.info(`🎯 Generated retrained AI ${signal.type} signal`, {
          price: signal.price,
          confidence: signal.confidence,
          consensusScore: consensus.consensusScore,
          modelsUsed: modelPredictions.map(p => p.modelName),
        });
        
        return signal;
      }

      return null;

    } catch (error) {
      logger.error('❌ Error generating retrained AI signal:', error);
      return null;
    }
  }

  /**
   * Get predictions from retrained models
   */
  private getRetrainedModelPredictions(
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): RetrainedModelPrediction[] {
    const predictions: RetrainedModelPrediction[] = [];

    Object.entries(this.trainedModels).forEach(([modelType, model]) => {
      try {
        const prediction = this.runModelPrediction(model, currentCandle, data, currentIndex);
        if (prediction) {
          predictions.push(prediction);
        }
      } catch (error) {
        logger.warn(`⚠️ Failed to get prediction from ${model.modelName}:`, error);
      }
    });

    return predictions;
  }

  /**
   * Run prediction using a specific retrained model
   */
  private runModelPrediction(
    model: TrainedModel,
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): RetrainedModelPrediction | null {
    
    const indicators = currentCandle.indicators;
    if (!indicators.rsi || !indicators.ema_12 || !indicators.ema_26) {
      return null;
    }

    // Extract features for prediction
    const features = this.extractFeatures(currentCandle, data, currentIndex);
    
    // Run model inference using trained weights
    const rawPrediction = this.runModelInference(model, features);
    
    // Convert to trading signal
    const signalType = this.predictionToSignal(rawPrediction);
    
    // Calculate confidence based on model performance and prediction strength
    const baseConfidence = model.finalMetrics.testAccuracy * 100;
    const predictionStrength = Math.abs(rawPrediction - 0.5) * 2; // 0-1 scale
    const confidence = baseConfidence * (0.7 + 0.3 * predictionStrength);

    return {
      modelName: model.modelName,
      prediction: rawPrediction,
      confidence: Math.min(100, confidence),
      signalType,
      modelVersion: model.version,
      featureImportance: model.featureImportance,
    };
  }

  /**
   * Extract features for model prediction
   */
  private extractFeatures(
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): number[] {
    const indicators = currentCandle.indicators;
    
    // Normalize features to 0-1 scale
    const features = [
      indicators.rsi / 100,
      Math.min(1, Math.max(0, (indicators.ema_12 - indicators.ema_26) / indicators.ema_26 + 0.5)),
      indicators.macd ? Math.min(1, Math.max(0, indicators.macd / 100 + 0.5)) : 0.5,
      indicators.volume_sma ? Math.min(1, currentCandle.volume / indicators.volume_sma / 2) : 0.5,
      indicators.bollinger_upper && indicators.bollinger_lower ? 
        (currentCandle.close - indicators.bollinger_lower) / (indicators.bollinger_upper - indicators.bollinger_lower) : 0.5,
    ];

    return features;
  }

  /**
   * Run model inference
   */
  private runModelInference(model: TrainedModel, features: number[]): number {
    // Simplified neural network inference
    const weights = Object.values(model.parameters.weights);
    
    let output = 0;
    features.forEach((feature, index) => {
      const weight = weights[index % weights.length] || 0.5;
      output += feature * weight;
    });
    
    // Apply sigmoid activation
    output = 1 / (1 + Math.exp(-output));
    
    return output;
  }

  /**
   * Convert prediction to signal
   */
  private predictionToSignal(prediction: number): 'BUY' | 'SELL' | 'HOLD' {
    if (prediction > 0.65) return 'BUY';
    if (prediction < 0.35) return 'SELL';
    return 'HOLD';
  }

  /**
   * Analyze consensus among model predictions
   */
  private analyzeModelConsensus(predictions: RetrainedModelPrediction[]): any {
    const buySignals = predictions.filter(p => p.signalType === 'BUY');
    const sellSignals = predictions.filter(p => p.signalType === 'SELL');
    const holdSignals = predictions.filter(p => p.signalType === 'HOLD');

    const totalModels = predictions.length;
    const buyWeight = buySignals.reduce((sum, p) => sum + this.getModelWeight(p.modelName), 0);
    const sellWeight = sellSignals.reduce((sum, p) => sum + this.getModelWeight(p.modelName), 0);

    let primarySignal: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let consensusScore = 0;

    if (buyWeight > sellWeight && buySignals.length >= totalModels * 0.5) {
      primarySignal = 'BUY';
      consensusScore = buyWeight;
    } else if (sellWeight > buyWeight && sellSignals.length >= totalModels * 0.5) {
      primarySignal = 'SELL';
      consensusScore = sellWeight;
    }

    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / totalModels;

    return {
      primarySignal,
      consensusScore,
      avgConfidence,
      modelBreakdown: {
        buy: buySignals.length,
        sell: sellSignals.length,
        hold: holdSignals.length,
      },
    };
  }

  /**
   * Generate final trading signal
   */
  private generateFinalSignal(
    consensus: any,
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): TradingSignal | null {
    
    if (consensus.primarySignal === 'HOLD') {
      return null;
    }

    const signal: TradingSignal = {
      id: createEventId(),
      timestamp: currentCandle.timestamp,
      symbol: this.config!.symbol,
      type: consensus.primarySignal,
      price: currentCandle.close,
      quantity: 0,
      confidence: consensus.avgConfidence,
      strategy: this.name,
      reason: `Retrained AI Consensus: ${consensus.consensusScore.toFixed(2)} (${consensus.modelBreakdown.buy}B/${consensus.modelBreakdown.sell}S/${consensus.modelBreakdown.hold}H)`,
    };

    return signal;
  }

  // Helper methods
  private getModelTypeFromName(modelName: string): string | null {
    if (modelName.toLowerCase().includes('transformer')) return 'transformer';
    if (modelName.toLowerCase().includes('lstm')) return 'lstm';
    if (modelName.toLowerCase().includes('smc')) return 'smc';
    return null;
  }

  private getModelWeight(modelName: string): number {
    const modelType = this.getModelTypeFromName(modelName);
    return modelType ? this.parameters.modelWeights[modelType] || 0.33 : 0.33;
  }

  /**
   * Get strategy description
   */
  public getDescription(): string {
    const modelCount = Object.keys(this.trainedModels).length;
    const avgAccuracy = Object.values(this.trainedModels).reduce((sum, model) => 
      sum + model.finalMetrics.testAccuracy, 0) / modelCount;
    
    return `Retrained AI Trading System using ${modelCount} models with ${(avgAccuracy * 100).toFixed(1)}% average accuracy`;
  }
}

// Export factory function
export function createRetrainedAITradingSystem(): RetrainedAITradingSystem {
  return new RetrainedAITradingSystem();
}

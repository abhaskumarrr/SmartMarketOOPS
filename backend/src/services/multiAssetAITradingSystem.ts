/**
 * Multi-Asset AI Trading System
 * Advanced trading system supporting multiple cryptocurrency pairs with cross-asset analysis
 */

import { 
  TradingStrategy,
  TradingSignal, 
  EnhancedMarketData, 
  BacktestConfig 
} from '../types/marketData';
import { TrainedModel } from './aiModelTrainer';
import { createMultiAssetDataProvider, CryptoPair } from './multiAssetDataProvider';
import { logger } from '../utils/logger';
import { createEventId } from '../types/events';
import fs from 'fs';
import path from 'path';

export interface MultiAssetPrediction {
  asset: CryptoPair;
  prediction: number;
  confidence: number;
  signalType: 'BUY' | 'SELL' | 'HOLD';
  modelConsensus: number;
  crossAssetScore: number;
  relativeStrength: number;
}

export interface PortfolioAllocation {
  btc: number;
  eth: number;
  sol: number;
  cash: number;
}

export interface MultiAssetSignal extends TradingSignal {
  targetAsset: CryptoPair;
  portfolioAllocation: PortfolioAllocation;
  crossAssetAnalysis: {
    correlations: { btc_eth: number; btc_sol: number; eth_sol: number };
    relativeStrengths: { btc: number; eth: number; sol: number };
    marketRegime: 'RISK_ON' | 'RISK_OFF' | 'NEUTRAL';
  };
}

export class MultiAssetAITradingSystem implements TradingStrategy {
  public readonly name = 'Multi_Asset_AI_Trading_System';
  public parameters: Record<string, any>;
  
  private config?: BacktestConfig;
  private dataProvider = createMultiAssetDataProvider();
  private trainedModels: { [modelName: string]: TrainedModel } = {};
  private lastDecisionTime: number = 0;
  private supportedAssets: CryptoPair[] = ['BTCUSD', 'ETHUSD', 'SOLUSD'];
  private currentPortfolio: PortfolioAllocation = { btc: 0, eth: 0, sol: 0, cash: 1 };

  constructor() {
    this.parameters = {
      // Multi-asset specific parameters
      enabledAssets: ['BTCUSD', 'ETHUSD', 'SOLUSD'],
      primaryAsset: 'BTCUSD',
      
      // Portfolio management
      maxPositionSize: 0.4, // Max 40% in any single asset
      minCashReserve: 0.1, // Keep 10% in cash
      rebalanceThreshold: 0.05, // Rebalance if allocation drifts >5%
      
      // Cross-asset analysis
      correlationThreshold: 0.7, // High correlation threshold
      relativeStrengthPeriod: 20, // 20 periods for RS calculation
      marketRegimeConfidence: 0.6, // Confidence for regime detection
      
      // Signal generation
      minConfidence: 70, // Higher confidence for multi-asset
      minCrossAssetScore: 0.6, // Minimum cross-asset agreement
      decisionCooldown: 600000, // 10 minutes between decisions
      
      // Model weights (asset-specific)
      modelWeights: {
        btc: { transformer: 0.4, lstm: 0.35, smc: 0.25 },
        eth: { transformer: 0.35, lstm: 0.4, smc: 0.25 },
        sol: { transformer: 0.3, lstm: 0.3, smc: 0.4 }, // SMC more important for alt-coins
      },
      
      // Risk management
      maxCorrelationExposure: 0.7, // Max exposure to correlated assets
      volatilityAdjustment: true,
      dynamicPositionSizing: true,
    };

    logger.info('🪙 Multi-Asset AI Trading System initialized', {
      supportedAssets: this.supportedAssets,
      portfolioManagement: 'enabled',
    });
  }

  /**
   * Initialize the multi-asset trading system
   */
  public async initialize(config: BacktestConfig): Promise<void> {
    this.config = config;
    this.lastDecisionTime = 0;
    
    // Load trained models for all assets
    await this.loadMultiAssetModels();
    
    // Initialize portfolio
    this.initializePortfolio();
    
    logger.info(`🎯 Initialized ${this.name} with multi-asset support`, {
      symbol: config.symbol,
      timeframe: config.timeframe,
      enabledAssets: this.parameters.enabledAssets,
      modelsLoaded: Object.keys(this.trainedModels).length,
    });
  }

  /**
   * Load trained models for multi-asset trading
   */
  private async loadMultiAssetModels(): Promise<void> {
    logger.info('📂 Loading multi-asset AI models...');

    const modelsDir = path.join(process.cwd(), 'trained_models');
    const modelTypes = ['transformer', 'lstm', 'smc'];

    for (const modelType of modelTypes) {
      try {
        const latestModelPath = path.join(modelsDir, `${modelType}_model_latest.json`);
        
        if (fs.existsSync(latestModelPath)) {
          const modelData = fs.readFileSync(latestModelPath, 'utf8');
          const trainedModel: TrainedModel = JSON.parse(modelData);
          
          if (this.validateModel(trainedModel)) {
            this.trainedModels[modelType] = trainedModel;
            logger.info(`✅ Loaded ${trainedModel.modelName} for multi-asset trading`, {
              testAccuracy: `${(trainedModel.finalMetrics.testAccuracy * 100).toFixed(1)}%`,
              version: trainedModel.version,
            });
          }
        } else {
          logger.warn(`⚠️ No trained model found for ${modelType}, using default`);
          this.trainedModels[modelType] = this.createDefaultModel(modelType);
        }
      } catch (error) {
        logger.error(`❌ Failed to load ${modelType} model:`, error);
        this.trainedModels[modelType] = this.createDefaultModel(modelType);
      }
    }

    const loadedCount = Object.keys(this.trainedModels).length;
    logger.info(`📊 Multi-asset model loading completed: ${loadedCount}/3 models loaded`);
  }

  /**
   * Initialize portfolio allocation
   */
  private initializePortfolio(): void {
    // Start with equal allocation across assets
    const assetCount = this.parameters.enabledAssets.length;
    const assetAllocation = (1 - this.parameters.minCashReserve) / assetCount;
    
    this.currentPortfolio = {
      btc: this.parameters.enabledAssets.includes('BTCUSD') ? assetAllocation : 0,
      eth: this.parameters.enabledAssets.includes('ETHUSD') ? assetAllocation : 0,
      sol: this.parameters.enabledAssets.includes('SOLUSD') ? assetAllocation : 0,
      cash: this.parameters.minCashReserve,
    };

    logger.info('💼 Portfolio initialized', { allocation: this.currentPortfolio });
  }

  /**
   * Generate multi-asset trading signal
   */
  public generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null {
    if (!this.config) {
      throw new Error('Strategy not initialized. Call initialize() first.');
    }

    if (currentIndex < 100) {
      return null;
    }

    const currentTime = Date.now();
    if (currentTime - this.lastDecisionTime < this.parameters.decisionCooldown) {
      return null;
    }

    try {
      // Analyze all assets
      const multiAssetPredictions = this.analyzeAllAssets(data, currentIndex);
      
      if (multiAssetPredictions.length === 0) {
        return null;
      }

      // Perform cross-asset analysis
      const crossAssetAnalysis = this.performCrossAssetAnalysis(multiAssetPredictions, data, currentIndex);
      
      // Determine optimal portfolio allocation
      const optimalAllocation = this.calculateOptimalAllocation(multiAssetPredictions, crossAssetAnalysis);
      
      // Generate portfolio rebalancing signal
      const signal = this.generatePortfolioSignal(
        multiAssetPredictions,
        crossAssetAnalysis,
        optimalAllocation,
        data[currentIndex]
      );
      
      if (signal && signal.confidence >= this.parameters.minConfidence) {
        this.lastDecisionTime = currentTime;
        this.updatePortfolioAllocation(optimalAllocation);
        
        logger.info(`🪙 Generated multi-asset ${signal.type} signal`, {
          targetAsset: (signal as MultiAssetSignal).targetAsset,
          confidence: signal.confidence,
          portfolioAllocation: (signal as MultiAssetSignal).portfolioAllocation,
          marketRegime: (signal as MultiAssetSignal).crossAssetAnalysis.marketRegime,
        });
        
        return signal;
      }

      return null;

    } catch (error) {
      logger.error('❌ Error generating multi-asset signal:', error);
      return null;
    }
  }

  /**
   * Analyze all supported assets
   */
  private analyzeAllAssets(data: EnhancedMarketData[], currentIndex: number): MultiAssetPrediction[] {
    const predictions: MultiAssetPrediction[] = [];
    
    // For this implementation, we'll simulate multi-asset analysis
    // In a real system, this would fetch data for all assets
    
    this.parameters.enabledAssets.forEach((asset: CryptoPair) => {
      const prediction = this.analyzeAsset(asset, data, currentIndex);
      if (prediction) {
        predictions.push(prediction);
      }
    });

    return predictions;
  }

  /**
   * Analyze a specific asset
   */
  private analyzeAsset(asset: CryptoPair, data: EnhancedMarketData[], currentIndex: number): MultiAssetPrediction | null {
    const currentCandle = data[currentIndex];
    const indicators = currentCandle.indicators;
    
    if (!indicators.rsi || !indicators.ema_12 || !indicators.ema_26) {
      return null;
    }

    // Get model predictions for this asset
    const modelPredictions = this.getAssetModelPredictions(asset, currentCandle, data, currentIndex);
    
    if (modelPredictions.length === 0) {
      return null;
    }

    // Calculate model consensus
    const avgPrediction = modelPredictions.reduce((sum, p) => sum + p.prediction, 0) / modelPredictions.length;
    const avgConfidence = modelPredictions.reduce((sum, p) => sum + p.confidence, 0) / modelPredictions.length;
    
    // Determine signal type
    const signalType = this.predictionToSignal(avgPrediction);
    
    // Calculate cross-asset score (simplified)
    const crossAssetScore = this.calculateCrossAssetScore(asset, data, currentIndex);
    
    // Calculate relative strength
    const relativeStrength = this.calculateAssetRelativeStrength(asset, data, currentIndex);
    
    // Calculate model consensus score
    const modelConsensus = this.calculateModelConsensus(modelPredictions);

    return {
      asset,
      prediction: avgPrediction,
      confidence: avgConfidence,
      signalType,
      modelConsensus,
      crossAssetScore,
      relativeStrength,
    };
  }

  /**
   * Get model predictions for a specific asset
   */
  private getAssetModelPredictions(
    asset: CryptoPair,
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): any[] {
    
    const predictions: any[] = [];
    const assetWeights = this.parameters.modelWeights[asset.substring(0, 3).toLowerCase()] || 
                        this.parameters.modelWeights.btc;

    Object.entries(this.trainedModels).forEach(([modelType, model]) => {
      try {
        const features = this.extractAssetFeatures(asset, currentCandle, data, currentIndex);
        const rawPrediction = this.runModelInference(model, features);
        const weight = assetWeights[modelType as keyof typeof assetWeights] || 0.33;
        
        predictions.push({
          modelType,
          prediction: rawPrediction,
          confidence: model.finalMetrics.testAccuracy * 100,
          weight,
        });
      } catch (error) {
        logger.warn(`⚠️ Failed to get ${modelType} prediction for ${asset}:`, error);
      }
    });

    return predictions;
  }

  /**
   * Extract features for a specific asset
   */
  private extractAssetFeatures(
    asset: CryptoPair,
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): number[] {
    
    const indicators = currentCandle.indicators;
    const config = this.dataProvider.getAssetConfig(asset);
    
    // Base features
    const features = [
      indicators.rsi / 100,
      Math.min(1, Math.max(0, (indicators.ema_12 - indicators.ema_26) / indicators.ema_26 + 0.5)),
      indicators.macd ? Math.min(1, Math.max(0, indicators.macd / 100 + 0.5)) : 0.5,
      indicators.volume_sma ? Math.min(1, currentCandle.volume / indicators.volume_sma / 2) : 0.5,
    ];
    
    // Asset-specific adjustments
    if (config) {
      // Add volatility profile adjustment
      const volatilityMultiplier = config.volatilityProfile === 'high' ? 1.2 : 
                                  config.volatilityProfile === 'low' ? 0.8 : 1.0;
      features.push(volatilityMultiplier);
      
      // Add category behavior
      const categoryScore = config.category === 'large-cap' ? 0.8 : 0.3;
      features.push(categoryScore);
    }

    return features;
  }

  /**
   * Run model inference
   */
  private runModelInference(model: TrainedModel, features: number[]): number {
    const weights = Object.values(model.parameters.weights);
    
    let output = 0;
    features.forEach((feature, index) => {
      const weight = weights[index % weights.length] || 0.5;
      output += feature * weight;
    });
    
    return 1 / (1 + Math.exp(-output));
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
   * Calculate cross-asset score
   */
  private calculateCrossAssetScore(asset: CryptoPair, data: EnhancedMarketData[], currentIndex: number): number {
    // Simplified cross-asset score based on market conditions
    const indicators = data[currentIndex].indicators;
    
    // Base score on technical indicators
    let score = 0.5;
    
    if (indicators.rsi) {
      if (indicators.rsi > 30 && indicators.rsi < 70) score += 0.2;
    }
    
    if (indicators.volume_sma) {
      if (data[currentIndex].volume > indicators.volume_sma) score += 0.1;
    }
    
    return Math.min(1, score);
  }

  /**
   * Calculate asset relative strength
   */
  private calculateAssetRelativeStrength(asset: CryptoPair, data: EnhancedMarketData[], currentIndex: number): number {
    if (currentIndex < this.parameters.relativeStrengthPeriod) return 0.5;
    
    const period = this.parameters.relativeStrengthPeriod;
    const currentPrice = data[currentIndex].close;
    const pastPrice = data[currentIndex - period].close;
    
    const assetReturn = (currentPrice - pastPrice) / pastPrice;
    
    // For simplification, compare against a benchmark (assume 0% return)
    return 0.5 + assetReturn;
  }

  /**
   * Calculate model consensus
   */
  private calculateModelConsensus(predictions: any[]): number {
    if (predictions.length === 0) return 0;
    
    const avgPrediction = predictions.reduce((sum, p) => sum + p.prediction, 0) / predictions.length;
    const variance = predictions.reduce((sum, p) => sum + Math.pow(p.prediction - avgPrediction, 2), 0) / predictions.length;
    
    // Higher consensus = lower variance
    return Math.max(0, 1 - variance * 4);
  }

  /**
   * Perform cross-asset analysis
   */
  private performCrossAssetAnalysis(predictions: MultiAssetPrediction[], data: EnhancedMarketData[], currentIndex: number): any {
    // Calculate correlations (simplified)
    const correlations = {
      btc_eth: 0.7 + (Math.random() - 0.5) * 0.4,
      btc_sol: 0.6 + (Math.random() - 0.5) * 0.4,
      eth_sol: 0.5 + (Math.random() - 0.5) * 0.4,
    };
    
    // Calculate relative strengths
    const relativeStrengths = {
      btc: predictions.find(p => p.asset === 'BTCUSD')?.relativeStrength || 0.5,
      eth: predictions.find(p => p.asset === 'ETHUSD')?.relativeStrength || 0.5,
      sol: predictions.find(p => p.asset === 'SOLUSD')?.relativeStrength || 0.5,
    };
    
    // Determine market regime
    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;
    const avgCrossAssetScore = predictions.reduce((sum, p) => sum + p.crossAssetScore, 0) / predictions.length;
    
    let marketRegime: 'RISK_ON' | 'RISK_OFF' | 'NEUTRAL' = 'NEUTRAL';
    
    if (avgConfidence > 75 && avgCrossAssetScore > 0.7) {
      marketRegime = 'RISK_ON';
    } else if (avgConfidence < 50 || avgCrossAssetScore < 0.4) {
      marketRegime = 'RISK_OFF';
    }

    return {
      correlations,
      relativeStrengths,
      marketRegime,
      avgConfidence,
      avgCrossAssetScore,
    };
  }

  /**
   * Calculate optimal portfolio allocation
   */
  private calculateOptimalAllocation(predictions: MultiAssetPrediction[], crossAssetAnalysis: any): PortfolioAllocation {
    const allocation: PortfolioAllocation = { btc: 0, eth: 0, sol: 0, cash: this.parameters.minCashReserve };
    const availableCapital = 1 - this.parameters.minCashReserve;
    
    // Calculate scores for each asset
    const assetScores: { [key: string]: number } = {};
    
    predictions.forEach(prediction => {
      const assetKey = prediction.asset.substring(0, 3).toLowerCase();
      const score = (prediction.confidence / 100) * prediction.modelConsensus * prediction.crossAssetScore;
      assetScores[assetKey] = score;
    });
    
    // Normalize scores
    const totalScore = Object.values(assetScores).reduce((sum, score) => sum + score, 0);
    
    if (totalScore > 0) {
      Object.entries(assetScores).forEach(([asset, score]) => {
        const baseAllocation = (score / totalScore) * availableCapital;
        const maxAllocation = Math.min(baseAllocation, this.parameters.maxPositionSize);
        
        allocation[asset as keyof PortfolioAllocation] = maxAllocation;
      });
    } else {
      // Equal allocation if no clear signals
      const equalAllocation = availableCapital / 3;
      allocation.btc = equalAllocation;
      allocation.eth = equalAllocation;
      allocation.sol = equalAllocation;
    }
    
    // Adjust for correlation exposure
    this.adjustForCorrelationRisk(allocation, crossAssetAnalysis.correlations);
    
    return allocation;
  }

  /**
   * Adjust allocation for correlation risk
   */
  private adjustForCorrelationRisk(allocation: PortfolioAllocation, correlations: any): void {
    // If assets are highly correlated, reduce exposure
    const highCorrelationThreshold = this.parameters.correlationThreshold;
    
    if (correlations.btc_eth > highCorrelationThreshold) {
      const reduction = 0.1;
      allocation.btc = Math.max(0, allocation.btc - reduction);
      allocation.eth = Math.max(0, allocation.eth - reduction);
      allocation.cash += reduction * 2;
    }
    
    if (correlations.btc_sol > highCorrelationThreshold) {
      const reduction = 0.05;
      allocation.btc = Math.max(0, allocation.btc - reduction);
      allocation.sol = Math.max(0, allocation.sol - reduction);
      allocation.cash += reduction * 2;
    }
  }

  /**
   * Generate portfolio rebalancing signal
   */
  private generatePortfolioSignal(
    predictions: MultiAssetPrediction[],
    crossAssetAnalysis: any,
    optimalAllocation: PortfolioAllocation,
    currentCandle: EnhancedMarketData
  ): MultiAssetSignal | null {
    
    // Find the strongest signal
    const strongestPrediction = predictions.reduce((strongest, current) => 
      current.confidence > strongest.confidence ? current : strongest
    );
    
    if (strongestPrediction.signalType === 'HOLD') {
      return null;
    }

    // Check if rebalancing is needed
    const rebalanceNeeded = this.isRebalanceNeeded(optimalAllocation);
    
    if (!rebalanceNeeded && strongestPrediction.confidence < this.parameters.minConfidence) {
      return null;
    }

    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;

    const signal: MultiAssetSignal = {
      id: createEventId(),
      timestamp: currentCandle.timestamp,
      symbol: this.config!.symbol,
      type: strongestPrediction.signalType,
      price: currentCandle.close,
      quantity: 0,
      confidence: avgConfidence,
      strategy: this.name,
      reason: `Multi-Asset: ${crossAssetAnalysis.marketRegime} regime, ${predictions.length} assets analyzed`,
      targetAsset: strongestPrediction.asset,
      portfolioAllocation: optimalAllocation,
      crossAssetAnalysis,
    };

    return signal;
  }

  /**
   * Check if portfolio rebalancing is needed
   */
  private isRebalanceNeeded(optimalAllocation: PortfolioAllocation): boolean {
    const threshold = this.parameters.rebalanceThreshold;
    
    return Math.abs(this.currentPortfolio.btc - optimalAllocation.btc) > threshold ||
           Math.abs(this.currentPortfolio.eth - optimalAllocation.eth) > threshold ||
           Math.abs(this.currentPortfolio.sol - optimalAllocation.sol) > threshold;
  }

  /**
   * Update portfolio allocation
   */
  private updatePortfolioAllocation(newAllocation: PortfolioAllocation): void {
    this.currentPortfolio = { ...newAllocation };
  }

  // Helper methods
  private validateModel(model: TrainedModel): boolean {
    return model.finalMetrics.testAccuracy >= 0.6;
  }

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
   * Get strategy description
   */
  public getDescription(): string {
    return `Multi-Asset AI Trading System supporting ${this.supportedAssets.join(', ')} with cross-asset analysis and portfolio optimization`;
  }
}

// Export factory function
export function createMultiAssetAITradingSystem(): MultiAssetAITradingSystem {
  return new MultiAssetAITradingSystem();
}

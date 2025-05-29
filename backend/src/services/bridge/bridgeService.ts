/**
 * Bridge Service
 * Integrates ML and Trading systems
 */

import { v4 as uuidv4 } from 'uuid';
import { createLogger, LogData } from '../../utils/logger';
import mlBridgeService from './mlBridgeService';
import signalGenerationService from '../trading/signalGenerationService';
import strategyExecutionService from '../trading/strategyExecutionService';
import riskManagementService from '../trading/riskManagementService';
import prisma from '../../utils/prismaClient';
import {
  PredictionInput,
  PredictionOutput,
  SignalGenerationRequest,
  BacktestRequest,
  BridgeHealth,
  TrainingRequest,
  ModelStatus
} from '../../types/bridge';
import { TradingSignal, SignalDirection, SignalStrength, SignalTimeframe, SignalType } from '../../types/signals';

// Create logger
const logger = createLogger('BridgeService');

/**
 * Bridge Service class
 * Connects ML and Trading systems
 */
class BridgeService {
  private healthStatus: BridgeHealth;
  private latencyHistory: {
    prediction: number[];
    signalGeneration: number[];
    endToEnd: number[];
  };
  
  /**
   * Creates a new Bridge Service instance
   */
  constructor() {
    this.healthStatus = {
      status: 'HEALTHY',
      mlSystemStatus: 'AVAILABLE',
      tradingSystemStatus: 'AVAILABLE',
      latency: {
        predictionAvg: 0,
        signalGenerationAvg: 0,
        endToEndAvg: 0
      },
      lastSyncedAt: new Date().toISOString(),
      activeModels: 0,
      activeStrategies: 0,
      errors: [],
      metrics: {
        predictionRequests1h: 0,
        signalGenerationRequests1h: 0,
        successRate: 100
      }
    };
    
    this.latencyHistory = {
      prediction: [],
      signalGeneration: [],
      endToEnd: []
    };
    
    // Initialize health monitoring
    this.startHealthMonitoring();
    
    logger.info('Bridge Service initialized');
  }
  
  /**
   * Get model prediction and generate trading signal
   * @param symbol - Trading symbol
   * @param timeframe - Timeframe
   * @param options - Additional options
   * @returns Generated trading signal
   */
  async getPredictionAndGenerateSignal(
    symbol: string,
    timeframe: string,
    options?: {
      modelVersion?: string;
      confidenceThreshold?: number;
      signalExpiry?: number;
    }
  ): Promise<TradingSignal> {
    try {
      const startTime = Date.now();
      logger.info(`Getting prediction and generating signal for ${symbol} on ${timeframe}`);
      
      // Get prediction from ML system
      const predictionStart = Date.now();
      const predictionInput: PredictionInput = {
        symbol,
        timeframe,
        modelVersion: options?.modelVersion
      };
      
      const prediction = await mlBridgeService.getPrediction(predictionInput);
      const predictionLatency = Date.now() - predictionStart;
      this.latencyHistory.prediction.push(predictionLatency);
      
      // Generate trading signal based on prediction
      const signalStart = Date.now();
      const signal = await this.convertPredictionToSignal(prediction, options);
      const signalLatency = Date.now() - signalStart;
      this.latencyHistory.signalGeneration.push(signalLatency);
      
      // Save signal to database
      await signalGenerationService.saveSignal(signal);
      
      // Update latency metrics
      const endToEndLatency = Date.now() - startTime;
      this.latencyHistory.endToEnd.push(endToEndLatency);
      this.updateLatencyAverages();
      
      // Update metrics
      this.healthStatus.metrics.predictionRequests1h++;
      this.healthStatus.metrics.signalGenerationRequests1h++;
      
      return signal;
    } catch (error) {
      const logData: LogData = {
        symbol,
        timeframe,
        options,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error getting prediction and generating signal for ${symbol}`, logData);
      
      // Update error metrics
      this.recordError('PREDICTION_AND_SIGNAL', error instanceof Error ? error.message : String(error));
      
      throw error;
    }
  }
  
  /**
   * Convert ML prediction to trading signal
   * @param prediction - ML prediction
   * @param options - Additional options
   * @returns Trading signal
   */
  private async convertPredictionToSignal(
    prediction: PredictionOutput,
    options?: {
      confidenceThreshold?: number;
      signalExpiry?: number;
    }
  ): Promise<TradingSignal> {
    try {
      logger.debug(`Converting prediction to signal for ${prediction.symbol}`);
      
      // Extract prediction data
      const { symbol, timeframe, values, timestamps, confidenceScores, metadata } = prediction;
      
      // Default expiry is 24 hours
      const expiryMinutes = options?.signalExpiry || 24 * 60;
      const expiresAt = new Date(new Date().getTime() + expiryMinutes * 60 * 1000);
      
      // Get latest prediction value and confidence
      const latestValue = values[values.length - 1];
      const latestConfidence = confidenceScores[confidenceScores.length - 1];
      const confidenceThreshold = options?.confidenceThreshold || 70;
      
      // Determine signal type and direction
      let type: SignalType = SignalType.HOLD;
      let direction: SignalDirection = SignalDirection.NEUTRAL;
      
      if (prediction.predictionType === 'DIRECTION') {
        // Direction prediction (1 = long, 0 = neutral, -1 = short)
        if (latestValue > 0.5 && latestConfidence >= confidenceThreshold) {
          type = SignalType.ENTRY;
          direction = SignalDirection.LONG;
        } else if (latestValue < -0.5 && latestConfidence >= confidenceThreshold) {
          type = SignalType.ENTRY;
          direction = SignalDirection.SHORT;
        } else {
          type = SignalType.HOLD;
          direction = SignalDirection.NEUTRAL;
        }
      } else if (prediction.predictionType === 'PRICE') {
        // Price prediction (compare with current price)
        // Get current price from market data service
        const currentPrice = await this.getCurrentPrice(symbol);
        
        if (latestValue > currentPrice * 1.01 && latestConfidence >= confidenceThreshold) {
          type = SignalType.ENTRY;
          direction = SignalDirection.LONG;
        } else if (latestValue < currentPrice * 0.99 && latestConfidence >= confidenceThreshold) {
          type = SignalType.ENTRY;
          direction = SignalDirection.SHORT;
        } else {
          type = SignalType.HOLD;
          direction = SignalDirection.NEUTRAL;
        }
      } else if (prediction.predictionType === 'PROBABILITY') {
        // Probability prediction (probability of up/down movement)
        if (latestValue > 0.6 && latestConfidence >= confidenceThreshold) {
          type = SignalType.ENTRY;
          direction = SignalDirection.LONG;
        } else if (latestValue < 0.4 && latestConfidence >= confidenceThreshold) {
          type = SignalType.ENTRY;
          direction = SignalDirection.SHORT;
        } else {
          type = SignalType.HOLD;
          direction = SignalDirection.NEUTRAL;
        }
      }
      
      // Determine signal strength based on confidence
      let strength: SignalStrength;
      if (latestConfidence >= 90) {
        strength = SignalStrength.VERY_STRONG;
      } else if (latestConfidence >= 80) {
        strength = SignalStrength.STRONG;
      } else if (latestConfidence >= 70) {
        strength = SignalStrength.MODERATE;
      } else if (latestConfidence >= 60) {
        strength = SignalStrength.WEAK;
      } else {
        strength = SignalStrength.VERY_WEAK;
      }
      
      // Determine timeframe category
      let signalTimeframe: SignalTimeframe;
      switch (timeframe.toLowerCase()) {
        case '1m':
        case '5m':
        case '15m':
          signalTimeframe = SignalTimeframe.VERY_SHORT;
          break;
        case '30m':
        case '1h':
        case '2h':
          signalTimeframe = SignalTimeframe.SHORT;
          break;
        case '4h':
        case '6h':
        case '8h':
          signalTimeframe = SignalTimeframe.MEDIUM;
          break;
        case '12h':
        case '1d':
          signalTimeframe = SignalTimeframe.LONG;
          break;
        case '3d':
        case '1w':
        case '1M':
          signalTimeframe = SignalTimeframe.VERY_LONG;
          break;
        default:
          signalTimeframe = SignalTimeframe.MEDIUM;
      }
      
      // Calculate risk-reward metrics
      const expectedReturn = latestConfidence / 100 * 3; // Simplified calculation
      const expectedRisk = (100 - latestConfidence) / 100 * 1.5; // Simplified calculation
      const riskRewardRatio = expectedReturn / expectedRisk;
      
      // Create signal object
      const signal: TradingSignal = {
        id: uuidv4(),
        symbol,
        type,
        direction,
        strength,
        timeframe: signalTimeframe,
        price: await this.getCurrentPrice(symbol),
        targetPrice: direction === SignalDirection.LONG ? latestValue : latestValue,
        stopLoss: this.calculateStopLoss(await this.getCurrentPrice(symbol), direction),
        confidenceScore: Math.round(latestConfidence),
        expectedReturn,
        expectedRisk,
        riskRewardRatio,
        generatedAt: new Date().toISOString(),
        expiresAt: expiresAt.toISOString(),
        source: `ML:${metadata.modelName}:${metadata.modelVersion}`,
        metadata: {
          predictionId: prediction.id,
          modelVersion: metadata.modelVersion,
          modelName: metadata.modelName,
          modelPerformance: metadata.performance,
          timeframe: timeframe,
          originalTimestamps: timestamps
        },
        predictionValues: {
          raw: values,
          timestamps,
          confidences: confidenceScores
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      return signal;
    } catch (error) {
      const logData: LogData = {
        predictionId: prediction.id,
        symbol: prediction.symbol,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error converting prediction to signal for ${prediction.symbol}`, logData);
      throw error;
    }
  }
  
  /**
   * Temporary method to get current price
   * This should be replaced with a proper market data service
   * @param symbol - Symbol
   * @returns Current price
   */
  private async getCurrentPrice(symbol: string): Promise<number> {
    // This should be replaced with a call to the market data service
    // For now, return a random price as a placeholder
    const basePrice = symbol.includes('BTC') ? 50000 : 
                     symbol.includes('ETH') ? 2000 : 
                     symbol.includes('SOL') ? 100 : 
                     symbol.includes('XRP') ? 0.5 : 
                     symbol.includes('ADA') ? 0.4 : 
                     symbol.includes('DOGE') ? 0.1 : 100;
    
    return basePrice * (0.95 + Math.random() * 0.1); // +/- 5% variation
  }
  
  /**
   * Calculate stop loss based on current price and direction
   * @param currentPrice - Current price
   * @param direction - Signal direction
   * @returns Stop loss price
   */
  private calculateStopLoss(currentPrice: number, direction: SignalDirection): number {
    // Simple calculation: 2% from entry for long, 2% above for short
    if (direction === SignalDirection.LONG) {
      return currentPrice * 0.98;
    } else if (direction === SignalDirection.SHORT) {
      return currentPrice * 1.02;
    }
    return currentPrice;
  }
  
  /**
   * Run backtest using ML predictions
   * @param request - Backtest request
   * @returns Backtest result
   */
  async runBacktest(request: BacktestRequest): Promise<any> {
    try {
      logger.info(`Running backtest for strategy ${request.strategyId} on ${request.symbol}`);
      
      // Get strategy configuration
      const strategy = await strategyExecutionService.getStrategy(request.strategyId);
      
      // Get historical predictions
      // This is a placeholder - we would need to implement a way to get historical predictions
      // or generate them from historical data
      
      // Convert backtest request to strategy backtest format
      const strategyBacktestConfig = {
        strategyId: request.strategyId,
        startDate: request.startDate,
        endDate: request.endDate,
        initialCapital: request.initialCapital,
        symbols: [request.symbol],
        includeFees: request.options?.includeFees || false,
        feePercentage: request.options?.feePercentage || 0.1,
        includeSlippage: request.options?.includeSlippage || false,
        slippagePercentage: request.options?.slippagePercentage || 0.05
      };
      
      // This would be implemented in the strategy execution service
      // For now, return a placeholder result
      return {
        id: uuidv4(),
        strategyId: request.strategyId,
        config: strategyBacktestConfig,
        performance: {
          totalPnL: 1250.75,
          totalPnLPercentage: 12.5,
          winRate: 0.65,
          totalTrades: 48,
          successfulTrades: 31,
          failedTrades: 17,
          averageHoldingTime: 12.5, // hours
          maxDrawdown: 8.2,
          sharpeRatio: 1.8,
          sortinoRatio: 2.2
        },
        trades: [],
        equityCurve: [],
        createdAt: new Date().toISOString()
      };
    } catch (error) {
      const logData: LogData = {
        request,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error running backtest for strategy ${request.strategyId}`, logData);
      
      // Update error metrics
      this.recordError('BACKTEST', error instanceof Error ? error.message : String(error));
      
      throw error;
    }
  }
  
  /**
   * Get system health status
   * @returns Health status
   */
  async getHealth(): Promise<BridgeHealth> {
    try {
      logger.debug('Getting bridge health status');
      
      // Update last synced timestamp
      this.healthStatus.lastSyncedAt = new Date().toISOString();
      
      return this.healthStatus;
    } catch (error) {
      logger.error('Error getting bridge health status', {
        error: error instanceof Error ? error.message : String(error)
      });
      
      // Return a degraded status
      return {
        ...this.healthStatus,
        status: 'DEGRADED',
        errors: [
          ...this.healthStatus.errors,
          {
            timestamp: new Date().toISOString(),
            component: 'BRIDGE',
            message: error instanceof Error ? error.message : String(error),
            count: 1
          }
        ]
      };
    }
  }
  
  /**
   * Start model training
   * @param request - Training request
   * @returns Training status
   */
  async startModelTraining(request: TrainingRequest): Promise<any> {
    try {
      logger.info(`Starting model training for ${request.symbol} on ${request.timeframe}`);
      
      // Start training using ML bridge service
      const trainingStatus = await mlBridgeService.startTraining(request);
      
      return trainingStatus;
    } catch (error) {
      const logData: LogData = {
        request,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error starting model training for ${request.symbol}`, logData);
      
      // Update error metrics
      this.recordError('TRAINING', error instanceof Error ? error.message : String(error));
      
      throw error;
    }
  }
  
  /**
   * Get available models
   * @returns Array of model status
   */
  async getAvailableModels(): Promise<ModelStatus[]> {
    try {
      logger.info('Getting available ML models');
      
      // Get models from ML bridge service
      const models = await mlBridgeService.getAllModels();
      
      // Update active models count
      this.healthStatus.activeModels = models.filter(m => m.status === 'ACTIVE').length;
      
      return models;
    } catch (error) {
      const logData: LogData = {
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error('Error getting available ML models', logData);
      
      // Update error metrics
      this.recordError('MODEL_LIST', error instanceof Error ? error.message : String(error));
      
      throw error;
    }
  }
  
  /**
   * Start health monitoring
   * @private
   */
  private startHealthMonitoring(): void {
    // Check ML system health every 5 minutes
    setInterval(async () => {
      try {
        const mlHealth = await mlBridgeService.checkConnection();
        this.healthStatus.mlSystemStatus = mlHealth.status === 'error' ? 'UNAVAILABLE' : 'AVAILABLE';
        
        // Check trading system health (simplified)
        try {
          await signalGenerationService.getRecentSignals(1);
          this.healthStatus.tradingSystemStatus = 'AVAILABLE';
        } catch (e) {
          this.healthStatus.tradingSystemStatus = 'DEGRADED';
        }
        
        // Update overall status
        if (this.healthStatus.mlSystemStatus === 'UNAVAILABLE' && this.healthStatus.tradingSystemStatus === 'UNAVAILABLE') {
          this.healthStatus.status = 'UNAVAILABLE';
        } else if (this.healthStatus.mlSystemStatus === 'DEGRADED' || this.healthStatus.tradingSystemStatus === 'DEGRADED' ||
                  this.healthStatus.mlSystemStatus === 'UNAVAILABLE' || this.healthStatus.tradingSystemStatus === 'UNAVAILABLE') {
          this.healthStatus.status = 'DEGRADED';
        } else {
          this.healthStatus.status = 'HEALTHY';
        }
        
        // Clean up old errors (keep only last 24 hours)
        const oneDayAgo = new Date();
        oneDayAgo.setDate(oneDayAgo.getDate() - 1);
        this.healthStatus.errors = this.healthStatus.errors.filter(
          error => new Date(error.timestamp) > oneDayAgo
        );
        
        // Get active strategies count
        const activeStrategies = await prisma.tradingStrategy.count({
          where: { isActive: true }
        });
        this.healthStatus.activeStrategies = activeStrategies;
        
        // Reset hourly metrics every hour
        const now = new Date();
        if (now.getMinutes() === 0) {
          this.healthStatus.metrics.predictionRequests1h = 0;
          this.healthStatus.metrics.signalGenerationRequests1h = 0;
        }
        
        // Calculate success rate
        const totalRequests = this.healthStatus.metrics.predictionRequests1h;
        const errorCount = this.healthStatus.errors
          .filter(e => new Date(e.timestamp) > new Date(Date.now() - 60 * 60 * 1000))
          .reduce((sum, e) => sum + e.count, 0);
          
        this.healthStatus.metrics.successRate = totalRequests > 0 
          ? Math.round(((totalRequests - errorCount) / totalRequests) * 100) 
          : 100;
      } catch (error) {
        logger.error('Error in health monitoring', {
          error: error instanceof Error ? error.message : String(error)
        });
        
        this.healthStatus.status = 'DEGRADED';
      }
    }, 5 * 60 * 1000); // Every 5 minutes
  }
  
  /**
   * Update latency averages
   * @private
   */
  private updateLatencyAverages(): void {
    // Keep only the last 100 entries
    if (this.latencyHistory.prediction.length > 100) {
      this.latencyHistory.prediction = this.latencyHistory.prediction.slice(-100);
    }
    
    if (this.latencyHistory.signalGeneration.length > 100) {
      this.latencyHistory.signalGeneration = this.latencyHistory.signalGeneration.slice(-100);
    }
    
    if (this.latencyHistory.endToEnd.length > 100) {
      this.latencyHistory.endToEnd = this.latencyHistory.endToEnd.slice(-100);
    }
    
    // Calculate averages
    this.healthStatus.latency.predictionAvg = this.calculateAverage(this.latencyHistory.prediction);
    this.healthStatus.latency.signalGenerationAvg = this.calculateAverage(this.latencyHistory.signalGeneration);
    this.healthStatus.latency.endToEndAvg = this.calculateAverage(this.latencyHistory.endToEnd);
  }
  
  /**
   * Calculate average of an array
   * @private
   * @param arr - Array of numbers
   * @returns Average
   */
  private calculateAverage(arr: number[]): number {
    if (arr.length === 0) return 0;
    return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);
  }
  
  /**
   * Record error in health status
   * @private
   * @param component - Component name
   * @param message - Error message
   */
  private recordError(component: string, message: string): void {
    // Check if error already exists
    const existingError = this.healthStatus.errors.find(
      e => e.component === component && e.message === message
    );
    
    if (existingError) {
      existingError.count++;
      existingError.timestamp = new Date().toISOString();
    } else {
      this.healthStatus.errors.push({
        timestamp: new Date().toISOString(),
        component,
        message,
        count: 1
      });
    }
    
    // Update overall status
    this.healthStatus.status = 'DEGRADED';
  }
}

// Create singleton instance
const bridgeService = new BridgeService();

export default bridgeService; 
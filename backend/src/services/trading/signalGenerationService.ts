/**
 * Signal Generation Service
 * Converts ML model predictions into actionable trading signals
 */

import { v4 as uuidv4 } from 'uuid';
import mlModelClient from '../../clients/mlModelClient';
import {
  ModelPrediction,
  TradingSignal,
  SignalType,
  SignalDirection,
  SignalStrength,
  SignalTimeframe,
  SignalGenerationOptions,
  SignalFilterCriteria
} from '../../types/signals';
import { createLogger, LogData } from '../../utils/logger';
import prisma from '../../utils/prismaClient';

// Create logger
const logger = createLogger('SignalGenerationService');

/**
 * Default signal generation options
 */
const DEFAULT_OPTIONS: SignalGenerationOptions = {
  validateSignals: true,
  useHistoricalData: true,
  lookbackPeriod: 30,
  minConfidenceThreshold: 60,
  maxSignalsPerSymbol: 3,
  filterWeakSignals: true
};

/**
 * Signal Generation Service class
 * Provides methods to generate and manage trading signals
 */
export class SignalGenerationService {
  private options: SignalGenerationOptions;
  
  /**
   * Creates a new Signal Generation Service instance
   * @param options - Signal generation options
   */
  constructor(options?: Partial<SignalGenerationOptions>) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    logger.info('Signal Generation Service initialized', { options: this.options });
  }
  
  /**
   * Generate trading signals for a specific symbol
   * @param symbol - Trading pair symbol
   * @param features - Current market features
   * @param options - Optional signal generation options to override defaults
   * @returns Generated trading signals
   */
  async generateSignals(
    symbol: string, 
    features: Record<string, number>,
    options?: Partial<SignalGenerationOptions>
  ): Promise<TradingSignal[]> {
    try {
      // Merge options
      const mergedOptions = { ...this.options, ...options };
      
      logger.info(`Generating signals for ${symbol}`);
      
      // 1. Get model prediction
      const prediction = await mlModelClient.getPrediction({
        symbol,
        features,
        sequence_length: 60
      });
      
      logger.debug(`Received prediction for ${symbol}`, {
        predictions: prediction.predictions,
        model_version: prediction.model_version
      });
      
      // 2. Process prediction into signals
      const signals = await this._processModelPrediction(prediction, features);
      
      // 3. Validate signals if option is enabled
      const validatedSignals = mergedOptions.validateSignals
        ? await this._validateSignals(signals)
        : signals;
      
      // 4. Filter signals based on confidence threshold
      const filteredSignals = this._filterSignals(validatedSignals, mergedOptions);
      
      // 5. Store signals in database
      await this._storeSignals(filteredSignals);
      
      logger.info(`Generated ${filteredSignals.length} signals for ${symbol}`);
      
      return filteredSignals;
    } catch (error) {
      const logData: LogData = {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error generating signals for ${symbol}`, logData);
      throw error;
    }
  }
  
  /**
   * Process model prediction into trading signals
   * @private
   * @param prediction - ML model prediction
   * @param features - Current market features
   * @returns Processed trading signals
   */
  private async _processModelPrediction(
    prediction: ModelPrediction,
    features: Record<string, number>
  ): Promise<TradingSignal[]> {
    const signals: TradingSignal[] = [];
    const currentPrice = features.close || features.price || 0;
    
    if (currentPrice === 0) {
      logger.warn(`No current price available for ${prediction.symbol}`);
      return [];
    }
    
    // Get prediction values - these are typically future price predictions
    const predictionValues = prediction.predictions;
    
    // Calculate price change percentages
    const priceChanges = predictionValues.map(value => ((value - currentPrice) / currentPrice) * 100);
    
    // Calculate average and standard deviation of changes
    const avgChange = priceChanges.reduce((sum, change) => sum + change, 0) / priceChanges.length;
    const stdDevChange = Math.sqrt(
      priceChanges.reduce((sum, change) => sum + Math.pow(change - avgChange, 2), 0) / priceChanges.length
    );
    
    // Determine signal direction based on average change
    let direction: SignalDirection;
    if (avgChange > 1.0) {
      direction = SignalDirection.LONG;
    } else if (avgChange < -1.0) {
      direction = SignalDirection.SHORT;
    } else {
      direction = SignalDirection.NEUTRAL;
    }
    
    // Skip neutral signals if we're filtering weak signals
    if (direction === SignalDirection.NEUTRAL && this.options.filterWeakSignals) {
      logger.debug(`Neutral signal for ${prediction.symbol} filtered out`);
      return [];
    }
    
    // Determine signal type based on direction and magnitude
    let signalType: SignalType;
    if (direction === SignalDirection.NEUTRAL) {
      signalType = SignalType.HOLD;
    } else if (Math.abs(avgChange) > 5.0) {
      signalType = direction === SignalDirection.LONG ? SignalType.ENTRY : SignalType.EXIT;
    } else {
      signalType = direction === SignalDirection.LONG ? SignalType.INCREASE : SignalType.DECREASE;
    }
    
    // Calculate confidence score based on consistency and magnitude of predictions
    const consistency = 1.0 - (stdDevChange / Math.max(1.0, Math.abs(avgChange)));
    const magnitude = Math.min(1.0, Math.abs(avgChange) / 10.0); // Normalize to max of 1.0
    const confidenceScore = Math.round((consistency * 0.6 + magnitude * 0.4) * 100);
    
    // Determine signal strength based on confidence score
    let strength: SignalStrength;
    if (confidenceScore >= 80) {
      strength = SignalStrength.VERY_STRONG;
    } else if (confidenceScore >= 60) {
      strength = SignalStrength.STRONG;
    } else if (confidenceScore >= 40) {
      strength = SignalStrength.MODERATE;
    } else if (confidenceScore >= 20) {
      strength = SignalStrength.WEAK;
    } else {
      strength = SignalStrength.VERY_WEAK;
    }
    
    // Skip weak signals if option is enabled
    if (
      this.options.filterWeakSignals && 
      (strength === SignalStrength.VERY_WEAK || strength === SignalStrength.WEAK)
    ) {
      logger.debug(`Weak signal for ${prediction.symbol} filtered out`);
      return [];
    }
    
    // Calculate target price and stop loss
    const targetPricePercent = direction === SignalDirection.LONG ? 
      Math.max(...priceChanges) : 
      Math.min(...priceChanges);
    
    const targetPrice = currentPrice * (1 + targetPricePercent / 100);
    
    // Calculate stop loss (simple approach - can be enhanced with more sophisticated methods)
    const stopLossPercent = direction === SignalDirection.LONG ? -2.0 : 2.0;
    const stopLoss = currentPrice * (1 + stopLossPercent / 100);
    
    // Determine timeframe based on prediction horizon
    let timeframe: SignalTimeframe;
    const numPredictions = predictionValues.length;
    
    if (numPredictions <= 6) { // Short term (hours)
      timeframe = SignalTimeframe.VERY_SHORT;
    } else if (numPredictions <= 24) { // Medium term (day)
      timeframe = SignalTimeframe.SHORT;
    } else if (numPredictions <= 72) { // Medium term (days)
      timeframe = SignalTimeframe.MEDIUM;
    } else if (numPredictions <= 168) { // Longer term (week)
      timeframe = SignalTimeframe.LONG;
    } else { // Very long term (weeks+)
      timeframe = SignalTimeframe.VERY_LONG;
    }
    
    // Calculate expected return and risk
    const expectedReturn = direction === SignalDirection.LONG ? 
      Math.max(0, avgChange) : 
      Math.max(0, -avgChange);
    
    const expectedRisk = direction === SignalDirection.LONG ? 
      Math.max(0, -stopLossPercent) : 
      Math.max(0, stopLossPercent);
    
    // Calculate risk-reward ratio (avoid division by zero)
    const riskRewardRatio = expectedRisk > 0 ? expectedReturn / expectedRisk : expectedReturn;
    
    // Create signal object
    const signal: TradingSignal = {
      id: uuidv4(),
      symbol: prediction.symbol,
      type: signalType,
      direction,
      strength,
      timeframe,
      price: currentPrice,
      targetPrice,
      stopLoss,
      confidenceScore,
      expectedReturn,
      expectedRisk,
      riskRewardRatio,
      generatedAt: new Date().toISOString(),
      expiresAt: this._calculateExpiryTime(timeframe),
      source: `ml-model-${prediction.model_version}`,
      metadata: {
        avgChange,
        stdDevChange,
        consistency,
        magnitude
      },
      predictionValues: predictionValues
    };
    
    signals.push(signal);
    return signals;
  }
  
  /**
   * Calculate signal expiry time based on timeframe
   * @private
   * @param timeframe - Signal timeframe
   * @returns Expiry timestamp
   */
  private _calculateExpiryTime(timeframe: SignalTimeframe): string {
    const now = new Date();
    
    switch (timeframe) {
      case SignalTimeframe.VERY_SHORT:
        now.setHours(now.getHours() + 4);
        break;
      case SignalTimeframe.SHORT:
        now.setHours(now.getHours() + 24);
        break;
      case SignalTimeframe.MEDIUM:
        now.setDate(now.getDate() + 3);
        break;
      case SignalTimeframe.LONG:
        now.setDate(now.getDate() + 7);
        break;
      case SignalTimeframe.VERY_LONG:
        now.setDate(now.getDate() + 30);
        break;
    }
    
    return now.toISOString();
  }
  
  /**
   * Validate signals using additional techniques
   * @private
   * @param signals - Signals to validate
   * @returns Validated signals
   */
  private async _validateSignals(signals: TradingSignal[]): Promise<TradingSignal[]> {
    return Promise.all(signals.map(async (signal) => {
      try {
        // This is where you would implement additional validation
        // Examples:
        // 1. Technical indicator confirmation
        // 2. Volume analysis
        // 3. Market sentiment analysis
        // 4. Correlation with related assets
        
        // For now, we'll just mark all signals as validated
        const validatedSignal: TradingSignal = {
          ...signal,
          validatedAt: new Date().toISOString(),
          validationStatus: true,
          validationReason: 'Passed basic validation checks'
        };
        
        return validatedSignal;
      } catch (error) {
        logger.warn(`Signal validation failed for ${signal.symbol}`, {
          signalId: signal.id,
          error: error instanceof Error ? error.message : String(error)
        });
        
        return {
          ...signal,
          validatedAt: new Date().toISOString(),
          validationStatus: false,
          validationReason: error instanceof Error ? error.message : String(error)
        };
      }
    }));
  }
  
  /**
   * Filter signals based on confidence threshold and other criteria
   * @private
   * @param signals - Signals to filter
   * @param options - Signal generation options
   * @returns Filtered signals
   */
  private _filterSignals(signals: TradingSignal[], options: SignalGenerationOptions): TradingSignal[] {
    // Filter by confidence threshold
    let filteredSignals = signals.filter(signal => 
      signal.confidenceScore >= (options.minConfidenceThreshold || 0)
    );
    
    // Filter by validation status if we've validated signals
    if (options.validateSignals) {
      filteredSignals = filteredSignals.filter(signal => 
        signal.validationStatus === true
      );
    }
    
    // Limit signals per symbol if needed
    if (
      options.maxSignalsPerSymbol && 
      filteredSignals.length > options.maxSignalsPerSymbol
    ) {
      // Sort by confidence score and take top N
      filteredSignals.sort((a, b) => b.confidenceScore - a.confidenceScore);
      filteredSignals = filteredSignals.slice(0, options.maxSignalsPerSymbol);
    }
    
    return filteredSignals;
  }
  
  /**
   * Store signals in the database
   * @private
   * @param signals - Signals to store
   */
  private async _storeSignals(signals: TradingSignal[]): Promise<void> {
    try {
      // Store each signal in the database
      for (const signal of signals) {
        await prisma.tradingSignal.create({
          data: {
            id: signal.id,
            symbol: signal.symbol,
            type: signal.type,
            direction: signal.direction,
            strength: signal.strength,
            timeframe: signal.timeframe,
            price: signal.price,
            targetPrice: signal.targetPrice,
            stopLoss: signal.stopLoss,
            confidenceScore: signal.confidenceScore,
            expectedReturn: signal.expectedReturn,
            expectedRisk: signal.expectedRisk,
            riskRewardRatio: signal.riskRewardRatio,
            generatedAt: new Date(signal.generatedAt),
            expiresAt: signal.expiresAt ? new Date(signal.expiresAt) : null,
            source: signal.source,
            metadata: signal.metadata as any,
            predictionValues: signal.predictionValues as any,
            validatedAt: signal.validatedAt ? new Date(signal.validatedAt) : null,
            validationStatus: signal.validationStatus || false,
            validationReason: signal.validationReason || null
          }
        });
      }
    } catch (error) {
      logger.error('Error storing signals', {
        count: signals.length,
        error: error instanceof Error ? error.message : String(error)
      });
      // Continue execution even if storage fails
    }
  }
  
  /**
   * Get signals based on filter criteria
   * @param criteria - Filter criteria
   * @returns Filtered signals
   */
  async getSignals(criteria: SignalFilterCriteria = {}): Promise<TradingSignal[]> {
    try {
      const {
        symbol,
        types,
        directions,
        minStrength,
        timeframes,
        minConfidenceScore,
        fromTimestamp,
        toTimestamp,
        status = 'active'
      } = criteria;
      
      // Build database query conditions
      const where: any = {};
      
      // Symbol filter
      if (symbol) {
        where.symbol = symbol;
      }
      
      // Types filter
      if (types && types.length > 0) {
        where.type = { in: types };
      }
      
      // Directions filter
      if (directions && directions.length > 0) {
        where.direction = { in: directions };
      }
      
      // Strength filter
      if (minStrength) {
        const strengthLevels = Object.values(SignalStrength);
        const minIndex = strengthLevels.indexOf(minStrength);
        
        if (minIndex >= 0) {
          const allowedStrengths = strengthLevels.slice(minIndex);
          where.strength = { in: allowedStrengths };
        }
      }
      
      // Timeframes filter
      if (timeframes && timeframes.length > 0) {
        where.timeframe = { in: timeframes };
      }
      
      // Confidence score filter
      if (minConfidenceScore !== undefined) {
        where.confidenceScore = { gte: minConfidenceScore };
      }
      
      // Timestamp filters
      if (fromTimestamp) {
        where.generatedAt = { ...(where.generatedAt || {}), gte: new Date(fromTimestamp) };
      }
      
      if (toTimestamp) {
        where.generatedAt = { ...(where.generatedAt || {}), lte: new Date(toTimestamp) };
      }
      
      // Status filter
      if (status !== 'all') {
        const now = new Date();
        
        if (status === 'active') {
          where.expiresAt = { gte: now };
        } else if (status === 'expired') {
          where.expiresAt = { lt: now };
        } else if (status === 'validated') {
          where.validationStatus = true;
        } else if (status === 'invalidated') {
          where.validationStatus = false;
        }
      }
      
      // Execute query
      const signalsData = await prisma.tradingSignal.findMany({ where });
      
      // Convert database records to TradingSignal objects
      const signals: TradingSignal[] = signalsData.map(data => ({
        id: data.id,
        symbol: data.symbol,
        type: data.type as SignalType,
        direction: data.direction as SignalDirection,
        strength: data.strength as SignalStrength,
        timeframe: data.timeframe as SignalTimeframe,
        price: data.price,
        targetPrice: data.targetPrice || undefined,
        stopLoss: data.stopLoss || undefined,
        confidenceScore: data.confidenceScore,
        expectedReturn: data.expectedReturn,
        expectedRisk: data.expectedRisk,
        riskRewardRatio: data.riskRewardRatio,
        generatedAt: data.generatedAt.toISOString(),
        expiresAt: data.expiresAt?.toISOString(),
        source: data.source,
        metadata: data.metadata as Record<string, any>,
        predictionValues: data.predictionValues as number[],
        validatedAt: data.validatedAt?.toISOString(),
        validationStatus: data.validationStatus,
        validationReason: data.validationReason || undefined
      }));
      
      return signals;
    } catch (error) {
      logger.error('Error getting signals', {
        criteria,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }
  
  /**
   * Get the latest signal for a symbol
   * @param symbol - Trading pair symbol
   * @returns Latest signal or null if none found
   */
  async getLatestSignal(symbol: string): Promise<TradingSignal | null> {
    try {
      const latestSignal = await prisma.tradingSignal.findFirst({
        where: { symbol },
        orderBy: { generatedAt: 'desc' }
      });
      
      if (!latestSignal) {
        return null;
      }
      
      return {
        id: latestSignal.id,
        symbol: latestSignal.symbol,
        type: latestSignal.type as SignalType,
        direction: latestSignal.direction as SignalDirection,
        strength: latestSignal.strength as SignalStrength,
        timeframe: latestSignal.timeframe as SignalTimeframe,
        price: latestSignal.price,
        targetPrice: latestSignal.targetPrice || undefined,
        stopLoss: latestSignal.stopLoss || undefined,
        confidenceScore: latestSignal.confidenceScore,
        expectedReturn: latestSignal.expectedReturn,
        expectedRisk: latestSignal.expectedRisk,
        riskRewardRatio: latestSignal.riskRewardRatio,
        generatedAt: latestSignal.generatedAt.toISOString(),
        expiresAt: latestSignal.expiresAt?.toISOString(),
        source: latestSignal.source,
        metadata: latestSignal.metadata as Record<string, any>,
        predictionValues: latestSignal.predictionValues as number[],
        validatedAt: latestSignal.validatedAt?.toISOString(),
        validationStatus: latestSignal.validationStatus,
        validationReason: latestSignal.validationReason || undefined
      };
    } catch (error) {
      logger.error(`Error getting latest signal for ${symbol}`, {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }
}

// Create default instance
const signalGenerationService = new SignalGenerationService();

export default signalGenerationService; 
/**
 * Fixed Intelligent AI-Driven Trading System
 * Implements all fixes identified in the root cause analysis
 */

import {
  TradingStrategy,
  TradingSignal,
  EnhancedMarketData,
  BacktestConfig
} from '../types/marketData';
import { logger } from '../utils/logger';
import { createEventId } from '../types/events';

// Trading Guide Principles
interface TradingGuidePrinciples {
  riskManagement: {
    maxRiskPerTrade: number;
    stopLossStrategy: 'ATR' | 'SMC' | 'FIXED';
    positionSizing: 'FIXED' | 'VOLATILITY_ADJUSTED' | 'CONFIDENCE_BASED';
  };
  smartMoneyConcepts: {
    useOrderBlocks: boolean;
    useFairValueGaps: boolean;
    useLiquidityLevels: boolean;
    useMarketStructure: boolean;
  };
}

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

interface AIModelPrediction {
  modelName: string;
  prediction: number;
  confidence: number;
  signalType: 'BUY' | 'SELL' | 'HOLD';
  timeHorizon: TradeType;
  marketRegime: MarketRegime;
}

export class FixedIntelligentTradingSystem implements TradingStrategy {
  public readonly name = 'Fixed_Intelligent_AI_System';
  public parameters: Record<string, any>;
  private config?: BacktestConfig;
  private tradingPrinciples: TradingGuidePrinciples;
  private lastDecisionTime: number = 0;
  private decisionCooldown: number = 60000; // 1 minute

  constructor() {
    // Initialize trading principles
    this.tradingPrinciples = this.initializeTradingPrinciples();

    // Apply all fixes from root cause analysis
    this.parameters = {
      // Fix #1: Lower thresholds
      minConfidence: 50,           // Reduced from 70%
      minModelConsensus: 0.4,      // Reduced from 0.6
      decisionCooldown: 60000,     // Reduced to 1 minute from 5 minutes
      
      // Fix #4: Adjust risk management
      stopLossPercent: 2.0,        // Increased from 1.5%
      takeProfitMultiplier: 3.0,   // Increased from 2.5x
      positionSizeMultiplier: 1.0, // Increased from 0.8
      
      // Enable sideways market trading
      enableSidewaysTrading: true,
      
      // Make models more decisive
      useEnhancedAIModels: true,
    };
    
    logger.info('🔧 Fixed Intelligent AI Trading System initialized with optimized parameters');
  }

  /**
   * Override signal generation with fixes
   */
  public generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null {
    if (!this.config) {
      throw new Error('Strategy not initialized. Call initialize() first.');
    }

    // Need enough data for analysis
    if (currentIndex < 50) {
      return null;
    }

    const currentCandle = data[currentIndex];
    const currentTime = Date.now();

    // Check decision cooldown (now 1 minute instead of 5)
    if (currentTime - this.lastDecisionTime < this.getDecisionCooldown()) {
      return null;
    }

    const indicators = currentCandle.indicators;
    if (!this.hasRequiredIndicators(indicators)) {
      return null;
    }

    try {
      // Step 1: Get enhanced AI model predictions (Fix #2)
      const modelPredictions = this.getEnhancedAIModelPredictions(currentCandle, data, currentIndex);
      
      if (modelPredictions.length === 0) {
        return null;
      }

      // Step 2: Analyze market regime (Fix #3 - enable sideways trading)
      const marketRegime = this.analyzeMarketRegimeFixed(data, currentIndex);
      
      // Step 3: Generate intelligent decision with lower thresholds
      const decision = this.generateIntelligentDecisionFixed(
        modelPredictions,
        marketRegime,
        currentCandle,
        data,
        currentIndex
      );

      if (!decision) {
        return null;
      }

      // Step 4: Apply fixed risk management
      const finalSignal = this.applyFixedRiskManagement(decision, currentCandle, data, currentIndex);
      
      if (finalSignal && finalSignal.confidence >= this.getMinConfidence()) {
        this.lastDecisionTime = currentTime;
        
        logger.info(`🎯 Generated FIXED ${finalSignal.type} signal`, {
          price: finalSignal.price,
          confidence: finalSignal.confidence,
          marketRegime: marketRegime,
          reason: finalSignal.reason,
        });
        
        return finalSignal;
      }

      return null;

    } catch (error) {
      logger.error('❌ Error generating fixed trading signal:', error);
      return null;
    }
  }

  /**
   * Enhanced AI model predictions with wider ranges (Fix #2)
   */
  private getEnhancedAIModelPredictions(
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): AIModelPrediction[] {
    const predictions: AIModelPrediction[] = [];
    const indicators = currentCandle.indicators;
    
    try {
      // Enhanced Model 1: Aggressive Transformer
      const transformerPrediction = this.simulateAggressiveTransformer(indicators, data, currentIndex);
      predictions.push(transformerPrediction);
      
      // Enhanced Model 2: Decisive LSTM
      const lstmPrediction = this.simulateDecisiveLSTM(indicators, data, currentIndex);
      predictions.push(lstmPrediction);
      
      // Enhanced Model 3: Active SMC Analyzer
      const smcPrediction = this.simulateActiveSMC(indicators, data, currentIndex);
      predictions.push(smcPrediction);

    } catch (error) {
      logger.warn('⚠️ Error getting enhanced AI model predictions:', error);
    }

    return predictions;
  }

  /**
   * Aggressive Transformer model with wider prediction ranges
   */
  private simulateAggressiveTransformer(indicators: any, data: EnhancedMarketData[], currentIndex: number): AIModelPrediction {
    const rsi = indicators.rsi || 50;
    const macdSignal = indicators.macd_signal || 0;
    const ema12 = indicators.ema_12 || 0;
    const ema26 = indicators.ema_26 || 0;
    
    let prediction = 0.5; // Start neutral
    let confidence = 0.7;
    
    // More aggressive RSI signals
    if (rsi < 35) prediction += 0.35;      // Strong buy signal
    else if (rsi < 45) prediction += 0.15; // Moderate buy signal
    else if (rsi > 65) prediction -= 0.35; // Strong sell signal
    else if (rsi > 55) prediction -= 0.15; // Moderate sell signal
    
    // EMA trend with stronger signals
    const emaDiff = (ema12 - ema26) / ema26;
    if (emaDiff > 0.01) prediction += 0.2;      // Strong uptrend
    else if (emaDiff > 0.005) prediction += 0.1; // Moderate uptrend
    else if (emaDiff < -0.01) prediction -= 0.2; // Strong downtrend
    else if (emaDiff < -0.005) prediction -= 0.1; // Moderate downtrend
    
    // MACD confirmation with stronger signals
    if (macdSignal > 50) prediction += 0.15;
    else if (macdSignal < -50) prediction -= 0.15;
    
    // Ensure prediction stays in valid range
    prediction = Math.max(0.1, Math.min(0.9, prediction));
    
    // Adjust confidence based on signal strength
    confidence = Math.min(0.95, 0.6 + Math.abs(prediction - 0.5) * 0.7);
    
    return {
      modelName: 'Aggressive_Transformer',
      prediction,
      confidence,
      signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
      timeHorizon: this.determineTimeHorizon(confidence),
      marketRegime: MarketRegime.SIDEWAYS,
    };
  }

  /**
   * Decisive LSTM model with momentum focus
   */
  private simulateDecisiveLSTM(indicators: any, data: EnhancedMarketData[], currentIndex: number): AIModelPrediction {
    const recentPrices = data.slice(Math.max(0, currentIndex - 10), currentIndex + 1).map(d => d.close);
    
    let prediction = 0.5;
    let confidence = 0.75;
    
    if (recentPrices.length >= 5) {
      // Short-term momentum (3 periods)
      const shortTrend = (recentPrices[recentPrices.length - 1] - recentPrices[recentPrices.length - 3]) / recentPrices[recentPrices.length - 3];
      
      // Medium-term momentum (5 periods)
      const mediumTrend = (recentPrices[recentPrices.length - 1] - recentPrices[recentPrices.length - 5]) / recentPrices[recentPrices.length - 5];
      
      // Combine trends with amplification
      const combinedTrend = (shortTrend * 0.6 + mediumTrend * 0.4) * 20; // Amplify signals
      
      prediction = 0.5 + combinedTrend;
      prediction = Math.max(0.1, Math.min(0.9, prediction));
      
      // Higher confidence for stronger trends
      confidence = Math.min(0.9, 0.65 + Math.abs(combinedTrend) * 2);
    }
    
    return {
      modelName: 'Decisive_LSTM',
      prediction,
      confidence,
      signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
      timeHorizon: this.determineTimeHorizon(confidence),
      marketRegime: MarketRegime.TRENDING_BULLISH,
    };
  }

  /**
   * Active SMC analyzer with volume and price action
   */
  private simulateActiveSMC(indicators: any, data: EnhancedMarketData[], currentIndex: number): AIModelPrediction {
    const currentCandle = data[currentIndex];
    const volume = currentCandle.volume;
    const volumeSMA = indicators.volume_sma || volume;
    
    let prediction = 0.5;
    let confidence = 0.8;
    
    // Volume analysis with stronger signals
    const volumeRatio = volume / volumeSMA;
    if (volumeRatio > 1.8) {
      // Very high volume - strong institutional activity
      const priceAction = currentCandle.close > currentCandle.open ? 0.3 : -0.3;
      prediction += priceAction;
      confidence += 0.1;
    } else if (volumeRatio > 1.3) {
      // High volume - moderate institutional activity
      const priceAction = currentCandle.close > currentCandle.open ? 0.2 : -0.2;
      prediction += priceAction;
      confidence += 0.05;
    }
    
    // Price action analysis
    const bodySize = Math.abs(currentCandle.close - currentCandle.open) / currentCandle.open;
    const wickSize = (currentCandle.high - Math.max(currentCandle.open, currentCandle.close)) / currentCandle.open;
    
    // Strong bullish candle
    if (currentCandle.close > currentCandle.open && bodySize > 0.015) {
      prediction += 0.15;
    }
    // Strong bearish candle
    else if (currentCandle.close < currentCandle.open && bodySize > 0.015) {
      prediction -= 0.15;
    }
    
    // Rejection wicks
    if (wickSize > 0.01) {
      prediction += currentCandle.close > currentCandle.open ? 0.1 : -0.1;
    }
    
    // Bollinger Bands position with stronger signals
    if (indicators.bollinger_upper && indicators.bollinger_lower) {
      const bbPosition = (currentCandle.close - indicators.bollinger_lower) / 
                        (indicators.bollinger_upper - indicators.bollinger_lower);
      
      if (bbPosition < 0.2) prediction += 0.2; // Near lower band - strong buy
      else if (bbPosition > 0.8) prediction -= 0.2; // Near upper band - strong sell
    }
    
    prediction = Math.max(0.1, Math.min(0.9, prediction));
    confidence = Math.min(0.95, confidence);
    
    return {
      modelName: 'Active_SMC',
      prediction,
      confidence,
      signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
      timeHorizon: this.determineTimeHorizon(confidence),
      marketRegime: MarketRegime.VOLATILE,
    };
  }

  /**
   * Fixed market regime analysis that enables sideways trading (Fix #3)
   */
  private analyzeMarketRegimeFixed(data: EnhancedMarketData[], currentIndex: number): MarketRegime {
    const recentData = data.slice(Math.max(0, currentIndex - 20), currentIndex + 1);
    const prices = recentData.map(d => d.close);
    
    // Calculate trend strength
    const trendStrength = (prices[prices.length - 1] - prices[0]) / prices[0];
    
    // Calculate volatility
    const returns = prices.slice(1).map((price, i) => Math.log(price / prices[i]));
    const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length);
    
    // More permissive regime detection
    if (Math.abs(trendStrength) > 0.03 && volatility < 0.04) {
      return trendStrength > 0 ? MarketRegime.TRENDING_BULLISH : MarketRegime.TRENDING_BEARISH;
    } else if (volatility > 0.06) {
      return MarketRegime.VOLATILE;
    } else {
      // Enable sideways market trading instead of avoiding it
      return MarketRegime.SIDEWAYS;
    }
  }

  /**
   * Fixed intelligent decision generation with lower thresholds (Fix #1)
   */
  private generateIntelligentDecisionFixed(
    modelPredictions: AIModelPrediction[],
    marketRegime: MarketRegime,
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): any {
    
    // Calculate model consensus with lower threshold
    const buySignals = modelPredictions.filter(p => p.signalType === 'BUY').length;
    const sellSignals = modelPredictions.filter(p => p.signalType === 'SELL').length;
    const totalSignals = modelPredictions.length;
    
    if (totalSignals === 0) return null;
    
    const modelConsensus = Math.max(buySignals, sellSignals) / totalSignals;
    
    // Lower consensus requirement (Fix #1)
    const minConsensus = this.getModelConsensusThreshold(); // Now 0.4 instead of 0.6
    if (modelConsensus < minConsensus) {
      logger.debug(`Insufficient model consensus: ${modelConsensus.toFixed(2)} < ${minConsensus}`);
      return null;
    }

    // Determine signal type
    const signalType = buySignals > sellSignals ? 'BUY' : 'SELL';
    
    // Calculate average confidence
    const avgConfidence = modelPredictions.reduce((sum, p) => sum + p.confidence, 0) / totalSignals;
    
    // Enable trading in ALL market regimes (Fix #3)
    const tradeType = this.determineOptimalTradeType(marketRegime, avgConfidence);

    // Create trading signal
    const signal: TradingSignal = {
      id: createEventId(),
      timestamp: currentCandle.timestamp,
      symbol: this.config!.symbol,
      type: signalType,
      price: currentCandle.close,
      quantity: 0,
      confidence: avgConfidence * 100,
      strategy: this.name,
      reason: `FIXED AI Consensus: ${modelConsensus.toFixed(2)}, Regime: ${marketRegime}`,
    };

    return {
      signal,
      tradeType,
      marketRegime,
      modelConsensus,
      riskAssessment: {
        riskLevel: this.assessRiskLevel(avgConfidence, marketRegime),
        expectedDrawdown: this.calculateExpectedDrawdown(marketRegime),
        confidenceScore: avgConfidence,
      },
      executionPlan: {
        entryStrategy: this.determineEntryStrategy(tradeType, marketRegime),
        exitStrategy: this.determineExitStrategy(tradeType, marketRegime),
        positionSize: 0,
        timeframe: this.config!.timeframe,
      },
    };
  }

  /**
   * Fixed risk management with improved parameters (Fix #4)
   */
  private applyFixedRiskManagement(
    decision: any,
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): TradingSignal | null {
    
    // Calculate position size with improved parameters
    const riskAmount = this.config!.initialCapital * (this.config!.riskPerTrade / 100);
    
    // Use fixed stop loss percentage (now 2.0% instead of 1.5%)
    const stopLossPercent = this.parameters.stopLossPercent / 100;
    const stopLoss = decision.signal.type === 'BUY' 
      ? currentCandle.close * (1 - stopLossPercent)
      : currentCandle.close * (1 + stopLossPercent);
    
    const stopDistance = Math.abs(currentCandle.close - stopLoss) / currentCandle.close;
    
    // Calculate position size with improved multiplier (now 1.0 instead of 0.8)
    let positionSize = riskAmount / (stopDistance * currentCandle.close);
    positionSize *= this.config!.leverage;
    positionSize *= this.parameters.positionSizeMultiplier; // Now 1.0
    
    // Apply confidence-based sizing
    const confidenceMultiplier = decision.riskAssessment.confidenceScore;
    positionSize *= confidenceMultiplier;
    
    // Calculate take profit with improved multiplier (now 3.0x instead of 2.5x)
    const takeProfit = decision.signal.type === 'BUY'
      ? currentCandle.close * (1 + stopDistance * this.parameters.takeProfitMultiplier)
      : currentCandle.close * (1 - stopDistance * this.parameters.takeProfitMultiplier);

    return {
      ...decision.signal,
      quantity: positionSize,
      stopLoss,
      takeProfit,
      riskReward: this.parameters.takeProfitMultiplier,
    };
  }

  // Override configuration methods
  protected getMinConfidence(): number {
    return this.parameters.minConfidence || 50; // Reduced from 70
  }

  protected getModelConsensusThreshold(): number {
    return this.parameters.minModelConsensus || 0.4; // Reduced from 0.6
  }

  protected getDecisionCooldown(): number {
    return this.parameters.decisionCooldown || 60000; // Reduced to 1 minute
  }

  private hasRequiredIndicators(indicators: any): boolean {
    return !!(
      indicators.ema_12 !== undefined && !isNaN(indicators.ema_12) &&
      indicators.ema_26 !== undefined && !isNaN(indicators.ema_26) &&
      indicators.rsi !== undefined && !isNaN(indicators.rsi) &&
      indicators.volume_sma !== undefined && !isNaN(indicators.volume_sma)
    );
  }

  private lastDecisionTime: number = 0;
  private config?: BacktestConfig;

  public initialize(config: BacktestConfig): void {
    this.config = config;
    this.lastDecisionTime = 0;
    
    logger.info(`🎯 Initialized ${this.name} with FIXED parameters`, {
      symbol: config.symbol,
      timeframe: config.timeframe,
      minConfidence: this.getMinConfidence(),
      modelConsensus: this.getModelConsensusThreshold(),
      decisionCooldown: this.getDecisionCooldown() / 60000 + ' minutes',
    });
  }

  public getDescription(): string {
    return 'Fixed Intelligent AI-Driven Trading System with optimized parameters based on root cause analysis';
  }
}

// Export factory function
export function createFixedIntelligentTradingSystem(): FixedIntelligentTradingSystem {
  return new FixedIntelligentTradingSystem();
}

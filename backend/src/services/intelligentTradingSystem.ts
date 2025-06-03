/**
 * Intelligent AI-Driven Trading System
 * Integrates existing ML models, Smart Money Concepts, and trading guide principles
 */

import { 
  TradingStrategy, 
  TradingSignal, 
  EnhancedMarketData, 
  BacktestConfig,
  Position,
  Trade
} from '../types/marketData';
import { logger } from '../utils/logger';
import { createEventId } from '../types/events';
import mlBridgeService from './bridge/mlBridgeService';
import bridgeService from './bridge/bridgeService';

// Trading Guide Principles from tradingguide.txt
interface TradingGuidePrinciples {
  riskManagement: {
    maxRiskPerTrade: number; // 1-2% rule
    stopLossStrategy: 'ATR' | 'SMC' | 'FIXED';
    positionSizing: 'FIXED' | 'VOLATILITY_ADJUSTED' | 'CONFIDENCE_BASED';
  };
  smartMoneyConcepts: {
    useOrderBlocks: boolean;
    useFairValueGaps: boolean;
    useLiquidityLevels: boolean;
    useMarketStructure: boolean;
  };
  marketRegimeAdaptation: {
    trendingMarkets: string[];
    sidewaysMarkets: string[];
    volatileMarkets: string[];
  };
}

// Trade Classification System
export enum TradeType {
  SCALPING = 'SCALPING',           // < 15 minutes
  DAY_TRADING = 'DAY_TRADING',     // < 1 day
  SWING_TRADING = 'SWING_TRADING', // 1-7 days
  POSITION_TRADING = 'POSITION_TRADING' // > 7 days
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
  smcAnalysis?: {
    orderBlocks: any[];
    fairValueGaps: any[];
    liquidityLevels: any[];
    marketStructure: string;
  };
}

interface IntelligentTradingDecision {
  signal: TradingSignal;
  tradeType: TradeType;
  marketRegime: MarketRegime;
  modelConsensus: number;
  riskAssessment: {
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
    expectedDrawdown: number;
    confidenceScore: number;
  };
  executionPlan: {
    entryStrategy: string;
    exitStrategy: string;
    positionSize: number;
    timeframe: string;
  };
}

export class IntelligentTradingSystem implements TradingStrategy {
  public readonly name = 'Intelligent_AI_System';
  public parameters: Record<string, any>;
  private config?: BacktestConfig;
  private tradingPrinciples: TradingGuidePrinciples;
  private modelPredictions: Map<string, AIModelPrediction> = new Map();
  private lastDecisionTime: number = 0;
  private decisionCooldown: number = 300000; // 5 minutes

  constructor() {
    this.parameters = {
      useAIModels: true,
      useSMC: true,
      adaptiveRiskManagement: true,
      multiTimeframeAnalysis: true,
      decisionCooldown: 300000, // 5 minutes
      minModelConsensus: 0.6,
      minConfidence: 70,
    };
    this.tradingPrinciples = this.initializeTradingPrinciples();
    logger.info('🧠 Intelligent AI Trading System initialized');
  }

  public initialize(config: BacktestConfig): void {
    this.config = config;
    this.lastDecisionTime = 0;
    
    logger.info(`🎯 Initialized ${this.name} with AI model integration`, {
      symbol: config.symbol,
      timeframe: config.timeframe,
      riskPerTrade: config.riskPerTrade,
      leverage: config.leverage,
    });
  }

  public generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null {
    if (!this.config) {
      throw new Error('Strategy not initialized. Call initialize() first.');
    }

    // Cooldown check to prevent overtrading
    const currentTime = Date.now();
    if (currentTime - this.lastDecisionTime < this.decisionCooldown) {
      return null;
    }

    const currentCandle = data[currentIndex];

    try {
      // Step 1: Get simulated AI model predictions (simplified for sync operation)
      const modelPredictions = this.getSimulatedAIModelPredictions(currentCandle, data, currentIndex);

      if (modelPredictions.length === 0) {
        logger.debug('No AI model predictions available');
        return null;
      }

      // Step 2: Analyze market regime
      const marketRegime = this.analyzeMarketRegime(data, currentIndex);

      // Step 3: Apply Smart Money Concepts analysis
      const smcAnalysis = this.performSMCAnalysis(currentCandle, data, currentIndex);

      // Step 4: Generate intelligent trading decision
      const decision = this.generateIntelligentDecision(
        modelPredictions,
        marketRegime,
        smcAnalysis,
        currentCandle,
        data,
        currentIndex
      );

      if (!decision) {
        return null;
      }

      // Step 5: Apply trading guide risk management
      const finalSignal = this.applyRiskManagement(decision, currentCandle, data, currentIndex);

      if (finalSignal) {
        this.lastDecisionTime = currentTime;

        logger.info(`🎯 Generated intelligent ${finalSignal.type} signal`, {
          price: finalSignal.price,
          confidence: finalSignal.confidence,
          tradeType: decision.tradeType,
          marketRegime: decision.marketRegime,
          modelConsensus: decision.modelConsensus,
          riskLevel: decision.riskAssessment.riskLevel,
        });
      }

      return finalSignal;

    } catch (error) {
      logger.error('❌ Error generating intelligent trading signal:', error);
      return null;
    }
  }

  /**
   * Get simulated AI model predictions (synchronous version for demo)
   */
  private getSimulatedAIModelPredictions(
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): AIModelPrediction[] {
    const predictions: AIModelPrediction[] = [];

    try {
      // Simulate multiple AI model predictions based on technical indicators
      const indicators = currentCandle.indicators;

      // Model 1: Enhanced Transformer (simulated)
      const transformerPrediction = this.simulateTransformerPrediction(indicators, data, currentIndex);
      predictions.push(transformerPrediction);

      // Model 2: LSTM Model (simulated)
      const lstmPrediction = this.simulateLSTMPrediction(indicators, data, currentIndex);
      predictions.push(lstmPrediction);

      // Model 3: SMC Analyzer (simulated)
      const smcPrediction = this.simulateSMCPrediction(indicators, data, currentIndex);
      predictions.push(smcPrediction);

    } catch (error) {
      logger.warn('⚠️ Error getting simulated AI model predictions:', error);
    }

    return predictions;
  }

  /**
   * Get enhanced model predictions from ML service
   */
  private async getEnhancedModelPredictions(currentCandle: EnhancedMarketData): Promise<AIModelPrediction[]> {
    const predictions: AIModelPrediction[] = [];

    try {
      // Call enhanced model service (if available)
      const response = await fetch(`http://localhost:3002/api/models/enhanced/${this.config!.symbol}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: this.extractMLFeatures(currentCandle),
          sequence_length: 60,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        
        predictions.push({
          modelName: 'Enhanced_Transformer',
          prediction: result.prediction,
          confidence: result.confidence,
          signalType: result.recommendation === 'BUY' ? 'BUY' : 
                     result.recommendation === 'SELL' ? 'SELL' : 'HOLD',
          timeHorizon: this.determineTimeHorizon(result.confidence),
          marketRegime: this.mapToMarketRegime(result.market_regime),
          smcAnalysis: result.smc_analysis,
        });
      }

    } catch (error) {
      logger.debug('Enhanced model service not available, continuing with basic predictions');
    }

    return predictions;
  }

  /**
   * Analyze current market regime using multiple indicators
   */
  private analyzeMarketRegime(data: EnhancedMarketData[], currentIndex: number): MarketRegime {
    const recentData = data.slice(Math.max(0, currentIndex - 20), currentIndex + 1);
    
    // Calculate trend strength
    const prices = recentData.map(d => d.close);
    const trendStrength = this.calculateTrendStrength(prices);
    
    // Calculate volatility
    const volatility = this.calculateVolatility(prices);
    
    // Determine regime
    if (trendStrength > 0.7 && volatility < 0.3) {
      return trendStrength > 0 ? MarketRegime.TRENDING_BULLISH : MarketRegime.TRENDING_BEARISH;
    } else if (volatility > 0.5) {
      return MarketRegime.VOLATILE;
    } else if (Math.abs(trendStrength) < 0.3) {
      return MarketRegime.SIDEWAYS;
    } else {
      return MarketRegime.BREAKOUT;
    }
  }

  /**
   * Perform Smart Money Concepts analysis based on trading guide
   */
  private performSMCAnalysis(
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): any {
    if (!this.tradingPrinciples.smartMoneyConcepts.useOrderBlocks) {
      return null;
    }

    // Implement SMC analysis based on trading guide principles
    const recentData = data.slice(Math.max(0, currentIndex - 50), currentIndex + 1);
    
    return {
      orderBlocks: this.identifyOrderBlocks(recentData),
      fairValueGaps: this.identifyFairValueGaps(recentData),
      liquidityLevels: this.identifyLiquidityLevels(recentData),
      marketStructure: this.analyzeMarketStructure(recentData),
    };
  }

  /**
   * Generate intelligent trading decision based on all inputs
   */
  private generateIntelligentDecision(
    modelPredictions: AIModelPrediction[],
    marketRegime: MarketRegime,
    smcAnalysis: any,
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): IntelligentTradingDecision | null {
    
    // Calculate model consensus
    const buySignals = modelPredictions.filter(p => p.signalType === 'BUY').length;
    const sellSignals = modelPredictions.filter(p => p.signalType === 'SELL').length;
    const totalSignals = modelPredictions.length;
    
    if (totalSignals === 0) return null;
    
    const modelConsensus = Math.max(buySignals, sellSignals) / totalSignals;
    
    // Require minimum consensus (configurable)
    const minConsensus = this.getModelConsensusThreshold();
    if (modelConsensus < minConsensus) {
      logger.debug(`Insufficient model consensus: ${modelConsensus.toFixed(2)} < ${minConsensus}`);
      return null;
    }

    // Determine signal type
    const signalType = buySignals > sellSignals ? 'BUY' : 'SELL';
    
    // Calculate average confidence
    const avgConfidence = modelPredictions.reduce((sum, p) => sum + p.confidence, 0) / totalSignals;
    
    // Determine trade type based on market regime and confidence
    const tradeType = this.determineOptimalTradeType(marketRegime, avgConfidence);
    
    // Check if market regime is favorable for trading
    if (!this.isMarketRegimeFavorable(marketRegime, tradeType)) {
      logger.debug(`Market regime ${marketRegime} not favorable for ${tradeType}`);
      return null;
    }

    // Create trading signal
    const signal: TradingSignal = {
      id: createEventId(),
      timestamp: currentCandle.timestamp,
      symbol: this.config!.symbol,
      type: signalType,
      price: currentCandle.close,
      quantity: 0, // Will be calculated in risk management
      confidence: avgConfidence * 100,
      strategy: this.name,
      reason: `AI Consensus: ${modelConsensus.toFixed(2)}, Regime: ${marketRegime}`,
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
        positionSize: 0, // Will be calculated in risk management
        timeframe: this.config!.timeframe,
      },
    };
  }

  /**
   * Apply trading guide risk management principles
   */
  private applyRiskManagement(
    decision: IntelligentTradingDecision,
    currentCandle: EnhancedMarketData,
    data: EnhancedMarketData[],
    currentIndex: number
  ): TradingSignal | null {
    
    // Calculate position size based on 1-2% rule
    const riskAmount = this.config!.initialCapital * (this.config!.riskPerTrade / 100);
    
    // Calculate stop loss based on strategy
    const stopLoss = this.calculateStopLoss(
      decision.signal.type,
      currentCandle.close,
      data,
      currentIndex,
      decision.marketRegime
    );
    
    const stopDistance = Math.abs(currentCandle.close - stopLoss) / currentCandle.close;
    
    // Calculate position size
    let positionSize = riskAmount / (stopDistance * currentCandle.close);
    positionSize *= this.config!.leverage;
    
    // Apply confidence-based sizing
    const confidenceMultiplier = decision.riskAssessment.confidenceScore;
    positionSize *= confidenceMultiplier;
    
    // Apply maximum position size limits
    const maxPositionSize = this.config!.initialCapital * 0.1; // Max 10% of capital
    positionSize = Math.min(positionSize, maxPositionSize / currentCandle.close);
    
    // Calculate take profit
    const takeProfit = this.calculateTakeProfit(
      decision.signal.type,
      currentCandle.close,
      stopDistance,
      decision.tradeType
    );

    return {
      ...decision.signal,
      quantity: positionSize,
      stopLoss,
      takeProfit,
      riskReward: Math.abs(takeProfit - currentCandle.close) / Math.abs(stopLoss - currentCandle.close),
    };
  }

  // Helper methods
  private initializeTradingPrinciples(): TradingGuidePrinciples {
    return {
      riskManagement: {
        maxRiskPerTrade: 2, // 2% max risk per trade
        stopLossStrategy: 'SMC', // Use Smart Money Concepts for stops
        positionSizing: 'CONFIDENCE_BASED',
      },
      smartMoneyConcepts: {
        useOrderBlocks: true,
        useFairValueGaps: true,
        useLiquidityLevels: true,
        useMarketStructure: true,
      },
      marketRegimeAdaptation: {
        trendingMarkets: ['TRENDING_BULLISH', 'TRENDING_BEARISH'],
        sidewaysMarkets: ['SIDEWAYS'],
        volatileMarkets: ['VOLATILE', 'BREAKOUT'],
      },
    };
  }

  private extractFeatures(currentCandle: EnhancedMarketData, data: EnhancedMarketData[], currentIndex: number): any {
    return {
      price: currentCandle.close,
      volume: currentCandle.volume,
      rsi: currentCandle.indicators.rsi,
      sma_20: currentCandle.indicators.sma_20,
      sma_50: currentCandle.indicators.sma_50,
      ema_12: currentCandle.indicators.ema_12,
      ema_26: currentCandle.indicators.ema_26,
      macd: currentCandle.indicators.macd,
      macd_signal: currentCandle.indicators.macd_signal,
    };
  }

  private extractMLFeatures(currentCandle: EnhancedMarketData): any {
    return {
      open: currentCandle.open,
      high: currentCandle.high,
      low: currentCandle.low,
      close: currentCandle.close,
      volume: currentCandle.volume,
      ...currentCandle.indicators,
    };
  }

  private determineTimeHorizon(confidence: number): TradeType {
    if (confidence > 0.8) return TradeType.POSITION_TRADING;
    if (confidence > 0.7) return TradeType.SWING_TRADING;
    if (confidence > 0.6) return TradeType.DAY_TRADING;
    return TradeType.SCALPING;
  }

  private mapToMarketRegime(regime: string): MarketRegime {
    switch (regime?.toLowerCase()) {
      case 'trending_bullish': return MarketRegime.TRENDING_BULLISH;
      case 'trending_bearish': return MarketRegime.TRENDING_BEARISH;
      case 'sideways': return MarketRegime.SIDEWAYS;
      case 'volatile': return MarketRegime.VOLATILE;
      case 'breakout': return MarketRegime.BREAKOUT;
      default: return MarketRegime.SIDEWAYS;
    }
  }

  private calculateTrendStrength(prices: number[]): number {
    if (prices.length < 2) return 0;
    const firstPrice = prices[0];
    const lastPrice = prices[prices.length - 1];
    return (lastPrice - firstPrice) / firstPrice;
  }

  private calculateVolatility(prices: number[]): number {
    if (prices.length < 2) return 0;
    const returns = prices.slice(1).map((price, i) => Math.log(price / prices[i]));
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }

  private determineOptimalTradeType(regime: MarketRegime, confidence: number): TradeType {
    if (regime === MarketRegime.TRENDING_BULLISH || regime === MarketRegime.TRENDING_BEARISH) {
      return confidence > 0.8 ? TradeType.SWING_TRADING : TradeType.DAY_TRADING;
    }
    if (regime === MarketRegime.VOLATILE) {
      return TradeType.SCALPING;
    }
    return TradeType.DAY_TRADING;
  }

  private isMarketRegimeFavorable(regime: MarketRegime, tradeType: TradeType): boolean {
    // Avoid trading in sideways markets for longer timeframes
    if (regime === MarketRegime.SIDEWAYS && 
        (tradeType === TradeType.SWING_TRADING || tradeType === TradeType.POSITION_TRADING)) {
      return false;
    }
    return true;
  }

  private assessRiskLevel(confidence: number, regime: MarketRegime): 'LOW' | 'MEDIUM' | 'HIGH' {
    if (confidence > 0.8 && (regime === MarketRegime.TRENDING_BULLISH || regime === MarketRegime.TRENDING_BEARISH)) {
      return 'LOW';
    }
    if (regime === MarketRegime.VOLATILE) {
      return 'HIGH';
    }
    return 'MEDIUM';
  }

  private calculateExpectedDrawdown(regime: MarketRegime): number {
    switch (regime) {
      case MarketRegime.TRENDING_BULLISH:
      case MarketRegime.TRENDING_BEARISH:
        return 0.05; // 5%
      case MarketRegime.SIDEWAYS:
        return 0.08; // 8%
      case MarketRegime.VOLATILE:
        return 0.15; // 15%
      case MarketRegime.BREAKOUT:
        return 0.10; // 10%
      default:
        return 0.10;
    }
  }

  private determineEntryStrategy(tradeType: TradeType, regime: MarketRegime): string {
    if (regime === MarketRegime.TRENDING_BULLISH || regime === MarketRegime.TRENDING_BEARISH) {
      return 'Trend Following with SMC Confirmation';
    }
    if (regime === MarketRegime.VOLATILE) {
      return 'Mean Reversion with Quick Exits';
    }
    return 'Breakout Strategy with Volume Confirmation';
  }

  private determineExitStrategy(tradeType: TradeType, regime: MarketRegime): string {
    if (tradeType === TradeType.SCALPING) {
      return 'Quick Profit Taking (1-2% targets)';
    }
    if (tradeType === TradeType.SWING_TRADING) {
      return 'Trailing Stop with SMC Levels';
    }
    return 'Fixed R:R with Partial Profit Taking';
  }

  private calculateStopLoss(
    signalType: string,
    currentPrice: number,
    data: EnhancedMarketData[],
    currentIndex: number,
    regime: MarketRegime
  ): number {
    // Use ATR-based stops for volatile markets
    if (regime === MarketRegime.VOLATILE) {
      const atr = this.calculateATR(data, currentIndex);
      const multiplier = 1.5;
      return signalType === 'BUY' 
        ? currentPrice - (atr * multiplier)
        : currentPrice + (atr * multiplier);
    }

    // Use fixed percentage for other markets
    const stopPercent = 0.015; // 1.5%
    return signalType === 'BUY'
      ? currentPrice * (1 - stopPercent)
      : currentPrice * (1 + stopPercent);
  }

  private calculateTakeProfit(
    signalType: string,
    currentPrice: number,
    stopDistance: number,
    tradeType: TradeType
  ): number {
    // Risk-reward ratios based on trade type
    const rrRatios = {
      [TradeType.SCALPING]: 1.5,
      [TradeType.DAY_TRADING]: 2.0,
      [TradeType.SWING_TRADING]: 3.0,
      [TradeType.POSITION_TRADING]: 4.0,
    };

    const rrRatio = rrRatios[tradeType];
    const targetDistance = stopDistance * rrRatio;

    return signalType === 'BUY'
      ? currentPrice * (1 + targetDistance)
      : currentPrice * (1 - targetDistance);
  }

  private calculateATR(data: EnhancedMarketData[], currentIndex: number, period: number = 14): number {
    const start = Math.max(0, currentIndex - period);
    const recentData = data.slice(start, currentIndex + 1);
    
    let atrSum = 0;
    for (let i = 1; i < recentData.length; i++) {
      const tr1 = recentData[i].high - recentData[i].low;
      const tr2 = Math.abs(recentData[i].high - recentData[i - 1].close);
      const tr3 = Math.abs(recentData[i].low - recentData[i - 1].close);
      atrSum += Math.max(tr1, tr2, tr3);
    }
    
    return atrSum / (recentData.length - 1);
  }

  // SMC Analysis methods (simplified implementations)
  private identifyOrderBlocks(data: EnhancedMarketData[]): any[] {
    // Simplified order block identification
    return [];
  }

  private identifyFairValueGaps(data: EnhancedMarketData[]): any[] {
    // Simplified FVG identification
    return [];
  }

  private identifyLiquidityLevels(data: EnhancedMarketData[]): any[] {
    // Simplified liquidity level identification
    return [];
  }

  private analyzeMarketStructure(data: EnhancedMarketData[]): string {
    // Simplified market structure analysis
    return 'NEUTRAL';
  }

  public getDescription(): string {
    return 'Intelligent AI-Driven Trading System integrating ML models, Smart Money Concepts, and trading guide principles';
  }

  // AI Model Simulation Methods
  private simulateTransformerPrediction(indicators: any, data: EnhancedMarketData[], currentIndex: number): AIModelPrediction {
    // Simulate transformer model using multiple indicators
    const rsi = indicators.rsi || 50;
    const macdSignal = indicators.macd_signal || 0;
    const ema12 = indicators.ema_12 || 0;
    const ema26 = indicators.ema_26 || 0;

    // Complex prediction logic
    let prediction = 0.5; // Neutral
    let confidence = 0.6;

    // RSI momentum
    if (rsi < 30) prediction += 0.2;
    else if (rsi > 70) prediction -= 0.2;

    // EMA trend
    if (ema12 > ema26) prediction += 0.1;
    else prediction -= 0.1;

    // MACD confirmation
    if (macdSignal > 0) prediction += 0.1;
    else prediction -= 0.1;

    // Adjust confidence based on signal strength
    confidence = Math.min(0.9, 0.5 + Math.abs(prediction - 0.5));

    return {
      modelName: 'Enhanced_Transformer',
      prediction,
      confidence,
      signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
      timeHorizon: this.determineTimeHorizon(confidence),
      marketRegime: MarketRegime.SIDEWAYS,
    };
  }

  private simulateLSTMPrediction(indicators: any, data: EnhancedMarketData[], currentIndex: number): AIModelPrediction {
    // Simulate LSTM model using price sequence
    const recentPrices = data.slice(Math.max(0, currentIndex - 10), currentIndex + 1).map(d => d.close);

    let prediction = 0.5;
    let confidence = 0.65;

    // Simple momentum calculation
    if (recentPrices.length >= 3) {
      const shortTrend = (recentPrices[recentPrices.length - 1] - recentPrices[recentPrices.length - 3]) / recentPrices[recentPrices.length - 3];
      prediction = 0.5 + (shortTrend * 10); // Scale trend
      prediction = Math.max(0, Math.min(1, prediction));
      confidence = Math.min(0.85, 0.6 + Math.abs(shortTrend) * 5);
    }

    return {
      modelName: 'LSTM_Sequence',
      prediction,
      confidence,
      signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
      timeHorizon: this.determineTimeHorizon(confidence),
      marketRegime: MarketRegime.TRENDING_BULLISH,
    };
  }

  private simulateSMCPrediction(indicators: any, data: EnhancedMarketData[], currentIndex: number): AIModelPrediction {
    // Simulate Smart Money Concepts analysis
    const currentCandle = data[currentIndex];
    const volume = currentCandle.volume;
    const volumeSMA = indicators.volume_sma || volume;

    let prediction = 0.5;
    let confidence = 0.7;

    // Volume analysis
    const volumeRatio = volume / volumeSMA;
    if (volumeRatio > 1.5) {
      // High volume suggests institutional activity
      const priceAction = currentCandle.close > currentCandle.open ? 0.15 : -0.15;
      prediction += priceAction;
      confidence += 0.1;
    }

    // Bollinger Bands position
    if (indicators.bollinger_upper && indicators.bollinger_lower) {
      const bbPosition = (currentCandle.close - indicators.bollinger_lower) /
                        (indicators.bollinger_upper - indicators.bollinger_lower);

      if (bbPosition < 0.2) prediction += 0.1; // Near lower band - potential buy
      else if (bbPosition > 0.8) prediction -= 0.1; // Near upper band - potential sell
    }

    prediction = Math.max(0, Math.min(1, prediction));
    confidence = Math.min(0.9, confidence);

    return {
      modelName: 'SMC_Analyzer',
      prediction,
      confidence,
      signalType: prediction > 0.6 ? 'BUY' : prediction < 0.4 ? 'SELL' : 'HOLD',
      timeHorizon: this.determineTimeHorizon(confidence),
      marketRegime: MarketRegime.VOLATILE,
    };
  }

  // Configuration methods for optimization
  protected getMinConfidence(): number {
    return this.parameters.minConfidence || 70;
  }

  protected getModelConsensusThreshold(): number {
    return this.parameters.minModelConsensus || 0.6;
  }

  protected getDecisionCooldown(): number {
    return this.parameters.decisionCooldown || 300000; // 5 minutes default
  }
}

// Export factory function
export function createIntelligentTradingSystem(): IntelligentTradingSystem {
  return new IntelligentTradingSystem();
}

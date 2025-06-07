/**
 * Enhanced Trading Decision Engine
 * Core ML-driven trading logic with ensemble models, confidence scoring, and intelligent entry/exit decisions
 * Optimized for small capital + high leverage + pinpoint entry/exit precision
 */

import { DataCollectorIntegration, ExtractedFeatures } from './DataCollectorIntegration';
import { MLTradingDecisionEngine } from './MLTradingDecisionEngine';
import { MultiTimeframeAnalysisEngine } from './MultiTimeframeAnalysisEngine';
import { EnhancedMLIntegrationService } from './EnhancedMLIntegrationService';
import { DeltaTradingBot } from './DeltaTradingBot';
import { logger } from '../utils/logger';

// Trading decision types
export type TradingAction = 'buy' | 'sell' | 'hold' | 'close_long' | 'close_short';

// Enhanced trading decision structure
export interface TradingDecision {
  action: TradingAction;
  confidence: number;           // 0-1 ML confidence score
  symbol: string;
  timestamp: number;
  
  // Entry/Exit details
  entryPrice?: number;
  exitPrice?: number;
  stopLoss: number;
  takeProfit: number;
  positionSize: number;         // Percentage of balance
  leverage: number;             // Leverage multiplier
  
  // ML model insights
  modelVotes: {
    lstm: { action: TradingAction; confidence: number };
    transformer: { action: TradingAction; confidence: number };
    ensemble: { action: TradingAction; confidence: number };
  };
  
  // Feature analysis
  keyFeatures: {
    fibonacciSignal: number;    // -1 to 1
    biasAlignment: number;      // 0 to 1
    candleStrength: number;     // 0 to 1
    volumeConfirmation: number; // 0 to 1
    marketTiming: number;       // 0 to 1
  };
  
  // Risk assessment
  riskScore: number;            // 0-1 (higher = riskier)
  maxDrawdown: number;          // Expected maximum drawdown
  winProbability: number;       // Estimated win probability
  
  // Execution details
  urgency: 'low' | 'medium' | 'high';
  timeToLive: number;           // Milliseconds until decision expires
  reasoning: string[];          // Human-readable decision factors
}

// Decision engine configuration
export interface DecisionEngineConfig {
  // Confidence thresholds
  minConfidenceThreshold: number;    // Minimum confidence for any trade (0.70)
  highConfidenceThreshold: number;   // High confidence threshold (0.85)
  
  // Position sizing (optimized for small capital + high leverage)
  basePositionSize: number;          // Base position size (3%)
  maxPositionSize: number;           // Maximum position size (8%)
  confidenceMultiplier: number;      // Position size multiplier based on confidence
  
  // Risk management (enhanced for pinpoint trading)
  baseLeverage: number;              // Base leverage (100x)
  maxLeverage: number;               // Maximum leverage (200x)
  stopLossBase: number;              // Base stop loss (1.2%)
  takeProfitBase: number;            // Base take profit (4%)
  
  // Model weights for ensemble voting
  modelWeights: {
    lstm: number;                    // LSTM weight (0.35)
    transformer: number;             // Transformer weight (0.40)
    ensemble: number;                // Ensemble weight (0.25)
  };
  
  // Feature importance weights
  featureWeights: {
    fibonacci: number;               // Fibonacci analysis weight
    bias: number;                    // Multi-timeframe bias weight
    candles: number;                 // Candle formation weight
    volume: number;                  // Volume analysis weight
    timing: number;                  // Market timing weight
  };
}

export class EnhancedTradingDecisionEngine {
  private dataIntegration: DataCollectorIntegration;
  private mlEngine: MLTradingDecisionEngine;
  private mtfAnalyzer: MultiTimeframeAnalysisEngine;
  private mlService: EnhancedMLIntegrationService;
  private tradingBot: DeltaTradingBot;
  
  // Enhanced configuration for small capital + high leverage
  private config: DecisionEngineConfig = {
    // Higher confidence thresholds for pinpoint entries
    minConfidenceThreshold: 0.70,
    highConfidenceThreshold: 0.85,
    
    // Optimized position sizing for small capital
    basePositionSize: 0.03,           // 3% base position
    maxPositionSize: 0.08,            // 8% maximum position
    confidenceMultiplier: 1.5,        // Scale with confidence
    
    // Enhanced leverage for maximum profit
    baseLeverage: 100,                // 100x base leverage
    maxLeverage: 200,                 // 200x maximum leverage
    stopLossBase: 0.012,              // 1.2% stop loss (tight for pinpoint entries)
    takeProfitBase: 0.040,            // 4% take profit (maximize profit)
    
    // Optimized model weights based on backtesting
    modelWeights: {
      lstm: 0.35,                     // LSTM for sequential patterns
      transformer: 0.40,              // Transformer for attention-based analysis
      ensemble: 0.25                  // Ensemble for stability
    },
    
    // Feature importance weights
    featureWeights: {
      fibonacci: 0.30,                // High weight for Fibonacci levels
      bias: 0.25,                     // Multi-timeframe bias importance
      candles: 0.20,                  // Candle formation analysis
      volume: 0.15,                   // Volume confirmation
      timing: 0.10                    // Market timing
    }
  };

  // Active decisions cache
  private activeDecisions: Map<string, TradingDecision> = new Map();
  private decisionHistory: TradingDecision[] = [];

  constructor() {
    this.dataIntegration = new DataCollectorIntegration();
    this.mtfAnalyzer = new MultiTimeframeAnalysisEngine();
    this.mlService = new EnhancedMLIntegrationService();
    this.tradingBot = new DeltaTradingBot();
    this.mlEngine = new MLTradingDecisionEngine(this.mtfAnalyzer, this.mlService, this.tradingBot);
  }

  /**
   * Initialize the enhanced trading decision engine
   */
  public async initialize(): Promise<void> {
    try {
      logger.info('🧠 Initializing Enhanced Trading Decision Engine...');
      
      // Initialize all components
      await this.dataIntegration.initialize();
      await this.mlEngine.initialize();
      
      // Connect data integration to ML engine
      this.dataIntegration.setMLEngine(this.mlEngine);
      
      logger.info('✅ Enhanced Trading Decision Engine initialized successfully');
      logger.info(`🎯 Configuration: Min Confidence ${(this.config.minConfidenceThreshold * 100).toFixed(0)}%, Max Position ${(this.config.maxPositionSize * 100).toFixed(0)}%, Max Leverage ${this.config.maxLeverage}x`);
      
    } catch (error: any) {
      logger.error('❌ Failed to initialize Enhanced Trading Decision Engine:', error.message);
      throw error;
    }
  }

  /**
   * Generate comprehensive trading decision for a symbol
   */
  public async generateTradingDecision(symbol: string): Promise<TradingDecision | null> {
    try {
      logger.debug(`🧠 Generating trading decision for ${symbol}...`);

      // Step 1: Extract ML features from multi-timeframe data
      const features = await this.dataIntegration.getRealTimeTradingFeatures(symbol);
      if (!features || features.dataQuality < 0.8) {
        logger.warn(`⚠️ Insufficient data quality for ${symbol}: ${features?.dataQuality || 0}`);
        return null;
      }

      // Step 2: Get ML model predictions
      const modelPredictions = await this.getMLModelPredictions(features);
      if (!modelPredictions) {
        logger.warn(`⚠️ Failed to get ML predictions for ${symbol}`);
        return null;
      }

      // Step 3: Calculate ensemble confidence and action
      const ensembleDecision = this.calculateEnsembleDecision(modelPredictions);
      
      // Step 4: Check confidence threshold
      if (ensembleDecision.confidence < this.config.minConfidenceThreshold) {
        logger.debug(`📊 Confidence too low for ${symbol}: ${(ensembleDecision.confidence * 100).toFixed(1)}%`);
        return null;
      }

      // Step 5: Analyze key features for decision support
      const keyFeatures = this.analyzeKeyFeatures(features);

      // Step 6: Calculate risk assessment
      const riskAssessment = this.calculateRiskAssessment(features, ensembleDecision);

      // Step 7: Determine position sizing and leverage
      const positionDetails = this.calculatePositionDetails(ensembleDecision.confidence, riskAssessment);

      // Step 8: Calculate stop loss and take profit levels
      const { stopLoss, takeProfit } = await this.calculateStopLossAndTakeProfit(
        symbol, 
        ensembleDecision.action, 
        ensembleDecision.confidence,
        features
      );

      // Step 9: Generate comprehensive trading decision
      const decision: TradingDecision = {
        action: ensembleDecision.action,
        confidence: ensembleDecision.confidence,
        symbol,
        timestamp: Date.now(),
        
        // Position details
        stopLoss,
        takeProfit,
        positionSize: positionDetails.size,
        leverage: positionDetails.leverage,
        
        // ML insights
        modelVotes: modelPredictions,
        keyFeatures,
        
        // Risk assessment
        riskScore: riskAssessment.score,
        maxDrawdown: riskAssessment.maxDrawdown,
        winProbability: riskAssessment.winProbability,
        
        // Execution details
        urgency: this.determineUrgency(ensembleDecision.confidence, keyFeatures),
        timeToLive: this.calculateTimeToLive(ensembleDecision.action, keyFeatures),
        reasoning: this.generateReasoningExplanation(features, ensembleDecision, keyFeatures)
      };

      // Cache the decision
      this.activeDecisions.set(symbol, decision);
      this.decisionHistory.push(decision);

      logger.info(`🎯 Trading decision generated for ${symbol}:`);
      logger.info(`   Action: ${decision.action.toUpperCase()}`);
      logger.info(`   Confidence: ${(decision.confidence * 100).toFixed(1)}%`);
      logger.info(`   Position Size: ${(decision.positionSize * 100).toFixed(1)}%`);
      logger.info(`   Leverage: ${decision.leverage}x`);
      logger.info(`   Risk Score: ${(decision.riskScore * 100).toFixed(1)}%`);

      return decision;

    } catch (error: any) {
      logger.error(`❌ Failed to generate trading decision for ${symbol}:`, error.message);
      return null;
    }
  }

  /**
   * Get the latest trading decision for a symbol
   */
  public getLatestDecision(symbol: string): TradingDecision | null {
    return this.activeDecisions.get(symbol) || null;
  }

  /**
   * Get decision history
   */
  public getDecisionHistory(limit: number = 100): TradingDecision[] {
    return this.decisionHistory.slice(-limit);
  }

  /**
   * Update configuration
   */
  public updateConfiguration(newConfig: Partial<DecisionEngineConfig>): void {
    this.config = { ...this.config, ...newConfig };
    logger.info('🔧 Enhanced Trading Decision Engine configuration updated');
  }

  /**
   * Get current configuration
   */
  public getConfiguration(): DecisionEngineConfig {
    return { ...this.config };
  }

  /**
   * Cleanup resources
   */
  public async cleanup(): Promise<void> {
    try {
      logger.info('🧹 Cleaning up Enhanced Trading Decision Engine...');
      await this.dataIntegration.cleanup();
      this.activeDecisions.clear();
      logger.info('✅ Enhanced Trading Decision Engine cleanup completed');
    } catch (error: any) {
      logger.error('❌ Error during Enhanced Trading Decision Engine cleanup:', error.message);
    }
  }

  // Private methods for decision logic

  /**
   * Get ML model predictions from all models
   */
  private async getMLModelPredictions(features: ExtractedFeatures): Promise<any> {
    try {
      // Convert features to ML input format
      const mlInput = this.convertFeaturesToMLInput(features);

      // Get predictions from all models
      const lstmPrediction = await this.mlService.predictWithLSTM(mlInput);
      const transformerPrediction = await this.mlService.predictWithTransformer(mlInput);
      const ensemblePrediction = await this.mlService.predictWithEnsemble(mlInput);

      return {
        lstm: {
          action: this.convertPredictionToAction(lstmPrediction),
          confidence: lstmPrediction.confidence || 0.5
        },
        transformer: {
          action: this.convertPredictionToAction(transformerPrediction),
          confidence: transformerPrediction.confidence || 0.5
        },
        ensemble: {
          action: this.convertPredictionToAction(ensemblePrediction),
          confidence: ensemblePrediction.confidence || 0.5
        }
      };

    } catch (error: any) {
      logger.error('❌ Failed to get ML model predictions:', error.message);
      return null;
    }
  }

  /**
   * Calculate ensemble decision from model votes
   */
  private calculateEnsembleDecision(modelPredictions: any): { action: TradingAction; confidence: number } {
    const { lstm, transformer, ensemble } = modelPredictions;
    const weights = this.config.modelWeights;

    // Calculate weighted confidence
    const weightedConfidence =
      (lstm.confidence * weights.lstm) +
      (transformer.confidence * weights.transformer) +
      (ensemble.confidence * weights.ensemble);

    // Determine action based on weighted voting
    const actionVotes = {
      buy: 0,
      sell: 0,
      hold: 0
    };

    // Weight the votes
    if (lstm.action === 'buy') actionVotes.buy += weights.lstm;
    else if (lstm.action === 'sell') actionVotes.sell += weights.lstm;
    else actionVotes.hold += weights.lstm;

    if (transformer.action === 'buy') actionVotes.buy += weights.transformer;
    else if (transformer.action === 'sell') actionVotes.sell += weights.transformer;
    else actionVotes.hold += weights.transformer;

    if (ensemble.action === 'buy') actionVotes.buy += weights.ensemble;
    else if (ensemble.action === 'sell') actionVotes.sell += weights.ensemble;
    else actionVotes.hold += weights.ensemble;

    // Determine final action
    let finalAction: TradingAction = 'hold';
    let maxVote = actionVotes.hold;

    if (actionVotes.buy > maxVote) {
      finalAction = 'buy';
      maxVote = actionVotes.buy;
    }
    if (actionVotes.sell > maxVote) {
      finalAction = 'sell';
    }

    // Adjust confidence based on vote consensus
    const consensusBonus = maxVote > 0.7 ? 0.1 : 0; // Bonus for strong consensus
    const finalConfidence = Math.min(1.0, weightedConfidence + consensusBonus);

    return { action: finalAction, confidence: finalConfidence };
  }

  /**
   * Analyze key features for decision support
   */
  private analyzeKeyFeatures(features: ExtractedFeatures): any {
    const weights = this.config.featureWeights;

    // Fibonacci signal strength
    const fibonacciSignal = this.calculateFibonacciSignal(features);

    // Bias alignment strength
    const biasAlignment = features.biasAlignment;

    // Candle formation strength
    const candleStrength = this.calculateCandleStrength(features);

    // Volume confirmation
    const volumeConfirmation = Math.min(1.0, features.volumeRatio / 2); // Normalize volume ratio

    // Market timing score
    const marketTiming = this.calculateMarketTiming(features);

    return {
      fibonacciSignal,
      biasAlignment,
      candleStrength,
      volumeConfirmation,
      marketTiming
    };
  }

  /**
   * Calculate risk assessment
   */
  private calculateRiskAssessment(features: ExtractedFeatures, decision: any): any {
    // Base risk from volatility
    const volatilityRisk = Math.min(1.0, features.volatility * 2);

    // Risk from data quality
    const dataQualityRisk = 1 - features.dataQuality;

    // Risk from confidence level
    const confidenceRisk = 1 - decision.confidence;

    // Risk from market timing
    const timingRisk = features.marketSession === 0 ? 0.2 : 0; // Higher risk during Asian session

    // Combined risk score
    const riskScore = Math.min(1.0,
      (volatilityRisk * 0.4) +
      (dataQualityRisk * 0.3) +
      (confidenceRisk * 0.2) +
      (timingRisk * 0.1)
    );

    // Estimate maximum drawdown based on risk
    const maxDrawdown = riskScore * 0.05; // 0-5% based on risk

    // Estimate win probability based on confidence and risk
    const winProbability = Math.max(0.5, decision.confidence * (1 - riskScore * 0.3));

    return {
      score: riskScore,
      maxDrawdown,
      winProbability
    };
  }

  /**
   * Calculate position details based on confidence and risk
   */
  private calculatePositionDetails(confidence: number, riskAssessment: any): any {
    // Position size based on confidence (higher confidence = larger position)
    const confidenceMultiplier = 1 + ((confidence - this.config.minConfidenceThreshold) * this.config.confidenceMultiplier);
    let positionSize = this.config.basePositionSize * confidenceMultiplier;

    // Adjust for risk (higher risk = smaller position)
    positionSize *= (1 - riskAssessment.score * 0.5);

    // Cap at maximum
    positionSize = Math.min(positionSize, this.config.maxPositionSize);

    // Leverage based on confidence and risk
    let leverage = this.config.baseLeverage;

    // Increase leverage for high confidence
    if (confidence > this.config.highConfidenceThreshold) {
      leverage = Math.min(this.config.maxLeverage, leverage * 1.5);
    }

    // Reduce leverage for high risk
    leverage *= (1 - riskAssessment.score * 0.3);

    // Ensure minimum leverage
    leverage = Math.max(50, Math.round(leverage));

    return {
      size: Math.round(positionSize * 10000) / 10000, // Round to 4 decimals
      leverage: Math.round(leverage)
    };
  }

  /**
   * Calculate stop loss and take profit levels
   */
  private async calculateStopLossAndTakeProfit(
    symbol: string,
    action: TradingAction,
    confidence: number,
    features: ExtractedFeatures
  ): Promise<{ stopLoss: number; takeProfit: number }> {

    // Get current price (using close price from features)
    const currentPrice = features.pricePosition; // This needs to be actual price

    // Base stop loss and take profit
    let stopLossPercent = this.config.stopLossBase;
    let takeProfitPercent = this.config.takeProfitBase;

    // Adjust based on confidence (higher confidence = tighter stops, higher targets)
    if (confidence > this.config.highConfidenceThreshold) {
      stopLossPercent *= 0.8;  // Tighter stop loss
      takeProfitPercent *= 1.3; // Higher take profit
    }

    // Adjust based on volatility
    const volatilityMultiplier = 1 + (features.volatility * 0.5);
    stopLossPercent *= volatilityMultiplier;
    takeProfitPercent *= volatilityMultiplier;

    // Calculate actual levels
    let stopLoss: number;
    let takeProfit: number;

    if (action === 'buy') {
      stopLoss = currentPrice * (1 - stopLossPercent);
      takeProfit = currentPrice * (1 + takeProfitPercent);
    } else { // sell
      stopLoss = currentPrice * (1 + stopLossPercent);
      takeProfit = currentPrice * (1 - takeProfitPercent);
    }

    return {
      stopLoss: Math.round(stopLoss * 100) / 100,
      takeProfit: Math.round(takeProfit * 100) / 100
    };
  }

  /**
   * Convert features to ML input format
   */
  private convertFeaturesToMLInput(features: ExtractedFeatures): number[] {
    return [
      // Fibonacci features (7)
      ...features.fibonacciProximity,
      features.nearestFibLevel,
      features.fibStrength,

      // Multi-timeframe bias features (6)
      features.bias4h,
      features.bias1h,
      features.bias15m,
      features.bias5m,
      features.overallBias,
      features.biasAlignment,

      // Candle formation features (7)
      features.bodyPercentage,
      features.wickPercentage,
      features.buyingPressure,
      features.sellingPressure,
      features.candleType,
      features.momentum,
      features.volatility,

      // Market context features (5)
      features.volume,
      features.volumeRatio,
      features.timeOfDay,
      features.marketSession,
      features.pricePosition
    ];
  }

  /**
   * Convert ML prediction to trading action
   */
  private convertPredictionToAction(prediction: any): TradingAction {
    if (!prediction || typeof prediction.action !== 'string') {
      return 'hold';
    }

    const action = prediction.action.toLowerCase();
    if (['buy', 'long'].includes(action)) return 'buy';
    if (['sell', 'short'].includes(action)) return 'sell';
    return 'hold';
  }

  /**
   * Calculate Fibonacci signal strength
   */
  private calculateFibonacciSignal(features: ExtractedFeatures): number {
    // Find the strongest Fibonacci level proximity
    const maxProximity = Math.max(...features.fibonacciProximity);

    // Combine with Fibonacci strength
    const signal = (maxProximity + features.fibStrength) / 2;

    // Convert to -1 to 1 range (negative for sell, positive for buy)
    return (signal - 0.5) * 2;
  }

  /**
   * Calculate candle formation strength
   */
  private calculateCandleStrength(features: ExtractedFeatures): number {
    // Combine multiple candle factors
    const bodyStrength = features.bodyPercentage;
    const pressureBalance = Math.abs(features.buyingPressure - features.sellingPressure);
    const momentumStrength = Math.abs(features.momentum);

    return (bodyStrength + pressureBalance + momentumStrength) / 3;
  }

  /**
   * Calculate market timing score
   */
  private calculateMarketTiming(features: ExtractedFeatures): number {
    // Higher score for active trading sessions
    let timingScore = 0.5; // Base score

    // Boost for European and American sessions
    if (features.marketSession >= 1) {
      timingScore += 0.3;
    }

    // Boost for overlap periods
    if (features.marketSession === 2) {
      timingScore += 0.2;
    }

    return Math.min(1.0, timingScore);
  }

  /**
   * Determine trade urgency
   */
  private determineUrgency(confidence: number, keyFeatures: any): 'low' | 'medium' | 'high' {
    if (confidence > this.config.highConfidenceThreshold && keyFeatures.biasAlignment > 0.8) {
      return 'high';
    }
    if (confidence > (this.config.minConfidenceThreshold + 0.1)) {
      return 'medium';
    }
    return 'low';
  }

  /**
   * Calculate time to live for decision
   */
  private calculateTimeToLive(action: TradingAction, keyFeatures: any): number {
    // Base TTL: 5 minutes
    let ttl = 5 * 60 * 1000;

    // Shorter TTL for high urgency trades
    if (keyFeatures.biasAlignment > 0.8) {
      ttl = 2 * 60 * 1000; // 2 minutes
    }

    // Longer TTL for hold actions
    if (action === 'hold') {
      ttl = 10 * 60 * 1000; // 10 minutes
    }

    return ttl;
  }

  /**
   * Generate human-readable reasoning explanation
   */
  private generateReasoningExplanation(
    features: ExtractedFeatures,
    decision: any,
    keyFeatures: any
  ): string[] {
    const reasoning: string[] = [];

    // Confidence explanation
    reasoning.push(`ML ensemble confidence: ${(decision.confidence * 100).toFixed(1)}%`);

    // Fibonacci analysis
    if (keyFeatures.fibonacciSignal > 0.3) {
      reasoning.push(`Strong Fibonacci support detected (${(keyFeatures.fibonacciSignal * 100).toFixed(0)}%)`);
    } else if (keyFeatures.fibonacciSignal < -0.3) {
      reasoning.push(`Strong Fibonacci resistance detected (${Math.abs(keyFeatures.fibonacciSignal * 100).toFixed(0)}%)`);
    }

    // Bias alignment
    if (keyFeatures.biasAlignment > 0.7) {
      reasoning.push(`Excellent multi-timeframe bias alignment (${(keyFeatures.biasAlignment * 100).toFixed(0)}%)`);
    } else if (keyFeatures.biasAlignment < 0.4) {
      reasoning.push(`Poor timeframe alignment - conflicting signals`);
    }

    // Candle strength
    if (keyFeatures.candleStrength > 0.7) {
      reasoning.push(`Strong candle formation pattern detected`);
    }

    // Volume confirmation
    if (keyFeatures.volumeConfirmation > 0.8) {
      reasoning.push(`High volume confirmation (${(features.volumeRatio).toFixed(1)}x average)`);
    } else if (keyFeatures.volumeConfirmation < 0.3) {
      reasoning.push(`Low volume - weak confirmation`);
    }

    // Market timing
    if (keyFeatures.marketTiming > 0.8) {
      reasoning.push(`Optimal market timing - active trading session`);
    } else if (keyFeatures.marketTiming < 0.4) {
      reasoning.push(`Suboptimal timing - low activity session`);
    }

    // Data quality
    if (features.dataQuality < 0.9) {
      reasoning.push(`Data quality: ${(features.dataQuality * 100).toFixed(0)}% - proceed with caution`);
    }

    return reasoning;
  }
}

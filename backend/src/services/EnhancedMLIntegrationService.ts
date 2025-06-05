/**
 * Enhanced ML Integration Service
 * Integrates with existing AdvancedIntelligenceSystem, EnhancedTradingPredictor, and SMCDetector
 */

import { MultiTimeframeAnalysisEngine, CrossTimeframeAnalysis } from './MultiTimeframeAnalysisEngine';
import { EnhancedMarketRegimeDetector, EnhancedRegimeAnalysis } from './EnhancedMarketRegimeDetector';
import { DeltaExchangeUnified, MultiTimeframeData } from './DeltaExchangeUnified';
import { logger } from '../utils/logger';

export interface MLPrediction {
  model_name: string;
  prediction_type: 'position_outcome' | 'market_regime' | 'price_direction' | 'volatility';
  prediction_value: number; // 0-1 for probabilities, actual values for prices
  confidence: number; // 0-1
  time_horizon: number; // minutes
  features_used: string[];
  model_version: string;
  timestamp: number;
}

export interface PositionOutcomePrediction {
  profit_probability: number; // 0-1
  expected_return: number; // percentage
  time_to_target: number; // minutes
  risk_score: number; // 0-1
  confidence: number; // 0-1
  contributing_factors: {
    trend_alignment: number;
    regime_compatibility: number;
    volatility_factor: number;
    momentum_score: number;
    technical_indicators: number;
  };
  recommendations: {
    position_size_multiplier: number; // 0.5-2.0
    hold_duration: number; // minutes
    exit_strategy: 'aggressive' | 'moderate' | 'conservative';
  };
}

export interface MarketRegimePrediction {
  current_regime: string;
  regime_probability: number; // 0-1
  transition_probability: number; // 0-1
  next_likely_regime: string;
  regime_duration: number; // minutes
  confidence: number; // 0-1
  regime_characteristics: {
    volatility_level: 'low' | 'medium' | 'high';
    trend_strength: number; // 0-1
    mean_reversion_tendency: number; // 0-1
  };
}

export interface EnhancedMLFeatures {
  price_features: {
    returns: number[];
    volatility: number;
    momentum: number;
    rsi: number;
    macd: number;
    atr_normalized: number;
  };
  volume_features: {
    volume_ratio: number;
    volume_trend: number;
    volume_volatility: number;
  };
  timeframe_features: {
    short_term_trend: number;
    medium_term_trend: number;
    long_term_trend: number;
    trend_alignment: number;
  };
  regime_features: {
    current_regime_score: number;
    regime_stability: number;
    transition_signals: number[];
  };
  sentiment_features: {
    market_sentiment: number;
    fear_greed_index: number;
    institutional_flow: number;
  };
}

export class EnhancedMLIntegrationService {
  private deltaService: DeltaExchangeUnified;
  private mtfAnalyzer: MultiTimeframeAnalysisEngine;
  private regimeDetector: EnhancedMarketRegimeDetector;
  private modelVersions: Map<string, string> = new Map();
  private predictionCache: Map<string, MLPrediction[]> = new Map();

  constructor(deltaService: DeltaExchangeUnified) {
    this.deltaService = deltaService;
    this.mtfAnalyzer = new MultiTimeframeAnalysisEngine(deltaService);
    this.regimeDetector = new EnhancedMarketRegimeDetector(deltaService);
    
    // Initialize model versions
    this.modelVersions.set('position_outcome_predictor', 'v2.1.0');
    this.modelVersions.set('regime_detector', 'v1.8.0');
    this.modelVersions.set('volatility_predictor', 'v1.5.0');
    this.modelVersions.set('rl_agent', 'v3.0.0');
  }

  /**
   * Predict position outcome using enhanced ML models
   */
  public async predictPositionOutcome(
    symbol: string,
    side: 'LONG' | 'SHORT',
    entryPrice: number,
    currentPrice: number,
    positionAge: number
  ): Promise<PositionOutcomePrediction> {
    try {
      logger.info(`🤖 Predicting position outcome for ${symbol} ${side}`);
      
      // Get market analysis
      const mtfAnalysis = await this.mtfAnalyzer.analyzeSymbol(symbol);
      const regimeAnalysis = await this.regimeDetector.detectRegime(symbol);
      const marketData = await this.deltaService.getMultiTimeframeData(symbol);
      
      // Extract enhanced features
      const features = this.extractEnhancedFeatures(marketData, mtfAnalysis, regimeAnalysis);
      
      // Calculate position metrics
      const priceChange = ((currentPrice - entryPrice) / entryPrice) * 100;
      const positionDirection = side === 'LONG' ? 1 : -1;
      const alignedReturn = priceChange * positionDirection;
      
      // Simulate advanced ML prediction (in production, this would call actual ML models)
      const prediction = this.simulateAdvancedPositionPrediction(
        features,
        alignedReturn,
        positionAge,
        mtfAnalysis,
        regimeAnalysis
      );
      
      logger.info(`✅ Position outcome predicted: ${(prediction.profit_probability * 100).toFixed(1)}% profit probability`);
      
      return prediction;
      
    } catch (error) {
      logger.error(`❌ Error predicting position outcome for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Predict market regime transitions
   */
  public async predictMarketRegime(symbol: string): Promise<MarketRegimePrediction> {
    try {
      logger.info(`🔮 Predicting market regime for ${symbol}`);
      
      const regimeAnalysis = await this.regimeDetector.detectRegime(symbol);
      const mtfAnalysis = await this.mtfAnalyzer.analyzeSymbol(symbol);
      const marketData = await this.deltaService.getMultiTimeframeData(symbol);
      
      // Extract features for regime prediction
      const features = this.extractEnhancedFeatures(marketData, mtfAnalysis, regimeAnalysis);
      
      // Simulate regime prediction (in production, this would use actual ML models)
      const prediction = this.simulateRegimePrediction(features, regimeAnalysis, mtfAnalysis);
      
      logger.info(`✅ Regime predicted: ${prediction.current_regime} (${(prediction.confidence * 100).toFixed(1)}% confidence)`);
      
      return prediction;
      
    } catch (error) {
      logger.error(`❌ Error predicting market regime for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Get ensemble prediction combining multiple models
   */
  public async getEnsemblePrediction(
    symbol: string,
    side: 'LONG' | 'SHORT',
    entryPrice: number,
    currentPrice: number
  ): Promise<{
    position_outcome: PositionOutcomePrediction;
    regime_prediction: MarketRegimePrediction;
    ensemble_confidence: number;
    recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  }> {
    try {
      logger.info(`🎯 Getting ensemble prediction for ${symbol}`);
      
      // Get individual predictions
      const positionOutcome = await this.predictPositionOutcome(
        symbol, side, entryPrice, currentPrice, Date.now()
      );
      
      const regimePrediction = await this.predictMarketRegime(symbol);
      
      // Calculate ensemble confidence
      const ensembleConfidence = (
        positionOutcome.confidence * 0.6 +
        regimePrediction.confidence * 0.4
      );
      
      // Generate ensemble recommendation
      const recommendation = this.generateEnsembleRecommendation(
        positionOutcome,
        regimePrediction,
        side
      );
      
      logger.info(`✅ Ensemble prediction: ${recommendation} (${(ensembleConfidence * 100).toFixed(1)}% confidence)`);
      
      return {
        position_outcome: positionOutcome,
        regime_prediction: regimePrediction,
        ensemble_confidence: ensembleConfidence,
        recommendation
      };
      
    } catch (error) {
      logger.error(`❌ Error getting ensemble prediction for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Extract enhanced features for ML models
   */
  private extractEnhancedFeatures(
    marketData: MultiTimeframeData,
    mtfAnalysis: CrossTimeframeAnalysis,
    regimeAnalysis: EnhancedRegimeAnalysis
  ): EnhancedMLFeatures {
    const hourlyData = marketData.timeframes['1h'];
    const dailyData = marketData.timeframes['1d'];
    
    // Price features
    const priceFeatures = {
      returns: this.calculateReturns(hourlyData?.candles || []),
      volatility: regimeAnalysis.volatility_metrics.price_volatility,
      momentum: mtfAnalysis.overallTrend.strength,
      rsi: hourlyData?.indicators.rsi || 50,
      macd: hourlyData?.indicators.macd?.macd || 0,
      atr_normalized: regimeAnalysis.volatility_metrics.atr_normalized
    };
    
    // Volume features
    const volumeFeatures = {
      volume_ratio: this.calculateVolumeRatio(hourlyData?.candles || []),
      volume_trend: this.calculateVolumeTrend(hourlyData?.candles || []),
      volume_volatility: regimeAnalysis.volatility_metrics.volume_volatility
    };
    
    // Timeframe features
    const timeframeFeatures = {
      short_term_trend: this.getTrendScore(mtfAnalysis.trends['15m']),
      medium_term_trend: this.getTrendScore(mtfAnalysis.trends['1h']),
      long_term_trend: this.getTrendScore(mtfAnalysis.trends['4h']),
      trend_alignment: mtfAnalysis.overallTrend.alignment
    };
    
    // Regime features
    const regimeFeatures = {
      current_regime_score: regimeAnalysis.confidence,
      regime_stability: this.calculateRegimeStability(regimeAnalysis),
      transition_signals: this.calculateTransitionSignals(regimeAnalysis)
    };
    
    // Sentiment features (simplified)
    const sentimentFeatures = {
      market_sentiment: 0.5, // Would integrate with sentiment analysis
      fear_greed_index: 0.5, // Would integrate with fear/greed index
      institutional_flow: 0.5 // Would integrate with institutional flow data
    };
    
    return {
      price_features: priceFeatures,
      volume_features: volumeFeatures,
      timeframe_features: timeframeFeatures,
      regime_features: regimeFeatures,
      sentiment_features: sentimentFeatures
    };
  }

  /**
   * Simulate advanced position prediction (placeholder for actual ML model)
   */
  private simulateAdvancedPositionPrediction(
    features: EnhancedMLFeatures,
    alignedReturn: number,
    positionAge: number,
    mtfAnalysis: CrossTimeframeAnalysis,
    regimeAnalysis: EnhancedRegimeAnalysis
  ): PositionOutcomePrediction {
    // Simulate sophisticated ML prediction
    let profitProbability = 0.5; // Base probability
    
    // Trend alignment factor
    const trendFactor = features.timeframe_features.trend_alignment;
    profitProbability += trendFactor * 0.3;
    
    // Regime compatibility
    const regimeFactor = features.regime_features.current_regime_score;
    profitProbability += regimeFactor * 0.2;
    
    // Technical indicators
    const rsi = features.price_features.rsi;
    if (rsi > 30 && rsi < 70) {
      profitProbability += 0.1; // Not in extreme territory
    }
    
    // Momentum factor
    profitProbability += features.price_features.momentum * 0.15;
    
    // Current performance factor
    if (alignedReturn > 0) {
      profitProbability += Math.min(0.2, alignedReturn / 100);
    } else {
      profitProbability += Math.max(-0.2, alignedReturn / 100);
    }
    
    profitProbability = Math.max(0.1, Math.min(0.9, profitProbability));
    
    // Calculate other metrics
    const expectedReturn = (profitProbability - 0.5) * 10; // -5% to +5%
    const timeToTarget = regimeAnalysis.duration_minutes;
    const riskScore = 1 - profitProbability;
    const confidence = Math.min(0.9, regimeFactor + mtfAnalysis.signals.confidence) / 2;
    
    return {
      profit_probability: profitProbability,
      expected_return: expectedReturn,
      time_to_target: timeToTarget,
      risk_score: riskScore,
      confidence: confidence,
      contributing_factors: {
        trend_alignment: trendFactor,
        regime_compatibility: regimeFactor,
        volatility_factor: features.price_features.volatility,
        momentum_score: features.price_features.momentum,
        technical_indicators: (rsi - 50) / 50 // Normalized RSI
      },
      recommendations: {
        position_size_multiplier: profitProbability > 0.7 ? 1.5 : profitProbability < 0.3 ? 0.5 : 1.0,
        hold_duration: timeToTarget,
        exit_strategy: profitProbability > 0.7 ? 'aggressive' : profitProbability < 0.4 ? 'conservative' : 'moderate'
      }
    };
  }

  /**
   * Simulate regime prediction (placeholder for actual ML model)
   */
  private simulateRegimePrediction(
    features: EnhancedMLFeatures,
    regimeAnalysis: EnhancedRegimeAnalysis,
    mtfAnalysis: CrossTimeframeAnalysis
  ): MarketRegimePrediction {
    const currentRegime = regimeAnalysis.current_regime;
    const volatility = features.price_features.volatility;
    const trendStrength = features.price_features.momentum;
    
    // Simulate regime transition probability
    let transitionProbability = 0.1; // Base 10% chance of regime change
    
    // High volatility increases transition probability
    if (volatility > 0.05) {
      transitionProbability += 0.3;
    }
    
    // Weak trends increase transition probability
    if (trendStrength < 0.3) {
      transitionProbability += 0.2;
    }
    
    transitionProbability = Math.min(0.8, transitionProbability);
    
    // Determine next likely regime
    let nextLikelyRegime = currentRegime;
    if (transitionProbability > 0.4) {
      if (currentRegime.includes('trending')) {
        nextLikelyRegime = 'sideways' as any;
      } else if (currentRegime === 'sideways') {
        nextLikelyRegime = trendStrength > 0.5 ? 'trending_bullish' as any : 'volatile' as any;
      } else if (currentRegime === 'volatile') {
        nextLikelyRegime = 'consolidation' as any;
      }
    }
    
    return {
      current_regime: currentRegime,
      regime_probability: regimeAnalysis.confidence,
      transition_probability: transitionProbability,
      next_likely_regime: nextLikelyRegime,
      regime_duration: regimeAnalysis.duration_minutes,
      confidence: regimeAnalysis.confidence,
      regime_characteristics: {
        volatility_level: volatility > 0.05 ? 'high' : volatility > 0.02 ? 'medium' : 'low',
        trend_strength: trendStrength,
        mean_reversion_tendency: currentRegime.includes('sideways') ? 0.8 : 0.3
      }
    };
  }

  /**
   * Generate ensemble recommendation
   */
  private generateEnsembleRecommendation(
    positionOutcome: PositionOutcomePrediction,
    regimePrediction: MarketRegimePrediction,
    side: 'LONG' | 'SHORT'
  ): 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL' {
    const profitProb = positionOutcome.profit_probability;
    const regimeCompatible = regimePrediction.regime_characteristics.trend_strength > 0.5;
    
    if (profitProb > 0.8 && regimeCompatible) {
      return side === 'LONG' ? 'STRONG_BUY' : 'STRONG_SELL';
    } else if (profitProb > 0.6) {
      return side === 'LONG' ? 'BUY' : 'SELL';
    } else if (profitProb < 0.3) {
      return side === 'LONG' ? 'SELL' : 'BUY';
    } else {
      return 'HOLD';
    }
  }

  // Helper methods
  private calculateReturns(candles: any[]): number[] {
    if (candles.length < 2) return [];
    
    return candles.slice(1).map((candle, i) => 
      (candle.close - candles[i].close) / candles[i].close
    );
  }

  private calculateVolumeRatio(candles: any[]): number {
    if (candles.length < 20) return 1;
    
    const recent = candles.slice(-5);
    const baseline = candles.slice(-20, -5);
    
    const recentAvg = recent.reduce((sum, c) => sum + c.volume, 0) / recent.length;
    const baselineAvg = baseline.reduce((sum, c) => sum + c.volume, 0) / baseline.length;
    
    return baselineAvg > 0 ? recentAvg / baselineAvg : 1;
  }

  private calculateVolumeTrend(candles: any[]): number {
    if (candles.length < 10) return 0;
    
    const volumes = candles.slice(-10).map(c => c.volume);
    const firstHalf = volumes.slice(0, 5);
    const secondHalf = volumes.slice(5);
    
    const firstAvg = firstHalf.reduce((sum, v) => sum + v, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((sum, v) => sum + v, 0) / secondHalf.length;
    
    return firstAvg > 0 ? (secondAvg - firstAvg) / firstAvg : 0;
  }

  private getTrendScore(trend: any): number {
    if (!trend) return 0;
    
    const directionScore = trend.direction === 'bullish' ? 1 : trend.direction === 'bearish' ? -1 : 0;
    return directionScore * trend.strength * trend.confidence;
  }

  private calculateRegimeStability(regimeAnalysis: EnhancedRegimeAnalysis): number {
    // Higher confidence and longer duration = more stable
    return regimeAnalysis.confidence * Math.min(1, regimeAnalysis.duration_minutes / 120);
  }

  private calculateTransitionSignals(regimeAnalysis: EnhancedRegimeAnalysis): number[] {
    // Simplified transition signals
    const volatilitySignal = regimeAnalysis.volatility_metrics.price_volatility > 0.05 ? 1 : 0;
    const trendSignal = regimeAnalysis.trend_strength.momentum_strength < 0.3 ? 1 : 0;
    const durationSignal = regimeAnalysis.duration_minutes > 300 ? 1 : 0; // Long duration suggests potential change
    
    return [volatilitySignal, trendSignal, durationSignal];
  }
}

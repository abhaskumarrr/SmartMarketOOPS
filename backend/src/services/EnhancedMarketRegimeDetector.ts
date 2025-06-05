/**
 * Enhanced Market Regime Detector
 * Integrates with existing SMC infrastructure and adds advanced regime detection
 */

import { MultiTimeframeAnalysisEngine, CrossTimeframeAnalysis, MarketRegime } from './MultiTimeframeAnalysisEngine';
import { DeltaExchangeUnified, MultiTimeframeData, TechnicalIndicators } from './DeltaExchangeUnified';
import { logger } from '../utils/logger';

// Enhanced regime types that extend our existing MarketRegime
export enum EnhancedMarketRegime {
  TRENDING_BULLISH = 'trending_bullish',
  TRENDING_BEARISH = 'trending_bearish',
  SIDEWAYS = 'sideways',
  VOLATILE = 'volatile',
  BREAKOUT_BULLISH = 'breakout_bullish',
  BREAKOUT_BEARISH = 'breakout_bearish',
  CONSOLIDATION = 'consolidation',
  ACCUMULATION = 'accumulation',
  DISTRIBUTION = 'distribution'
}

export interface VolatilityMetrics {
  atr_normalized: number;
  price_volatility: number;
  volume_volatility: number;
  garch_volatility?: number;
}

export interface TrendStrengthMetrics {
  adx: number;
  ma_slope: number;
  trend_consistency: number;
  momentum_strength: number;
}

export interface RegimeChangeDetection {
  change_detected: boolean;
  confidence: number;
  previous_regime: EnhancedMarketRegime;
  new_regime: EnhancedMarketRegime;
  change_timestamp: number;
  cusum_statistic?: number;
  bayesian_probability?: number;
}

export interface EnhancedRegimeAnalysis {
  current_regime: EnhancedMarketRegime;
  confidence: number;
  duration_minutes: number;
  volatility_metrics: VolatilityMetrics;
  trend_strength: TrendStrengthMetrics;
  regime_change: RegimeChangeDetection | null;
  trading_recommendations: {
    strategy_type: 'trend_following' | 'mean_reversion' | 'breakout' | 'scalping';
    risk_multiplier: number;
    optimal_timeframes: string[];
    position_sizing_factor: number;
  };
  smc_context: {
    order_block_strength: number;
    liquidity_levels: number;
    market_structure_quality: number;
  };
}

export class EnhancedMarketRegimeDetector {
  private deltaService: DeltaExchangeUnified;
  private mtfAnalyzer: MultiTimeframeAnalysisEngine;
  private regimeHistory: Map<string, EnhancedRegimeAnalysis[]> = new Map();
  private changeDetectionWindow: number = 50; // Number of periods for change detection
  private cusumThreshold: number = 5.0; // CUSUM threshold for regime change
  
  constructor(deltaService: DeltaExchangeUnified) {
    this.deltaService = deltaService;
    this.mtfAnalyzer = new MultiTimeframeAnalysisEngine(deltaService);
  }

  /**
   * Perform comprehensive regime detection with SMC integration
   */
  public async detectRegime(symbol: string): Promise<EnhancedRegimeAnalysis> {
    try {
      logger.info(`🔍 Enhanced regime detection for ${symbol}`);
      
      // Get multi-timeframe analysis
      const mtfAnalysis = await this.mtfAnalyzer.analyzeSymbol(symbol);
      
      // Get multi-timeframe data for detailed analysis
      const data = await this.deltaService.getMultiTimeframeData(symbol);
      
      // Calculate volatility metrics
      const volatilityMetrics = this.calculateVolatilityMetrics(data);
      
      // Calculate trend strength metrics
      const trendStrength = this.calculateTrendStrengthMetrics(data, mtfAnalysis);
      
      // Detect regime based on comprehensive analysis
      const currentRegime = this.classifyRegime(mtfAnalysis, volatilityMetrics, trendStrength);
      
      // Calculate regime confidence
      const confidence = this.calculateRegimeConfidence(mtfAnalysis, volatilityMetrics, trendStrength);
      
      // Detect regime changes
      const regimeChange = this.detectRegimeChange(symbol, currentRegime);
      
      // Estimate regime duration
      const duration = this.estimateRegimeDuration(currentRegime, volatilityMetrics);
      
      // Generate trading recommendations
      const tradingRecommendations = this.generateTradingRecommendations(
        currentRegime, 
        volatilityMetrics, 
        trendStrength
      );
      
      // Get SMC context (integrate with existing SMC system)
      const smcContext = this.getSMCContext(data);
      
      const analysis: EnhancedRegimeAnalysis = {
        current_regime: currentRegime,
        confidence,
        duration_minutes: duration,
        volatility_metrics: volatilityMetrics,
        trend_strength: trendStrength,
        regime_change: regimeChange,
        trading_recommendations: tradingRecommendations,
        smc_context: smcContext
      };
      
      // Store in history for change detection
      this.updateRegimeHistory(symbol, analysis);
      
      logger.info(`✅ Regime detected: ${currentRegime} (${(confidence * 100).toFixed(1)}% confidence)`);
      logger.info(`📊 Strategy: ${tradingRecommendations.strategy_type}, Risk: ${tradingRecommendations.risk_multiplier}x`);
      
      return analysis;
      
    } catch (error) {
      logger.error(`❌ Error detecting regime for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Calculate comprehensive volatility metrics
   */
  private calculateVolatilityMetrics(data: MultiTimeframeData): VolatilityMetrics {
    const hourlyData = data.timeframes['1h'];
    const dailyData = data.timeframes['1d'];
    
    let atr_normalized = 0;
    let price_volatility = 0;
    let volume_volatility = 0;
    
    if (hourlyData?.candles && hourlyData.candles.length > 20) {
      const candles = hourlyData.candles.slice(-20);
      const currentPrice = candles[candles.length - 1].close;
      
      // ATR normalized by price
      if (hourlyData.indicators.atr) {
        atr_normalized = hourlyData.indicators.atr / currentPrice;
      }
      
      // Price volatility (standard deviation of returns)
      const returns = candles.map((candle, i) => 
        i > 0 ? (candle.close - candles[i - 1].close) / candles[i - 1].close : 0
      ).slice(1);
      
      const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
      price_volatility = Math.sqrt(
        returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length
      );
      
      // Volume volatility
      const volumes = candles.map(c => c.volume);
      const meanVolume = volumes.reduce((sum, v) => sum + v, 0) / volumes.length;
      volume_volatility = Math.sqrt(
        volumes.reduce((sum, v) => sum + Math.pow(v - meanVolume, 2), 0) / volumes.length
      ) / meanVolume;
    }
    
    return {
      atr_normalized,
      price_volatility,
      volume_volatility
    };
  }

  /**
   * Calculate trend strength metrics including ADX-like calculations
   */
  private calculateTrendStrengthMetrics(
    data: MultiTimeframeData, 
    mtfAnalysis: CrossTimeframeAnalysis
  ): TrendStrengthMetrics {
    const hourlyData = data.timeframes['1h'];
    
    let adx = 0;
    let ma_slope = 0;
    let trend_consistency = 0;
    let momentum_strength = 0;
    
    if (hourlyData?.candles && hourlyData.candles.length > 14) {
      const candles = hourlyData.candles.slice(-14);
      
      // Calculate ADX-like indicator
      adx = this.calculateADX(candles);
      
      // Moving average slope
      const prices = candles.map(c => c.close);
      const firstHalf = prices.slice(0, 7);
      const secondHalf = prices.slice(7);
      const firstAvg = firstHalf.reduce((sum, p) => sum + p, 0) / firstHalf.length;
      const secondAvg = secondHalf.reduce((sum, p) => sum + p, 0) / secondHalf.length;
      ma_slope = (secondAvg - firstAvg) / firstAvg;
      
      // Trend consistency from MTF analysis
      trend_consistency = mtfAnalysis.overallTrend.alignment;
      
      // Momentum strength
      momentum_strength = mtfAnalysis.overallTrend.strength;
    }
    
    return {
      adx,
      ma_slope,
      trend_consistency,
      momentum_strength
    };
  }

  /**
   * Calculate ADX-like trend strength indicator
   */
  private calculateADX(candles: any[], period: number = 14): number {
    if (candles.length < period + 1) return 0;
    
    let dmPlus = 0;
    let dmMinus = 0;
    let trSum = 0;
    
    for (let i = 1; i < candles.length; i++) {
      const high = candles[i].high;
      const low = candles[i].low;
      const prevHigh = candles[i - 1].high;
      const prevLow = candles[i - 1].low;
      const prevClose = candles[i - 1].close;
      
      // True Range
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      trSum += tr;
      
      // Directional Movement
      const upMove = high - prevHigh;
      const downMove = prevLow - low;
      
      if (upMove > downMove && upMove > 0) {
        dmPlus += upMove;
      }
      if (downMove > upMove && downMove > 0) {
        dmMinus += downMove;
      }
    }
    
    if (trSum === 0) return 0;
    
    const diPlus = (dmPlus / trSum) * 100;
    const diMinus = (dmMinus / trSum) * 100;
    const dx = Math.abs(diPlus - diMinus) / (diPlus + diMinus) * 100;
    
    return dx;
  }

  /**
   * Classify regime based on comprehensive metrics
   */
  private classifyRegime(
    mtfAnalysis: CrossTimeframeAnalysis,
    volatility: VolatilityMetrics,
    trendStrength: TrendStrengthMetrics
  ): EnhancedMarketRegime {
    const { overallTrend } = mtfAnalysis;
    
    // High volatility regime
    if (volatility.price_volatility > 0.05 || volatility.atr_normalized > 0.03) {
      return EnhancedMarketRegime.VOLATILE;
    }
    
    // Strong trending regimes
    if (trendStrength.adx > 25 && trendStrength.trend_consistency > 0.7) {
      if (overallTrend.direction === 'bullish') {
        return EnhancedMarketRegime.TRENDING_BULLISH;
      } else if (overallTrend.direction === 'bearish') {
        return EnhancedMarketRegime.TRENDING_BEARISH;
      }
    }
    
    // Breakout detection
    if (trendStrength.momentum_strength > 0.8 && volatility.volume_volatility > 0.3) {
      if (overallTrend.direction === 'bullish') {
        return EnhancedMarketRegime.BREAKOUT_BULLISH;
      } else if (overallTrend.direction === 'bearish') {
        return EnhancedMarketRegime.BREAKOUT_BEARISH;
      }
    }
    
    // Consolidation (low volatility + weak trend)
    if (volatility.price_volatility < 0.02 && trendStrength.adx < 20) {
      return EnhancedMarketRegime.CONSOLIDATION;
    }
    
    // Default to sideways
    return EnhancedMarketRegime.SIDEWAYS;
  }

  /**
   * Calculate regime confidence score
   */
  private calculateRegimeConfidence(
    mtfAnalysis: CrossTimeframeAnalysis,
    volatility: VolatilityMetrics,
    trendStrength: TrendStrengthMetrics
  ): number {
    let confidence = 0.5; // Base confidence
    
    // MTF analysis confidence
    confidence += mtfAnalysis.overallTrend.confidence * 0.3;
    
    // Trend strength confidence
    confidence += (trendStrength.adx / 100) * 0.2;
    
    // Consistency bonus
    confidence += trendStrength.trend_consistency * 0.2;
    
    // Volatility clarity (either very high or very low is clear)
    const volatilityClarity = Math.abs(volatility.price_volatility - 0.025) / 0.025;
    confidence += Math.min(0.3, volatilityClarity) * 0.3;
    
    return Math.min(1, confidence);
  }

  /**
   * Detect regime changes using CUSUM-like algorithm
   */
  private detectRegimeChange(symbol: string, currentRegime: EnhancedMarketRegime): RegimeChangeDetection | null {
    const history = this.regimeHistory.get(symbol) || [];
    
    if (history.length < 2) {
      return null;
    }
    
    const previousRegime = history[history.length - 1].current_regime;
    
    if (currentRegime !== previousRegime) {
      // Simple regime change detection
      // In production, implement CUSUM or Bayesian changepoint detection
      return {
        change_detected: true,
        confidence: 0.8,
        previous_regime: previousRegime,
        new_regime: currentRegime,
        change_timestamp: Date.now()
      };
    }
    
    return null;
  }

  /**
   * Estimate regime duration based on regime type and volatility
   */
  private estimateRegimeDuration(regime: EnhancedMarketRegime, volatility: VolatilityMetrics): number {
    const baseDurations: Record<EnhancedMarketRegime, number> = {
      [EnhancedMarketRegime.TRENDING_BULLISH]: 240,
      [EnhancedMarketRegime.TRENDING_BEARISH]: 240,
      [EnhancedMarketRegime.VOLATILE]: 30,
      [EnhancedMarketRegime.BREAKOUT_BULLISH]: 60,
      [EnhancedMarketRegime.BREAKOUT_BEARISH]: 60,
      [EnhancedMarketRegime.CONSOLIDATION]: 180,
      [EnhancedMarketRegime.SIDEWAYS]: 120,
      [EnhancedMarketRegime.ACCUMULATION]: 300,
      [EnhancedMarketRegime.DISTRIBUTION]: 300
    };
    
    let duration = baseDurations[regime];
    
    // Adjust based on volatility
    if (volatility.price_volatility > 0.05) {
      duration *= 0.5; // High volatility shortens regime duration
    } else if (volatility.price_volatility < 0.01) {
      duration *= 1.5; // Low volatility extends regime duration
    }
    
    return Math.round(duration);
  }

  /**
   * Generate trading recommendations based on regime
   */
  private generateTradingRecommendations(
    regime: EnhancedMarketRegime,
    volatility: VolatilityMetrics,
    trendStrength: TrendStrengthMetrics
  ): any {
    const recommendations: Record<EnhancedMarketRegime, any> = {
      [EnhancedMarketRegime.TRENDING_BULLISH]: {
        strategy_type: 'trend_following',
        risk_multiplier: 1.2,
        optimal_timeframes: ['1h', '4h'],
        position_sizing_factor: 1.0
      },
      [EnhancedMarketRegime.TRENDING_BEARISH]: {
        strategy_type: 'trend_following',
        risk_multiplier: 1.2,
        optimal_timeframes: ['1h', '4h'],
        position_sizing_factor: 1.0
      },
      [EnhancedMarketRegime.VOLATILE]: {
        strategy_type: 'scalping',
        risk_multiplier: 0.5,
        optimal_timeframes: ['1m', '5m'],
        position_sizing_factor: 0.5
      },
      [EnhancedMarketRegime.BREAKOUT_BULLISH]: {
        strategy_type: 'breakout',
        risk_multiplier: 1.5,
        optimal_timeframes: ['15m', '1h'],
        position_sizing_factor: 1.2
      },
      [EnhancedMarketRegime.BREAKOUT_BEARISH]: {
        strategy_type: 'breakout',
        risk_multiplier: 1.5,
        optimal_timeframes: ['15m', '1h'],
        position_sizing_factor: 1.2
      },
      [EnhancedMarketRegime.CONSOLIDATION]: {
        strategy_type: 'mean_reversion',
        risk_multiplier: 0.8,
        optimal_timeframes: ['5m', '15m'],
        position_sizing_factor: 0.8
      },
      [EnhancedMarketRegime.SIDEWAYS]: {
        strategy_type: 'mean_reversion',
        risk_multiplier: 0.8,
        optimal_timeframes: ['15m', '1h'],
        position_sizing_factor: 0.8
      },
      [EnhancedMarketRegime.ACCUMULATION]: {
        strategy_type: 'trend_following',
        risk_multiplier: 1.0,
        optimal_timeframes: ['4h', '1d'],
        position_sizing_factor: 1.1
      },
      [EnhancedMarketRegime.DISTRIBUTION]: {
        strategy_type: 'trend_following',
        risk_multiplier: 1.0,
        optimal_timeframes: ['4h', '1d'],
        position_sizing_factor: 0.9
      }
    };
    
    return recommendations[regime];
  }

  /**
   * Get SMC context (placeholder for integration with existing SMC system)
   */
  private getSMCContext(data: MultiTimeframeData): any {
    // This would integrate with the existing SMC detection system
    // For now, return basic metrics
    return {
      order_block_strength: 0.7,
      liquidity_levels: 3,
      market_structure_quality: 0.8
    };
  }

  /**
   * Update regime history for change detection
   */
  private updateRegimeHistory(symbol: string, analysis: EnhancedRegimeAnalysis): void {
    if (!this.regimeHistory.has(symbol)) {
      this.regimeHistory.set(symbol, []);
    }
    
    const history = this.regimeHistory.get(symbol)!;
    history.push(analysis);
    
    // Keep only recent history for change detection
    if (history.length > this.changeDetectionWindow) {
      history.shift();
    }
  }
}

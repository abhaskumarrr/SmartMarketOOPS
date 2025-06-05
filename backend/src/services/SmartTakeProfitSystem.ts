/**
 * Smart Take Profit System
 * Dynamic take profit levels based on market conditions, support/resistance, and trend strength
 */

import { EnhancedMarketRegimeDetector, EnhancedRegimeAnalysis } from './EnhancedMarketRegimeDetector';
import { MultiTimeframeAnalysisEngine, CrossTimeframeAnalysis } from './MultiTimeframeAnalysisEngine';
import { DeltaExchangeUnified, MultiTimeframeData } from './DeltaExchangeUnified';
import { logger } from '../utils/logger';

export interface TakeProfitLevel {
  level: number; // 1, 2, 3, etc.
  target_price: number;
  percentage: number; // Percentage of position to close
  distance_percent: number; // Distance from entry in %
  confidence: number; // 0-1
  reasoning: string;
  level_type: 'support_resistance' | 'fibonacci' | 'atr_based' | 'trend_target' | 'regime_based';
}

export interface SmartTakeProfitConfig {
  max_levels: number; // Maximum number of TP levels
  partial_percentages: number[]; // [25, 50, 75] - percentages to close at each level
  base_targets: number[]; // [1.5, 3.0, 5.0] - base target percentages
  use_support_resistance: boolean;
  use_fibonacci: boolean;
  use_trend_targets: boolean;
  use_regime_adjustment: boolean;
  min_target_distance: number; // Minimum distance between targets (%)
  max_target_distance: number; // Maximum target distance (%)
}

export interface SmartTakeProfit {
  levels: TakeProfitLevel[];
  total_confidence: number;
  strategy_type: 'aggressive' | 'moderate' | 'conservative';
  market_context: {
    regime: string;
    trend_strength: number;
    volatility: number;
    support_resistance_quality: number;
  };
  execution_plan: {
    immediate_targets: TakeProfitLevel[];
    conditional_targets: TakeProfitLevel[];
    trailing_activation: number; // Price level to activate trailing
  };
}

export interface Position {
  symbol: string;
  side: 'LONG' | 'SHORT';
  entry_price: number;
  current_price: number;
  size: number;
  entry_time: number;
}

export class SmartTakeProfitSystem {
  private deltaService: DeltaExchangeUnified;
  private mtfAnalyzer: MultiTimeframeAnalysisEngine;
  private regimeDetector: EnhancedMarketRegimeDetector;
  private defaultConfig: SmartTakeProfitConfig;

  constructor(deltaService: DeltaExchangeUnified) {
    this.deltaService = deltaService;
    this.mtfAnalyzer = new MultiTimeframeAnalysisEngine(deltaService);
    this.regimeDetector = new EnhancedMarketRegimeDetector(deltaService);
    
    this.defaultConfig = {
      max_levels: 4,
      partial_percentages: [25, 35, 25, 15], // Progressive scaling out
      base_targets: [1.5, 3.0, 5.0, 8.0], // Base target percentages
      use_support_resistance: true,
      use_fibonacci: true,
      use_trend_targets: true,
      use_regime_adjustment: true,
      min_target_distance: 0.5, // 0.5%
      max_target_distance: 15.0 // 15%
    };
  }

  /**
   * Calculate smart take profit levels for a position
   */
  public async calculateTakeProfit(
    position: Position,
    config: Partial<SmartTakeProfitConfig> = {}
  ): Promise<SmartTakeProfit> {
    try {
      const finalConfig = { ...this.defaultConfig, ...config };
      
      logger.info(`🎯 Calculating smart take profit for ${position.symbol}`);
      
      // Get market analysis
      const mtfAnalysis = await this.mtfAnalyzer.analyzeSymbol(position.symbol);
      const regimeAnalysis = await this.regimeDetector.detectRegime(position.symbol);
      const marketData = await this.deltaService.getMultiTimeframeData(position.symbol);
      
      // Calculate support/resistance levels
      const supportResistanceLevels = finalConfig.use_support_resistance
        ? this.calculateSupportResistanceLevels(marketData, position)
        : [];
      
      // Calculate Fibonacci levels
      const fibonacciLevels = finalConfig.use_fibonacci
        ? this.calculateFibonacciLevels(marketData, position)
        : [];
      
      // Calculate trend-based targets
      const trendTargets = finalConfig.use_trend_targets
        ? this.calculateTrendTargets(position, mtfAnalysis)
        : [];
      
      // Calculate regime-adjusted targets
      const regimeTargets = finalConfig.use_regime_adjustment
        ? this.calculateRegimeTargets(position, regimeAnalysis, finalConfig)
        : [];
      
      // Combine and optimize levels
      const allLevels = [
        ...supportResistanceLevels,
        ...fibonacciLevels,
        ...trendTargets,
        ...regimeTargets
      ];
      
      // Select optimal levels
      const optimizedLevels = this.optimizeTakeProfitLevels(
        allLevels,
        finalConfig,
        position
      );
      
      // Determine strategy type
      const strategyType = this.determineStrategyType(regimeAnalysis, mtfAnalysis);
      
      // Calculate market context
      const marketContext = {
        regime: regimeAnalysis.current_regime,
        trend_strength: mtfAnalysis.overallTrend.strength,
        volatility: regimeAnalysis.volatility_metrics.price_volatility,
        support_resistance_quality: this.assessSupportResistanceQuality(marketData)
      };
      
      // Create execution plan
      const executionPlan = this.createExecutionPlan(optimizedLevels, strategyType);
      
      // Calculate total confidence
      const totalConfidence = this.calculateTotalConfidence(
        optimizedLevels,
        mtfAnalysis,
        regimeAnalysis
      );
      
      const smartTakeProfit: SmartTakeProfit = {
        levels: optimizedLevels,
        total_confidence: totalConfidence,
        strategy_type: strategyType,
        market_context: marketContext,
        execution_plan: executionPlan
      };
      
      logger.info(`✅ Smart take profit calculated: ${optimizedLevels.length} levels`);
      logger.info(`📊 Strategy: ${strategyType}, Confidence: ${(totalConfidence * 100).toFixed(1)}%`);
      
      return smartTakeProfit;
      
    } catch (error) {
      logger.error(`❌ Error calculating smart take profit for ${position.symbol}:`, error);
      throw error;
    }
  }

  /**
   * Calculate support/resistance levels
   */
  private calculateSupportResistanceLevels(
    marketData: MultiTimeframeData,
    position: Position
  ): TakeProfitLevel[] {
    const levels: TakeProfitLevel[] = [];
    const dailyData = marketData.timeframes['1d'];
    const hourlyData = marketData.timeframes['1h'];
    
    if (!dailyData?.candles || !hourlyData?.candles) return levels;
    
    // Find recent highs and lows
    const recentCandles = dailyData.candles.slice(-20);
    const highs = recentCandles.map(c => c.high);
    const lows = recentCandles.map(c => c.low);
    
    // Identify resistance levels (for long positions) or support levels (for short positions)
    if (position.side === 'LONG') {
      const resistanceLevels = this.findResistanceLevels(highs, position.current_price);
      
      resistanceLevels.forEach((level, index) => {
        const distancePercent = ((level - position.entry_price) / position.entry_price) * 100;
        
        if (distancePercent > 0.5 && distancePercent < 15) {
          levels.push({
            level: index + 1,
            target_price: level,
            percentage: 25, // Will be optimized later
            distance_percent: distancePercent,
            confidence: 0.8,
            reasoning: `Resistance level at $${level.toFixed(2)}`,
            level_type: 'support_resistance'
          });
        }
      });
    } else {
      const supportLevels = this.findSupportLevels(lows, position.current_price);
      
      supportLevels.forEach((level, index) => {
        const distancePercent = ((position.entry_price - level) / position.entry_price) * 100;
        
        if (distancePercent > 0.5 && distancePercent < 15) {
          levels.push({
            level: index + 1,
            target_price: level,
            percentage: 25,
            distance_percent: distancePercent,
            confidence: 0.8,
            reasoning: `Support level at $${level.toFixed(2)}`,
            level_type: 'support_resistance'
          });
        }
      });
    }
    
    return levels;
  }

  /**
   * Calculate Fibonacci retracement levels
   */
  private calculateFibonacciLevels(
    marketData: MultiTimeframeData,
    position: Position
  ): TakeProfitLevel[] {
    const levels: TakeProfitLevel[] = [];
    const hourlyData = marketData.timeframes['1h'];
    
    if (!hourlyData?.candles || hourlyData.candles.length < 50) return levels;
    
    // Find recent swing high and low
    const recentCandles = hourlyData.candles.slice(-50);
    const high = Math.max(...recentCandles.map(c => c.high));
    const low = Math.min(...recentCandles.map(c => c.low));
    const range = high - low;
    
    // Fibonacci extension levels
    const fibLevels = [1.272, 1.414, 1.618, 2.0];
    
    fibLevels.forEach((fib, index) => {
      let targetPrice: number;
      
      if (position.side === 'LONG') {
        targetPrice = position.entry_price + (range * (fib - 1));
      } else {
        targetPrice = position.entry_price - (range * (fib - 1));
      }
      
      const distancePercent = Math.abs((targetPrice - position.entry_price) / position.entry_price) * 100;
      
      if (distancePercent > 1 && distancePercent < 12) {
        levels.push({
          level: index + 1,
          target_price: targetPrice,
          percentage: 25,
          distance_percent: distancePercent,
          confidence: 0.7,
          reasoning: `Fibonacci ${fib} extension at $${targetPrice.toFixed(2)}`,
          level_type: 'fibonacci'
        });
      }
    });
    
    return levels;
  }

  /**
   * Calculate trend-based targets
   */
  private calculateTrendTargets(
    position: Position,
    mtfAnalysis: CrossTimeframeAnalysis
  ): TakeProfitLevel[] {
    const levels: TakeProfitLevel[] = [];
    const trendStrength = mtfAnalysis.overallTrend.strength;
    const trendDirection = mtfAnalysis.overallTrend.direction;
    
    // Only create trend targets if trend aligns with position
    const isAligned = (position.side === 'LONG' && trendDirection === 'bullish') ||
                     (position.side === 'SHORT' && trendDirection === 'bearish');
    
    if (!isAligned || trendStrength < 0.5) return levels;
    
    // Calculate trend-based targets
    const baseTargets = [2.0, 4.0, 7.0];
    const trendMultiplier = 1 + trendStrength; // 1.5 to 2.0
    
    baseTargets.forEach((baseTarget, index) => {
      const adjustedTarget = baseTarget * trendMultiplier;
      const targetPrice = position.side === 'LONG'
        ? position.entry_price * (1 + adjustedTarget / 100)
        : position.entry_price * (1 - adjustedTarget / 100);
      
      levels.push({
        level: index + 1,
        target_price: targetPrice,
        percentage: 30,
        distance_percent: adjustedTarget,
        confidence: trendStrength,
        reasoning: `Trend target ${adjustedTarget.toFixed(1)}% (strength: ${(trendStrength * 100).toFixed(1)}%)`,
        level_type: 'trend_target'
      });
    });
    
    return levels;
  }

  /**
   * Calculate regime-adjusted targets
   */
  private calculateRegimeTargets(
    position: Position,
    regimeAnalysis: EnhancedRegimeAnalysis,
    config: SmartTakeProfitConfig
  ): TakeProfitLevel[] {
    const levels: TakeProfitLevel[] = [];
    const regime = regimeAnalysis.current_regime;
    const recommendations = regimeAnalysis.trading_recommendations;
    
    // Adjust base targets based on regime
    let adjustedTargets = [...config.base_targets];
    
    switch (regime) {
      case 'trending_bullish':
      case 'trending_bearish':
        adjustedTargets = adjustedTargets.map(t => t * 1.3); // 30% higher targets in trending markets
        break;
      
      case 'volatile':
        adjustedTargets = adjustedTargets.map(t => t * 0.7); // 30% lower targets in volatile markets
        break;
      
      case 'breakout_bullish':
      case 'breakout_bearish':
        adjustedTargets = adjustedTargets.map(t => t * 1.5); // 50% higher targets for breakouts
        break;
      
      case 'consolidation':
      case 'sideways':
        adjustedTargets = adjustedTargets.map(t => t * 0.8); // 20% lower targets in ranging markets
        break;
    }
    
    adjustedTargets.forEach((target, index) => {
      const targetPrice = position.side === 'LONG'
        ? position.entry_price * (1 + target / 100)
        : position.entry_price * (1 - target / 100);
      
      levels.push({
        level: index + 1,
        target_price: targetPrice,
        percentage: config.partial_percentages[index] || 25,
        distance_percent: target,
        confidence: regimeAnalysis.confidence,
        reasoning: `${regime} regime target ${target.toFixed(1)}%`,
        level_type: 'regime_based'
      });
    });
    
    return levels;
  }

  /**
   * Optimize take profit levels
   */
  private optimizeTakeProfitLevels(
    allLevels: TakeProfitLevel[],
    config: SmartTakeProfitConfig,
    position: Position
  ): TakeProfitLevel[] {
    // Sort by distance
    allLevels.sort((a, b) => a.distance_percent - b.distance_percent);
    
    // Remove duplicates and select best levels
    const optimizedLevels: TakeProfitLevel[] = [];
    let lastDistance = 0;
    
    for (const level of allLevels) {
      // Ensure minimum distance between levels
      if (level.distance_percent - lastDistance >= config.min_target_distance) {
        optimizedLevels.push({
          ...level,
          level: optimizedLevels.length + 1,
          percentage: config.partial_percentages[optimizedLevels.length] || 25
        });
        
        lastDistance = level.distance_percent;
        
        if (optimizedLevels.length >= config.max_levels) break;
      }
    }
    
    return optimizedLevels;
  }

  /**
   * Determine strategy type based on market conditions
   */
  private determineStrategyType(
    regimeAnalysis: EnhancedRegimeAnalysis,
    mtfAnalysis: CrossTimeframeAnalysis
  ): 'aggressive' | 'moderate' | 'conservative' {
    const volatility = regimeAnalysis.volatility_metrics.price_volatility;
    const trendStrength = mtfAnalysis.overallTrend.strength;
    const confidence = regimeAnalysis.confidence;
    
    if (volatility > 0.05 || confidence < 0.5) {
      return 'conservative';
    } else if (trendStrength > 0.7 && confidence > 0.8) {
      return 'aggressive';
    } else {
      return 'moderate';
    }
  }

  /**
   * Create execution plan
   */
  private createExecutionPlan(
    levels: TakeProfitLevel[],
    strategyType: 'aggressive' | 'moderate' | 'conservative'
  ): any {
    const immediateTargets = levels.slice(0, 2); // First 2 levels
    const conditionalTargets = levels.slice(2); // Remaining levels
    
    // Trailing activation at 50% of first target
    const trailingActivation = levels.length > 0 
      ? levels[0].target_price * 0.5 
      : 0;
    
    return {
      immediate_targets: immediateTargets,
      conditional_targets: conditionalTargets,
      trailing_activation: trailingActivation
    };
  }

  /**
   * Calculate total confidence
   */
  private calculateTotalConfidence(
    levels: TakeProfitLevel[],
    mtfAnalysis: CrossTimeframeAnalysis,
    regimeAnalysis: EnhancedRegimeAnalysis
  ): number {
    if (levels.length === 0) return 0;
    
    const avgLevelConfidence = levels.reduce((sum, level) => sum + level.confidence, 0) / levels.length;
    const mtfConfidence = mtfAnalysis.signals.confidence;
    const regimeConfidence = regimeAnalysis.confidence;
    
    return (avgLevelConfidence + mtfConfidence + regimeConfidence) / 3;
  }

  // Helper methods
  private findResistanceLevels(highs: number[], currentPrice: number): number[] {
    const levels: number[] = [];
    const sortedHighs = [...highs].sort((a, b) => b - a);
    
    for (const high of sortedHighs) {
      if (high > currentPrice && !levels.some(level => Math.abs(level - high) / high < 0.01)) {
        levels.push(high);
        if (levels.length >= 3) break;
      }
    }
    
    return levels;
  }

  private findSupportLevels(lows: number[], currentPrice: number): number[] {
    const levels: number[] = [];
    const sortedLows = [...lows].sort((a, b) => a - b);
    
    for (const low of sortedLows) {
      if (low < currentPrice && !levels.some(level => Math.abs(level - low) / low < 0.01)) {
        levels.push(low);
        if (levels.length >= 3) break;
      }
    }
    
    return levels;
  }

  private assessSupportResistanceQuality(marketData: MultiTimeframeData): number {
    // Simplified quality assessment
    return 0.7; // Default moderate quality
  }
}

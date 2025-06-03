/**
 * Enhanced Trend Following Strategy
 * Professional-grade strategy with trend analysis, market regime detection, and dynamic risk management
 */

import { 
  TradingStrategy, 
  TradingSignal, 
  EnhancedMarketData, 
  BacktestConfig 
} from '../types/marketData';
import { technicalAnalysis } from '../utils/technicalAnalysis';
import { logger } from '../utils/logger';
import { createEventId } from '../types/events';

export interface EnhancedTrendParams {
  // Trend Detection
  trendPeriod: number;
  trendThreshold: number; // Minimum slope for trend
  
  // Entry Signals
  fastEMA: number;
  slowEMA: number;
  rsiPeriod: number;
  rsiOverbought: number;
  rsiOversold: number;
  
  // Market Regime
  volatilityPeriod: number;
  volumeConfirmation: number;
  
  // Risk Management
  baseStopLoss: number;
  dynamicStopLoss: boolean;
  takeProfitMultiplier: number;
  maxPositionSize: number;
  
  // Filters
  minTrendStrength: number;
  minConfidence: number;
  antiWhipsawPeriod: number; // Minimum bars between signals
}

export class EnhancedTrendStrategy implements TradingStrategy {
  public readonly name = 'Enhanced_Trend';
  public parameters: EnhancedTrendParams;
  private config?: BacktestConfig;
  private lastSignalIndex: number = -1;

  constructor(parameters?: Partial<EnhancedTrendParams>) {
    this.parameters = {
      // Trend Detection
      trendPeriod: 20,
      trendThreshold: 0.001, // 0.1% minimum slope
      
      // Entry Signals
      fastEMA: 12,
      slowEMA: 26,
      rsiPeriod: 14,
      rsiOverbought: 75,
      rsiOversold: 25,
      
      // Market Regime
      volatilityPeriod: 20,
      volumeConfirmation: 1.5,
      
      // Risk Management
      baseStopLoss: 1.5, // Tighter stops
      dynamicStopLoss: true,
      takeProfitMultiplier: 3, // 3:1 R/R minimum
      maxPositionSize: 0.8, // Max 80% of calculated size
      
      // Filters
      minTrendStrength: 0.6,
      minConfidence: 70,
      antiWhipsawPeriod: 6, // Minimum 6 bars between signals
      
      ...parameters,
    };
  }

  public initialize(config: BacktestConfig): void {
    this.config = config;
    this.lastSignalIndex = -1;
    
    logger.info(`🎯 Initialized ${this.name} strategy`, {
      parameters: this.parameters,
      symbol: config.symbol,
      timeframe: config.timeframe,
    });
  }

  public generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null {
    if (!this.config) {
      throw new Error('Strategy not initialized. Call initialize() first.');
    }

    // Need enough data for analysis
    if (currentIndex < Math.max(this.parameters.trendPeriod, this.parameters.slowEMA, this.parameters.volatilityPeriod)) {
      return null;
    }

    // Anti-whipsaw filter
    if (currentIndex - this.lastSignalIndex < this.parameters.antiWhipsawPeriod) {
      return null;
    }

    const currentCandle = data[currentIndex];
    const indicators = currentCandle.indicators;

    // Check if we have required indicators
    if (!this.hasRequiredIndicators(indicators)) {
      return null;
    }

    // 1. Analyze market regime
    const marketRegime = this.analyzeMarketRegime(data, currentIndex);
    
    // Only trade in trending markets
    if (marketRegime.type !== 'trending') {
      return null;
    }

    // 2. Detect trend direction
    const trendAnalysis = this.analyzeTrend(data, currentIndex);
    
    if (trendAnalysis.strength < this.parameters.minTrendStrength) {
      return null;
    }

    // 3. Generate entry signal
    const signal = this.generateEntrySignal(data, currentIndex, trendAnalysis, marketRegime);
    
    if (!signal) {
      return null;
    }

    // 4. Apply filters and enhance signal
    const enhancedSignal = this.enhanceSignal(signal, data, currentIndex, trendAnalysis, marketRegime);
    
    if (enhancedSignal.confidence < this.parameters.minConfidence) {
      return null;
    }

    this.lastSignalIndex = currentIndex;

    logger.info(`📊 Generated ${enhancedSignal.type} signal`, {
      price: enhancedSignal.price,
      confidence: enhancedSignal.confidence,
      trendDirection: trendAnalysis.direction,
      trendStrength: trendAnalysis.strength.toFixed(2),
      marketRegime: marketRegime.type,
    });

    return enhancedSignal;
  }

  private hasRequiredIndicators(indicators: any): boolean {
    return !!(
      indicators.ema_12 !== undefined && !isNaN(indicators.ema_12) &&
      indicators.ema_26 !== undefined && !isNaN(indicators.ema_26) &&
      indicators.rsi !== undefined && !isNaN(indicators.rsi) &&
      indicators.volume_sma !== undefined && !isNaN(indicators.volume_sma)
    );
  }

  private analyzeMarketRegime(data: EnhancedMarketData[], currentIndex: number): any {
    const period = this.parameters.volatilityPeriod;
    const recentData = data.slice(currentIndex - period + 1, currentIndex + 1);
    
    // Calculate volatility
    const returns = recentData.slice(1).map((candle, i) => 
      Math.log(candle.close / recentData[i].close)
    );
    
    const volatility = Math.sqrt(
      returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length
    ) * Math.sqrt(252); // Annualized

    // Calculate trend consistency
    const prices = recentData.map(d => d.close);
    const trendConsistency = this.calculateTrendConsistency(prices);
    
    // Determine regime
    let type = 'sideways';
    if (volatility > 0.4 && trendConsistency > 0.7) {
      type = 'trending';
    } else if (volatility > 0.6) {
      type = 'volatile';
    }

    return {
      type,
      volatility,
      trendConsistency,
    };
  }

  private analyzeTrend(data: EnhancedMarketData[], currentIndex: number): any {
    const period = this.parameters.trendPeriod;
    const recentData = data.slice(currentIndex - period + 1, currentIndex + 1);
    
    // Linear regression for trend
    const prices = recentData.map(d => d.close);
    const { slope, r2 } = this.calculateLinearRegression(prices);
    
    // Normalize slope as percentage per period
    const normalizedSlope = slope / prices[0];
    
    // Determine direction and strength
    let direction = 'neutral';
    if (normalizedSlope > this.parameters.trendThreshold) {
      direction = 'up';
    } else if (normalizedSlope < -this.parameters.trendThreshold) {
      direction = 'down';
    }

    const strength = Math.min(r2 * Math.abs(normalizedSlope) * 100, 1);

    return {
      direction,
      strength,
      slope: normalizedSlope,
      r2,
    };
  }

  private generateEntrySignal(
    data: EnhancedMarketData[], 
    currentIndex: number, 
    trendAnalysis: any, 
    marketRegime: any
  ): TradingSignal | null {
    const currentCandle = data[currentIndex];
    const previousCandle = data[currentIndex - 1];
    const indicators = currentCandle.indicators;
    const prevIndicators = previousCandle.indicators;

    // EMA crossover in trend direction only
    const currentFastAboveSlow = indicators.ema_12! > indicators.ema_26!;
    const previousFastAboveSlow = prevIndicators.ema_12! > prevIndicators.ema_26!;

    // Bullish signal: EMA crossover + uptrend + RSI not overbought
    if (!previousFastAboveSlow && currentFastAboveSlow && 
        trendAnalysis.direction === 'up' && 
        indicators.rsi! < this.parameters.rsiOverbought) {
      
      return {
        id: createEventId(),
        timestamp: currentCandle.timestamp,
        symbol: this.config!.symbol,
        type: 'BUY',
        price: currentCandle.close,
        quantity: 0,
        confidence: 60, // Base confidence
        strategy: this.name,
        reason: 'Bullish EMA crossover in uptrend',
      };
    }

    // Bearish signal: EMA crossover + downtrend + RSI not oversold
    if (previousFastAboveSlow && !currentFastAboveSlow && 
        trendAnalysis.direction === 'down' && 
        indicators.rsi! > this.parameters.rsiOversold) {
      
      return {
        id: createEventId(),
        timestamp: currentCandle.timestamp,
        symbol: this.config!.symbol,
        type: 'SELL',
        price: currentCandle.close,
        quantity: 0,
        confidence: 60, // Base confidence
        strategy: this.name,
        reason: 'Bearish EMA crossover in downtrend',
      };
    }

    return null;
  }

  private enhanceSignal(
    signal: TradingSignal, 
    data: EnhancedMarketData[], 
    currentIndex: number,
    trendAnalysis: any,
    marketRegime: any
  ): TradingSignal {
    const currentCandle = data[currentIndex];
    const indicators = currentCandle.indicators;
    let confidence = signal.confidence;

    // 1. Trend strength bonus
    confidence += trendAnalysis.strength * 20;

    // 2. Market regime bonus
    if (marketRegime.type === 'trending') {
      confidence += 10;
    }

    // 3. Volume confirmation
    const volumeRatio = currentCandle.volume / indicators.volume_sma!;
    if (volumeRatio > this.parameters.volumeConfirmation) {
      confidence += 15;
    }

    // 4. RSI position
    if (signal.type === 'BUY' && indicators.rsi! < 50) {
      confidence += 10;
    }
    if (signal.type === 'SELL' && indicators.rsi! > 50) {
      confidence += 10;
    }

    // 5. Price momentum
    const momentum = (currentCandle.close - data[currentIndex - 5].close) / data[currentIndex - 5].close;
    if ((signal.type === 'BUY' && momentum > 0) || (signal.type === 'SELL' && momentum < 0)) {
      confidence += 10;
    }

    confidence = Math.min(confidence, 100);

    // Calculate dynamic position size
    const quantity = this.calculateDynamicPositionSize(
      currentCandle.close, 
      confidence, 
      trendAnalysis, 
      marketRegime
    );

    // Calculate dynamic stop loss
    const stopLoss = this.calculateDynamicStopLoss(
      signal.type, 
      currentCandle.close, 
      data, 
      currentIndex, 
      trendAnalysis
    );

    const stopDistance = Math.abs(currentCandle.close - stopLoss) / currentCandle.close;
    const takeProfit = signal.type === 'BUY'
      ? currentCandle.close * (1 + stopDistance * this.parameters.takeProfitMultiplier)
      : currentCandle.close * (1 - stopDistance * this.parameters.takeProfitMultiplier);

    return {
      ...signal,
      quantity,
      confidence,
      stopLoss,
      takeProfit,
      riskReward: this.parameters.takeProfitMultiplier,
      reason: `${signal.reason} (Confidence: ${confidence.toFixed(0)}%, Trend: ${trendAnalysis.strength.toFixed(2)}, Vol: ${volumeRatio.toFixed(1)}x)`,
    };
  }

  private calculateDynamicPositionSize(
    price: number, 
    confidence: number, 
    trendAnalysis: any, 
    marketRegime: any
  ): number {
    if (!this.config) return 0;

    // Base position size
    const riskAmount = this.config.initialCapital * (this.config.riskPerTrade / 100);
    const stopDistance = price * (this.parameters.baseStopLoss / 100);
    let positionSize = riskAmount / stopDistance;

    // Apply leverage
    positionSize *= this.config.leverage;

    // Scale by confidence
    const confidenceMultiplier = confidence / 100;
    positionSize *= confidenceMultiplier;

    // Scale by trend strength
    positionSize *= (0.5 + trendAnalysis.strength * 0.5);

    // Apply maximum position size limit
    positionSize *= this.parameters.maxPositionSize;

    return Math.max(positionSize, 0.001);
  }

  private calculateDynamicStopLoss(
    signalType: string, 
    price: number, 
    data: EnhancedMarketData[], 
    currentIndex: number,
    trendAnalysis: any
  ): number {
    if (!this.parameters.dynamicStopLoss) {
      return signalType === 'BUY' 
        ? price * (1 - this.parameters.baseStopLoss / 100)
        : price * (1 + this.parameters.baseStopLoss / 100);
    }

    // Calculate ATR-based stop
    const atrPeriod = 14;
    const recentData = data.slice(currentIndex - atrPeriod, currentIndex + 1);
    const atr = this.calculateATR(recentData);
    
    // Use 1.5x ATR as stop distance, but minimum of base stop loss
    const atrStopDistance = Math.max(
      (atr / price) * 1.5,
      this.parameters.baseStopLoss / 100
    );

    return signalType === 'BUY'
      ? price * (1 - atrStopDistance)
      : price * (1 + atrStopDistance);
  }

  // Helper methods
  private calculateTrendConsistency(prices: number[]): number {
    let consistentMoves = 0;
    const totalMoves = prices.length - 1;
    
    for (let i = 1; i < prices.length; i++) {
      const currentMove = prices[i] > prices[i - 1] ? 1 : -1;
      const overallTrend = prices[prices.length - 1] > prices[0] ? 1 : -1;
      
      if (currentMove === overallTrend) {
        consistentMoves++;
      }
    }
    
    return consistentMoves / totalMoves;
  }

  private calculateLinearRegression(prices: number[]): { slope: number; r2: number } {
    const n = prices.length;
    const x = Array.from({ length: n }, (_, i) => i);
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = prices.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * prices[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumYY = prices.reduce((sum, yi) => sum + yi * yi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate R²
    const yMean = sumY / n;
    const ssRes = prices.reduce((sum, yi, i) => {
      const predicted = slope * x[i] + intercept;
      return sum + Math.pow(yi - predicted, 2);
    }, 0);
    const ssTot = prices.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
    const r2 = 1 - (ssRes / ssTot);
    
    return { slope, r2: Math.max(0, r2) };
  }

  private calculateATR(data: EnhancedMarketData[]): number {
    let atrSum = 0;
    
    for (let i = 1; i < data.length; i++) {
      const tr1 = data[i].high - data[i].low;
      const tr2 = Math.abs(data[i].high - data[i - 1].close);
      const tr3 = Math.abs(data[i].low - data[i - 1].close);
      atrSum += Math.max(tr1, tr2, tr3);
    }
    
    return atrSum / (data.length - 1);
  }

  public getDescription(): string {
    return `Enhanced Trend Following Strategy with market regime detection, dynamic risk management, and anti-whipsaw filters`;
  }

  public getParameters(): EnhancedTrendParams {
    return { ...this.parameters };
  }

  public updateParameters(newParams: Partial<EnhancedTrendParams>): void {
    this.parameters = { ...this.parameters, ...newParams };
    logger.info(`📊 Updated ${this.name} strategy parameters`, this.parameters);
  }
}

// Export factory function
export function createEnhancedTrendStrategy(params?: Partial<EnhancedTrendParams>): EnhancedTrendStrategy {
  return new EnhancedTrendStrategy(params);
}

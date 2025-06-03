#!/usr/bin/env node

/**
 * Optimized Trading Strategy - Enhanced Signal Generation
 * Based on performance analysis of current system
 */

import { TradingSignal, BacktestConfig } from '../types/marketData';
import { logger } from '../utils/logger';

// Extended trading signal with additional optimization data
interface OptimizedTradingSignal extends TradingSignal {
  marketRegime?: string;
  volatility?: number;
  volumeStrength?: number;
}

class OptimizedTradingStrategy {
  
  /**
   * Enhanced signal generation with market regime awareness
   */
  public generateOptimizedSignal(candle: any, config: BacktestConfig, index: number, marketData: any[]): OptimizedTradingSignal | null {
    // Market regime detection
    const marketRegime = this.detectMarketRegime(marketData, index);
    const volatility = this.calculateVolatility(marketData, index);
    const volume = this.analyzeVolumeProfile(marketData, index);
    
    // Enhanced technical analysis
    const technicalSignals = this.getTechnicalSignals(candle, marketData, index);
    const momentum = this.calculateMomentum(marketData, index);
    const support_resistance = this.findSupportResistance(marketData, index);
    
    // Signal strength calculation
    let signalStrength = 0;
    let signalType: 'BUY' | 'SELL' | null = null;
    let reason = '';

    // 1. Trend Following Signals (Strong Trends)
    if (marketRegime === 'TRENDING') {
      if (momentum.direction === 'UP' && technicalSignals.rsi < 70 && volume.strength > 1.2) {
        signalType = 'BUY';
        signalStrength += 30;
        reason = 'Strong uptrend with momentum confirmation';
      } else if (momentum.direction === 'DOWN' && technicalSignals.rsi > 30 && volume.strength > 1.2) {
        signalType = 'SELL';
        signalStrength += 30;
        reason = 'Strong downtrend with momentum confirmation';
      }
    }

    // 2. Mean Reversion Signals (Ranging Markets)
    else if (marketRegime === 'RANGING') {
      if (technicalSignals.rsi < 30 && candle.close <= support_resistance.support * 1.005) {
        signalType = 'BUY';
        signalStrength += 25;
        reason = 'Oversold bounce in ranging market';
      } else if (technicalSignals.rsi > 70 && candle.close >= support_resistance.resistance * 0.995) {
        signalType = 'SELL';
        signalStrength += 25;
        reason = 'Overbought rejection in ranging market';
      }
    }

    // 3. Breakout Signals (High Volume)
    if (volume.strength > 2.0 && volatility.percentile > 80) {
      const priceChange = (candle.close - candle.open) / candle.open;
      if (priceChange > 0.01) {
        signalType = 'BUY';
        signalStrength += 20;
        reason += ' + High volume breakout';
      } else if (priceChange < -0.01) {
        signalType = 'SELL';
        signalStrength += 20;
        reason += ' + High volume breakdown';
      }
    }

    // 4. Multi-timeframe Confirmation
    const higherTimeframeSignal = this.getHigherTimeframeSignal(marketData, index);
    if (higherTimeframeSignal && higherTimeframeSignal === signalType) {
      signalStrength += 15;
      reason += ' + Higher timeframe confirmation';
    }

    // Signal filtering - only trade high-confidence signals
    if (!signalType || signalStrength < 40) {
      return null;
    }

    // Dynamic position sizing based on volatility and confidence
    const baseRisk = config.initialCapital * (config.riskPerTrade / 100);
    const volatilityAdjustment = Math.max(0.5, Math.min(2.0, 1 / volatility.normalized));
    const confidenceAdjustment = signalStrength / 100;
    
    let quantity = (baseRisk * volatilityAdjustment * confidenceAdjustment) / candle.close;
    quantity = Math.max(quantity, 0.001); // Minimum quantity
    
    // Dynamic stop loss based on volatility
    const atr = volatility.atr;
    const stopLossDistance = Math.max(atr * 2, candle.close * 0.02); // Min 2% or 2x ATR
    
    const stopLoss = signalType === 'BUY'
      ? candle.close - stopLossDistance
      : candle.close + stopLossDistance;

    const takeProfit = signalType === 'BUY'
      ? candle.close + (stopLossDistance * 2.5) // 2.5:1 risk/reward
      : candle.close - (stopLossDistance * 2.5);

    return {
      id: `optimized_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: candle.timestamp,
      symbol: config.symbol,
      type: signalType,
      price: candle.close,
      quantity: quantity,
      confidence: signalStrength,
      strategy: 'OPTIMIZED_STRATEGY',
      reason,
      stopLoss,
      takeProfit,
      riskReward: 2.5,
      marketRegime,
      volatility: volatility.normalized,
      volumeStrength: volume.strength,
    };
  }

  /**
   * Detect market regime (trending vs ranging)
   */
  private detectMarketRegime(data: any[], index: number): 'TRENDING' | 'RANGING' | 'VOLATILE' {
    if (index < 50) return 'RANGING';
    
    const lookback = 20;
    const prices = data.slice(index - lookback, index).map(d => d.close);
    
    // Calculate trend strength using linear regression
    const trendStrength = this.calculateTrendStrength(prices);
    const volatility = this.calculateRangeVolatility(prices);
    
    if (Math.abs(trendStrength) > 0.7 && volatility < 0.03) {
      return 'TRENDING';
    } else if (volatility > 0.05) {
      return 'VOLATILE';
    } else {
      return 'RANGING';
    }
  }

  /**
   * Calculate comprehensive volatility metrics
   */
  private calculateVolatility(data: any[], index: number) {
    if (index < 20) return { normalized: 1, atr: 0, percentile: 50 };
    
    const lookback = 14;
    const recent = data.slice(index - lookback, index);
    
    // Average True Range
    let atrSum = 0;
    for (let i = 1; i < recent.length; i++) {
      const high = recent[i].high;
      const low = recent[i].low;
      const prevClose = recent[i - 1].close;
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      atrSum += tr;
    }
    const atr = atrSum / (recent.length - 1);
    
    // Normalized volatility (0-2 scale)
    const currentPrice = data[index].close;
    const normalized = Math.min(2, (atr / currentPrice) * 100);
    
    // Volatility percentile
    const longerPeriod = data.slice(Math.max(0, index - 100), index);
    const volatilities = longerPeriod.map((_, i) => {
      if (i < 14) return 0;
      return this.calculateVolatility(longerPeriod, i).normalized;
    }).filter(v => v > 0);
    
    const percentile = (volatilities.filter(v => v < normalized).length / volatilities.length) * 100;
    
    return { normalized, atr, percentile };
  }

  /**
   * Analyze volume profile and strength
   */
  private analyzeVolumeProfile(data: any[], index: number) {
    if (index < 20) return { strength: 1, trend: 'NEUTRAL' };
    
    const lookback = 20;
    const recent = data.slice(index - lookback, index);
    const currentVolume = data[index].volume;
    const avgVolume = recent.reduce((sum, d) => sum + d.volume, 0) / recent.length;
    
    const strength = currentVolume / avgVolume;
    
    // Volume trend analysis
    const firstHalf = recent.slice(0, 10).reduce((sum, d) => sum + d.volume, 0) / 10;
    const secondHalf = recent.slice(10).reduce((sum, d) => sum + d.volume, 0) / 10;
    const trend = secondHalf > firstHalf * 1.1 ? 'INCREASING' : 
                  secondHalf < firstHalf * 0.9 ? 'DECREASING' : 'NEUTRAL';
    
    return { strength, trend };
  }

  /**
   * Get comprehensive technical signals
   */
  private getTechnicalSignals(candle: any, data: any[], index: number) {
    if (index < 50) return { rsi: 50, macd: 0, bb_position: 0.5 };
    
    // RSI calculation
    const rsi = this.calculateRSI(data, index, 14);
    
    // MACD calculation
    const macd = this.calculateMACD(data, index);
    
    // Bollinger Bands position
    const bb_position = this.calculateBollingerPosition(data, index);
    
    return { rsi, macd, bb_position };
  }

  /**
   * Calculate momentum indicators
   */
  private calculateMomentum(data: any[], index: number) {
    if (index < 20) return { direction: 'NEUTRAL', strength: 0 };
    
    const current = data[index].close;
    const past5 = data[index - 5].close;
    const past10 = data[index - 10].close;
    const past20 = data[index - 20].close;
    
    const shortMomentum = (current - past5) / past5;
    const mediumMomentum = (current - past10) / past10;
    const longMomentum = (current - past20) / past20;
    
    const avgMomentum = (shortMomentum + mediumMomentum + longMomentum) / 3;
    
    const direction = avgMomentum > 0.005 ? 'UP' : 
                     avgMomentum < -0.005 ? 'DOWN' : 'NEUTRAL';
    
    const strength = Math.abs(avgMomentum) * 100;
    
    return { direction, strength, short: shortMomentum, medium: mediumMomentum, long: longMomentum };
  }

  /**
   * Find dynamic support and resistance levels
   */
  private findSupportResistance(data: any[], index: number) {
    if (index < 50) return { support: data[index].low, resistance: data[index].high };
    
    const lookback = 50;
    const recent = data.slice(index - lookback, index);
    
    // Find pivot points
    const highs = recent.map(d => d.high).sort((a, b) => b - a);
    const lows = recent.map(d => d.low).sort((a, b) => a - b);
    
    // Support: average of lowest 20% of lows
    const supportLevels = lows.slice(0, Math.floor(lows.length * 0.2));
    const support = supportLevels.reduce((sum, l) => sum + l, 0) / supportLevels.length;
    
    // Resistance: average of highest 20% of highs
    const resistanceLevels = highs.slice(0, Math.floor(highs.length * 0.2));
    const resistance = resistanceLevels.reduce((sum, h) => sum + h, 0) / resistanceLevels.length;
    
    return { support, resistance };
  }

  /**
   * Get higher timeframe signal confirmation
   */
  private getHigherTimeframeSignal(data: any[], index: number): 'BUY' | 'SELL' | null {
    if (index < 100) return null;
    
    // Simulate higher timeframe by using every 4th candle (1h vs 15m)
    const htfData = data.filter((_, i) => i % 4 === 0);
    const htfIndex = Math.floor(index / 4);
    
    if (htfIndex < 20) return null;
    
    const htfCurrent = htfData[htfIndex].close;
    const htfPast = htfData[htfIndex - 10].close;
    const htfMomentum = (htfCurrent - htfPast) / htfPast;
    
    return htfMomentum > 0.02 ? 'BUY' : htfMomentum < -0.02 ? 'SELL' : null;
  }

  // Helper calculation methods
  private calculateTrendStrength(prices: number[]): number {
    const n = prices.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = prices;
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const avgPrice = sumY / n;
    
    return slope / avgPrice; // Normalized slope
  }

  private calculateRangeVolatility(prices: number[]): number {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  private calculateRSI(data: any[], index: number, period: number = 14): number {
    if (index < period) return 50;
    
    const prices = data.slice(index - period, index + 1).map(d => d.close);
    let gains = 0, losses = 0;
    
    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgGain / (avgLoss || 0.001);
    
    return 100 - (100 / (1 + rs));
  }

  private calculateMACD(data: any[], index: number): number {
    if (index < 26) return 0;
    
    const prices = data.slice(index - 26, index + 1).map(d => d.close);
    const ema12 = this.calculateEMA(prices, 12);
    const ema26 = this.calculateEMA(prices, 26);
    
    return ema12 - ema26;
  }

  private calculateEMA(prices: number[], period: number): number {
    const multiplier = 2 / (period + 1);
    let ema = prices[0];
    
    for (let i = 1; i < prices.length; i++) {
      ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
    }
    
    return ema;
  }

  private calculateBollingerPosition(data: any[], index: number): number {
    if (index < 20) return 0.5;
    
    const prices = data.slice(index - 20, index + 1).map(d => d.close);
    const sma = prices.reduce((a, b) => a + b, 0) / prices.length;
    const variance = prices.reduce((sum, p) => sum + Math.pow(p - sma, 2), 0) / prices.length;
    const stdDev = Math.sqrt(variance);
    
    const upperBand = sma + (stdDev * 2);
    const lowerBand = sma - (stdDev * 2);
    const currentPrice = data[index].close;
    
    return (currentPrice - lowerBand) / (upperBand - lowerBand);
  }
}

export { OptimizedTradingStrategy };

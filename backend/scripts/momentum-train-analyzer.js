/**
 * MOMENTUM TRAIN ANALYZER
 * 
 * Professional 4-tier timeframe analysis for catching momentum trains:
 * 
 * 🚂 MOMENTUM TRAIN STRATEGY:
 * 1. 4H - TREND DIRECTION: Where is the train going?
 * 2. 1H - TREND CONFIRMATION: Is the train accelerating?
 * 3. 15M - ENTRY TIMING: When to jump on the train?
 * 4. 5M - PRECISION ENTRY: Exact entry point
 * 
 * Based on professional trading research and multi-timeframe confluence.
 */

const DefensiveTradingUtils = require('./defensive-trading-utils');

class MomentumTrainAnalyzer {
  constructor(dataManager) {
    this.dataManager = dataManager;
    
    // Professional timeframe hierarchy
    this.timeframes = {
      trend: '4h',        // TREND DIRECTION (Where is the train going?)
      confirmation: '1h', // TREND CONFIRMATION (Is the train accelerating?)
      entry: '15m',       // ENTRY TIMING (When to jump on the train?)
      precision: '5m'     // PRECISION ENTRY (Exact entry point)
    };
    
    // Momentum detection thresholds
    this.thresholds = {
      trendStrength: 0.7,      // 70% directional moves for strong trend
      momentumAcceleration: 0.02, // 2% acceleration for momentum
      entryConfirmation: 0.6,   // 60% alignment for entry
      precisionTiming: 0.8      // 80% precision for exact entry
    };
  }

  /**
   * Analyze momentum train across all timeframes
   */
  async analyzeMomentumTrain(symbol, currentPrice) {
    try {
      console.log(`🚂 ANALYZING MOMENTUM TRAIN for ${symbol} at $${currentPrice.toFixed(2)}`);
      
      // Step 1: 4H TREND DIRECTION - Where is the train going?
      const trendDirection = this.analyzeTrendDirection(symbol);
      
      // Step 2: 1H TREND CONFIRMATION - Is the train accelerating?
      const trendConfirmation = this.analyzeTrendConfirmation(symbol, trendDirection);
      
      // Step 3: 15M ENTRY TIMING - When to jump on the train?
      const entryTiming = this.analyzeEntryTiming(symbol, trendDirection, trendConfirmation);
      
      // Step 4: 5M PRECISION ENTRY - Exact entry point
      const precisionEntry = this.analyzePrecisionEntry(symbol, currentPrice, entryTiming);
      
      // Calculate overall momentum train signal
      const momentumSignal = this.calculateMomentumTrainSignal({
        trendDirection,
        trendConfirmation,
        entryTiming,
        precisionEntry,
        currentPrice
      });
      
      console.log(`🎯 MOMENTUM TRAIN ANALYSIS COMPLETE:`);
      console.log(`   4H Trend: ${trendDirection.direction} (${(trendDirection.strength * 100).toFixed(1)}%)`);
      console.log(`   1H Confirmation: ${trendConfirmation.status} (${(trendConfirmation.acceleration * 100).toFixed(1)}%)`);
      console.log(`   15M Entry: ${entryTiming.signal} (${(entryTiming.confidence * 100).toFixed(1)}%)`);
      console.log(`   5M Precision: ${precisionEntry.timing} (${(precisionEntry.precision * 100).toFixed(1)}%)`);
      console.log(`   🚂 Train Signal: ${momentumSignal.action} (${(momentumSignal.confidence * 100).toFixed(1)}%)`);
      
      return momentumSignal;
      
    } catch (error) {
      console.error(`❌ Momentum train analysis error for ${symbol}: ${error.message}`);
      return null;
    }
  }

  /**
   * Step 1: 4H TREND DIRECTION - Where is the train going?
   */
  analyzeTrendDirection(symbol) {
    const candles = this.dataManager.getHistoricalData(symbol, this.timeframes.trend);
    
    if (!candles || candles.length < 20) {
      return { direction: 'neutral', strength: 0, confidence: 0 };
    }
    
    // Analyze last 20 candles for trend direction
    const recent20 = candles.slice(-20);
    const prices = recent20.map(c => parseFloat(c.close));
    
    // Calculate trend strength using linear regression
    const trendStrength = this.calculateTrendStrength(prices);
    
    // Determine direction
    let direction = 'neutral';
    if (trendStrength > this.thresholds.trendStrength) {
      direction = 'bullish';
    } else if (trendStrength < -this.thresholds.trendStrength) {
      direction = 'bearish';
    }
    
    return {
      direction,
      strength: Math.abs(trendStrength),
      confidence: Math.min(Math.abs(trendStrength) * 1.5, 1.0),
      timeframe: '4h',
      analysis: 'TREND DIRECTION'
    };
  }

  /**
   * Step 2: 1H TREND CONFIRMATION - Is the train accelerating?
   */
  analyzeTrendConfirmation(symbol, trendDirection) {
    const candles = this.dataManager.getHistoricalData(symbol, this.timeframes.confirmation);
    
    if (!candles || candles.length < 10) {
      return { status: 'neutral', acceleration: 0, alignment: false };
    }
    
    // Analyze last 10 candles for acceleration
    const recent10 = candles.slice(-10);
    const prices = recent10.map(c => parseFloat(c.close));
    
    // Calculate momentum acceleration
    const acceleration = this.calculateMomentumAcceleration(prices);
    
    // Check alignment with 4H trend
    const alignment = this.checkTrendAlignment(trendDirection.direction, acceleration);
    
    let status = 'neutral';
    if (alignment && Math.abs(acceleration) > this.thresholds.momentumAcceleration) {
      status = acceleration > 0 ? 'accelerating_up' : 'accelerating_down';
    }
    
    return {
      status,
      acceleration: Math.abs(acceleration),
      alignment,
      timeframe: '1h',
      analysis: 'TREND CONFIRMATION'
    };
  }

  /**
   * Step 3: 15M ENTRY TIMING - When to jump on the train?
   */
  analyzeEntryTiming(symbol, trendDirection, trendConfirmation) {
    const candles = this.dataManager.getHistoricalData(symbol, this.timeframes.entry);
    
    if (!candles || candles.length < 5) {
      return { signal: 'wait', confidence: 0, timing: 'poor' };
    }
    
    // Analyze last 5 candles for entry timing
    const recent5 = candles.slice(-5);
    const prices = recent5.map(c => parseFloat(c.close));
    
    // Calculate entry confidence based on alignment
    const entryConfidence = this.calculateEntryConfidence(
      trendDirection,
      trendConfirmation,
      prices
    );
    
    let signal = 'wait';
    let timing = 'poor';
    
    if (entryConfidence > this.thresholds.entryConfirmation) {
      if (trendDirection.direction === 'bullish' && trendConfirmation.alignment) {
        signal = 'enter_long';
        timing = 'good';
      } else if (trendDirection.direction === 'bearish' && trendConfirmation.alignment) {
        signal = 'enter_short';
        timing = 'good';
      }
    }
    
    return {
      signal,
      confidence: entryConfidence,
      timing,
      timeframe: '15m',
      analysis: 'ENTRY TIMING'
    };
  }

  /**
   * Step 4: 5M PRECISION ENTRY - Exact entry point
   */
  analyzePrecisionEntry(symbol, currentPrice, entryTiming) {
    const candles = this.dataManager.getHistoricalData(symbol, this.timeframes.precision);
    
    if (!candles || candles.length < 3) {
      return { timing: 'wait', precision: 0, action: 'hold' };
    }
    
    // Analyze last 3 candles for precision timing
    const recent3 = candles.slice(-3);
    const prices = recent3.map(c => parseFloat(c.close));
    
    // Calculate precision score
    const precision = this.calculatePrecisionScore(prices, currentPrice, entryTiming);
    
    let timing = 'wait';
    let action = 'hold';
    
    if (precision > this.thresholds.precisionTiming && entryTiming.signal !== 'wait') {
      timing = 'execute';
      action = entryTiming.signal;
    }
    
    return {
      timing,
      precision,
      action,
      timeframe: '5m',
      analysis: 'PRECISION ENTRY'
    };
  }

  /**
   * Calculate overall momentum train signal
   */
  calculateMomentumTrainSignal({ trendDirection, trendConfirmation, entryTiming, precisionEntry, currentPrice }) {
    // Weight each timeframe analysis
    const weights = {
      trend: 0.4,        // 40% weight for 4H trend
      confirmation: 0.3, // 30% weight for 1H confirmation
      entry: 0.2,        // 20% weight for 15M entry
      precision: 0.1     // 10% weight for 5M precision
    };
    
    // Calculate weighted confidence
    const weightedConfidence = 
      (trendDirection.confidence * weights.trend) +
      (trendConfirmation.acceleration * weights.confirmation) +
      (entryTiming.confidence * weights.entry) +
      (precisionEntry.precision * weights.precision);
    
    // Determine action
    let action = 'wait';
    if (precisionEntry.timing === 'execute' && weightedConfidence > 0.7) {
      action = precisionEntry.action;
    }
    
    return {
      action,
      confidence: weightedConfidence,
      direction: trendDirection.direction,
      momentum: trendConfirmation.status,
      timing: entryTiming.timing,
      precision: precisionEntry.timing,
      currentPrice,
      analysis: 'MOMENTUM TRAIN COMPLETE'
    };
  }

  /**
   * Calculate trend strength using linear regression
   */
  calculateTrendStrength(prices) {
    if (prices.length < 2) return 0;
    
    const n = prices.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = prices;
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const avgPrice = sumY / n;
    
    // Normalize slope relative to average price
    return slope / avgPrice;
  }

  /**
   * Calculate momentum acceleration
   */
  calculateMomentumAcceleration(prices) {
    if (prices.length < 3) return 0;
    
    const recent = prices.slice(-3);
    const change1 = (recent[1] - recent[0]) / recent[0];
    const change2 = (recent[2] - recent[1]) / recent[1];
    
    return change2 - change1; // Acceleration = change in momentum
  }

  /**
   * Check trend alignment between timeframes
   */
  checkTrendAlignment(trendDirection, acceleration) {
    if (trendDirection === 'bullish' && acceleration > 0) return true;
    if (trendDirection === 'bearish' && acceleration < 0) return true;
    return false;
  }

  /**
   * Calculate entry confidence
   */
  calculateEntryConfidence(trendDirection, trendConfirmation, prices) {
    let confidence = 0.5; // Base confidence
    
    // Add confidence for strong trend
    if (trendDirection.strength > 0.7) confidence += 0.2;
    
    // Add confidence for trend confirmation
    if (trendConfirmation.alignment) confidence += 0.2;
    
    // Add confidence for price momentum
    if (prices.length >= 3) {
      const momentum = (prices[prices.length - 1] - prices[0]) / prices[0];
      if (Math.abs(momentum) > 0.01) confidence += 0.1;
    }
    
    return Math.min(confidence, 1.0);
  }

  /**
   * Calculate precision score for exact entry
   */
  calculatePrecisionScore(prices, currentPrice, entryTiming) {
    if (entryTiming.signal === 'wait') return 0;
    
    let precision = 0.5; // Base precision
    
    // Add precision for price action alignment
    if (prices.length >= 2) {
      const lastMove = (prices[prices.length - 1] - prices[prices.length - 2]) / prices[prices.length - 2];
      
      if (entryTiming.signal === 'enter_long' && lastMove > 0) precision += 0.3;
      if (entryTiming.signal === 'enter_short' && lastMove < 0) precision += 0.3;
    }
    
    // Add precision for entry timing confidence
    precision += entryTiming.confidence * 0.2;
    
    return Math.min(precision, 1.0);
  }
}

module.exports = MomentumTrainAnalyzer;

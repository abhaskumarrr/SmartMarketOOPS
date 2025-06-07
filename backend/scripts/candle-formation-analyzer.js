/**
 * CANDLE FORMATION ANALYZER
 * 
 * Analyzes real-time candle formation using shorter timeframes to understand
 * how longer timeframe candles are being built internally.
 * 
 * 🕯️ CANDLE ANATOMY ANALYSIS:
 * - Body: Open vs Close battle (buying/selling pressure)
 * - Upper Wick: Rejection at higher prices (selling pressure)
 * - Lower Wick: Rejection at lower prices (buying pressure)
 * - Color: Green (bullish) vs Red (bearish)
 * 
 * 🔍 INTRA-CANDLE BEHAVIOR:
 * - 4H Candle = 16 × 15M candles = 48 × 5M candles
 * - 1H Candle = 4 × 15M candles = 12 × 5M candles
 * - 15M Candle = 3 × 5M candles
 * 
 * Shorter timeframes reveal HOW the longer candle is forming in real-time.
 */

const DefensiveTradingUtils = require('./defensive-trading-utils');

class CandleFormationAnalyzer {
  constructor(dataManager) {
    this.dataManager = dataManager;
    
    // Candle formation relationships
    this.candleRelationships = {
      '4h': { subCandles: '15m', ratio: 16 },  // 4H = 16 × 15M
      '1h': { subCandles: '15m', ratio: 4 },   // 1H = 4 × 15M  
      '15m': { subCandles: '5m', ratio: 3 }    // 15M = 3 × 5M
    };
    
    // Candle anatomy thresholds
    this.thresholds = {
      strongBody: 0.7,        // 70% of range is body (strong momentum)
      longWick: 0.3,          // 30% of range is wick (rejection)
      doji: 0.1,              // 10% body size (indecision)
      hammer: 2.0,            // Wick 2x body size (reversal signal)
      wickRejection: 0.5      // 50% wick retracement (strong rejection)
    };
  }

  /**
   * Analyze current candle formation across all timeframes
   */
  async analyzeCurrentCandleFormation(symbol, currentPrice) {
    try {
      console.log(`🕯️ ANALYZING CANDLE FORMATION for ${symbol} at $${currentPrice.toFixed(2)}`);
      console.log(`════════════════════════════════════════════════════════════════════════════════`);
      
      const formations = {};
      
      // Analyze each timeframe's current candle formation
      for (const [timeframe, relationship] of Object.entries(this.candleRelationships)) {
        const formation = await this.analyzeTimeframeFormation(symbol, timeframe, currentPrice);
        formations[timeframe] = formation;
        
        console.log(`🕯️ ${timeframe.toUpperCase()} CANDLE FORMATION:`);
        console.log(`   Status: ${formation.status}`);
        console.log(`   Body: ${formation.bodyType} (${(formation.bodyPercent * 100).toFixed(1)}%)`);
        console.log(`   Upper Wick: ${(formation.upperWickPercent * 100).toFixed(1)}% | Lower Wick: ${(formation.lowerWickPercent * 100).toFixed(1)}%`);
        console.log(`   Momentum: ${formation.momentum} | Pressure: ${formation.pressure}`);
        console.log(`   Prediction: ${formation.nextCandlePrediction}`);
        console.log(``);
      }
      
      // Analyze intra-candle behavior
      const intraCandleAnalysis = this.analyzeIntraCandleBehavior(formations, currentPrice);
      
      console.log(`🔍 INTRA-CANDLE BEHAVIOR ANALYSIS:`);
      console.log(`   4H Formation: ${intraCandleAnalysis.fourHourInsight}`);
      console.log(`   1H Formation: ${intraCandleAnalysis.oneHourInsight}`);
      console.log(`   15M Formation: ${intraCandleAnalysis.fifteenMinInsight}`);
      console.log(`   Overall Signal: ${intraCandleAnalysis.overallSignal}`);
      console.log(`   Next Move Prediction: ${intraCandleAnalysis.nextMovePrediction}`);
      
      return {
        formations,
        intraCandleAnalysis,
        currentPrice,
        timestamp: Date.now()
      };
      
    } catch (error) {
      console.error(`❌ Candle formation analysis error for ${symbol}: ${error.message}`);
      return null;
    }
  }

  /**
   * Analyze specific timeframe candle formation
   */
  async analyzeTimeframeFormation(symbol, timeframe, currentPrice) {
    const candles = this.dataManager.getHistoricalData(symbol, timeframe);
    
    if (!candles || candles.length < 2) {
      return this.getEmptyFormation();
    }
    
    // Get current forming candle (last candle)
    const currentCandle = candles[candles.length - 1];
    const previousCandle = candles[candles.length - 2];
    
    // Calculate candle anatomy
    const anatomy = this.calculateCandleAnatomy(currentCandle, currentPrice);
    
    // Determine candle type and characteristics
    const candleType = this.determineCandleType(anatomy);
    
    // Analyze momentum and pressure
    const momentum = this.analyzeMomentum(currentCandle, previousCandle, currentPrice);
    
    // Predict next candle behavior
    const prediction = this.predictNextCandle(anatomy, candleType, momentum);
    
    return {
      timeframe,
      status: this.getCandleStatus(anatomy, currentPrice),
      bodyType: candleType.type,
      bodyPercent: anatomy.bodyPercent,
      upperWickPercent: anatomy.upperWickPercent,
      lowerWickPercent: anatomy.lowerWickPercent,
      momentum: momentum.direction,
      pressure: momentum.pressure,
      nextCandlePrediction: prediction,
      anatomy,
      candleType,
      rawData: {
        current: currentCandle,
        previous: previousCandle,
        currentPrice
      }
    };
  }

  /**
   * Calculate detailed candle anatomy
   */
  calculateCandleAnatomy(candle, currentPrice) {
    const open = parseFloat(candle.open);
    const high = parseFloat(candle.high);
    const low = parseFloat(candle.low);
    const close = currentPrice; // Use current price as forming close
    
    // Calculate range and body
    const totalRange = high - low;
    const bodySize = Math.abs(close - open);
    const upperWick = high - Math.max(open, close);
    const lowerWick = Math.min(open, close) - low;
    
    // Calculate percentages
    const bodyPercent = totalRange > 0 ? bodySize / totalRange : 0;
    const upperWickPercent = totalRange > 0 ? upperWick / totalRange : 0;
    const lowerWickPercent = totalRange > 0 ? lowerWick / totalRange : 0;
    
    return {
      open,
      high,
      low,
      close,
      totalRange,
      bodySize,
      upperWick,
      lowerWick,
      bodyPercent,
      upperWickPercent,
      lowerWickPercent,
      isBullish: close > open,
      isBearish: close < open,
      isDoji: bodyPercent < this.thresholds.doji
    };
  }

  /**
   * Determine candle type based on anatomy
   */
  determineCandleType(anatomy) {
    const { bodyPercent, upperWickPercent, lowerWickPercent, isBullish, isBearish, isDoji } = anatomy;
    
    // Doji patterns (indecision)
    if (isDoji) {
      if (upperWickPercent > this.thresholds.longWick && lowerWickPercent > this.thresholds.longWick) {
        return { type: 'Long-Legged Doji', signal: 'indecision', strength: 'high' };
      } else if (upperWickPercent > this.thresholds.longWick) {
        return { type: 'Dragonfly Doji', signal: 'bullish_reversal', strength: 'medium' };
      } else if (lowerWickPercent > this.thresholds.longWick) {
        return { type: 'Gravestone Doji', signal: 'bearish_reversal', strength: 'medium' };
      }
      return { type: 'Doji', signal: 'indecision', strength: 'low' };
    }
    
    // Hammer patterns (reversal)
    if (lowerWickPercent > bodyPercent * this.thresholds.hammer) {
      return { 
        type: isBullish ? 'Hammer' : 'Hanging Man', 
        signal: 'bullish_reversal', 
        strength: 'high' 
      };
    }
    
    // Shooting star patterns (reversal)
    if (upperWickPercent > bodyPercent * this.thresholds.hammer) {
      return { 
        type: isBullish ? 'Inverted Hammer' : 'Shooting Star', 
        signal: 'bearish_reversal', 
        strength: 'high' 
      };
    }
    
    // Strong body candles (continuation)
    if (bodyPercent > this.thresholds.strongBody) {
      return { 
        type: isBullish ? 'Strong Bullish' : 'Strong Bearish', 
        signal: isBullish ? 'bullish_continuation' : 'bearish_continuation', 
        strength: 'high' 
      };
    }
    
    // Regular candles
    return { 
      type: isBullish ? 'Bullish' : 'Bearish', 
      signal: isBullish ? 'bullish' : 'bearish', 
      strength: 'medium' 
    };
  }

  /**
   * Analyze momentum and buying/selling pressure
   */
  analyzeMomentum(currentCandle, previousCandle, currentPrice) {
    const currentOpen = parseFloat(currentCandle.open);
    const previousClose = parseFloat(previousCandle.close);
    const currentHigh = parseFloat(currentCandle.high);
    const currentLow = parseFloat(currentCandle.low);
    
    // Gap analysis
    const gap = currentOpen - previousClose;
    const gapPercent = Math.abs(gap) / previousClose;
    
    // Current momentum
    const momentum = currentPrice - currentOpen;
    const momentumPercent = momentum / currentOpen;
    
    // Pressure analysis
    const buyingPressure = (currentPrice - currentLow) / (currentHigh - currentLow);
    const sellingPressure = 1 - buyingPressure;
    
    return {
      direction: momentum > 0 ? 'bullish' : 'bearish',
      strength: Math.abs(momentumPercent),
      gap: gap > 0 ? 'gap_up' : gap < 0 ? 'gap_down' : 'no_gap',
      gapPercent,
      pressure: buyingPressure > 0.6 ? 'buying' : sellingPressure > 0.6 ? 'selling' : 'balanced',
      buyingPressure,
      sellingPressure
    };
  }

  /**
   * Predict next candle behavior based on current formation
   */
  predictNextCandle(anatomy, candleType, momentum) {
    const { signal, strength } = candleType;
    const { direction, pressure } = momentum;
    
    // Reversal signals
    if (signal.includes('reversal')) {
      if (signal.includes('bullish')) {
        return 'Next candle likely bullish - reversal signal detected';
      } else {
        return 'Next candle likely bearish - reversal signal detected';
      }
    }
    
    // Continuation signals
    if (signal.includes('continuation')) {
      return `Next candle likely ${direction} - strong continuation momentum`;
    }
    
    // Indecision signals
    if (signal === 'indecision') {
      return 'Next candle direction uncertain - market indecision';
    }
    
    // Pressure-based prediction
    if (pressure === 'buying') {
      return 'Next candle likely bullish - strong buying pressure';
    } else if (pressure === 'selling') {
      return 'Next candle likely bearish - strong selling pressure';
    }
    
    return 'Next candle direction neutral - balanced market conditions';
  }

  /**
   * Analyze intra-candle behavior across timeframes
   */
  analyzeIntraCandleBehavior(formations, currentPrice) {
    const fourHour = formations['4h'];
    const oneHour = formations['1h'];
    const fifteenMin = formations['15m'];
    
    // Analyze how shorter timeframes are building longer timeframes
    const fourHourInsight = this.getIntraCandleInsight(fourHour, oneHour, '4H candle being built by 1H movements');
    const oneHourInsight = this.getIntraCandleInsight(oneHour, fifteenMin, '1H candle being built by 15M movements');
    const fifteenMinInsight = this.getFormationInsight(fifteenMin, '15M candle formation');
    
    // Overall signal synthesis
    const overallSignal = this.synthesizeOverallSignal(formations);
    const nextMovePrediction = this.predictNextMove(formations, currentPrice);
    
    return {
      fourHourInsight,
      oneHourInsight,
      fifteenMinInsight,
      overallSignal,
      nextMovePrediction
    };
  }

  /**
   * Get intra-candle insight between timeframes
   */
  getIntraCandleInsight(longerTF, shorterTF, description) {
    if (!longerTF || !shorterTF) {
      return `${description}: Insufficient data`;
    }
    
    const alignment = longerTF.momentum === shorterTF.momentum ? 'aligned' : 'diverging';
    const strength = longerTF.bodyPercent > 0.5 ? 'strong' : 'weak';
    
    return `${description}: ${alignment} momentum, ${strength} ${longerTF.bodyType} formation`;
  }

  /**
   * Get formation insight for single timeframe
   */
  getFormationInsight(formation, description) {
    if (!formation) {
      return `${description}: No data`;
    }
    
    return `${description}: ${formation.bodyType} with ${formation.pressure} pressure`;
  }

  /**
   * Synthesize overall signal from all timeframes
   */
  synthesizeOverallSignal(formations) {
    const signals = Object.values(formations).map(f => f?.momentum).filter(Boolean);
    
    if (signals.length === 0) return 'No signal';
    
    const bullishCount = signals.filter(s => s === 'bullish').length;
    const bearishCount = signals.filter(s => s === 'bearish').length;
    
    if (bullishCount > bearishCount) {
      return `Bullish (${bullishCount}/${signals.length} timeframes)`;
    } else if (bearishCount > bullishCount) {
      return `Bearish (${bearishCount}/${signals.length} timeframes)`;
    } else {
      return 'Mixed signals - market indecision';
    }
  }

  /**
   * Predict next move based on all formations
   */
  predictNextMove(formations, currentPrice) {
    const predictions = Object.values(formations)
      .map(f => f?.nextCandlePrediction)
      .filter(Boolean);
    
    if (predictions.length === 0) return 'No prediction available';
    
    const bullishPredictions = predictions.filter(p => p.includes('bullish')).length;
    const bearishPredictions = predictions.filter(p => p.includes('bearish')).length;
    
    if (bullishPredictions > bearishPredictions) {
      return `Upward move expected - ${bullishPredictions}/${predictions.length} timeframes bullish`;
    } else if (bearishPredictions > bullishPredictions) {
      return `Downward move expected - ${bearishPredictions}/${predictions.length} timeframes bearish`;
    } else {
      return 'Sideways movement expected - mixed timeframe signals';
    }
  }

  /**
   * Get candle status description
   */
  getCandleStatus(anatomy, currentPrice) {
    const progress = anatomy.totalRange > 0 ? 
      ((currentPrice - anatomy.low) / anatomy.totalRange * 100).toFixed(1) : 0;
    
    return `Forming ${anatomy.isBullish ? 'bullish' : 'bearish'} candle (${progress}% of range)`;
  }

  /**
   * Get empty formation for error cases
   */
  getEmptyFormation() {
    return {
      status: 'No data',
      bodyType: 'Unknown',
      bodyPercent: 0,
      upperWickPercent: 0,
      lowerWickPercent: 0,
      momentum: 'neutral',
      pressure: 'unknown',
      nextCandlePrediction: 'Insufficient data for prediction'
    };
  }
}

module.exports = CandleFormationAnalyzer;

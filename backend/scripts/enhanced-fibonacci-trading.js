/**
 * Enhanced Multi-Timeframe Fibonacci Trading Strategy
 * Implements institutional-grade market structure analysis with Fibonacci retracements
 */

const { DeltaExchangeUnified } = require('../dist/services/DeltaExchangeUnified.js');
const DefensiveTradingUtils = require('./defensive-trading-utils');
const DefensiveMultiTimeframeAnalyzer = require('./defensive-multi-timeframe-analyzer');
const CCXTMarketDataService = require('./ccxt-market-data-service');

class EnhancedFibonacciTrader {
  constructor() {
    this.config = {
      symbols: ['BTCUSD', 'ETHUSD'],
      riskPerTrade: 2.0,        // 2% risk per trade
      maxPositions: 2,          // Max 2 concurrent positions
      
      // Multi-timeframe settings
      dailyTimeframe: '1d',     // Daily chart for market structure
      entryTimeframe: '15m',    // 15M chart for entry signals
      confirmationTimeframe: '4h', // 4H chart for bias confirmation
      scalpTimeframe: '5m',     // 5M chart for scalping analysis
      
      // Fibonacci levels (key institutional levels)
      fibLevels: {
        0: 1.0,      // 100% (swing high/low)
        236: 0.764,  // 76.4% (0.236 retracement)
        382: 0.618,  // 61.8% (0.382 retracement) 
        500: 0.5,    // 50% (0.5 retracement)
        618: 0.382,  // 38.2% (0.618 retracement)
        786: 0.214,  // 21.4% (0.786 retracement)
        1000: 0.0    // 0% (full retracement)
      },
      
      // Trade classification thresholds
      tradeTypes: {
        scalp: { timeHorizon: 'minutes', fibTolerance: 0.002, minVolume: 1000 },
        day: { timeHorizon: 'hours', fibTolerance: 0.005, minVolume: 500 },
        swing: { timeHorizon: 'days', fibTolerance: 0.01, minVolume: 200 }
      },
      
      confluenceThreshold: 0.75, // 75% minimum confluence for more opportunities
      fibTolerance: 0.005,       // 0.5% tolerance for Fibonacci levels
      minSwingSize: 0.02,        // 2% minimum swing size for significance
      lookbackPeriod: 30,        // 30 bars (1 month of daily data)
      swingLookback: 5,          // 5 periods for swing detection
      maxSwingAge: 720           // Maximum 720 hours (30 days) for swing relevance
    };

    this.deltaService = null;
    this.ccxtService = new CCXTMarketDataService(); // Primary data source
    this.balance = { availableBalance: 0 };
    this.activePositions = new Map();
    this.marketStructure = new Map(); // Store market structure for each symbol
    this.defensiveAnalyzer = null; // Defensive multi-timeframe analyzer
  }

  /**
   * Initialize Delta Exchange connection
   */
  async initialize() {
    try {
      // Initialize with testnet credentials
      this.deltaService = new DeltaExchangeUnified({
        apiKey: process.env.DELTA_EXCHANGE_API_KEY,
        apiSecret: process.env.DELTA_EXCHANGE_API_SECRET,
        testnet: true
      });

      await this.deltaService.initialize();

      // Initialize CCXT service
      console.log(`🔧 Initializing CCXT market data service...`);
      await this.ccxtService.initialize();

      // Initialize defensive analyzer with both services
      this.defensiveAnalyzer = new DefensiveMultiTimeframeAnalyzer(this.deltaService);

      // Get account balance
      const balanceData = await this.deltaService.getBalance();

      // Find USD balance - handle different response formats
      let availableBalance = 0;
      if (Array.isArray(balanceData)) {
        const usdBalance = balanceData.find(b => b.asset_symbol === 'USD' || b.asset_id === 3);
        availableBalance = parseFloat(usdBalance?.available_balance || '0');
      } else if (balanceData && typeof balanceData === 'object') {
        // Handle direct balance object
        availableBalance = parseFloat(balanceData.available_balance || balanceData.availableBalance || '0');
      }

      this.balance.availableBalance = availableBalance;

      console.log(`✅ Enhanced Fibonacci Trader initialized`);
      console.log(`💰 Available Balance: $${this.balance.availableBalance.toFixed(2)}`);

      return true;
    } catch (error) {
      console.error(`❌ Failed to initialize: ${error.message}`);
      return false;
    }
  }

  /**
   * Get current price using CCXT with Delta Exchange fallback
   */
  async getCurrentPrice(symbol) {
    try {
      // Try CCXT first (more reliable)
      const ccxtPrice = await this.ccxtService.getCurrentPrice(symbol);
      if (ccxtPrice && ccxtPrice > 0) {
        return ccxtPrice;
      }

      // Fallback to Delta Exchange
      console.warn(`⚠️ CCXT failed for ${symbol}, trying Delta Exchange fallback`);
      const productId = this.getProductId(symbol);
      if (productId && this.deltaService) {
        const ticker = await this.deltaService.getTicker(productId);
        if (ticker && ticker.close) {
          return parseFloat(ticker.close);
        }
      }

      console.error(`❌ Failed to get current price for ${symbol} from all sources`);
      return null;
    } catch (error) {
      console.error(`❌ Error getting current price for ${symbol}: ${error.message}`);
      return null;
    }
  }

  /**
   * Analyze daily chart market structure and identify swing points
   */
  async analyzeDailyMarketStructure(symbol) {
    try {
      // Get daily OHLC data for swing analysis
      const dailyCandles = await this.deltaService.getCandles(
        this.getProductId(symbol), 
        this.config.dailyTimeframe, 
        this.config.lookbackPeriod
      );

      if (!dailyCandles || dailyCandles.length < 20) {
        throw new Error(`Insufficient daily data for ${symbol}`);
      }

      // Identify significant swing highs and lows
      const swingPoints = this.identifySwingPoints(dailyCandles);
      
      // Calculate Fibonacci retracement levels
      const fibLevels = this.calculateFibonacciLevels(swingPoints);
      
      // Determine current market bias
      const marketBias = this.determineMarketBias(dailyCandles, swingPoints);
      
      // Store market structure
      this.marketStructure.set(symbol, {
        swingPoints,
        fibLevels,
        marketBias,
        lastUpdate: Date.now(),
        timeframe: this.config.dailyTimeframe
      });

      console.log(`📊 ${symbol} Daily Market Structure (Recent ${this.config.lookbackPeriod} periods):`);
      console.log(`   Trend: ${marketBias.trend} (${marketBias.strength})`);
      console.log(`   Significant Swing High: $${swingPoints.lastHigh.price.toFixed(2)}`);
      console.log(`   Significant Swing Low: $${swingPoints.lastLow.price.toFixed(2)}`);
      console.log(`   Swing Range: $${swingPoints.swingRange.toFixed(2)} (${swingPoints.swingPercent.toFixed(1)}%)`);
      console.log(`   Key Fib Levels: ${Object.keys(fibLevels).slice(0, 4).map(k => `${k}% ($${fibLevels[k].toFixed(2)})`).join(', ')}`);

      return this.marketStructure.get(symbol);
    } catch (error) {
      console.error(`❌ Failed to analyze daily structure for ${symbol}: ${error.message}`);
      return null;
    }
  }

  /**
   * Identify significant swing highs and lows within recent timeframe
   * Focus on the most significant swing in the past month (30 daily candles)
   */
  identifySwingPoints(candles) {
    if (candles.length < 10) {
      return { highs: [], lows: [], lastHigh: null, lastLow: null, swingRange: 0 };
    }

    const swingHighs = [];
    const swingLows = [];
    const lookback = this.config.swingLookback; // 5 periods for pivot detection

    // Only analyze recent candles (last 30 for daily, proportionally for other timeframes)
    const recentCandles = candles.slice(-this.config.lookbackPeriod);

    // Find pivot points in recent data
    for (let i = lookback; i < recentCandles.length - lookback; i++) {
      const currentHigh = parseFloat(recentCandles[i].high);
      const currentLow = parseFloat(recentCandles[i].low);

      // Check for swing high (higher than surrounding candles)
      let isSwingHigh = true;
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i && parseFloat(recentCandles[j].high) >= currentHigh) {
          isSwingHigh = false;
          break;
        }
      }

      if (isSwingHigh) {
        swingHighs.push({
          index: i,
          price: currentHigh,
          time: recentCandles[i].time,
          candle: recentCandles[i]
        });
      }

      // Check for swing low (lower than surrounding candles)
      let isSwingLow = true;
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i && parseFloat(recentCandles[j].low) <= currentLow) {
          isSwingLow = false;
          break;
        }
      }

      if (isSwingLow) {
        swingLows.push({
          index: i,
          price: currentLow,
          time: recentCandles[i].time,
          candle: recentCandles[i]
        });
      }
    }

    // Find the most significant swing (largest range) in recent data
    let significantHigh = null;
    let significantLow = null;
    let maxSwingRange = 0;

    // Try different combinations of recent highs and lows
    const recentHighs = swingHighs.slice(-5); // Last 5 swing highs
    const recentLows = swingLows.slice(-5);   // Last 5 swing lows

    for (const high of recentHighs) {
      for (const low of recentLows) {
        const swingRange = Math.abs(high.price - low.price);
        const swingPercent = swingRange / Math.min(high.price, low.price);

        // Enhanced filtering for realistic swings
        const isReasonableRange = swingPercent >= this.config.minSwingSize && swingPercent <= 0.5; // Max 50% swing
        const isRecentEnough = Math.abs(high.index - low.index) <= 20; // Within 20 periods of each other

        if (isReasonableRange && isRecentEnough && swingRange > maxSwingRange) {
          maxSwingRange = swingRange;
          significantHigh = high;
          significantLow = low;
        }
      }
    }

    // Fallback: if no significant swing found, use most recent highs/lows
    if (!significantHigh && swingHighs.length > 0) {
      significantHigh = swingHighs[swingHighs.length - 1];
    }
    if (!significantLow && swingLows.length > 0) {
      significantLow = swingLows[swingLows.length - 1];
    }

    return {
      highs: swingHighs,
      lows: swingLows,
      lastHigh: significantHigh,
      lastLow: significantLow,
      swingRange: significantHigh && significantLow ? Math.abs(significantHigh.price - significantLow.price) : 0,
      swingPercent: significantHigh && significantLow ?
        (Math.abs(significantHigh.price - significantLow.price) / Math.min(significantHigh.price, significantLow.price)) * 100 : 0
    };
  }

  /**
   * Calculate Fibonacci retracement levels from swing points
   */
  calculateFibonacciLevels(swingPoints) {
    const { lastHigh, lastLow } = swingPoints;
    
    if (!lastHigh || !lastLow) {
      return {};
    }

    const range = lastHigh.price - lastLow.price;
    const fibLevels = {};
    
    // Calculate retracement levels (from high to low)
    if (lastHigh.time > lastLow.time) {
      // Downtrend: calculate retracements from high
      Object.entries(this.config.fibLevels).forEach(([key, ratio]) => {
        fibLevels[key] = lastHigh.price - (range * (1 - ratio));
      });
    } else {
      // Uptrend: calculate retracements from low
      Object.entries(this.config.fibLevels).forEach(([key, ratio]) => {
        fibLevels[key] = lastLow.price + (range * ratio);
      });
    }
    
    return fibLevels;
  }

  /**
   * Determine market bias from price action and structure
   */
  determineMarketBias(candles, swingPoints) {
    const recentCandles = candles.slice(-20); // Last 20 periods
    const { lastHigh, lastLow } = swingPoints;
    
    // Determine trend direction
    let trend = 'sideways';
    if (lastHigh && lastLow) {
      if (lastHigh.time > lastLow.time) {
        trend = 'bearish';
      } else {
        trend = 'bullish';
      }
    }
    
    // Calculate momentum
    const firstPrice = parseFloat(recentCandles[0].close);
    const lastPrice = parseFloat(recentCandles[recentCandles.length - 1].close);
    const momentum = ((lastPrice - firstPrice) / firstPrice) * 100;
    
    return {
      trend,
      momentum,
      strength: Math.abs(momentum) > 2 ? 'strong' : 'weak'
    };
  }

  /**
   * Analyze 4H timeframe for bias confirmation
   */
  async analyze4HBias(symbol) {
    try {
      const candles4h = await this.deltaService.getCandles(
        this.getProductId(symbol),
        this.config.confirmationTimeframe,
        50
      );

      if (!candles4h || candles4h.length < 10) {
        return { bias: 'neutral', confidence: 0, timeframe: '4h' };
      }

      // Simple trend analysis on 4H
      const recent = candles4h.slice(-10);
      const ema20 = this.calculateEMA(recent.map(c => parseFloat(c.close)), 20);
      const currentPrice = parseFloat(recent[recent.length - 1].close);
      
      let bias = 'neutral';
      let confidence = 0;
      
      if (currentPrice > ema20) {
        bias = 'bullish';
        confidence = Math.min(((currentPrice - ema20) / ema20) * 100, 100);
      } else {
        bias = 'bearish';
        confidence = Math.min(((ema20 - currentPrice) / ema20) * 100, 100);
      }

      return { bias, confidence, timeframe: '4h' };
    } catch (error) {
      console.error(`❌ Failed to analyze 4H bias for ${symbol}: ${error.message}`);
      return { bias: 'neutral', confidence: 0, timeframe: '4h' };
    }
  }

  /**
   * Calculate Exponential Moving Average
   */
  calculateEMA(prices, period) {
    if (prices.length < period) return prices[prices.length - 1];
    
    const multiplier = 2 / (period + 1);
    let ema = prices[0];
    
    for (let i = 1; i < prices.length; i++) {
      ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
    }
    
    return ema;
  }

  /**
   * Analyze multi-timeframe entry signals with Fibonacci confluence (DEFENSIVE VERSION)
   */
  async analyzeEntrySignals(symbol, currentPrice) {
    try {
      // Use defensive analyzer for multi-timeframe analysis
      const defensiveAnalysis = await this.defensiveAnalyzer.analyzeMultiTimeframeBias(symbol, currentPrice);

      if (!defensiveAnalysis || !defensiveAnalysis.isValid) {
        console.log(`⚠️ Defensive analysis failed or invalid for ${symbol}`);
        return null;
      }

      const marketStructure = this.marketStructure.get(symbol);
      if (!marketStructure) {
        console.log(`⚠️ No market structure data for ${symbol}`);
        return null;
      }

      // Check if current price is near significant Fibonacci level
      const fibSignal = this.checkFibonacciLevels(currentPrice, marketStructure.fibLevels);

      if (!fibSignal.isValid) {
        return null;
      }

      // Extract bias data from defensive analysis
      const bias4h = DefensiveTradingUtils.safeGet(defensiveAnalysis, 'timeframes.4h',
        DefensiveTradingUtils.createSafeBias('neutral', 0, '4h'));
      const bias15m = DefensiveTradingUtils.safeGet(defensiveAnalysis, 'timeframes.15m',
        DefensiveTradingUtils.createSafeBias('neutral', 0, '15m'));
      const bias5m = DefensiveTradingUtils.safeGet(defensiveAnalysis, 'timeframes.5m',
        DefensiveTradingUtils.createSafeBias('neutral', 0, '5m'));

      // Calculate confluence using defensive analysis
      const confluenceScore = DefensiveTradingUtils.safeGet(defensiveAnalysis, 'confluence', 0);

      // Classify trade type based on multi-timeframe analysis
      const tradeClassification = this.classifyTradeMultiTimeframe(fibSignal, marketStructure, bias4h, bias15m, bias5m);

      console.log(`🔍 ${symbol} Defensive Multi-Timeframe Analysis:`);
      console.log(`   Fib Level: ${fibSignal.level}% ($${fibSignal.price.toFixed(2)}) - Distance: ${(fibSignal.distance * 100).toFixed(2)}%`);
      console.log(`   4H Bias: ${bias4h.bias} (${bias4h.confidence.toFixed(0)}%)`);
      console.log(`   15M Bias: ${bias15m.bias} (${bias15m.confidence.toFixed(0)}%)`);
      console.log(`   5M Bias: ${bias5m.bias} (${bias5m.confidence.toFixed(0)}%)`);
      console.log(`   Confluence: ${(confluenceScore * 100).toFixed(1)}%`);
      console.log(`   Data Quality: ${defensiveAnalysis.dataQuality}`);

      return {
        signal: fibSignal,
        confluence: confluenceScore,
        tradeType: tradeClassification,
        bias4h,
        bias15m,
        bias5m,
        fvgs: { '15m': [], '5m': [] }, // Simplified for now
        orderBlocks: { '15m': [], '5m': [] }, // Simplified for now
        isValid: confluenceScore >= this.config.confluenceThreshold,
        defensiveAnalysis // Include full defensive analysis
      };

    } catch (error) {
      console.error(`❌ Failed to analyze entry signals for ${symbol}: ${error.message}`);
      return null;
    }
  }

  /**
   * Check if current price is near significant Fibonacci levels
   */
  checkFibonacciLevels(currentPrice, fibLevels) {
    const tolerance = this.config.fibTolerance;

    for (const [level, price] of Object.entries(fibLevels)) {
      const distance = Math.abs(currentPrice - price) / price;

      if (distance <= tolerance) {
        // Determine signal direction based on level and market structure
        const levelNum = parseInt(level);
        let direction = 'wait';
        let strength = 0;

        // Key institutional levels
        if ([382, 500, 618].includes(levelNum)) {
          strength = 90; // High probability levels
          direction = levelNum <= 500 ? 'buy' : 'sell';
        } else if ([236, 786].includes(levelNum)) {
          strength = 75; // Medium probability levels
          direction = levelNum === 236 ? 'sell' : 'buy';
        }

        return {
          isValid: true,
          level: level,
          price: price,
          distance: distance,
          direction: direction,
          strength: strength,
          fibRatio: this.config.fibLevels[level]
        };
      }
    }

    return { isValid: false };
  }

  /**
   * Identify Fair Value Gaps (FVGs) in price action
   */
  identifyFairValueGaps(candles) {
    const fvgs = [];

    for (let i = 2; i < candles.length; i++) {
      const prev = candles[i-2];
      const current = candles[i-1];
      const next = candles[i];

      const prevHigh = parseFloat(prev.high);
      const prevLow = parseFloat(prev.low);
      const nextHigh = parseFloat(next.high);
      const nextLow = parseFloat(next.low);

      // Bullish FVG: gap between previous low and next high
      if (prevLow > nextHigh) {
        fvgs.push({
          type: 'bullish',
          high: prevLow,
          low: nextHigh,
          index: i,
          time: current.time
        });
      }

      // Bearish FVG: gap between previous high and next low
      if (prevHigh < nextLow) {
        fvgs.push({
          type: 'bearish',
          high: nextLow,
          low: prevHigh,
          index: i,
          time: current.time
        });
      }
    }

    // Return only recent FVGs (last 10 periods)
    return fvgs.slice(-10);
  }

  /**
   * Identify institutional order blocks
   */
  identifyOrderBlocks(candles) {
    const orderBlocks = [];

    for (let i = 5; i < candles.length - 1; i++) {
      const current = candles[i];
      const next = candles[i + 1];

      const currentClose = parseFloat(current.close);
      const currentOpen = parseFloat(current.open);
      const nextClose = parseFloat(next.close);
      const nextOpen = parseFloat(next.open);

      // Bullish order block: strong bullish candle followed by continuation
      if (currentClose > currentOpen && nextClose > nextOpen &&
          (currentClose - currentOpen) / currentOpen > 0.01) {
        orderBlocks.push({
          type: 'bullish',
          high: parseFloat(current.high),
          low: parseFloat(current.low),
          index: i,
          time: current.time,
          strength: (currentClose - currentOpen) / currentOpen * 100
        });
      }

      // Bearish order block: strong bearish candle followed by continuation
      if (currentClose < currentOpen && nextClose < nextOpen &&
          (currentOpen - currentClose) / currentOpen > 0.01) {
        orderBlocks.push({
          type: 'bearish',
          high: parseFloat(current.high),
          low: parseFloat(current.low),
          index: i,
          time: current.time,
          strength: (currentOpen - currentClose) / currentOpen * 100
        });
      }
    }

    return orderBlocks.slice(-5); // Return last 5 order blocks
  }

  /**
   * Analyze timeframe bias using simple trend analysis
   */
  analyzeTimeframeBias(candles, timeframe) {
    if (!candles || candles.length < 5) {
      return { bias: 'neutral', confidence: 0, timeframe };
    }

    const recent = candles.slice(-Math.min(20, candles.length));
    const prices = recent.map(c => parseFloat(c.close));
    const ema10 = this.calculateEMA(prices, Math.min(10, prices.length));
    const ema20 = this.calculateEMA(prices, Math.min(20, prices.length));
    const currentPrice = prices[prices.length - 1];

    let bias = 'neutral';
    let confidence = 0;

    // Determine bias based on EMA relationship and price position
    if (currentPrice > ema10 && ema10 > ema20) {
      bias = 'bullish';
      confidence = Math.min(((currentPrice - ema20) / ema20) * 100, 100);
    } else if (currentPrice < ema10 && ema10 < ema20) {
      bias = 'bearish';
      confidence = Math.min(((ema20 - currentPrice) / ema20) * 100, 100);
    } else {
      confidence = 25; // Neutral but some confidence
    }

    return { bias, confidence: Math.abs(confidence), timeframe };
  }

  /**
   * Calculate multi-timeframe confluence score
   */
  calculateMultiTimeframeConfluence(factors) {
    let score = 0;
    const weights = {
      fibonacci: 0.30,    // 30% weight for Fibonacci levels
      bias4h: 0.20,       // 20% weight for 4H bias
      bias15m: 0.15,      // 15% weight for 15M bias
      bias3m: 0.10,       // 10% weight for 3M bias
      structure: 0.15,    // 15% weight for market structure
      fvg: 0.05,         // 5% weight for FVGs
      orderBlock: 0.05    // 5% weight for order blocks
    };

    // Fibonacci level strength
    if (factors.fibSignal.isValid) {
      score += (factors.fibSignal.strength / 100) * weights.fibonacci;
    }

    // Multi-timeframe bias alignment
    const biases = [factors.bias4h, factors.bias15m, factors.bias3m];
    const alignedBiases = biases.filter(b => b.bias === factors.fibSignal.direction.replace('buy', 'bullish').replace('sell', 'bearish'));

    // 4H bias (strongest weight)
    if (factors.bias4h.confidence > 40) {
      const alignment = factors.bias4h.bias === factors.fibSignal.direction.replace('buy', 'bullish').replace('sell', 'bearish') ? 1 : 0.3;
      score += (factors.bias4h.confidence / 100) * weights.bias4h * alignment;
    }

    // 15M bias
    if (factors.bias15m.confidence > 30) {
      const alignment = factors.bias15m.bias === factors.fibSignal.direction.replace('buy', 'bullish').replace('sell', 'bearish') ? 1 : 0.3;
      score += (factors.bias15m.confidence / 100) * weights.bias15m * alignment;
    }

    // 5M bias
    if (factors.bias5m.confidence > 25) {
      const alignment = factors.bias5m.bias === factors.fibSignal.direction.replace('buy', 'bullish').replace('sell', 'bearish') ? 1 : 0.3;
      score += (factors.bias5m.confidence / 100) * weights.bias3m * alignment;
    }

    // Market structure alignment
    const structureAlignment = this.checkStructureAlignment(factors.marketStructure, factors.fibSignal);
    score += structureAlignment * weights.structure;

    // Multi-timeframe FVG bonus
    const totalFVGs = factors.fvgs15m.length + factors.fvgs5m.length;
    if (totalFVGs > 0) {
      score += Math.min(totalFVGs * 0.2, 1.0) * weights.fvg;
    }

    // Multi-timeframe order block bonus
    const totalOrderBlocks = factors.orderBlocks15m.length + factors.orderBlocks5m.length;
    if (totalOrderBlocks > 0) {
      score += Math.min(totalOrderBlocks * 0.3, 1.0) * weights.orderBlock;
    }

    return Math.min(score, 1.0); // Cap at 100%
  }

  /**
   * Classify trade type based on multi-timeframe analysis
   */
  classifyTradeMultiTimeframe(fibSignal, marketStructure, bias4h, bias15m, bias5m) {
    const { swingRange, swingPercent } = marketStructure.swingPoints;
    const volatility = swingPercent / 100; // Convert to decimal

    // Count aligned timeframes
    const targetBias = fibSignal.direction.replace('buy', 'bullish').replace('sell', 'bearish');
    const alignedTimeframes = [bias4h, bias15m, bias5m].filter(b => b.bias === targetBias).length;

    // High volatility + strong multi-timeframe alignment = swing trade
    if (volatility > 0.05 && alignedTimeframes >= 2 && bias4h.confidence > 60) {
      return {
        type: 'swing',
        timeHorizon: 'days',
        expectedDuration: '2-5 days',
        riskReward: '1:4',
        confidence: 'high'
      };
    }

    // Medium volatility + some alignment = day trade
    if (volatility > 0.02 && alignedTimeframes >= 1 && bias15m.confidence > 40) {
      return {
        type: 'day',
        timeHorizon: 'hours',
        expectedDuration: '4-12 hours',
        riskReward: '1:3',
        confidence: 'medium'
      };
    }

    // Low volatility + 5M alignment = scalp trade
    if (bias5m.confidence > 30) {
      return {
        type: 'scalp',
        timeHorizon: 'minutes',
        expectedDuration: '15-60 minutes',
        riskReward: '1:2',
        confidence: 'low'
      };
    }

    // Default to day trade
    return {
      type: 'day',
      timeHorizon: 'hours',
      expectedDuration: '4-12 hours',
      riskReward: '1:3',
      confidence: 'low'
    };
  }

  /**
   * Check alignment between market structure and Fibonacci signal
   */
  checkStructureAlignment(marketStructure, fibSignal) {
    if (!fibSignal.isValid) return 0;

    const { marketBias } = marketStructure;

    // Check if signal direction aligns with market bias
    if ((marketBias.trend === 'bullish' && fibSignal.direction === 'buy') ||
        (marketBias.trend === 'bearish' && fibSignal.direction === 'sell')) {
      return marketBias.strength === 'strong' ? 1.0 : 0.7;
    }

    // Counter-trend signals get lower score
    return 0.3;
  }

  /**
   * Classify trade type based on market conditions
   */
  classifyTrade(fibSignal, marketStructure, bias4h) {
    const { swingRange } = marketStructure.swingPoints;
    const volatility = swingRange / marketStructure.swingPoints.lastHigh.price;

    // High volatility + strong bias = swing trade
    if (volatility > 0.05 && bias4h.confidence > 70) {
      return {
        type: 'swing',
        timeHorizon: 'days',
        expectedDuration: '2-5 days',
        riskReward: '1:4'
      };
    }

    // Medium volatility + medium bias = day trade
    if (volatility > 0.02 && bias4h.confidence > 40) {
      return {
        type: 'day',
        timeHorizon: 'hours',
        expectedDuration: '4-12 hours',
        riskReward: '1:3'
      };
    }

    // Low volatility = scalp trade
    return {
      type: 'scalp',
      timeHorizon: 'minutes',
      expectedDuration: '15-60 minutes',
      riskReward: '1:2'
    };
  }

  /**
   * Execute enhanced Fibonacci-based trade
   */
  async executeFibonacciTrade(symbol, currentPrice, entryAnalysis) {
    try {
      const { signal, tradeType, confluence } = entryAnalysis;

      // Calculate position size based on trade type and risk
      const positionSize = this.calculatePositionSize(tradeType, signal);

      // Determine stop loss and take profit based on Fibonacci levels
      const { stopLoss, takeProfit } = this.calculateFibonacciTargets(signal, tradeType, currentPrice);

      // Execute the trade
      const orderParams = {
        symbol: symbol,
        side: signal.direction === 'buy' ? 'buy' : 'sell',
        size: positionSize.contracts,
        price: currentPrice,
        stopLoss: stopLoss,
        takeProfit: takeProfit,
        leverage: positionSize.leverage
      };

      console.log(`🎯 ENHANCED FIBONACCI TRADE EXECUTION:`);
      console.log(`   Symbol: ${symbol} | Type: ${tradeType.type.toUpperCase()}`);
      console.log(`   Entry: $${currentPrice.toFixed(2)} | Fib Level: ${signal.level}% ($${signal.price.toFixed(2)})`);
      console.log(`   Confluence: ${(confluence * 100).toFixed(1)}%`);
      console.log(`   Position: ${positionSize.contracts} contracts | Leverage: ${positionSize.leverage}x`);
      console.log(`   Stop Loss: $${stopLoss.toFixed(2)} | Take Profit: $${takeProfit.toFixed(2)}`);
      console.log(`   Risk/Reward: ${tradeType.riskReward} | Duration: ${tradeType.expectedDuration}`);

      // Place order on Delta Exchange
      const order = await this.deltaService.placeOrder(orderParams);

      if (order && order.id) {
        const trade = {
          tradeId: `FIB_${Date.now()}`,
          orderId: order.id,
          symbol: symbol,
          side: signal.direction,
          entryPrice: currentPrice,
          contracts: positionSize.contracts,
          stopLoss: stopLoss,
          takeProfit: takeProfit,
          tradeType: tradeType.type,
          fibLevel: signal.level,
          confluence: confluence,
          timestamp: Date.now()
        };

        this.activePositions.set(trade.tradeId, trade);

        console.log(`✅ FIBONACCI TRADE EXECUTED SUCCESSFULLY!`);
        console.log(`   Trade ID: ${trade.tradeId} | Order ID: ${order.id}`);

        return trade;
      }

      throw new Error('Failed to place order');
    } catch (error) {
      console.error(`❌ Failed to execute Fibonacci trade: ${error.message}`);
      return null;
    }
  }

  /**
   * Calculate position size based on trade type and risk management
   */
  calculatePositionSize(tradeType, fibSignal) {
    const baseRisk = this.config.riskPerTrade / 100; // Convert to decimal
    const balance = this.balance.availableBalance;

    // Adjust risk based on trade type and Fibonacci level strength
    let adjustedRisk = baseRisk;

    // Higher risk for high-probability Fibonacci levels
    if ([382, 500, 618].includes(parseInt(fibSignal.level))) {
      adjustedRisk *= 1.2; // 20% increase for key levels
    }

    // Adjust leverage based on trade type
    let leverage;
    switch (tradeType.type) {
      case 'scalp':
        leverage = 25; // Conservative for scalping
        break;
      case 'day':
        leverage = 15; // Moderate for day trading
        break;
      case 'swing':
        leverage = 10; // Conservative for swing trading
        break;
      default:
        leverage = 15;
    }

    // Calculate position value
    const riskAmount = adjustedRisk * balance;
    const maxPositionValue = riskAmount * leverage;

    // Calculate contracts (assuming 0.001 BTC or 0.01 ETH per contract)
    const contractSize = fibSignal.level.includes('BTC') ? 0.001 : 0.01;
    const contractValue = contractSize * parseFloat(fibSignal.price);
    const contracts = Math.floor(maxPositionValue / contractValue);

    return {
      contracts: Math.max(1, Math.min(contracts, 50)), // Min 1, max 50 contracts
      leverage: leverage,
      riskAmount: riskAmount,
      positionValue: contracts * contractValue
    };
  }

  /**
   * Calculate stop loss and take profit based on Fibonacci levels
   */
  calculateFibonacciTargets(fibSignal, tradeType, currentPrice) {
    const fibLevels = Object.values(this.config.fibLevels);
    const currentLevel = this.config.fibLevels[fibSignal.level];

    let stopLoss, takeProfit;

    if (fibSignal.direction === 'buy') {
      // For buy signals, stop loss below next Fibonacci level
      const nextLowerLevel = fibLevels.find(level => level < currentLevel);
      stopLoss = currentPrice * (1 - (currentLevel - (nextLowerLevel || 0)) * 0.5);

      // Take profit at next higher Fibonacci level
      const nextHigherLevel = fibLevels.find(level => level > currentLevel);
      takeProfit = currentPrice * (1 + ((nextHigherLevel || 1) - currentLevel) * 0.8);
    } else {
      // For sell signals, stop loss above next Fibonacci level
      const nextHigherLevel = fibLevels.find(level => level > currentLevel);
      stopLoss = currentPrice * (1 + ((nextHigherLevel || 1) - currentLevel) * 0.5);

      // Take profit at next lower Fibonacci level
      const nextLowerLevel = fibLevels.find(level => level < currentLevel);
      takeProfit = currentPrice * (1 - (currentLevel - (nextLowerLevel || 0)) * 0.8);
    }

    // Adjust targets based on trade type
    const multiplier = tradeType.type === 'swing' ? 1.5 : tradeType.type === 'day' ? 1.2 : 1.0;

    return {
      stopLoss: stopLoss,
      takeProfit: fibSignal.direction === 'buy' ?
        takeProfit * multiplier :
        takeProfit / multiplier
    };
  }

  /**
   * Main trading loop with enhanced Fibonacci analysis
   */
  async startEnhancedTrading() {
    console.log(`🚀 STARTING ENHANCED FIBONACCI TRADING SYSTEM`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`📊 Strategy: Multi-Timeframe Fibonacci + Market Structure Analysis`);
    console.log(`🎯 Confluence Threshold: ${(this.config.confluenceThreshold * 100).toFixed(0)}%`);
    console.log(`⚡ Trade Types: Scalping, Day Trading, Swing Trading`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);

    let cycleCount = 0;

    while (true) {
      try {
        cycleCount++;
        console.log(`\n📊 Enhanced Trading Cycle ${cycleCount}`);

        // Update balance
        const balanceData = await this.deltaService.getBalance();
        this.balance.availableBalance = balanceData.availableBalance;

        for (const symbol of this.config.symbols) {
          // Skip if max positions reached
          if (this.activePositions.size >= this.config.maxPositions) {
            console.log(`⏸️ Maximum positions reached (${this.activePositions.size}/${this.config.maxPositions})`);
            break;
          }

          // Analyze daily market structure (update every hour)
          await this.analyzeDailyMarketStructure(symbol);

          // Get current price
          const currentPrice = await this.getCurrentPrice(symbol);
          if (!currentPrice) continue;

          console.log(`💹 ${symbol} Current Price: $${currentPrice.toFixed(2)}`);

          // Analyze entry signals with Fibonacci confluence
          const entryAnalysis = await this.analyzeEntrySignals(symbol, currentPrice);

          if (entryAnalysis && entryAnalysis.isValid) {
            console.log(`🎯 ${symbol} FIBONACCI SIGNAL DETECTED:`);
            console.log(`   Fib Level: ${entryAnalysis.signal.level}% ($${entryAnalysis.signal.price.toFixed(2)})`);
            console.log(`   Direction: ${entryAnalysis.signal.direction.toUpperCase()}`);
            console.log(`   Trade Type: ${entryAnalysis.tradeType.type.toUpperCase()}`);
            console.log(`   Confluence: ${(entryAnalysis.confluence * 100).toFixed(1)}%`);

            // Execute Fibonacci-based trade
            const trade = await this.executeFibonacciTrade(symbol, currentPrice, entryAnalysis);

            if (trade) {
              console.log(`✅ ENHANCED FIBONACCI TRADE EXECUTED: ${trade.tradeId}`);
            }
          } else {
            console.log(`⏸️ ${symbol} No valid Fibonacci signals (confluence < ${(this.config.confluenceThreshold * 100).toFixed(0)}%)`);
          }
        }

        // Manage active positions
        await this.manageActivePositions();

        // Display trading status
        this.displayTradingStatus();

        // Wait before next cycle
        await this.sleep(30000); // 30 seconds

      } catch (error) {
        console.error(`❌ Error in trading cycle: ${error.message}`);
        await this.sleep(10000); // Wait 10 seconds on error
      }
    }
  }

  /**
   * Get current price for symbol
   */
  async getCurrentPrice(symbol) {
    try {
      const ticker = await this.deltaService.getTicker(this.getProductId(symbol));
      return parseFloat(ticker.mark_price || ticker.last_price);
    } catch (error) {
      console.error(`❌ Failed to get price for ${symbol}: ${error.message}`);
      return null;
    }
  }

  /**
   * Manage active positions
   */
  async manageActivePositions() {
    if (this.activePositions.size === 0) return;

    console.log(`🎛️ Managing ${this.activePositions.size} active position(s)...`);

    for (const [tradeId, trade] of this.activePositions) {
      try {
        // Check order status
        const orderStatus = await this.deltaService.getOrderStatus(trade.orderId);

        if (orderStatus && (orderStatus.state === 'filled' || orderStatus.state === 'closed')) {
          console.log(`✅ Position ${tradeId} closed`);
          this.activePositions.delete(tradeId);
        }
      } catch (error) {
        console.error(`❌ Error managing position ${tradeId}: ${error.message}`);
      }
    }
  }

  /**
   * Display current trading status
   */
  displayTradingStatus() {
    console.log(`\n📊 ENHANCED FIBONACCI TRADING STATUS`);
    console.log(`────────────────────────────────────────────────────────────`);
    console.log(`💰 Balance: $${(this.balance.availableBalance || 0).toFixed(2)}`);
    console.log(`🎛️ Active Positions: ${this.activePositions.size}/${this.config.maxPositions}`);

    if (this.activePositions.size > 0) {
      console.log(`📋 Active Trades:`);
      for (const [tradeId, trade] of this.activePositions) {
        console.log(`   ${trade.symbol} ${trade.side.toUpperCase()} @ $${trade.entryPrice.toFixed(2)} (${trade.tradeType.toUpperCase()}) - Fib ${trade.fibLevel}%`);
      }
    }
    console.log(`────────────────────────────────────────────────────────────`);
  }

  /**
   * Sleep utility
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current price for symbol using Delta Exchange API
   */
  async getCurrentPrice(symbol) {
    try {
      const marketData = await this.deltaService.getMarketData(symbol);
      return parseFloat(marketData.mark_price || marketData.last_price);
    } catch (error) {
      console.error(`❌ Failed to get price for ${symbol}: ${error.message}`);
      return null;
    }
  }

  /**
   * Get order status from Delta Exchange
   */
  async getOrderStatus(orderId) {
    try {
      return await this.deltaService.getOrder(orderId);
    } catch (error) {
      console.error(`❌ Failed to get order status for ${orderId}: ${error.message}`);
      return null;
    }
  }

  /**
   * Get ticker data for symbol
   */
  async getTicker(productId) {
    try {
      const symbol = this.getSymbolByProductId(productId);
      if (!symbol) {
        throw new Error(`Symbol not found for product ID: ${productId}`);
      }
      return await this.deltaService.getMarketData(symbol);
    } catch (error) {
      console.error(`❌ Failed to get ticker for product ${productId}: ${error.message}`);
      return null;
    }
  }

  /**
   * Get symbol by product ID
   */
  getSymbolByProductId(productId) {
    const productMap = {
      84: 'BTCUSD',    // Testnet BTC
      1699: 'ETHUSD'   // Testnet ETH
    };
    return productMap[productId];
  }

  /**
   * Get product ID for symbol
   */
  getProductId(symbol) {
    const productMap = {
      'BTCUSD': 84,  // Testnet BTC
      'ETHUSD': 1699 // Testnet ETH
    };
    return productMap[symbol];
  }
}

module.exports = EnhancedFibonacciTrader;

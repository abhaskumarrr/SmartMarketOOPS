const DefensiveTradingUtils = require('./defensive-trading-utils');
const CCXTMarketDataService = require('./ccxt-market-data-service');

/**
 * Defensive Multi-Timeframe Analyzer
 * 
 * Professional-grade multi-timeframe analysis with comprehensive error handling
 * and defensive programming patterns to prevent system failures.
 * 
 * Features:
 * - Null safety for all operations
 * - Graceful degradation on API failures
 * - Fallback mechanisms for missing data
 * - Error boundaries for async operations
 * - Data validation and sanitization
 */
class DefensiveMultiTimeframeAnalyzer {
  constructor(deltaService = null) {
    this.deltaService = deltaService; // Keep for backward compatibility
    this.ccxtService = new CCXTMarketDataService(); // Primary data source
    this.cache = new Map(); // Cache for reducing API calls
    this.lastUpdate = new Map(); // Track last update times
    this.cacheTimeout = 30000; // 30 seconds cache timeout
    this.initialized = false;
  }

  /**
   * Initialize CCXT service
   */
  async initialize() {
    if (!this.initialized) {
      console.log(`🔧 Initializing CCXT Market Data Service...`);
      this.initialized = await this.ccxtService.initialize();

      if (this.initialized) {
        console.log(`✅ CCXT service initialized successfully`);
        // Test connections
        await this.ccxtService.testConnections();
      } else {
        console.warn(`⚠️ CCXT service initialization failed, using fallback`);
      }
    }
    return this.initialized;
  }

  /**
   * Analyze multi-timeframe bias with comprehensive error handling
   */
  async analyzeMultiTimeframeBias(symbol, currentPrice) {
    try {
      // Initialize CCXT if not already done
      await this.initialize();

      console.log(`🔍 Starting CCXT-powered multi-timeframe analysis for ${symbol}`);

      // Validate inputs
      if (!symbol || !currentPrice || currentPrice <= 0) {
        console.warn(`⚠️ Invalid inputs for ${symbol}: price=${currentPrice}`);
        return this.createFallbackAnalysis(symbol, 'invalid-inputs');
      }

      // Get multi-timeframe data with defensive API calls
      const timeframes = ['4h', '15m', '5m'];
      const candlePromises = timeframes.map(tf => 
        this.getSafeCandleData(symbol, tf, 50)
      );

      // Use Promise.allSettled to prevent one failure from breaking everything
      const candleResults = await DefensiveTradingUtils.safePromiseAllSettled(
        candlePromises,
        timeframes.map(tf => this.createFallbackCandleData(symbol, tf))
      );

      // Extract candle data with validation
      const [candles4h, candles15m, candles5m] = candleResults;

      // Validate candle data
      const validationResults = candleResults.map((candles, index) => ({
        timeframe: timeframes[index],
        validation: DefensiveTradingUtils.validateCandleData(candles),
        data: candles
      }));

      // Analyze bias for each timeframe with error handling
      const biasPromises = validationResults.map(result => 
        this.analyzeSafeTimeframeBias(result.data, result.timeframe)
      );

      const biasResults = await DefensiveTradingUtils.safePromiseAllSettled(
        biasPromises,
        timeframes.map(tf => DefensiveTradingUtils.createSafeBias('neutral', 0, tf))
      );

      // Extract bias data safely
      const [bias4h, bias15m, bias5m] = biasResults;

      // Ensure all bias objects are valid
      const safeBias4h = this.validateBiasObject(bias4h, '4h');
      const safeBias15m = this.validateBiasObject(bias15m, '15m');
      const safeBias5m = this.validateBiasObject(bias5m, '5m');

      // Calculate confluence with error handling
      const confluence = this.calculateSafeConfluence({
        bias4h: safeBias4h,
        bias15m: safeBias15m,
        bias5m: safeBias5m,
        currentPrice
      });

      // Create comprehensive analysis result
      const analysis = {
        symbol,
        currentPrice,
        timestamp: Date.now(),
        timeframes: {
          '4h': safeBias4h,
          '15m': safeBias15m,
          '5m': safeBias5m
        },
        confluence,
        dataQuality: this.assessDataQuality(validationResults),
        isValid: this.validateAnalysisResult(safeBias4h, safeBias15m, safeBias5m),
        errors: this.collectAnalysisErrors(validationResults, biasResults)
      };

      console.log(`✅ Defensive analysis completed for ${symbol}:`);
      console.log(`   4H: ${safeBias4h.bias} (${safeBias4h.confidence.toFixed(0)}%)`);
      console.log(`   15M: ${safeBias15m.bias} (${safeBias15m.confidence.toFixed(0)}%)`);
      console.log(`   5M: ${safeBias5m.bias} (${safeBias5m.confidence.toFixed(0)}%)`);
      console.log(`   Confluence: ${(confluence * 100).toFixed(1)}%`);

      return analysis;

    } catch (error) {
      console.error(`❌ Critical error in multi-timeframe analysis for ${symbol}: ${error.message}`);
      return this.createFallbackAnalysis(symbol, 'critical-error', error);
    }
  }

  /**
   * Get candle data with CCXT and defensive API handling
   */
  async getSafeCandleData(symbol, timeframe, limit = 50) {
    const cacheKey = `${symbol}_${timeframe}_${limit}`;

    // Check cache first
    if (this.isCacheValid(cacheKey)) {
      console.log(`📋 Using cached data for ${symbol} ${timeframe}`);
      return this.cache.get(cacheKey);
    }

    // Try CCXT first (primary data source)
    if (this.initialized) {
      const candles = await DefensiveTradingUtils.safeApiCall(
        async () => {
          console.log(`📡 Fetching ${symbol} ${timeframe} candles via CCXT (${limit} bars)`);
          return await this.ccxtService.getCandles(symbol, timeframe, limit);
        },
        3, // 3 retries
        1000, // 1 second delay
        [] // empty fallback
      );

      // Validate CCXT data
      const validation = DefensiveTradingUtils.validateCandleData(candles);
      if (validation.isValid && candles.length > 0) {
        this.cache.set(cacheKey, candles);
        this.lastUpdate.set(cacheKey, Date.now());
        console.log(`✅ Got ${candles.length} candles from CCXT for ${symbol} ${timeframe}`);
        return candles;
      } else {
        console.warn(`⚠️ CCXT data invalid for ${symbol} ${timeframe}, trying fallback`);
      }
    }

    // Fallback to Delta Exchange if CCXT fails
    if (this.deltaService) {
      const productId = this.getProductIdSafe(symbol);
      if (productId) {
        const candles = await DefensiveTradingUtils.safeApiCall(
          async () => {
            console.log(`📡 Fallback: Fetching ${symbol} ${timeframe} via Delta Exchange (${limit} bars)`);
            return await this.deltaService.getCandles(productId, timeframe, limit);
          },
          2, // 2 retries for fallback
          1500, // 1.5 second delay
          this.createFallbackCandleData(symbol, timeframe)
        );

        const validation = DefensiveTradingUtils.validateCandleData(candles);
        if (validation.isValid) {
          this.cache.set(cacheKey, candles);
          this.lastUpdate.set(cacheKey, Date.now());
          console.log(`✅ Got ${candles.length} candles from Delta Exchange fallback`);
          return candles;
        }
      }
    }

    // Final fallback
    console.warn(`⚠️ All data sources failed for ${symbol} ${timeframe}, using empty fallback`);
    return this.createFallbackCandleData(symbol, timeframe);
  }

  /**
   * Analyze timeframe bias with comprehensive error handling
   */
  async analyzeSafeTimeframeBias(candles, timeframe) {
    try {
      // Validate input data
      const validation = DefensiveTradingUtils.validateCandleData(candles);
      if (!validation.isValid) {
        console.warn(`⚠️ Invalid candle data for ${timeframe}: ${validation.reason}`);
        return DefensiveTradingUtils.createSafeBias('neutral', 0, timeframe);
      }

      // Ensure minimum data requirements
      if (candles.length < 5) {
        console.warn(`⚠️ Insufficient data for ${timeframe}: ${candles.length} candles`);
        return DefensiveTradingUtils.createSafeBias('neutral', 25, timeframe);
      }

      // Safe price extraction
      const prices = candles.map(candle => {
        const close = DefensiveTradingUtils.safeGet(candle, 'close', 0);
        return DefensiveTradingUtils.safeCalculation(() => parseFloat(close), 0);
      }).filter(price => price > 0);

      if (prices.length < 5) {
        console.warn(`⚠️ Insufficient valid prices for ${timeframe}: ${prices.length} prices`);
        return DefensiveTradingUtils.createSafeBias('neutral', 20, timeframe);
      }

      // Calculate EMAs with error handling
      const ema10 = this.calculateSafeEMA(prices, Math.min(10, prices.length));
      const ema20 = this.calculateSafeEMA(prices, Math.min(20, prices.length));
      const currentPrice = DefensiveTradingUtils.safeArrayAccess(prices, -1, 0);

      if (!ema10 || !ema20 || !currentPrice) {
        console.warn(`⚠️ EMA calculation failed for ${timeframe}`);
        return DefensiveTradingUtils.createSafeBias('neutral', 15, timeframe);
      }

      // Determine bias with safe calculations
      let bias = 'neutral';
      let confidence = 0;

      const priceDiff = DefensiveTradingUtils.safeCalculation(() => {
        if (currentPrice > ema10 && ema10 > ema20) {
          bias = 'bullish';
          return ((currentPrice - ema20) / ema20) * 100;
        } else if (currentPrice < ema10 && ema10 < ema20) {
          bias = 'bearish';
          return ((ema20 - currentPrice) / ema20) * 100;
        } else {
          return 25; // Neutral confidence
        }
      }, 25);

      confidence = Math.min(Math.abs(priceDiff), 100);

      return DefensiveTradingUtils.createSafeBias(bias, confidence, timeframe);

    } catch (error) {
      console.error(`❌ Error analyzing ${timeframe} bias: ${error.message}`);
      return DefensiveTradingUtils.createSafeBias('neutral', 0, timeframe);
    }
  }

  /**
   * Calculate EMA with error handling
   */
  calculateSafeEMA(prices, period) {
    return DefensiveTradingUtils.safeCalculation(() => {
      if (!Array.isArray(prices) || prices.length < period || period <= 0) {
        return null;
      }

      const multiplier = 2 / (period + 1);
      let ema = prices[0];

      for (let i = 1; i < prices.length; i++) {
        ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
      }

      return ema;
    }, null);
  }

  /**
   * Validate bias object structure
   */
  validateBiasObject(bias, timeframe) {
    if (!bias || typeof bias !== 'object') {
      return DefensiveTradingUtils.createSafeBias('neutral', 0, timeframe);
    }

    return {
      bias: DefensiveTradingUtils.validateBias(bias.bias),
      confidence: DefensiveTradingUtils.validateConfidence(bias.confidence),
      timeframe: DefensiveTradingUtils.validateTimeframe(bias.timeframe || timeframe),
      timestamp: bias.timestamp || Date.now(),
      isValid: true
    };
  }

  /**
   * Calculate confluence with error handling
   */
  calculateSafeConfluence(factors) {
    return DefensiveTradingUtils.safeCalculation(() => {
      const weights = { bias4h: 0.5, bias15m: 0.3, bias5m: 0.2 };
      let score = 0;

      // 4H bias contribution
      if (factors.bias4h && factors.bias4h.confidence > 20) {
        score += (factors.bias4h.confidence / 100) * weights.bias4h;
      }

      // 15M bias contribution
      if (factors.bias15m && factors.bias15m.confidence > 15) {
        score += (factors.bias15m.confidence / 100) * weights.bias15m;
      }

      // 5M bias contribution
      if (factors.bias5m && factors.bias5m.confidence > 10) {
        score += (factors.bias5m.confidence / 100) * weights.bias5m;
      }

      return Math.min(score, 1.0);
    }, 0);
  }

  /**
   * Create fallback analysis for error cases
   */
  createFallbackAnalysis(symbol, reason, error = null) {
    throw new Error(`Analysis failed for ${symbol}: ${reason}`);
  }

  /**
   * Create fallback candle data
   */
  createFallbackCandleData(symbol, timeframe) {
    throw new Error(`Candle data unavailable for ${symbol} ${timeframe}`);
  }

  /**
   * Check if cache is valid
   */
  isCacheValid(cacheKey) {
    if (!this.cache.has(cacheKey) || !this.lastUpdate.has(cacheKey)) {
      return false;
    }

    const lastUpdate = this.lastUpdate.get(cacheKey);
    return (Date.now() - lastUpdate) < this.cacheTimeout;
  }

  /**
   * Get product ID safely
   */
  getProductIdSafe(symbol) {
    const productMap = {
      'BTCUSD': 84,    // Testnet BTC
      'ETHUSD': 1699   // Testnet ETH
    };
    return productMap[symbol] || null;
  }

  /**
   * Assess data quality
   */
  assessDataQuality(validationResults) {
    const validCount = validationResults.filter(r => r.validation.isValid).length;
    const totalCount = validationResults.length;
    
    if (validCount === totalCount) return 'excellent';
    if (validCount >= totalCount * 0.7) return 'good';
    if (validCount >= totalCount * 0.5) return 'fair';
    return 'poor';
  }

  /**
   * Validate analysis result
   */
  validateAnalysisResult(bias4h, bias15m, bias5m) {
    return bias4h.isValid && bias15m.isValid && bias5m.isValid;
  }

  /**
   * Collect analysis errors
   */
  collectAnalysisErrors(validationResults, biasResults) {
    const errors = [];
    
    validationResults.forEach((result, index) => {
      if (!result.validation.isValid) {
        errors.push(`${result.timeframe}: ${result.validation.reason}`);
      }
    });

    return errors;
  }
}

module.exports = DefensiveMultiTimeframeAnalyzer;

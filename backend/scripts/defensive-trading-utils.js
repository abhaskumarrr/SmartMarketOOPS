/**
 * Defensive Programming Utilities for Trading Systems
 * 
 * Professional-grade error handling and null safety patterns
 * based on industry best practices for algorithmic trading systems.
 * 
 * Research-backed defensive programming techniques:
 * - Null safety patterns
 * - Graceful degradation
 * - Error boundaries
 * - Fallback mechanisms
 * - Data validation
 */

class DefensiveTradingUtils {
  /**
   * Safe object property access with fallback
   * Prevents "Cannot read properties of undefined" errors
   */
  static safeGet(obj, path, fallback = null) {
    try {
      const keys = path.split('.');
      let result = obj;
      
      for (const key of keys) {
        if (result === null || result === undefined) {
          return fallback;
        }
        result = result[key];
      }
      
      return result !== undefined ? result : fallback;
    } catch (error) {
      return fallback;
    }
  }

  /**
   * Safe bias object creation with validation
   * Ensures all bias objects have required properties
   */
  static createSafeBias(bias = 'neutral', confidence = 0, timeframe = 'unknown') {
    return {
      bias: this.validateBias(bias),
      confidence: this.validateConfidence(confidence),
      timeframe: this.validateTimeframe(timeframe),
      timestamp: Date.now(),
      isValid: true
    };
  }

  /**
   * Validate bias value
   */
  static validateBias(bias) {
    const validBiases = ['bullish', 'bearish', 'neutral'];
    return validBiases.includes(bias) ? bias : 'neutral';
  }

  /**
   * Validate confidence value
   */
  static validateConfidence(confidence) {
    const num = parseFloat(confidence);
    if (isNaN(num)) return 0;
    return Math.max(0, Math.min(100, num));
  }

  /**
   * Validate timeframe value
   */
  static validateTimeframe(timeframe) {
    const validTimeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'];
    return validTimeframes.includes(timeframe) ? timeframe : 'unknown';
  }

  /**
   * Safe array access with bounds checking
   */
  static safeArrayAccess(array, index, fallback = null) {
    if (!Array.isArray(array) || array.length === 0) {
      return fallback;
    }
    
    const safeIndex = index < 0 ? array.length + index : index;
    
    if (safeIndex < 0 || safeIndex >= array.length) {
      return fallback;
    }
    
    return array[safeIndex];
  }

  /**
   * Safe numeric calculation with validation
   */
  static safeCalculation(calculation, fallback = 0) {
    try {
      const result = calculation();
      
      if (typeof result !== 'number' || isNaN(result) || !isFinite(result)) {
        return fallback;
      }
      
      return result;
    } catch (error) {
      return fallback;
    }
  }

  /**
   * Safe API call with retry mechanism
   */
  static async safeApiCall(apiFunction, retries = 3, delay = 1000, fallback = null) {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const result = await apiFunction();
        
        if (result !== null && result !== undefined) {
          return result;
        }
      } catch (error) {
        console.warn(`API call attempt ${attempt}/${retries} failed: ${error.message}`);
        
        if (attempt === retries) {
          console.error(`All ${retries} API call attempts failed, returning fallback`);
          return fallback;
        }
        
        // Exponential backoff
        await this.sleep(delay * Math.pow(2, attempt - 1));
      }
    }
    
    return fallback;
  }

  /**
   * Safe promise resolution with timeout
   */
  static async safePromiseWithTimeout(promise, timeoutMs = 10000, fallback = null) {
    try {
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Promise timeout')), timeoutMs);
      });
      const result = await Promise.race([promise, timeoutPromise]);
      return result;
    } catch (error) {
      throw new Error(`Promise failed or timed out: ${error.message}`);
    }
  }

  /**
   * Safe multi-promise resolution with individual error handling
   */
  static async safePromiseAllSettled(promises, fallbacks = []) {
    try {
      const results = await Promise.allSettled(promises);
      
      return results.map((result, index) => {
        if (result.status === 'fulfilled') {
          return result.value !== undefined ? result.value : fallbacks[index] || null;
        } else {
          console.warn(`Promise ${index} rejected: ${result.reason?.message || 'Unknown error'}`);
          return fallbacks[index] || null;
        }
      });
    } catch (error) {
      console.error(`Promise.allSettled failed: ${error.message}`);
      return fallbacks;
    }
  }

  /**
   * Validate candle data structure
   */
  static validateCandleData(candles) {
    if (!Array.isArray(candles) || candles.length === 0) {
      return { isValid: false, reason: 'No candle data provided' };
    }

    const requiredFields = ['open', 'high', 'low', 'close', 'time'];
    const invalidCandles = [];

    for (let i = 0; i < candles.length; i++) {
      const candle = candles[i];
      
      for (const field of requiredFields) {
        if (candle[field] === undefined || candle[field] === null) {
          invalidCandles.push({ index: i, field, value: candle[field] });
        }
      }
    }

    if (invalidCandles.length > 0) {
      return { 
        isValid: false, 
        reason: 'Invalid candle data', 
        invalidCandles: invalidCandles.slice(0, 5) // Show first 5 errors
      };
    }

    return { isValid: true, count: candles.length };
  }

  /**
   * Create error-resistant market data object
   */
  static createSafeMarketData(symbol, price = 0, timestamp = Date.now()) {
    return {
      symbol: symbol || 'UNKNOWN',
      price: this.safeCalculation(() => parseFloat(price), 0),
      timestamp,
      isValid: price > 0,
      source: 'defensive-fallback'
    };
  }

  /**
   * Sleep utility for delays
   */
  static sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Create comprehensive error report
   */
  static createErrorReport(error, context = {}) {
    return {
      message: error.message || 'Unknown error',
      stack: error.stack || 'No stack trace',
      context,
      timestamp: Date.now(),
      type: error.constructor.name || 'Error'
    };
  }

  /**
   * Safe JSON parsing
   */
  static safeJsonParse(jsonString, fallback = {}) {
    try {
      const result = JSON.parse(jsonString);
      return result !== null ? result : fallback;
    } catch (error) {
      return fallback;
    }
  }

  /**
   * Validate and sanitize trading parameters
   */
  static validateTradingParams(params) {
    const defaults = {
      symbol: 'UNKNOWN',
      side: 'buy',
      size: 0,
      price: 0,
      leverage: 1,
      stopLoss: 0,
      takeProfit: 0
    };

    const sanitized = { ...defaults };

    // Validate each parameter
    if (params.symbol && typeof params.symbol === 'string') {
      sanitized.symbol = params.symbol.toUpperCase();
    }

    if (['buy', 'sell'].includes(params.side)) {
      sanitized.side = params.side;
    }

    sanitized.size = this.safeCalculation(() => Math.abs(parseFloat(params.size)), 0);
    sanitized.price = this.safeCalculation(() => Math.abs(parseFloat(params.price)), 0);
    sanitized.leverage = this.safeCalculation(() => Math.max(1, Math.min(100, parseFloat(params.leverage))), 1);
    sanitized.stopLoss = this.safeCalculation(() => Math.abs(parseFloat(params.stopLoss)), 0);
    sanitized.takeProfit = this.safeCalculation(() => Math.abs(parseFloat(params.takeProfit)), 0);

    return {
      params: sanitized,
      isValid: sanitized.size > 0 && sanitized.price > 0,
      errors: this.validateTradingParamsErrors(sanitized)
    };
  }

  /**
   * Check for trading parameter errors
   */
  static validateTradingParamsErrors(params) {
    const errors = [];

    if (params.size <= 0) errors.push('Invalid size: must be greater than 0');
    if (params.price <= 0) errors.push('Invalid price: must be greater than 0');
    if (params.leverage < 1 || params.leverage > 100) errors.push('Invalid leverage: must be between 1 and 100');
    if (params.symbol === 'UNKNOWN') errors.push('Invalid symbol: must be provided');

    return errors;
  }
}

module.exports = DefensiveTradingUtils;

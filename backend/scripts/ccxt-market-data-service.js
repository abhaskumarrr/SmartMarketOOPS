/**
 * CCXT Market Data Service
 * 
 * Professional-grade market data fetching using CCXT library
 * with multiple exchange support and comprehensive error handling.
 * 
 * Features:
 * - Multi-exchange support (Binance, Bybit, OKX, Delta Exchange)
 * - Automatic failover between exchanges
 * - Standardized data format
 * - Built-in caching and rate limiting
 * - Professional error handling
 */

const ccxt = require('ccxt');
const DefensiveTradingUtils = require('./defensive-trading-utils');

class CCXTMarketDataService {
  constructor() {
    this.exchanges = new Map();
    this.cache = new Map();
    this.cacheTimeout = 30000; // 30 seconds
    this.lastUpdate = new Map();
    
    // Exchange priority order (most reliable first)
    this.exchangePriority = [
      'binance',
      'bybit', 
      'okx',
      'delta'  // Delta Exchange as fallback
    ];
    
    // Symbol mapping for different exchanges
    this.symbolMapping = {
      'BTCUSD': {
        'binance': 'BTC/USDT',
        'bybit': 'BTC/USDT',
        'okx': 'BTC/USDT',
        'delta': 'BTC/USD'
      },
      'ETHUSD': {
        'binance': 'ETH/USDT',
        'bybit': 'ETH/USDT', 
        'okx': 'ETH/USDT',
        'delta': 'ETH/USD'
      }
    };
  }

  /**
   * Initialize CCXT exchanges
   */
  async initialize() {
    try {
      console.log(`🔧 Initializing CCXT Market Data Service...`);
      
      // Initialize Binance (most reliable)
      try {
        this.exchanges.set('binance', new ccxt.binance({
          sandbox: false, // Use live data
          enableRateLimit: true,
          timeout: 10000
        }));
        console.log(`✅ Binance exchange initialized`);
      } catch (error) {
        console.warn(`⚠️ Failed to initialize Binance: ${error.message}`);
      }

      // Initialize Bybit
      try {
        this.exchanges.set('bybit', new ccxt.bybit({
          sandbox: false,
          enableRateLimit: true,
          timeout: 10000
        }));
        console.log(`✅ Bybit exchange initialized`);
      } catch (error) {
        console.warn(`⚠️ Failed to initialize Bybit: ${error.message}`);
      }

      // Initialize OKX
      try {
        this.exchanges.set('okx', new ccxt.okx({
          sandbox: false,
          enableRateLimit: true,
          timeout: 10000
        }));
        console.log(`✅ OKX exchange initialized`);
      } catch (error) {
        console.warn(`⚠️ Failed to initialize OKX: ${error.message}`);
      }

      // Initialize Delta Exchange (if credentials available)
      if (process.env.DELTA_EXCHANGE_API_KEY && process.env.DELTA_EXCHANGE_API_SECRET) {
        try {
          this.exchanges.set('delta', new ccxt.delta({
            apiKey: process.env.DELTA_EXCHANGE_API_KEY,
            secret: process.env.DELTA_EXCHANGE_API_SECRET,
            sandbox: true, // Use testnet
            enableRateLimit: true,
            timeout: 10000
          }));
          console.log(`✅ Delta Exchange initialized (testnet)`);
        } catch (error) {
          console.warn(`⚠️ Failed to initialize Delta Exchange: ${error.message}`);
        }
      }

      const initializedCount = this.exchanges.size;
      console.log(`🚀 CCXT Service initialized with ${initializedCount} exchanges`);
      
      return initializedCount > 0;
    } catch (error) {
      console.error(`❌ Failed to initialize CCXT service: ${error.message}`);
      return false;
    }
  }

  /**
   * Get current price with multi-exchange fallback
   */
  async getCurrentPrice(symbol) {
    const cacheKey = `price_${symbol}`;
    
    // Check cache first
    if (this.isCacheValid(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    // Try exchanges in priority order
    for (const exchangeId of this.exchangePriority) {
      const exchange = this.exchanges.get(exchangeId);
      if (!exchange) continue;

      const mappedSymbol = this.getMappedSymbol(symbol, exchangeId);
      if (!mappedSymbol) continue;

      try {
        console.log(`📡 Fetching ${symbol} price from ${exchangeId}...`);
        
        const ticker = await exchange.fetchTicker(mappedSymbol);
        const price = ticker.last || ticker.close;
        
        if (price && price > 0) {
          // Cache the result
          this.cache.set(cacheKey, price);
          this.lastUpdate.set(cacheKey, Date.now());
          
          console.log(`✅ Got ${symbol} price: $${price.toFixed(2)} from ${exchangeId}`);
          return price;
        }
      } catch (error) {
        console.warn(`⚠️ Failed to get price from ${exchangeId}: ${error.message}`);
        continue;
      }
    }

    console.error(`❌ Failed to get price for ${symbol} from all exchanges`);
    return null;
  }

  /**
   * Get OHLCV candles with multi-exchange fallback
   */
  async getCandles(symbol, timeframe, limit = 50) {
    const cacheKey = `candles_${symbol}_${timeframe}_${limit}`;
    
    // Check cache first
    if (this.isCacheValid(cacheKey)) {
      console.log(`📋 Using cached candles for ${symbol} ${timeframe}`);
      return this.cache.get(cacheKey);
    }

    // Try exchanges in priority order
    for (const exchangeId of this.exchangePriority) {
      const exchange = this.exchanges.get(exchangeId);
      if (!exchange) continue;

      const mappedSymbol = this.getMappedSymbol(symbol, exchangeId);
      if (!mappedSymbol) continue;

      try {
        console.log(`📡 Fetching ${symbol} ${timeframe} candles from ${exchangeId}...`);
        
        // Check if exchange supports the timeframe
        if (!exchange.timeframes || !exchange.timeframes[timeframe]) {
          console.warn(`⚠️ ${exchangeId} doesn't support ${timeframe} timeframe`);
          continue;
        }

        const ohlcv = await exchange.fetchOHLCV(mappedSymbol, timeframe, undefined, limit);
        
        if (ohlcv && ohlcv.length > 0) {
          // Convert to standard format
          const candles = ohlcv.map(([timestamp, open, high, low, close, volume]) => ({
            time: timestamp,
            open: open.toString(),
            high: high.toString(),
            low: low.toString(),
            close: close.toString(),
            volume: volume.toString()
          }));

          // Validate candle data
          const validation = DefensiveTradingUtils.validateCandleData(candles);
          if (validation.isValid) {
            // Cache the result
            this.cache.set(cacheKey, candles);
            this.lastUpdate.set(cacheKey, Date.now());
            
            console.log(`✅ Got ${candles.length} ${symbol} ${timeframe} candles from ${exchangeId}`);
            return candles;
          } else {
            console.warn(`⚠️ Invalid candle data from ${exchangeId}: ${validation.reason}`);
            continue;
          }
        }
      } catch (error) {
        console.warn(`⚠️ Failed to get candles from ${exchangeId}: ${error.message}`);
        continue;
      }
    }

    console.error(`❌ Failed to get candles for ${symbol} ${timeframe} from all exchanges`);
    return [];
  }

  /**
   * Get mapped symbol for specific exchange
   */
  getMappedSymbol(symbol, exchangeId) {
    const mapping = this.symbolMapping[symbol];
    if (!mapping) {
      console.warn(`⚠️ No symbol mapping found for ${symbol}`);
      return null;
    }

    const mappedSymbol = mapping[exchangeId];
    if (!mappedSymbol) {
      console.warn(`⚠️ No symbol mapping found for ${symbol} on ${exchangeId}`);
      return null;
    }

    return mappedSymbol;
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
   * Get exchange status
   */
  getExchangeStatus() {
    const status = {};
    
    for (const [exchangeId, exchange] of this.exchanges) {
      status[exchangeId] = {
        initialized: !!exchange,
        rateLimit: exchange.rateLimit || 'unknown',
        timeout: exchange.timeout || 'unknown',
        sandbox: exchange.sandbox || false
      };
    }

    return status;
  }

  /**
   * Clear cache
   */
  clearCache() {
    this.cache.clear();
    this.lastUpdate.clear();
    console.log(`🧹 CCXT cache cleared`);
  }

  /**
   * Get supported timeframes for an exchange
   */
  getSupportedTimeframes(exchangeId) {
    const exchange = this.exchanges.get(exchangeId);
    if (!exchange || !exchange.timeframes) {
      return [];
    }

    return Object.keys(exchange.timeframes);
  }

  /**
   * Test connection to all exchanges
   */
  async testConnections() {
    console.log(`🔍 Testing connections to all exchanges...`);
    const results = {};

    for (const [exchangeId, exchange] of this.exchanges) {
      try {
        await exchange.loadMarkets();
        results[exchangeId] = { status: 'connected', error: null };
        console.log(`✅ ${exchangeId} connection successful`);
      } catch (error) {
        results[exchangeId] = { status: 'failed', error: error.message };
        console.error(`❌ ${exchangeId} connection failed: ${error.message}`);
      }
    }

    return results;
  }

  /**
   * Get available symbols for an exchange
   */
  async getAvailableSymbols(exchangeId) {
    const exchange = this.exchanges.get(exchangeId);
    if (!exchange) {
      return [];
    }

    try {
      await exchange.loadMarkets();
      return Object.keys(exchange.markets);
    } catch (error) {
      console.error(`❌ Failed to get symbols for ${exchangeId}: ${error.message}`);
      return [];
    }
  }

  /**
   * Get candles before a specific timestamp (for time-synchronized fetching)
   */
  async getCandlesBefore(symbol, timeframe, limit, beforeTimestamp) {
    for (const exchangeId of this.exchangePriority) {
      const exchange = this.exchanges.get(exchangeId);
      if (!exchange) continue;

      const mappedSymbol = this.getMappedSymbol(symbol, exchangeId);
      if (!mappedSymbol) continue;

      try {
        console.log(`📡 Fetching ${symbol} ${timeframe} candles before ${new Date(beforeTimestamp).toISOString()} from ${exchangeId}...`);

        // Calculate since timestamp (beforeTimestamp - duration needed)
        const timeframeDuration = this.getTimeframeDuration(timeframe);
        const sinceTimestamp = beforeTimestamp - (limit * timeframeDuration);

        const ohlcv = await exchange.fetchOHLCV(mappedSymbol, timeframe, sinceTimestamp, limit);

        if (ohlcv && ohlcv.length > 0) {
          // Filter candles that are actually before the specified timestamp
          const filteredCandles = ohlcv
            .filter(candle => candle[0] < beforeTimestamp)
            .map(candle => ({
              time: candle[0],
              open: candle[1].toString(),
              high: candle[2].toString(),
              low: candle[3].toString(),
              close: candle[4].toString(),
              volume: candle[5].toString()
            }));

          console.log(`✅ Got ${filteredCandles.length} ${symbol} ${timeframe} candles before timestamp from ${exchangeId}`);
          return filteredCandles;
        }
      } catch (error) {
        console.warn(`⚠️ Failed to fetch ${symbol} ${timeframe} candles before timestamp from ${exchangeId}: ${error.message}`);
        continue;
      }
    }

    console.error(`❌ Failed to fetch ${symbol} ${timeframe} candles before timestamp from all exchanges`);
    return [];
  }

  /**
   * Get timeframe duration in milliseconds
   */
  getTimeframeDuration(timeframe) {
    const durations = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000
    };

    return durations[timeframe] || 60 * 1000; // Default to 1 minute
  }
}

module.exports = CCXTMarketDataService;

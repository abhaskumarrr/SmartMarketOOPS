/**
 * Intelligent Market Data Manager
 * 
 * Optimized data fetching strategy:
 * - Fetch historical data ONCE on startup
 * - Only update current forming candle in real-time
 * - Historical zones/levels remain static
 * - Massive reduction in API calls and processing
 */

const CCXTMarketDataService = require('./ccxt-market-data-service');
const DefensiveTradingUtils = require('./defensive-trading-utils');

class IntelligentMarketDataManager {
  constructor() {
    this.ccxtService = new CCXTMarketDataService();
    
    // Historical data storage (fetched once)
    this.historicalData = new Map(); // symbol_timeframe -> candles array
    this.lastHistoricalUpdate = new Map(); // symbol_timeframe -> timestamp
    
    // Current candle tracking
    this.currentCandles = new Map(); // symbol_timeframe -> current forming candle
    this.lastCandleUpdate = new Map(); // symbol_timeframe -> timestamp
    
    // Market zones (calculated once from historical data)
    this.marketZones = new Map(); // symbol -> { fibLevels, swingPoints, etc. }
    
    // Configuration - TIME-SYNCHRONIZED DATA FETCHING
    this.config = {
      analysisTimeframeDays: 30,     // Analyze last 30 days
      currentCandleUpdateInterval: 5000,  // Update current candle every 5 seconds
      historicalRefreshInterval: 3600000, // Refresh historical data every hour
      priceUpdateInterval: 1000,     // Update current price every 1 second

      // Time-synchronized bar counts for 30 days - PROFESSIONAL 4-TIER HIERARCHY
      timeframeBars: {
        '1d': 30,      // 30 days - Daily structure analysis
        '4h': 180,     // 30 days × 6 bars per day - TREND DIRECTION (Where is the train going?)
        '1h': 720,     // 30 days × 24 bars per day - TREND CONFIRMATION (Is the train accelerating?)
        '15m': 2880,   // 30 days × 96 bars per day - ENTRY TIMING (When to jump on the train?)
        '5m': 8640     // 30 days × 288 bars per day - PRECISION ENTRY (Exact entry point)
      },

      // Exchange API limits (max candles per request)
      exchangeLimits: {
        binance: 1000,
        bybit: 1000,
        okx: 300
      }
    };
    
    this.initialized = false;
  }

  /**
   * Initialize the intelligent data manager
   */
  async initialize() {
    try {
      console.log(`🧠 Initializing Intelligent Market Data Manager...`);
      
      // Initialize CCXT service
      await this.ccxtService.initialize();
      
      // Fetch initial historical data for all symbols and timeframes
      await this.fetchInitialHistoricalData();
      
      // Calculate market zones from historical data
      await this.calculateMarketZones();
      
      // Start real-time current candle updates
      this.startCurrentCandleUpdates();
      
      // Start current price updates
      this.startCurrentPriceUpdates();
      
      this.initialized = true;
      console.log(`✅ Intelligent Market Data Manager initialized`);
      
      return true;
    } catch (error) {
      console.error(`❌ Failed to initialize Intelligent Market Data Manager: ${error.message}`);
      return false;
    }
  }

  /**
   * Fetch TIME-SYNCHRONIZED historical data ONCE for all symbols and timeframes
   */
  async fetchInitialHistoricalData() {
    const symbols = ['BTCUSD', 'ETHUSD'];
    const timeframes = ['1d', '4h', '1h', '15m', '5m']; // Added missing 1H timeframe

    console.log(`📚 Fetching TIME-SYNCHRONIZED historical data for ${this.config.analysisTimeframeDays} days...`);
    console.log(`🎯 PROFESSIONAL 4-TIER TIMEFRAME HIERARCHY:`);
    console.log(`   4H: ${this.config.timeframeBars['4h']} bars - TREND DIRECTION (Where is the train going?)`);
    console.log(`   1H: ${this.config.timeframeBars['1h']} bars - TREND CONFIRMATION (Is the train accelerating?)`);
    console.log(`   15M: ${this.config.timeframeBars['15m']} bars - ENTRY TIMING (When to jump on the train?)`);
    console.log(`   5M: ${this.config.timeframeBars['5m']} bars - PRECISION ENTRY (Exact entry point)`);
    console.log(`   1D: ${this.config.timeframeBars['1d']} bars - Daily structure analysis`);
    
    for (const symbol of symbols) {
      for (const timeframe of timeframes) {
        try {
          const requiredBars = this.config.timeframeBars[timeframe];
          console.log(`📡 Fetching ${symbol} ${timeframe} historical data (${requiredBars} bars for ${this.config.analysisTimeframeDays} days)...`);

          // Use multiple API calls if required bars exceed exchange limits
          const candles = await this.fetchCompleteSynchronizedData(symbol, timeframe, requiredBars);
          
          if (candles && candles.length > 0) {
            const key = `${symbol}_${timeframe}`;
            this.historicalData.set(key, candles);
            this.lastHistoricalUpdate.set(key, Date.now());
            
            console.log(`✅ Stored ${candles.length} historical candles for ${symbol} ${timeframe}`);
          } else {
            console.warn(`⚠️ No historical data received for ${symbol} ${timeframe}`);
          }
          
          // Small delay to respect rate limits
          await this.sleep(100);
          
        } catch (error) {
          console.error(`❌ Failed to fetch historical data for ${symbol} ${timeframe}: ${error.message}`);
        }
      }
    }
    
    console.log(`✅ Initial historical data fetch completed`);
  }

  /**
   * Fetch complete time-synchronized data using multiple API calls if needed
   */
  async fetchCompleteSynchronizedData(symbol, timeframe, requiredBars) {
    const exchangeLimit = this.config.exchangeLimits.binance; // Use primary exchange limit

    if (requiredBars <= exchangeLimit) {
      // Single API call sufficient
      console.log(`   Single API call: ${requiredBars} bars (within ${exchangeLimit} limit)`);
      return await this.ccxtService.getCandles(symbol, timeframe, requiredBars);
    }

    // Multiple API calls needed
    const numberOfCalls = Math.ceil(requiredBars / exchangeLimit);
    console.log(`   Multiple API calls needed: ${numberOfCalls} calls to fetch ${requiredBars} bars`);

    let allCandles = [];
    let lastTimestamp = null;

    for (let i = 0; i < numberOfCalls; i++) {
      try {
        const barsToFetch = Math.min(exchangeLimit, requiredBars - allCandles.length);

        console.log(`   📡 API Call ${i + 1}/${numberOfCalls}: Fetching ${barsToFetch} bars...`);

        let candles;
        if (lastTimestamp) {
          // Fetch candles before the last timestamp to avoid gaps
          candles = await this.ccxtService.getCandlesBefore(symbol, timeframe, barsToFetch, lastTimestamp);
        } else {
          // First call - fetch most recent candles
          candles = await this.ccxtService.getCandles(symbol, timeframe, barsToFetch);
        }

        if (!candles || candles.length === 0) {
          console.warn(`   ⚠️ No more candles available for ${symbol} ${timeframe}`);
          break;
        }

        // Sort candles by timestamp (oldest first)
        candles.sort((a, b) => a.time - b.time);

        // Add to beginning of array (we're fetching backwards)
        if (lastTimestamp) {
          allCandles = [...candles, ...allCandles];
        } else {
          allCandles = candles;
        }

        // Update last timestamp for next iteration
        lastTimestamp = candles[0].time;

        console.log(`   ✅ Fetched ${candles.length} candles | Total: ${allCandles.length}/${requiredBars}`);

        // Add small delay to avoid rate limiting
        if (i < numberOfCalls - 1) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }

      } catch (error) {
        console.error(`   ❌ API Call ${i + 1} failed: ${error.message}`);
        break;
      }
    }

    // Sort final result by timestamp (oldest first)
    allCandles.sort((a, b) => a.time - b.time);

    console.log(`   🎯 Time-synchronized fetch complete: ${allCandles.length}/${requiredBars} bars (${((allCandles.length/requiredBars)*100).toFixed(1)}%)`);

    return allCandles;
  }

  /**
   * Calculate market zones from historical data (done once)
   */
  async calculateMarketZones() {
    console.log(`🎯 Calculating market zones from historical data...`);
    
    const symbols = ['BTCUSD', 'ETHUSD'];
    
    for (const symbol of symbols) {
      try {
        // Get daily data for swing analysis
        const dailyCandles = this.getHistoricalData(symbol, '1d');
        
        if (dailyCandles && dailyCandles.length >= 30) {
          // Calculate swing points
          const swingPoints = this.calculateSwingPoints(dailyCandles);
          
          // Calculate Fibonacci levels
          const fibLevels = this.calculateFibonacciLevels(swingPoints);
          
          // Store market zones
          this.marketZones.set(symbol, {
            swingPoints,
            fibLevels,
            calculatedAt: Date.now(),
            basedOnCandles: dailyCandles.length
          });
          
          console.log(`✅ Market zones calculated for ${symbol}:`);
          console.log(`   Swing Range: $${swingPoints.swingRange?.toFixed(2) || 'N/A'}`);
          console.log(`   Fib Levels: ${Object.keys(fibLevels).length} levels`);
        }
        
      } catch (error) {
        console.error(`❌ Failed to calculate market zones for ${symbol}: ${error.message}`);
      }
    }
  }

  /**
   * Start real-time current candle updates (only the forming candle)
   */
  startCurrentCandleUpdates() {
    console.log(`⚡ Starting real-time current candle updates (${this.config.currentCandleUpdateInterval/1000}s interval)...`);
    
    setInterval(async () => {
      await this.updateCurrentCandles();
    }, this.config.currentCandleUpdateInterval);
  }

  /**
   * Start current price updates (for real-time price tracking)
   */
  startCurrentPriceUpdates() {
    console.log(`💰 Starting real-time price updates (${this.config.priceUpdateInterval/1000}s interval)...`);
    
    setInterval(async () => {
      await this.updateCurrentPrices();
    }, this.config.priceUpdateInterval);
  }

  /**
   * Update only the current forming candles
   */
  async updateCurrentCandles() {
    const symbols = ['BTCUSD', 'ETHUSD'];
    const timeframes = ['4h', '15m', '5m']; // Skip daily as it changes slowly
    
    for (const symbol of symbols) {
      for (const timeframe of timeframes) {
        try {
          // Get just the latest candle (limit=1)
          const latestCandles = await this.ccxtService.getCandles(symbol, timeframe, 1);
          
          if (latestCandles && latestCandles.length > 0) {
            const currentCandle = latestCandles[0];
            const key = `${symbol}_${timeframe}`;
            
            // Update the current forming candle
            this.currentCandles.set(key, currentCandle);
            this.lastCandleUpdate.set(key, Date.now());
            
            // Update the last candle in historical data if it's the same timestamp
            const historicalCandles = this.historicalData.get(key);
            if (historicalCandles && historicalCandles.length > 0) {
              const lastHistorical = historicalCandles[historicalCandles.length - 1];
              
              if (lastHistorical.time === currentCandle.time) {
                // Update the forming candle
                historicalCandles[historicalCandles.length - 1] = currentCandle;
              } else {
                // New candle formed, add it and remove oldest
                historicalCandles.push(currentCandle);
                const maxBars = this.config.timeframeBars[timeframe] || 100;
                if (historicalCandles.length > maxBars) {
                  historicalCandles.shift(); // Remove oldest
                }
              }
            }
          }
          
        } catch (error) {
          console.warn(`⚠️ Failed to update current candle for ${symbol} ${timeframe}: ${error.message}`);
        }
      }
    }
  }

  /**
   * Update current prices for real-time tracking
   */
  async updateCurrentPrices() {
    const symbols = ['BTCUSD', 'ETHUSD'];
    
    for (const symbol of symbols) {
      try {
        const currentPrice = await this.ccxtService.getCurrentPrice(symbol);
        
        if (currentPrice && currentPrice > 0) {
          // Store current price with timestamp
          this.currentCandles.set(`${symbol}_price`, {
            price: currentPrice,
            timestamp: Date.now()
          });
        }
        
      } catch (error) {
        console.warn(`⚠️ Failed to update current price for ${symbol}: ${error.message}`);
      }
    }
  }

  /**
   * Get historical data (cached, no API call)
   */
  getHistoricalData(symbol, timeframe) {
    const key = `${symbol}_${timeframe}`;
    return this.historicalData.get(key) || [];
  }

  /**
   * Get current price (cached, updated every second)
   */
  getCurrentPrice(symbol) {
    const priceData = this.currentCandles.get(`${symbol}_price`);
    return priceData ? priceData.price : null;
  }

  /**
   * Get market zones (calculated once from historical data)
   */
  getMarketZones(symbol) {
    return this.marketZones.get(symbol) || null;
  }

  /**
   * Get combined data (historical + current forming candle)
   */
  getCombinedCandleData(symbol, timeframe) {
    const historical = this.getHistoricalData(symbol, timeframe);
    const currentKey = `${symbol}_${timeframe}`;
    const current = this.currentCandles.get(currentKey);
    
    if (!historical || historical.length === 0) {
      return [];
    }
    
    // Return historical data (current candle is already updated in historical array)
    return historical;
  }

  /**
   * Calculate swing points from historical data
   */
  calculateSwingPoints(candles) {
    // Simplified swing point calculation
    if (!candles || candles.length < 10) {
      return { swingRange: 0, lastHigh: null, lastLow: null };
    }

    let highestPrice = 0;
    let lowestPrice = Infinity;
    let lastHigh = null;
    let lastLow = null;

    for (const candle of candles) {
      const high = parseFloat(candle.high);
      const low = parseFloat(candle.low);

      if (high > highestPrice) {
        highestPrice = high;
        lastHigh = { price: high, time: candle.time };
      }

      if (low < lowestPrice) {
        lowestPrice = low;
        lastLow = { price: low, time: candle.time };
      }
    }

    return {
      swingRange: highestPrice - lowestPrice,
      lastHigh,
      lastLow
    };
  }

  /**
   * Calculate Fibonacci levels from swing points
   */
  calculateFibonacciLevels(swingPoints) {
    const { lastHigh, lastLow } = swingPoints;
    
    if (!lastHigh || !lastLow) {
      return {};
    }

    const range = lastHigh.price - lastLow.price;
    const fibRatios = {
      0: 1.0,      // 100%
      236: 0.764,  // 76.4%
      382: 0.618,  // 61.8%
      500: 0.5,    // 50%
      618: 0.382,  // 38.2%
      786: 0.214,  // 21.4%
      1000: 0.0    // 0%
    };

    const fibLevels = {};
    
    // Calculate retracement levels
    Object.entries(fibRatios).forEach(([key, ratio]) => {
      if (lastHigh.time > lastLow.time) {
        // Downtrend: calculate from high
        fibLevels[key] = lastHigh.price - (range * (1 - ratio));
      } else {
        // Uptrend: calculate from low
        fibLevels[key] = lastLow.price + (range * ratio);
      }
    });

    return fibLevels;
  }

  /**
   * Get system statistics
   */
  getSystemStats() {
    return {
      historicalDataSets: this.historicalData.size,
      currentCandles: this.currentCandles.size,
      marketZones: this.marketZones.size,
      lastUpdate: Math.max(...Array.from(this.lastCandleUpdate.values())),
      initialized: this.initialized
    };
  }

  /**
   * Sleep utility
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = IntelligentMarketDataManager;

#!/usr/bin/env node
/**
 * Intelligent Trading System Backtest
 * Tests the enhanced trading system with improved short selling detection
 * Uses the same logic as the live trading system but with historical data
 */

const { DeltaExchangeUnified } = require('../dist/services/DeltaExchangeUnified');
const { logger } = require('../dist/utils/logger');

class IntelligentTradingBacktest {
  constructor() {
    // Load environment variables
    require('dotenv').config();
    
    // Initialize Delta Exchange service for data fetching
    this.deltaService = new DeltaExchangeUnified({
      apiKey: process.env.DELTA_EXCHANGE_API_KEY,
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET,
      testnet: true
    });

    // OPTIMIZED backtest configuration based on research
    this.config = {
      symbols: ['BTCUSD', 'ETHUSD'],
      initialCapital: 1000, // $1000 starting capital
      maxConcurrentPositions: 2,
      riskPerTrade: 1, // 1% risk per trade (OPTIMIZED)
      stopLossPercentage: 1, // 1% stop loss (OPTIMIZED)
      takeProfitRatio: 4, // 4:1 risk/reward ratio (OPTIMIZED)
      minConfidenceThreshold: 0.65, // 65% confidence required (OPTIMIZED)
      maxLeverage: { 'BTCUSD': 100, 'ETHUSD': 100 },
      minOrderSize: 1,
      backtestPeriod: 30, // 30 days
      timeframe: '1h' // 1 hour candles
    };

    // Portfolio tracking
    this.portfolio = {
      initialBalance: this.config.initialCapital,
      currentBalance: this.config.initialCapital,
      availableBalance: this.config.initialCapital,
      activePositions: new Map(),
      totalPnL: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      longTrades: 0,
      shortTrades: 0,
      longWins: 0,
      shortWins: 0,
      maxDrawdown: 0,
      peakBalance: this.config.initialCapital
    };

    // Market analysis data
    this.marketData = new Map();
    this.priceHistory = new Map();
    this.trades = [];
  }

  /**
   * Sleep utility function
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Fetch historical data for backtesting
   */
  async fetchHistoricalData(symbol, days = 30) {
    try {
      logger.info(`📊 Fetching ${days} days of REAL historical data for ${symbol}...`);

      // Try CoinGecko API for real market data first
      try {
        const realData = await this.fetchRealMarketData(symbol, days);
        if (realData && realData.length > 0) {
          logger.info(`✅ Fetched ${realData.length} REAL market candles for ${symbol}`);
          return realData;
        }
      } catch (realDataError) {
        logger.warn(`⚠️ CoinGecko API failed for ${symbol}:`, realDataError.message);
      }

      // Try Delta Exchange as backup
      try {
        const candlesNeeded = days * 24;
        const data = await this.deltaService.getCandleData(symbol, '1h', candlesNeeded);

        if (data && data.length > 0) {
          logger.info(`✅ Fetched ${data.length} real candles from Delta Exchange for ${symbol}`);
          return data;
        }
      } catch (deltaError) {
        logger.warn(`⚠️ Delta Exchange data not available for ${symbol}`);
      }

      // Generate realistic simulated data as last resort
      logger.warn(`⚠️ No real data available, generating realistic simulation for ${symbol}...`);
      const simulatedData = this.generateRealisticMarketData(symbol, days);
      logger.info(`✅ Generated ${simulatedData.length} simulated candles for ${symbol}`);
      return simulatedData;

    } catch (error) {
      logger.error(`❌ Failed to fetch historical data for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Fetch real market data from CoinGecko API
   */
  async fetchRealMarketData(symbol, days) {
    const axios = require('axios');

    // Map symbols to CoinGecko IDs
    const coinGeckoIds = {
      'BTCUSD': 'bitcoin',
      'ETHUSD': 'ethereum',
      'SOLUSD': 'solana'
    };

    const coinId = coinGeckoIds[symbol];
    if (!coinId) {
      throw new Error(`CoinGecko ID not found for symbol: ${symbol}`);
    }

    try {
      // CoinGecko API endpoint for OHLC data
      const url = `https://api.coingecko.com/api/v3/coins/${coinId}/ohlc`;
      const params = {
        vs_currency: 'usd',
        days: days
      };

      logger.info(`🌐 Fetching real data from CoinGecko for ${symbol} (${coinId})...`);

      const response = await axios.get(url, {
        params,
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'SmartMarketOOPS-Backtest/1.0'
        }
      });

      if (!response.data || !Array.isArray(response.data)) {
        throw new Error('Invalid response format from CoinGecko');
      }

      // Convert CoinGecko OHLC format to our format
      const candles = response.data.map(ohlc => ({
        timestamp: ohlc[0], // Unix timestamp in milliseconds
        open: ohlc[1],
        high: ohlc[2],
        low: ohlc[3],
        close: ohlc[4],
        volume: Math.random() * 1000000 // CoinGecko OHLC doesn't include volume, so we estimate
      }));

      // Sort by timestamp to ensure chronological order
      candles.sort((a, b) => a.timestamp - b.timestamp);

      logger.info(`✅ Successfully fetched ${candles.length} real OHLC candles from CoinGecko`);
      logger.info(`📅 Data range: ${new Date(candles[0].timestamp).toISOString()} to ${new Date(candles[candles.length - 1].timestamp).toISOString()}`);
      logger.info(`💰 Price range: $${Math.min(...candles.map(c => c.low)).toFixed(2)} - $${Math.max(...candles.map(c => c.high)).toFixed(2)}`);

      return candles;

    } catch (error) {
      if (error.response) {
        logger.error(`CoinGecko API error: ${error.response.status} - ${error.response.statusText}`);
        if (error.response.status === 429) {
          logger.error('Rate limit exceeded. Consider adding delays between requests.');
        }
      }
      throw error;
    }
  }

  /**
   * Generate realistic market data for backtesting
   */
  generateRealisticMarketData(symbol, days) {
    const candlesNeeded = days * 24; // 1 hour candles
    const data = [];

    // Starting prices based on current market levels
    let currentPrice = symbol === 'BTCUSD' ? 105000 : 2600; // Realistic starting prices

    // Market parameters for realistic simulation
    const volatility = 0.02; // 2% hourly volatility
    const trendStrength = 0.001; // Slight trend component
    const meanReversion = 0.95; // Mean reversion factor

    // Generate realistic price movements with trends and reversals
    for (let i = 0; i < candlesNeeded; i++) {
      const timestamp = Date.now() - ((candlesNeeded - i) * 60 * 60 * 1000); // 1 hour intervals

      // Add trend component (creates longer-term directional moves)
      const trendComponent = Math.sin(i / 100) * trendStrength;

      // Add random volatility
      const randomComponent = (Math.random() - 0.5) * volatility;

      // Add mean reversion (prevents prices from going too extreme)
      const meanReversionComponent = (1 - meanReversion) * (Math.random() - 0.5) * 0.01;

      // Calculate price change
      const priceChange = (trendComponent + randomComponent + meanReversionComponent) * currentPrice;

      // Create OHLC data with realistic intrabar movement
      const open = currentPrice;
      const close = currentPrice + priceChange;

      // Generate realistic high/low based on volatility
      const intraDayVolatility = Math.abs(priceChange) * (1 + Math.random());
      const high = Math.max(open, close) + intraDayVolatility;
      const low = Math.min(open, close) - intraDayVolatility;

      // Generate realistic volume
      const baseVolume = 1000000;
      const volumeVariation = Math.random() * 0.5 + 0.75; // 75% to 125% of base
      const volume = baseVolume * volumeVariation;

      data.push({
        timestamp: timestamp,
        open: parseFloat(open.toFixed(2)),
        high: parseFloat(high.toFixed(2)),
        low: parseFloat(low.toFixed(2)),
        close: parseFloat(close.toFixed(2)),
        volume: parseFloat(volume.toFixed(0))
      });

      currentPrice = close;
    }

    // Add some realistic market events (sudden moves)
    this.addMarketEvents(data);

    return data;
  }

  /**
   * Add realistic market events to simulated data
   */
  addMarketEvents(data) {
    const numEvents = Math.floor(data.length / 100); // About 1 event per 100 candles

    for (let i = 0; i < numEvents; i++) {
      const eventIndex = Math.floor(Math.random() * data.length);
      const candle = data[eventIndex];

      // Create a significant price movement (news event simulation)
      const eventMagnitude = (Math.random() - 0.5) * 0.1; // Up to 10% move
      const priceChange = candle.close * eventMagnitude;

      // Apply the event
      candle.close += priceChange;
      candle.high = Math.max(candle.high, candle.close);
      candle.low = Math.min(candle.low, candle.close);

      // Propagate some of the effect to subsequent candles
      for (let j = eventIndex + 1; j < Math.min(eventIndex + 5, data.length); j++) {
        const decayFactor = 1 - ((j - eventIndex) / 5);
        const residualChange = priceChange * decayFactor * 0.3;

        data[j].open += residualChange;
        data[j].close += residualChange;
        data[j].high += residualChange;
        data[j].low += residualChange;
      }
    }
  }

  /**
   * Perform technical analysis on price data (same as live system)
   */
  performTechnicalAnalysis(symbol, history, currentPrice) {
    if (history.length < 10) {
      return {
        trend: 'neutral',
        confidence: 0.5,
        signal: 'wait',
        reason: 'Insufficient data for analysis',
        rsi: 50,
        volatility: 0,
        momentum: 0,
        sma10: currentPrice,
        sma20: currentPrice,
        currentPrice: currentPrice
      };
    }

    // Calculate moving averages
    const prices = history.map(h => h.close);
    const sma10 = this.calculateSMA(prices, 10);
    const sma20 = this.calculateSMA(prices, Math.min(20, prices.length));
    
    // Calculate RSI
    const rsi = this.calculateRSI(prices, 14);
    
    // Calculate volatility
    const volatility = this.calculateVolatility(prices);
    
    // Determine trend
    let trend = 'neutral';
    let confidence = 0.5;
    let signal = 'wait';
    let reason = 'Market analysis in progress';
    
    // Enhanced trend analysis - more sensitive to direction changes
    const smaSpread = ((sma10 - sma20) / sma20) * 100;
    const priceVsSma10 = ((currentPrice - sma10) / sma10) * 100;
    const priceVsSma20 = ((currentPrice - sma20) / sma20) * 100;
    
    // Bullish conditions
    if (sma10 > sma20 || currentPrice > sma10) {
      if (sma10 > sma20 && currentPrice > sma10) {
        trend = 'strong_bullish';
        confidence += 0.25;
      } else {
        trend = 'bullish';
        confidence += 0.15;
      }
    }
    // Bearish conditions - ENHANCED for better short detection
    else if (sma10 < sma20 || currentPrice < sma10) {
      if (sma10 < sma20 && currentPrice < sma10) {
        trend = 'strong_bearish';
        confidence += 0.25;
      } else {
        trend = 'bearish';
        confidence += 0.15;
      }
    }
    
    // Price momentum analysis
    const recentPrices = prices.slice(-5);
    const momentum = (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0];
    const momentumPercent = momentum * 100;
    
    // ENHANCED RSI + Momentum analysis for better signal generation
    
    // LONG (BUY) signals
    if (rsi < 35 && (trend.includes('bullish') || momentum > 0.01)) {
      signal = 'buy';
      confidence += 0.2;
      reason = `Oversold RSI (${rsi.toFixed(1)}) with ${trend} trend`;
    }
    // Additional bullish momentum signal
    else if (momentum > 0.03 && rsi < 60 && currentPrice > sma20) {
      signal = 'buy';
      confidence += 0.15;
      reason = `Strong bullish momentum (+${momentumPercent.toFixed(2)}%) with price above SMA20`;
    }
    
    // SHORT (SELL) signals - ENHANCED for downtrend detection
    else if (rsi > 65 && (trend.includes('bearish') || momentum < -0.01)) {
      signal = 'sell';
      confidence += 0.2;
      reason = `Overbought RSI (${rsi.toFixed(1)}) with ${trend} trend - SHORT OPPORTUNITY`;
    }
    // Additional bearish momentum signal - KEY ENHANCEMENT
    else if (momentum < -0.02 && rsi > 40 && currentPrice < sma20) {
      signal = 'sell';
      confidence += 0.18;
      reason = `Strong bearish momentum (${momentumPercent.toFixed(2)}%) with price below SMA20 - SHORT SIGNAL`;
    }
    // Bearish breakdown signal
    else if (currentPrice < sma10 && sma10 < sma20 && rsi < 50) {
      signal = 'sell';
      confidence += 0.15;
      reason = `Bearish breakdown: Price below both SMAs with declining RSI - SHORT ENTRY`;
    }
    // Resistance rejection signal
    else if (rsi > 55 && momentum < -0.015 && currentPrice < sma10) {
      signal = 'sell';
      confidence += 0.12;
      reason = `Resistance rejection with negative momentum - SHORT OPPORTUNITY`;
    }
    
    // Momentum confirmation
    if (Math.abs(momentum) > 0.02) {
      if (momentum > 0 && signal === 'buy') {
        confidence += 0.1;
        reason += ` (Strong upward momentum: +${momentumPercent.toFixed(2)}%)`;
      } else if (momentum < 0 && signal === 'sell') {
        confidence += 0.1;
        reason += ` (Strong downward momentum: ${momentumPercent.toFixed(2)}%)`;
      }
    }
    
    // Volatility adjustment
    if (volatility > 0.05) {
      confidence -= 0.08;
      reason += ` (High volatility: ${(volatility * 100).toFixed(2)}%)`;
    } else if (volatility < 0.02) {
      confidence += 0.05;
    }
    
    // Ensure confidence is within bounds
    confidence = Math.max(0, Math.min(1, confidence));
    
    return {
      trend,
      confidence,
      signal,
      reason,
      rsi,
      volatility,
      momentum,
      sma10,
      sma20,
      currentPrice
    };
  }

  /**
   * Calculate Simple Moving Average
   */
  calculateSMA(prices, period) {
    if (prices.length < period) return prices[prices.length - 1];
    
    const slice = prices.slice(-period);
    return slice.reduce((sum, price) => sum + price, 0) / slice.length;
  }

  /**
   * Calculate RSI (Relative Strength Index)
   */
  calculateRSI(prices, period = 14) {
    if (prices.length < period + 1) return 50;
    
    const gains = [];
    const losses = [];
    
    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    const avgGain = gains.slice(-period).reduce((sum, gain) => sum + gain, 0) / period;
    const avgLoss = losses.slice(-period).reduce((sum, loss) => sum + loss, 0) / period;
    
    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  /**
   * Calculate price volatility
   */
  calculateVolatility(prices) {
    if (prices.length < 2) return 0;
    
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Calculate optimal position size based on risk management
   */
  calculatePositionSize(symbol, currentPrice, side) {
    const riskAmount = this.portfolio.availableBalance * (this.config.riskPerTrade / 100);
    const leverage = this.config.maxLeverage[symbol] || 50;

    // Calculate stop loss distance
    const stopLossDistance = currentPrice * (this.config.stopLossPercentage / 100);

    // Calculate position size based on risk
    let positionSize = riskAmount / stopLossDistance;

    // Apply leverage
    positionSize = Math.floor(positionSize * leverage);

    // Ensure minimum order size
    positionSize = Math.max(positionSize, this.config.minOrderSize);

    // Ensure we don't exceed available balance
    const maxPositionValue = this.portfolio.availableBalance * 0.8;
    const positionValue = (positionSize * currentPrice) / leverage;

    if (positionValue > maxPositionValue) {
      positionSize = Math.floor((maxPositionValue * leverage) / currentPrice);
    }

    return positionSize;
  }

  /**
   * Execute a trade in the backtest
   */
  executeBacktestTrade(symbol, analysis, timestamp) {
    try {
      // Check if we already have a position in this symbol
      if (this.portfolio.activePositions.has(symbol)) {
        return false;
      }

      // Check if we've reached max concurrent positions
      if (this.portfolio.activePositions.size >= this.config.maxConcurrentPositions) {
        return false;
      }

      // Check confidence threshold
      if (analysis.confidence < this.config.minConfidenceThreshold) {
        return false;
      }

      // Check if signal is actionable
      if (analysis.signal === 'wait') {
        return false;
      }

      const currentPrice = analysis.currentPrice;
      const side = analysis.signal; // 'buy' or 'sell'

      // Calculate position size
      const positionSize = this.calculatePositionSize(symbol, currentPrice, side);

      if (positionSize < this.config.minOrderSize) {
        return false;
      }

      // Check available balance
      const requiredMargin = (positionSize * currentPrice) / this.config.maxLeverage[symbol];
      if (requiredMargin > this.portfolio.availableBalance * 0.8) {
        return false;
      }

      // Calculate stop loss and take profit levels
      const stopLossPrice = side === 'buy'
        ? currentPrice * (1 - this.config.stopLossPercentage / 100)
        : currentPrice * (1 + this.config.stopLossPercentage / 100);

      const takeProfitPrice = side === 'buy'
        ? currentPrice * (1 + (this.config.stopLossPercentage * this.config.takeProfitRatio) / 100)
        : currentPrice * (1 - (this.config.stopLossPercentage * this.config.takeProfitRatio) / 100);

      // Create position record
      const position = {
        symbol: symbol,
        side: side,
        size: positionSize,
        entryPrice: currentPrice,
        currentPrice: currentPrice,
        stopLossPrice: stopLossPrice,
        takeProfitPrice: takeProfitPrice,
        unrealizedPnL: 0,
        entryTime: timestamp,
        lastUpdate: timestamp,
        analysis: analysis,
        status: 'active'
      };

      this.portfolio.activePositions.set(symbol, position);

      // Update available balance
      this.portfolio.availableBalance -= requiredMargin;

      // Track trade statistics
      this.portfolio.totalTrades++;
      if (side === 'buy') {
        this.portfolio.longTrades++;
      } else {
        this.portfolio.shortTrades++;
      }

      logger.info(`✅ BACKTEST TRADE: ${side.toUpperCase()} ${positionSize} ${symbol} @ $${currentPrice.toFixed(2)}`);
      logger.info(`   Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
      logger.info(`   Reason: ${analysis.reason}`);
      logger.info(`   Stop Loss: $${stopLossPrice.toFixed(2)}`);
      logger.info(`   Take Profit: $${takeProfitPrice.toFixed(2)}`);

      return true;

    } catch (error) {
      logger.error(`❌ Failed to execute backtest trade for ${symbol}:`, error);
      return false;
    }
  }

  /**
   * Check and close positions based on stop loss/take profit
   */
  checkPositionExits(currentData, timestamp) {
    const positionsToClose = [];

    for (const [symbol, position] of this.portfolio.activePositions) {
      const currentPrice = currentData[symbol];
      if (!currentPrice) continue;

      // Update position
      position.currentPrice = currentPrice;
      position.lastUpdate = timestamp;

      // Calculate unrealized PnL
      const priceDiff = position.side === 'buy'
        ? currentPrice - position.entryPrice
        : position.entryPrice - currentPrice;

      position.unrealizedPnL = (priceDiff / position.entryPrice) * 100;

      // Check stop loss
      const shouldStopLoss = position.side === 'buy'
        ? currentPrice <= position.stopLossPrice
        : currentPrice >= position.stopLossPrice;

      if (shouldStopLoss) {
        positionsToClose.push({ symbol, reason: 'stop_loss', price: currentPrice });
        continue;
      }

      // Check take profit
      const shouldTakeProfit = position.side === 'buy'
        ? currentPrice >= position.takeProfitPrice
        : currentPrice <= position.takeProfitPrice;

      if (shouldTakeProfit) {
        positionsToClose.push({ symbol, reason: 'take_profit', price: currentPrice });
        continue;
      }
    }

    // Close positions
    for (const { symbol, reason, price } of positionsToClose) {
      this.closeBacktestPosition(symbol, reason, price, timestamp);
    }
  }

  /**
   * Close a position in the backtest
   */
  closeBacktestPosition(symbol, reason, exitPrice, timestamp) {
    try {
      const position = this.portfolio.activePositions.get(symbol);
      if (!position) return false;

      // Calculate final PnL
      const priceDiff = position.side === 'buy'
        ? exitPrice - position.entryPrice
        : position.entryPrice - exitPrice;

      const pnlPercent = (priceDiff / position.entryPrice) * 100;
      const pnlAmount = (pnlPercent / 100) * (position.size * position.entryPrice / this.config.maxLeverage[symbol]);

      // Update portfolio
      this.portfolio.currentBalance += pnlAmount;
      this.portfolio.totalPnL += pnlPercent;

      // Return margin to available balance
      const margin = (position.size * position.entryPrice) / this.config.maxLeverage[symbol];
      this.portfolio.availableBalance += margin;

      // Track win/loss statistics
      if (pnlAmount > 0) {
        this.portfolio.winningTrades++;
        if (position.side === 'buy') {
          this.portfolio.longWins++;
        } else {
          this.portfolio.shortWins++;
        }
      } else {
        this.portfolio.losingTrades++;
      }

      // Update max drawdown
      if (this.portfolio.currentBalance > this.portfolio.peakBalance) {
        this.portfolio.peakBalance = this.portfolio.currentBalance;
      }

      const drawdown = ((this.portfolio.peakBalance - this.portfolio.currentBalance) / this.portfolio.peakBalance) * 100;
      this.portfolio.maxDrawdown = Math.max(this.portfolio.maxDrawdown, drawdown);

      // Record trade
      const trade = {
        symbol: symbol,
        side: position.side,
        entryPrice: position.entryPrice,
        exitPrice: exitPrice,
        size: position.size,
        entryTime: position.entryTime,
        exitTime: timestamp,
        pnlPercent: pnlPercent,
        pnlAmount: pnlAmount,
        reason: reason,
        confidence: position.analysis.confidence
      };

      this.trades.push(trade);

      // Remove from active positions
      this.portfolio.activePositions.delete(symbol);

      logger.info(`🔄 POSITION CLOSED: ${position.side} ${symbol} (${reason})`);
      logger.info(`   Entry: $${position.entryPrice.toFixed(2)} → Exit: $${exitPrice.toFixed(2)}`);
      logger.info(`   P&L: ${pnlPercent.toFixed(2)}% ($${pnlAmount.toFixed(2)})`);
      logger.info(`   Balance: $${this.portfolio.currentBalance.toFixed(2)}`);

      return true;

    } catch (error) {
      logger.error(`❌ Failed to close backtest position for ${symbol}:`, error);
      return false;
    }
  }

  /**
   * Run the complete backtest
   */
  async runBacktest() {
    try {
      logger.info('🧠 Starting Intelligent Trading System Backtest...');
      logger.info('═'.repeat(60));
      logger.info(`💰 Initial Capital: $${this.config.initialCapital}`);
      logger.info(`📊 Symbols: ${this.config.symbols.join(', ')}`);
      logger.info(`📅 Period: ${this.config.backtestPeriod} days`);
      logger.info(`🎯 Min Confidence: ${(this.config.minConfidenceThreshold * 100).toFixed(0)}%`);
      logger.info(`⚡ Risk Per Trade: ${this.config.riskPerTrade}%`);
      logger.info(`🔴 Enhanced Short Selling: ENABLED`);
      logger.info('═'.repeat(60));

      // Wait for Delta service to be ready
      let retries = 0;
      while (!this.deltaService.isReady() && retries < 10) {
        logger.info(`⏳ Waiting for Delta Exchange service... (${retries + 1}/10)`);
        await this.sleep(2000);
        retries++;
      }

      if (!this.deltaService.isReady()) {
        throw new Error('Delta Exchange service failed to initialize');
      }

      // Fetch historical data for all symbols
      const historicalData = {};
      for (const symbol of this.config.symbols) {
        const data = await this.fetchHistoricalData(symbol, this.config.backtestPeriod);
        if (!data) {
          throw new Error(`Failed to fetch data for ${symbol}`);
        }
        historicalData[symbol] = data;

        // Add delay between API calls to respect rate limits
        if (this.config.symbols.indexOf(symbol) < this.config.symbols.length - 1) {
          logger.info('⏳ Waiting 2 seconds before next API call...');
          await this.sleep(2000);
        }
      }

      // Get the minimum data length to ensure all symbols have data for the same period
      const minLength = Math.min(...Object.values(historicalData).map(data => data.length));
      logger.info(`📊 Processing ${minLength} candles for backtest...`);

      // Run backtest simulation
      for (let i = 20; i < minLength; i++) { // Start at 20 to have enough history for analysis
        const timestamp = new Date(historicalData[this.config.symbols[0]][i].timestamp);

        // Build price history for each symbol
        for (const symbol of this.config.symbols) {
          if (!this.priceHistory.has(symbol)) {
            this.priceHistory.set(symbol, []);
          }

          const history = this.priceHistory.get(symbol);
          const candle = historicalData[symbol][i];

          history.push({
            price: candle.close,
            close: candle.close,
            timestamp: candle.timestamp,
            volume: candle.volume || 0
          });

          // Keep only last 100 data points
          if (history.length > 100) {
            history.splice(0, history.length - 100);
          }
        }

        // Check for position exits first
        const currentPrices = {};
        for (const symbol of this.config.symbols) {
          currentPrices[symbol] = historicalData[symbol][i].close;
        }
        this.checkPositionExits(currentPrices, timestamp);

        // Analyze each symbol for new trading opportunities
        for (const symbol of this.config.symbols) {
          // Skip if we already have a position in this symbol
          if (this.portfolio.activePositions.has(symbol)) {
            continue;
          }

          // Skip if we've reached max concurrent positions
          if (this.portfolio.activePositions.size >= this.config.maxConcurrentPositions) {
            continue;
          }

          const history = this.priceHistory.get(symbol);
          const currentPrice = historicalData[symbol][i].close;

          // Perform technical analysis
          const analysis = this.performTechnicalAnalysis(symbol, history, currentPrice);

          // Execute trade if conditions are met
          if (analysis.signal !== 'wait' && analysis.confidence >= this.config.minConfidenceThreshold) {
            this.executeBacktestTrade(symbol, analysis, timestamp);
          }
        }

        // Log progress every 100 candles
        if (i % 100 === 0) {
          const progress = ((i / minLength) * 100).toFixed(1);
          logger.info(`📈 Progress: ${progress}% | Balance: $${this.portfolio.currentBalance.toFixed(2)} | Active Positions: ${this.portfolio.activePositions.size}`);
        }
      }

      // Close any remaining positions at the end
      const finalPrices = {};
      for (const symbol of this.config.symbols) {
        finalPrices[symbol] = historicalData[symbol][minLength - 1].close;
      }

      for (const symbol of this.portfolio.activePositions.keys()) {
        this.closeBacktestPosition(symbol, 'backtest_end', finalPrices[symbol], new Date());
      }

      // Generate and display results
      this.displayBacktestResults();

    } catch (error) {
      logger.error('❌ Backtest failed:', error);
      throw error;
    }
  }

  /**
   * Display comprehensive backtest results
   */
  displayBacktestResults() {
    const totalReturn = ((this.portfolio.currentBalance - this.portfolio.initialBalance) / this.portfolio.initialBalance) * 100;
    const winRate = this.portfolio.totalTrades > 0 ? (this.portfolio.winningTrades / this.portfolio.totalTrades) * 100 : 0;
    const longWinRate = this.portfolio.longTrades > 0 ? (this.portfolio.longWins / this.portfolio.longTrades) * 100 : 0;
    const shortWinRate = this.portfolio.shortTrades > 0 ? (this.portfolio.shortWins / this.portfolio.shortTrades) * 100 : 0;
    const avgTradeReturn = this.portfolio.totalTrades > 0 ? this.portfolio.totalPnL / this.portfolio.totalTrades : 0;

    logger.info('\n🎉 INTELLIGENT TRADING BACKTEST RESULTS');
    logger.info('═'.repeat(80));
    logger.info(`💰 PERFORMANCE SUMMARY:`);
    logger.info(`   Initial Capital: $${this.portfolio.initialBalance.toFixed(2)}`);
    logger.info(`   Final Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
    logger.info(`   Total Return: ${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`);
    logger.info(`   Max Drawdown: ${this.portfolio.maxDrawdown.toFixed(2)}%`);
    logger.info('');
    logger.info(`📊 TRADING STATISTICS:`);
    logger.info(`   Total Trades: ${this.portfolio.totalTrades}`);
    logger.info(`   Winning Trades: ${this.portfolio.winningTrades}`);
    logger.info(`   Losing Trades: ${this.portfolio.losingTrades}`);
    logger.info(`   Overall Win Rate: ${winRate.toFixed(1)}%`);
    logger.info(`   Average Trade Return: ${avgTradeReturn.toFixed(2)}%`);
    logger.info('');
    logger.info(`🔴 SHORT SELLING PERFORMANCE:`);
    logger.info(`   Long Trades: ${this.portfolio.longTrades} (Win Rate: ${longWinRate.toFixed(1)}%)`);
    logger.info(`   Short Trades: ${this.portfolio.shortTrades} (Win Rate: ${shortWinRate.toFixed(1)}%)`);
    logger.info(`   Short Trade Ratio: ${this.portfolio.totalTrades > 0 ? ((this.portfolio.shortTrades / this.portfolio.totalTrades) * 100).toFixed(1) : 0}%`);
    logger.info('');

    // Display top trades
    if (this.trades.length > 0) {
      const sortedTrades = this.trades.sort((a, b) => b.pnlPercent - a.pnlPercent);
      const topWinners = sortedTrades.slice(0, 3);
      const topLosers = sortedTrades.slice(-3).reverse();

      logger.info(`🏆 TOP WINNING TRADES:`);
      topWinners.forEach((trade, i) => {
        logger.info(`   ${i + 1}. ${trade.side.toUpperCase()} ${trade.symbol}: +${trade.pnlPercent.toFixed(2)}% (${trade.reason})`);
      });

      logger.info(`📉 TOP LOSING TRADES:`);
      topLosers.forEach((trade, i) => {
        logger.info(`   ${i + 1}. ${trade.side.toUpperCase()} ${trade.symbol}: ${trade.pnlPercent.toFixed(2)}% (${trade.reason})`);
      });
    }

    logger.info('═'.repeat(80));

    // Performance assessment
    if (totalReturn > 20) {
      logger.info('🎉 EXCELLENT PERFORMANCE! The enhanced system shows strong profitability.');
    } else if (totalReturn > 10) {
      logger.info('✅ GOOD PERFORMANCE! The system demonstrates solid returns.');
    } else if (totalReturn > 0) {
      logger.info('📈 POSITIVE PERFORMANCE! The system is profitable but could be optimized.');
    } else {
      logger.info('⚠️ NEEDS IMPROVEMENT! Consider adjusting parameters or strategy.');
    }

    if (this.portfolio.shortTrades > 0) {
      logger.info(`🔴 SHORT SELLING ANALYSIS: ${this.portfolio.shortTrades} short trades executed with ${shortWinRate.toFixed(1)}% win rate.`);
      if (shortWinRate > longWinRate) {
        logger.info('🎯 Short selling outperformed long trades - excellent downtrend detection!');
      }
    } else {
      logger.info('⚠️ NO SHORT TRADES: Consider lowering confidence threshold or improving bearish signal detection.');
    }

    logger.info('═'.repeat(80));
  }
}

// Main execution
async function main() {
  const backtest = new IntelligentTradingBacktest();

  try {
    await backtest.runBacktest();

  } catch (error) {
    logger.error('❌ Failed to run intelligent trading backtest:', error);
    process.exit(1);
  }
}

// Run the backtest
if (require.main === module) {
  main().catch(error => {
    logger.error('❌ Unhandled error:', error);
    process.exit(1);
  });
}

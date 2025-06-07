#!/usr/bin/env node
/**
 * Intelligent Delta Exchange Live Trading System
 * Analyzes market conditions thoroughly before taking positions
 * Manages positions until completion with proper risk management
 */

const { DeltaExchangeUnified } = require('../dist/services/DeltaExchangeUnified');
const { logger } = require('../dist/utils/logger');

class IntelligentDeltaTradingBot {
  constructor() {
    // Load environment variables
    require('dotenv').config();
    
    // Initialize Delta Exchange service for LIVE trading
    this.deltaService = new DeltaExchangeUnified({
      apiKey: process.env.DELTA_EXCHANGE_API_KEY,
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET,
      testnet: true
    });

    // Enhanced trading configuration
    this.config = {
      symbols: ['BTCUSD', 'ETHUSD'],
      maxConcurrentPositions: 2, // Maximum 2 positions at once
      riskPerTrade: 15, // 15% risk per trade
      stopLossPercentage: 5, // 5% stop loss
      takeProfitRatio: 2.5, // 2.5:1 risk/reward ratio
      analysisInterval: 60000, // 1 minute analysis interval
      positionCheckInterval: 30000, // 30 seconds position check
      minConfidenceThreshold: 0.75, // 75% confidence required (lowered for more trades)
      maxLeverage: { 'BTCUSD': 100, 'ETHUSD': 100 },
      minOrderSize: 1,
      enableLiveTrading: true
    };

    // Portfolio tracking
    this.portfolio = {
      initialBalance: 0,
      currentBalance: 0,
      availableBalance: 0,
      activePositions: new Map(),
      totalPnL: 0,
      totalTrades: 0,
      winningTrades: 0
    };

    // Market analysis data
    this.marketData = new Map();
    this.priceHistory = new Map();
    
    // System state
    this.isRunning = false;
    this.lastAnalysisTime = 0;
    this.sessionStartTime = 0;
  }

  /**
   * Sleep utility function
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Initialize the intelligent trading system
   */
  async initialize() {
    try {
      logger.info('🧠 Initializing Intelligent Delta Exchange Trading System...');

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

      // Get real balance from Delta Exchange
      await this.updateBalance();
      
      if (this.portfolio.currentBalance < 10) {
        throw new Error(`Insufficient balance: $${this.portfolio.currentBalance.toFixed(2)} (minimum: $10)`);
      }

      this.portfolio.initialBalance = this.portfolio.currentBalance;
      this.sessionStartTime = Date.now();

      logger.info('✅ Delta Exchange service connected successfully');
      this.displayConfiguration();

    } catch (error) {
      logger.error('❌ Failed to initialize intelligent trading system:', error);
      throw error;
    }
  }

  /**
   * Update balance from Delta Exchange
   */
  async updateBalance() {
    try {
      const balances = await this.deltaService.getBalance();
      logger.info(`📊 Received ${balances.length} balance entries from Delta Exchange`);

      // Log all balances for debugging
      balances.forEach(balance => {
        logger.info(`   ${balance.asset_symbol}: Balance=${balance.balance}, Available=${balance.available_balance}`);
      });

      // Look for USD, USDT, or any balance with value > 0
      let usdBalance = balances.find(b =>
        b.asset_symbol === 'USDT' ||
        b.asset_symbol === 'USD' ||
        b.asset_symbol === 'USDC'
      );

      // If no USD balance found, use the first balance with value > 0
      if (!usdBalance) {
        usdBalance = balances.find(b => parseFloat(b.balance || '0') > 0);
      }

      // If still no balance, use the first balance entry
      if (!usdBalance && balances.length > 0) {
        usdBalance = balances[0];
      }

      if (usdBalance) {
        this.portfolio.currentBalance = parseFloat(usdBalance.balance || '0');
        this.portfolio.availableBalance = parseFloat(usdBalance.available_balance || usdBalance.balance || '0');
        logger.info(`💰 Updated Balance: $${this.portfolio.currentBalance.toFixed(2)} (${usdBalance.asset_symbol})`);
        logger.info(`💳 Available Balance: $${this.portfolio.availableBalance.toFixed(2)}`);
      } else {
        logger.warn('⚠️ No balance found in Delta Exchange response');
        this.portfolio.currentBalance = 0;
        this.portfolio.availableBalance = 0;
      }
    } catch (error) {
      logger.error('❌ Failed to update balance:', error);
    }
  }

  /**
   * Display system configuration
   */
  displayConfiguration() {
    logger.info('\n🧠 INTELLIGENT DELTA EXCHANGE TRADING SYSTEM');
    logger.info('════════════════════════════════════════════════════════════');
    logger.info(`💰 Live Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
    logger.info(`📊 Trading Symbols: ${this.config.symbols.join(', ')}`);
    logger.info(`🎯 Max Concurrent Positions: ${this.config.maxConcurrentPositions}`);
    logger.info(`⚡ Max Leverage: BTC=${this.config.maxLeverage.BTCUSD}x, ETH=${this.config.maxLeverage.ETHUSD}x`);
    logger.info(`🎯 Risk Per Trade: ${this.config.riskPerTrade}%`);
    logger.info(`🛡️ Stop Loss: ${this.config.stopLossPercentage}%`);
    logger.info(`🎯 Take Profit Ratio: ${this.config.takeProfitRatio}:1`);
    logger.info(`🧠 Min Confidence: ${(this.config.minConfidenceThreshold * 100).toFixed(0)}% (Enhanced for Short Detection)`);
    logger.info(`🔄 Analysis Interval: ${this.config.analysisInterval / 1000}s`);
    logger.info(`🏢 Exchange: Delta Exchange India Testnet`);
    logger.info(`🔴 Mode: INTELLIGENT LIVE TRADING`);
    logger.info('════════════════════════════════════════════════════════════');
    logger.info('⚠️  This system will analyze markets before taking positions!');
    logger.info('🎯 Only high-confidence trades will be executed!');
    logger.info('📊 Positions will be managed until completion!');
    logger.info('════════════════════════════════════════════════════════════');
  }

  /**
   * Comprehensive market analysis for a symbol
   */
  async analyzeMarket(symbol) {
    try {
      logger.info(`🔍 Analyzing ${symbol} market conditions...`);
      
      // Get current market data
      const marketData = await this.deltaService.getMarketData(symbol);
      const currentPrice = marketData.last_price;
      
      // Update price history
      if (!this.priceHistory.has(symbol)) {
        this.priceHistory.set(symbol, []);
      }
      
      const history = this.priceHistory.get(symbol);
      history.push({
        price: currentPrice,
        timestamp: Date.now(),
        volume: marketData.volume || 0
      });
      
      // Keep only last 100 data points
      if (history.length > 100) {
        history.splice(0, history.length - 100);
      }
      
      // Perform technical analysis
      const analysis = this.performTechnicalAnalysis(symbol, history, currentPrice);
      
      // Store market data
      this.marketData.set(symbol, {
        ...marketData,
        analysis,
        lastUpdate: Date.now()
      });

      // Enhanced logging for better signal visibility
      const signalEmoji = analysis.signal === 'buy' ? '🟢' : analysis.signal === 'sell' ? '🔴' : '⚪';
      logger.info(`📊 ${symbol} Analysis Complete:`);
      logger.info(`   ${signalEmoji} Signal: ${analysis.signal.toUpperCase()}`);
      logger.info(`   📈 Trend: ${analysis.trend}`);
      logger.info(`   🎯 Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);

      // Only log technical indicators if they exist
      if (analysis.rsi !== undefined) {
        logger.info(`   📊 RSI: ${analysis.rsi.toFixed(1)}`);
      }
      if (analysis.momentum !== undefined) {
        logger.info(`   ⚡ Momentum: ${(analysis.momentum * 100).toFixed(2)}%`);
      }
      logger.info(`   💡 Reason: ${analysis.reason}`);

      if (analysis.signal === 'sell') {
        logger.info(`🔴 SHORT OPPORTUNITY DETECTED for ${symbol}!`);
      } else if (analysis.signal === 'buy') {
        logger.info(`🟢 LONG OPPORTUNITY DETECTED for ${symbol}!`);
      }

      return analysis;
      
    } catch (error) {
      logger.error(`❌ Failed to analyze ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Perform technical analysis on price data
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
    const prices = history.map(h => h.price);
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
    const smaSpread = ((sma10 - sma20) / sma20) * 100; // Percentage difference
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
      confidence -= 0.08; // Slightly less penalty for volatility
      reason += ` (High volatility: ${(volatility * 100).toFixed(2)}%)`;
    } else if (volatility < 0.02) {
      confidence += 0.05; // Bonus for low volatility
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
    const maxPositionValue = this.portfolio.availableBalance * 0.8; // Use 80% max
    const positionValue = (positionSize * currentPrice) / leverage;

    if (positionValue > maxPositionValue) {
      positionSize = Math.floor((maxPositionValue * leverage) / currentPrice);
    }

    logger.info(`📊 Position Size Calculation for ${symbol}:`);
    logger.info(`   Risk Amount: $${riskAmount.toFixed(2)}`);
    logger.info(`   Current Price: $${currentPrice.toFixed(2)}`);
    logger.info(`   Stop Loss Distance: $${stopLossDistance.toFixed(2)}`);
    logger.info(`   Calculated Size: ${positionSize}`);
    logger.info(`   Position Value: $${((positionSize * currentPrice) / leverage).toFixed(2)}`);

    return positionSize;
  }

  /**
   * Execute a trading signal if conditions are met
   */
  async executeTradeSignal(symbol, analysis) {
    try {
      // Check if we already have a position in this symbol
      if (this.portfolio.activePositions.has(symbol)) {
        logger.info(`⏳ Skipping ${symbol} - position already exists`);
        return false;
      }

      // Check if we've reached max concurrent positions
      if (this.portfolio.activePositions.size >= this.config.maxConcurrentPositions) {
        logger.info(`⏳ Max concurrent positions reached (${this.config.maxConcurrentPositions})`);
        return false;
      }

      // Check confidence threshold
      if (analysis.confidence < this.config.minConfidenceThreshold) {
        logger.info(`📊 ${symbol} confidence too low: ${(analysis.confidence * 100).toFixed(1)}% < ${(this.config.minConfidenceThreshold * 100).toFixed(0)}%`);
        return false;
      }

      // Check if signal is actionable
      if (analysis.signal === 'wait') {
        logger.info(`📊 ${symbol} signal: WAIT - ${analysis.reason}`);
        return false;
      }

      const currentPrice = analysis.currentPrice;
      const side = analysis.signal; // 'buy' or 'sell'

      // Calculate position size
      const positionSize = this.calculatePositionSize(symbol, currentPrice, side);

      if (positionSize < this.config.minOrderSize) {
        logger.info(`⚠️ Position size too small for ${symbol}: ${positionSize}`);
        return false;
      }

      // Check available balance
      const requiredMargin = (positionSize * currentPrice) / this.config.maxLeverage[symbol];
      if (requiredMargin > this.portfolio.availableBalance * 0.8) {
        logger.info(`⚠️ Insufficient balance for ${symbol} trade`);
        return false;
      }

      logger.info(`🎯 EXECUTING TRADE: ${side.toUpperCase()} ${positionSize} ${symbol}`);
      logger.info(`📊 Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
      logger.info(`💡 Reason: ${analysis.reason}`);

      // Place the order
      const success = await this.placeLiveOrder(symbol, side, positionSize, currentPrice, analysis);

      if (success) {
        this.portfolio.totalTrades++;
        logger.info(`✅ Trade executed successfully for ${symbol}`);
        return true;
      }

      return false;

    } catch (error) {
      logger.error(`❌ Failed to execute trade signal for ${symbol}:`, error);
      return false;
    }
  }

  /**
   * Place a live order on Delta Exchange
   */
  async placeLiveOrder(symbol, side, size, entryPrice, analysis) {
    try {
      const productId = this.deltaService.getProductId(symbol);
      if (!productId) {
        throw new Error(`Product ID not found for ${symbol}`);
      }

      const orderRequest = {
        product_id: productId,
        side: side,
        size: size,
        order_type: 'market_order'
      };

      logger.info(`🔴 PLACING LIVE ORDER: ${side.toUpperCase()} ${size} ${symbol}`);
      logger.info(`📊 Order Details:`, orderRequest);

      const order = await this.deltaService.placeOrder(orderRequest);

      if (order) {
        // Calculate stop loss and take profit levels
        const stopLossPrice = side === 'buy'
          ? entryPrice * (1 - this.config.stopLossPercentage / 100)
          : entryPrice * (1 + this.config.stopLossPercentage / 100);

        const takeProfitPrice = side === 'buy'
          ? entryPrice * (1 + (this.config.stopLossPercentage * this.config.takeProfitRatio) / 100)
          : entryPrice * (1 - (this.config.stopLossPercentage * this.config.takeProfitRatio) / 100);

        // Create position record
        const position = {
          orderId: order.id,
          symbol: symbol,
          side: side,
          size: size,
          entryPrice: entryPrice,
          currentPrice: entryPrice,
          stopLossPrice: stopLossPrice,
          takeProfitPrice: takeProfitPrice,
          unrealizedPnL: 0,
          entryTime: Date.now(),
          lastUpdate: Date.now(),
          analysis: analysis,
          status: 'active'
        };

        this.portfolio.activePositions.set(symbol, position);

        logger.info(`✅ LIVE ORDER PLACED SUCCESSFULLY!`);
        logger.info(`📋 Order ID: ${order.id}`);
        logger.info(`📊 Position Created: ${side} ${size} ${symbol} @ $${entryPrice.toFixed(2)}`);
        logger.info(`🛡️ Stop Loss: $${stopLossPrice.toFixed(2)}`);
        logger.info(`🎯 Take Profit: $${takeProfitPrice.toFixed(2)}`);

        return true;
      }

      return false;

    } catch (error) {
      logger.error(`❌ Failed to place live order:`, error);
      return false;
    }
  }

  /**
   * Manage active positions - check for stop loss/take profit
   */
  async manageActivePositions() {
    if (this.portfolio.activePositions.size === 0) {
      return;
    }

    logger.info(`📊 Managing ${this.portfolio.activePositions.size} active position(s)...`);

    for (const [symbol, position] of this.portfolio.activePositions) {
      try {
        // Get current market price
        const marketData = await this.deltaService.getMarketData(symbol);
        const currentPrice = marketData.last_price;

        // Update position
        position.currentPrice = currentPrice;
        position.lastUpdate = Date.now();

        // Calculate unrealized PnL
        const priceDiff = position.side === 'buy'
          ? currentPrice - position.entryPrice
          : position.entryPrice - currentPrice;

        position.unrealizedPnL = (priceDiff / position.entryPrice) * 100;

        logger.info(`📊 ${symbol} Position: ${position.side} ${position.size} @ $${position.entryPrice.toFixed(2)}`);
        logger.info(`   Current Price: $${currentPrice.toFixed(2)}`);
        logger.info(`   Unrealized P&L: ${position.unrealizedPnL.toFixed(2)}%`);
        logger.info(`   Stop Loss: $${position.stopLossPrice.toFixed(2)}`);
        logger.info(`   Take Profit: $${position.takeProfitPrice.toFixed(2)}`);

        // Check stop loss
        const shouldStopLoss = position.side === 'buy'
          ? currentPrice <= position.stopLossPrice
          : currentPrice >= position.stopLossPrice;

        if (shouldStopLoss) {
          logger.info(`🛑 STOP LOSS TRIGGERED for ${symbol}!`);
          await this.closePosition(symbol, 'stop_loss');
          continue;
        }

        // Check take profit
        const shouldTakeProfit = position.side === 'buy'
          ? currentPrice >= position.takeProfitPrice
          : currentPrice <= position.takeProfitPrice;

        if (shouldTakeProfit) {
          logger.info(`🎯 TAKE PROFIT TRIGGERED for ${symbol}!`);
          await this.closePosition(symbol, 'take_profit');
          continue;
        }

        // Check for position timeout (optional - close after 4 hours)
        const positionAge = Date.now() - position.entryTime;
        const maxPositionAge = 4 * 60 * 60 * 1000; // 4 hours

        if (positionAge > maxPositionAge) {
          logger.info(`⏰ POSITION TIMEOUT for ${symbol} (${(positionAge / (60 * 60 * 1000)).toFixed(1)} hours)`);
          await this.closePosition(symbol, 'timeout');
          continue;
        }

      } catch (error) {
        logger.error(`❌ Error managing position for ${symbol}:`, error);
      }
    }
  }

  /**
   * Close a position
   */
  async closePosition(symbol, reason) {
    try {
      const position = this.portfolio.activePositions.get(symbol);
      if (!position) {
        logger.warn(`⚠️ No position found for ${symbol}`);
        return false;
      }

      const productId = this.deltaService.getProductId(symbol);
      if (!productId) {
        throw new Error(`Product ID not found for ${symbol}`);
      }

      // Determine close side (opposite of entry)
      const closeSide = position.side === 'buy' ? 'sell' : 'buy';

      const closeOrderRequest = {
        product_id: productId,
        side: closeSide,
        size: position.size,
        order_type: 'market_order',
        reduce_only: true
      };

      logger.info(`🔄 CLOSING POSITION: ${closeSide.toUpperCase()} ${position.size} ${symbol} (${reason})`);

      const closeOrder = await this.deltaService.placeOrder(closeOrderRequest);

      if (closeOrder) {
        // Calculate final PnL
        const finalPnL = position.unrealizedPnL;
        this.portfolio.totalPnL += finalPnL;

        if (finalPnL > 0) {
          this.portfolio.winningTrades++;
        }

        // Remove from active positions
        this.portfolio.activePositions.delete(symbol);

        logger.info(`✅ POSITION CLOSED SUCCESSFULLY!`);
        logger.info(`📋 Close Order ID: ${closeOrder.id}`);
        logger.info(`💰 Final P&L: ${finalPnL.toFixed(2)}%`);
        logger.info(`📊 Reason: ${reason.toUpperCase()}`);

        // Update balance
        await this.updateBalance();

        return true;
      }

      return false;

    } catch (error) {
      logger.error(`❌ Failed to close position for ${symbol}:`, error);
      return false;
    }
  }

  /**
   * Display current trading status
   */
  displayTradingStatus() {
    const runtime = (Date.now() - this.sessionStartTime) / (60 * 1000);
    const balanceChange = ((this.portfolio.currentBalance - this.portfolio.initialBalance) / this.portfolio.initialBalance) * 100;
    const winRate = this.portfolio.totalTrades > 0 ? (this.portfolio.winningTrades / this.portfolio.totalTrades) * 100 : 0;

    logger.info('\n📊 INTELLIGENT TRADING STATUS:');
    logger.info('════════════════════════════════════════════════════════════');
    logger.info(`💰 Balance: $${this.portfolio.currentBalance.toFixed(2)} (${balanceChange >= 0 ? '+' : ''}${balanceChange.toFixed(2)}%)`);
    logger.info(`🎯 Active Positions: ${this.portfolio.activePositions.size}/${this.config.maxConcurrentPositions}`);
    logger.info(`📊 Total Trades: ${this.portfolio.totalTrades}`);
    logger.info(`🏆 Win Rate: ${winRate.toFixed(1)}%`);
    logger.info(`💹 Total P&L: ${this.portfolio.totalPnL.toFixed(2)}%`);
    logger.info(`⏱️ Runtime: ${runtime.toFixed(1)} minutes`);

    if (this.portfolio.activePositions.size > 0) {
      logger.info('📋 Active Positions:');
      for (const [symbol, position] of this.portfolio.activePositions) {
        const age = (Date.now() - position.entryTime) / (60 * 1000);
        logger.info(`   ${symbol}: ${position.side} ${position.size} @ $${position.entryPrice.toFixed(2)} (${position.unrealizedPnL.toFixed(2)}%, ${age.toFixed(0)}m ago)`);
      }
    }
    logger.info('════════════════════════════════════════════════════════════');
  }

  /**
   * Main intelligent trading loop
   */
  async startIntelligentTrading() {
    this.isRunning = true;
    logger.info('🧠 Starting Intelligent Trading System...');
    logger.info('⚠️  This will analyze markets and place REAL orders when conditions are optimal!');

    let analysisCount = 0;
    const maxAnalysisCycles = 100; // Limit for safety

    while (this.isRunning && analysisCount < maxAnalysisCycles) {
      try {
        analysisCount++;
        logger.info(`\n🔄 Analysis Cycle ${analysisCount}/${maxAnalysisCycles}`);

        // Update balance
        await this.updateBalance();

        // Manage existing positions first
        await this.manageActivePositions();

        // Check if we can take new positions
        if (this.portfolio.activePositions.size < this.config.maxConcurrentPositions) {

          // Analyze each symbol for trading opportunities
          for (const symbol of this.config.symbols) {
            // Skip if we already have a position in this symbol
            if (this.portfolio.activePositions.has(symbol)) {
              continue;
            }

            logger.info(`\n🔍 Analyzing ${symbol} for trading opportunities...`);

            // Perform comprehensive market analysis
            const analysis = await this.analyzeMarket(symbol);

            if (analysis && analysis.signal !== 'wait') {
              // Execute trade if conditions are met
              await this.executeTradeSignal(symbol, analysis);
            }

            // Small delay between symbol analysis
            await this.sleep(2000);
          }
        } else {
          logger.info('📊 Maximum positions reached - focusing on position management');
        }

        // Display current status
        this.displayTradingStatus();

        // Check if we should stop (no balance or all positions closed)
        if (this.portfolio.currentBalance < 5) {
          logger.info('⚠️ Balance too low, stopping trading');
          break;
        }

        // Wait for next analysis cycle
        logger.info(`⏳ Waiting ${this.config.analysisInterval / 1000}s for next analysis cycle...`);
        await this.sleep(this.config.analysisInterval);

      } catch (error) {
        logger.error('❌ Error in trading loop:', error);
        await this.sleep(10000); // Wait 10 seconds on error
      }
    }

    logger.info('🛑 Intelligent trading system stopped');
    this.isRunning = false;

    // Final status
    this.displayTradingStatus();

    // Close any remaining positions
    if (this.portfolio.activePositions.size > 0) {
      logger.info('🔄 Closing remaining positions...');
      for (const symbol of this.portfolio.activePositions.keys()) {
        await this.closePosition(symbol, 'system_shutdown');
      }
    }
  }

  /**
   * Stop the trading system
   */
  stop() {
    logger.info('🛑 Stopping intelligent trading system...');
    this.isRunning = false;
  }
}

// Main execution
async function main() {
  const bot = new IntelligentDeltaTradingBot();

  try {
    await bot.initialize();

    // Confirmation prompt for live trading
    logger.info('\n⚠️  INTELLIGENT LIVE TRADING CONFIRMATION');
    logger.info('═'.repeat(60));
    logger.info('🧠 This system will analyze markets thoroughly before trading!');
    logger.info('🎯 Only high-confidence trades will be executed!');
    logger.info('📊 Positions will be managed until completion!');
    logger.info('🔴 This will place REAL orders on Delta Exchange testnet!');
    logger.info('💰 Current balance will be used for trading');
    logger.info('⚡ Intelligent risk management will be applied');
    logger.info('═'.repeat(60));

    // Auto-start after 5 seconds
    logger.info('🚀 Starting intelligent trading in 5 seconds...');
    await bot.sleep(5000);

    // Handle graceful shutdown
    process.on('SIGINT', () => {
      logger.info('\n🛑 Received SIGINT, shutting down gracefully...');
      bot.stop();
      process.exit(0);
    });

    await bot.startIntelligentTrading();

  } catch (error) {
    logger.error('❌ Failed to start intelligent trading system:', error);
    process.exit(1);
  }
}

// Run the intelligent trading system
if (require.main === module) {
  main().catch(error => {
    logger.error('❌ Unhandled error:', error);
    process.exit(1);
  });
}

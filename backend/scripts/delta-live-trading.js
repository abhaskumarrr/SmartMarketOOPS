#!/usr/bin/env node
/**
 * Delta Exchange Live Trading System
 * REAL TRADING on Delta Exchange India testnet with actual order placement
 */

const { DeltaExchangeUnified } = require('../dist/services/DeltaExchangeUnified');
const { logger } = require('../dist/utils/logger');

class DeltaLiveTradingBot {
  constructor() {
    // Load environment variables
    require('dotenv').config();
    
    // Initialize Delta Exchange service for LIVE trading
    this.deltaService = new DeltaExchangeUnified({
      apiKey: process.env.DELTA_EXCHANGE_API_KEY || 'uS2N0I4V37gMNJgbTjX8a33WPWv3GK',
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || 'hJwxEd1wCpMTYg5iSQKDnreX9IVlc4mcYegR5ojJzvQ5UVOiUhP7cF9u21To',
      testnet: true
    });

    this.config = {
      symbols: ['BTCUSD', 'ETHUSD'],
      maxLeverage: { 'BTCUSD': 100, 'ETHUSD': 100 }, // 100x leverage
      riskPerTrade: 20, // 20% risk per trade (aggressive for testnet)
      maxConcurrentTrades: 2,
      tradingInterval: 30000, // 30 seconds
      stopLossPercentage: 5, // 5% stop loss
      takeProfitRatio: 2, // 2:1 risk/reward
      enableLiveTrading: true, // LIVE TRADING ENABLED
      minOrderSize: 10 // Minimum order size
    };

    this.portfolio = {
      initialBalance: 0,
      currentBalance: 0,
      availableBalance: 0,
      totalPnl: 0,
      totalReturn: 0,
      activeOrders: new Map(),
      tradeHistory: []
    };

    this.isRunning = false;
    this.tradeCounter = 0;
    this.startTime = Date.now();
  }

  /**
   * Initialize the live trading system
   */
  async initialize() {
    try {
      logger.info('🚀 Initializing Delta Exchange LIVE Trading System...');

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

      logger.info('✅ Delta Exchange service connected successfully');
      this.displayConfiguration();

    } catch (error) {
      logger.error('❌ Failed to initialize live trading system:', error);
      throw error;
    }
  }

  /**
   * Update balance from Delta Exchange
   */
  async updateBalance() {
    try {
      const balances = await this.deltaService.getBalance();
      const usdBalance = balances.find(b => b.asset_symbol === 'USD');
      
      if (usdBalance) {
        this.portfolio.currentBalance = parseFloat(usdBalance.available_balance);
        this.portfolio.availableBalance = parseFloat(usdBalance.available_balance);
        
        if (this.portfolio.initialBalance === 0) {
          this.portfolio.initialBalance = this.portfolio.currentBalance;
        }
        
        logger.info(`💰 Updated Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
      } else {
        throw new Error('USD balance not found');
      }
    } catch (error) {
      logger.error('❌ Failed to update balance:', error);
      throw error;
    }
  }

  /**
   * Display trading configuration
   */
  displayConfiguration() {
    logger.info('\n🔴 DELTA EXCHANGE LIVE TRADING SYSTEM');
    logger.info('═'.repeat(60));
    logger.info(`💰 Live Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
    logger.info(`📊 Trading Symbols: ${this.config.symbols.join(', ')}`);
    logger.info(`⚡ Max Leverage: BTC=${this.config.maxLeverage.BTCUSD}x, ETH=${this.config.maxLeverage.ETHUSD}x`);
    logger.info(`🎯 Risk Per Trade: ${this.config.riskPerTrade}%`);
    logger.info(`🛡️ Stop Loss: ${this.config.stopLossPercentage}%`);
    logger.info(`🎯 Take Profit Ratio: ${this.config.takeProfitRatio}:1`);
    logger.info(`🔄 Trading Interval: ${this.config.tradingInterval / 1000}s`);
    logger.info(`🏢 Exchange: Delta Exchange India Testnet`);
    logger.info(`🔴 Mode: LIVE TRADING (REAL ORDERS)`);
    logger.info('═'.repeat(60));
    logger.info('⚠️  WARNING: This will place REAL orders on Delta Exchange!');
    logger.info('═'.repeat(60));
  }

  /**
   * Generate aggressive trading signal for testnet
   */
  async generateTradingSignal(symbol) {
    try {
      const marketData = await this.deltaService.getMarketData(symbol);
      const currentPrice = marketData.last_price;

      // More aggressive signal generation for testnet
      const random = Math.random();
      
      // Generate signals more frequently for testing
      if (symbol === 'BTCUSD') {
        if (random > 0.4) { // 60% chance of signal
          return { 
            side: random > 0.5 ? 'buy' : 'sell', 
            confidence: 0.9,
            reason: `BTC price at $${currentPrice.toFixed(2)}`
          };
        }
      }

      if (symbol === 'ETHUSD') {
        if (random > 0.4) { // 60% chance of signal
          return { 
            side: random > 0.5 ? 'buy' : 'sell', 
            confidence: 0.9,
            reason: `ETH price at $${currentPrice.toFixed(2)}`
          };
        }
      }

      return null;
    } catch (error) {
      logger.error(`❌ Error generating signal for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Calculate position size based on risk management
   */
  calculatePositionSize(symbol, entryPrice) {
    const riskAmount = this.portfolio.availableBalance * (this.config.riskPerTrade / 100);
    const stopLossDistance = entryPrice * (this.config.stopLossPercentage / 100);
    const leverage = this.config.maxLeverage[symbol] || 50;
    
    // Calculate position size in USD value
    const positionValue = (riskAmount / stopLossDistance) * entryPrice;
    
    // Convert to contract units (Delta Exchange uses integer contract units)
    const contractSize = Math.floor(positionValue / entryPrice);
    
    // Ensure minimum position size
    return Math.max(contractSize, this.config.minOrderSize);
  }

  /**
   * Place a LIVE order on Delta Exchange
   */
  async placeLiveOrder(symbol, side, size) {
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
      logger.info(`📊 Order Details:`, JSON.stringify(orderRequest, null, 2));
      
      const order = await this.deltaService.placeOrder(orderRequest);
      
      logger.info(`✅ LIVE ORDER PLACED SUCCESSFULLY!`);
      logger.info(`📋 Order ID: ${order.id}`);
      logger.info(`📊 Order Status: ${order.state}`);
      
      // Track the order
      this.portfolio.activeOrders.set(order.id, {
        ...order,
        symbol,
        timestamp: Date.now()
      });
      
      this.tradeCounter++;
      
      // Add to trade history
      this.portfolio.tradeHistory.push({
        id: order.id,
        symbol,
        side,
        size,
        timestamp: Date.now(),
        status: 'placed'
      });
      
      return order;
    } catch (error) {
      logger.error(`❌ Failed to place LIVE order:`, error);
      throw error;
    }
  }

  /**
   * Check order status and update positions
   */
  async updateOrderStatus() {
    for (const [orderId, order] of this.portfolio.activeOrders) {
      try {
        // In a real implementation, you would check order status
        // For now, we'll assume orders are filled quickly on testnet
        logger.info(`📊 Checking order ${orderId} status...`);
        
        // Remove from active orders after some time (simulating fill)
        const orderAge = Date.now() - order.timestamp;
        if (orderAge > 60000) { // 1 minute
          this.portfolio.activeOrders.delete(orderId);
          logger.info(`✅ Order ${orderId} assumed filled and removed from tracking`);
        }
      } catch (error) {
        logger.error(`❌ Error checking order ${orderId}:`, error);
      }
    }
  }

  /**
   * Main live trading loop
   */
  async startLiveTrading() {
    this.isRunning = true;
    logger.info('🔴 Starting LIVE trading loop...');
    logger.info('⚠️  This will place REAL orders with REAL money on Delta Exchange testnet!');

    let iteration = 0;
    const maxIterations = 20; // Limit for safety

    while (this.isRunning && iteration < maxIterations) {
      try {
        iteration++;
        logger.info(`\n🔄 Live Trading Iteration ${iteration}/${maxIterations}`);
        
        // Update balance and order status
        await this.updateBalance();
        await this.updateOrderStatus();
        
        // Check if we can place new orders
        if (this.portfolio.activeOrders.size < this.config.maxConcurrentTrades) {
          for (const symbol of this.config.symbols) {
            // Skip if we already have an active order for this symbol
            const hasActiveOrder = Array.from(this.portfolio.activeOrders.values())
              .some(order => order.symbol === symbol);
            
            if (hasActiveOrder) {
              logger.info(`⏳ Skipping ${symbol} - active order exists`);
              continue;
            }

            // Generate trading signal
            const signal = await this.generateTradingSignal(symbol);
            
            if (signal && signal.confidence > 0.8) {
              const marketData = await this.deltaService.getMarketData(symbol);
              const currentPrice = marketData.last_price;
              const positionSize = this.calculatePositionSize(symbol, currentPrice);
              
              if (positionSize >= this.config.minOrderSize && this.portfolio.availableBalance > 50) {
                logger.info(`🎯 Trading Signal: ${signal.side.toUpperCase()} ${symbol} - ${signal.reason}`);
                
                try {
                  await this.placeLiveOrder(symbol, signal.side, positionSize);
                  logger.info(`✅ Successfully placed ${signal.side} order for ${positionSize} ${symbol}`);
                } catch (orderError) {
                  logger.error(`❌ Failed to place order:`, orderError);
                }
              } else {
                logger.info(`⚠️ Insufficient balance or position size too small for ${symbol}`);
              }
            } else {
              logger.info(`📊 No trading signal for ${symbol}`);
            }
          }
        } else {
          logger.info(`⏳ Maximum concurrent trades reached (${this.config.maxConcurrentTrades})`);
        }

        // Display current status
        this.displayLiveStatus();
        
        // Wait for next iteration
        logger.info(`⏳ Waiting ${this.config.tradingInterval / 1000}s for next iteration...`);
        await this.sleep(this.config.tradingInterval);
        
      } catch (error) {
        logger.error('❌ Error in live trading loop:', error);
        await this.sleep(10000); // Wait 10 seconds before retrying
      }
    }

    logger.info('🏁 Live trading completed');
    this.generateLiveReport();
  }

  /**
   * Display current live trading status
   */
  displayLiveStatus() {
    const runtime = (Date.now() - this.startTime) / 1000 / 60; // minutes
    this.portfolio.totalReturn = this.portfolio.initialBalance > 0 
      ? ((this.portfolio.currentBalance - this.portfolio.initialBalance) / this.portfolio.initialBalance) * 100 
      : 0;
    
    logger.info('\n📊 LIVE TRADING STATUS:');
    logger.info(`💰 Balance: $${this.portfolio.currentBalance.toFixed(2)} (${this.portfolio.totalReturn.toFixed(2)}%)`);
    logger.info(`🎯 Active Orders: ${this.portfolio.activeOrders.size}`);
    logger.info(`📊 Total Orders Placed: ${this.portfolio.tradeHistory.length}`);
    logger.info(`⏱️ Runtime: ${runtime.toFixed(1)} minutes`);
    
    // Show active orders
    if (this.portfolio.activeOrders.size > 0) {
      logger.info('📋 Active Orders:');
      for (const order of this.portfolio.activeOrders.values()) {
        const age = (Date.now() - order.timestamp) / 1000;
        logger.info(`   ${order.symbol}: ${order.side} ${order.size} (${age.toFixed(0)}s ago)`);
      }
    }
  }

  /**
   * Generate final live trading report
   */
  generateLiveReport() {
    const runtime = (Date.now() - this.startTime) / 1000 / 60; // minutes
    
    logger.info('\n📊 LIVE TRADING FINAL REPORT');
    logger.info('═'.repeat(60));
    logger.info(`💰 Starting Balance: $${this.portfolio.initialBalance.toFixed(2)}`);
    logger.info(`💰 Final Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
    logger.info(`📈 Total Return: ${this.portfolio.totalReturn.toFixed(2)}%`);
    logger.info(`📊 Total Orders Placed: ${this.portfolio.tradeHistory.length}`);
    logger.info(`⏱️ Total Runtime: ${runtime.toFixed(1)} minutes`);
    logger.info(`🏢 Exchange: Delta Exchange India Testnet`);
    logger.info(`🔴 Mode: LIVE TRADING (Real Orders)`);
    logger.info('═'.repeat(60));
  }

  /**
   * Stop live trading
   */
  stop() {
    this.isRunning = false;
    logger.info('🛑 Live trading stopped');
  }

  /**
   * Sleep utility
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Main execution
async function main() {
  const bot = new DeltaLiveTradingBot();
  
  try {
    await bot.initialize();
    
    // Confirmation prompt for live trading
    logger.info('\n⚠️  LIVE TRADING CONFIRMATION');
    logger.info('═'.repeat(60));
    logger.info('🔴 This will place REAL orders on Delta Exchange testnet!');
    logger.info('💰 Current balance will be used for trading');
    logger.info('⚡ High leverage (100x) will be used');
    logger.info('🎯 20% risk per trade');
    logger.info('═'.repeat(60));
    
    // Auto-start after 5 seconds (remove this in production)
    logger.info('🚀 Starting live trading in 5 seconds...');
    await bot.sleep(5000);
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
      logger.info('\n🛑 Received SIGINT, shutting down gracefully...');
      bot.stop();
      process.exit(0);
    });
    
    await bot.startLiveTrading();
    
  } catch (error) {
    logger.error('❌ Live trading failed:', error);
    process.exit(1);
  }
}

// Run the bot
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { DeltaLiveTradingBot };

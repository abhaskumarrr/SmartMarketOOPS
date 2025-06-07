#!/usr/bin/env node
/**
 * Delta Exchange India Testnet Trading Bot
 * Live trading on testnet with real market data and intelligent risk management
 */

import { DeltaExchangeUnified, DeltaOrderRequest } from '../src/services/DeltaExchangeUnified';
import { logger } from '../src/utils/logger';

interface TradingConfig {
  symbols: string[];
  initialBalance: number;
  maxLeverage: Record<string, number>;
  riskPerTrade: number; // Percentage
  maxConcurrentTrades: number;
  tradingInterval: number; // milliseconds
  stopLossPercentage: number;
  takeProfitRatio: number;
  enableLiveTrading: boolean;
}

interface Position {
  id: number;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  currentPrice: number;
  leverage: number;
  pnl: number;
  pnlPercentage: number;
  entryTime: number;
  stopLoss?: number;
  takeProfit?: number;
}

interface TradingStats {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  totalPnl: number;
  winRate: number;
  maxDrawdown: number;
  currentDrawdown: number;
  startTime: number;
  lastTradeTime: number;
}

class DeltaTestnetTradingBot {
  private deltaService: DeltaExchangeUnified;
  private config: TradingConfig;
  private positions: Map<string, Position> = new Map();
  private stats: TradingStats;
  private isRunning = false;
  private currentBalance = 0;
  private peakBalance = 0;

  constructor() {
    // Initialize Delta Exchange service
    this.deltaService = new DeltaExchangeUnified({
      apiKey: process.env.DELTA_EXCHANGE_API_KEY!,
      apiSecret: process.env.DELTA_EXCHANGE_API_SECRET!,
      testnet: true
    });

    // Trading configuration optimized for testnet
    this.config = {
      symbols: ['BTCUSD', 'ETHUSD'],
      initialBalance: 69.65, // Use actual testnet balance
      maxLeverage: {
        'BTCUSD': 100, // 100x leverage for BTC
        'ETHUSD': 100  // 100x leverage for ETH
      },
      riskPerTrade: 20, // 20% risk per trade (aggressive for testnet)
      maxConcurrentTrades: 2,
      tradingInterval: 30000, // 30 seconds
      stopLossPercentage: 5, // 5% stop loss
      takeProfitRatio: 2, // 2:1 risk/reward ratio
      enableLiveTrading: true
    };

    // Initialize stats
    this.stats = {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      totalPnl: 0,
      winRate: 0,
      maxDrawdown: 0,
      currentDrawdown: 0,
      startTime: Date.now(),
      lastTradeTime: 0
    };
  }

  /**
   * Initialize the trading bot
   */
  async initialize(): Promise<void> {
    try {
      logger.info('🚀 Initializing Delta Exchange Testnet Trading Bot...');

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

      // Get current balance
      await this.updateBalance();
      
      logger.info('✅ Delta Exchange service initialized successfully');
      logger.info(`💰 Current Balance: $${this.currentBalance.toFixed(2)}`);

      // Display configuration
      this.displayConfiguration();

    } catch (error) {
      logger.error('❌ Failed to initialize trading bot:', error);
      throw error;
    }
  }

  /**
   * Update current balance from Delta Exchange
   */
  async updateBalance(): Promise<void> {
    try {
      const balances = await this.deltaService.getBalance();
      const usdBalance = balances.find(b => b.asset_symbol === 'USD');
      
      if (usdBalance) {
        this.currentBalance = parseFloat(usdBalance.available_balance);
        if (this.currentBalance > this.peakBalance) {
          this.peakBalance = this.currentBalance;
        }
      } else {
        logger.warn('⚠️ USD balance not found, using config initial balance');
        this.currentBalance = this.config.initialBalance;
      }
    } catch (error) {
      logger.error('❌ Failed to update balance:', error);
      this.currentBalance = this.config.initialBalance;
    }
  }

  /**
   * Display trading configuration
   */
  displayConfiguration(): void {
    logger.info('\n🎯 DELTA EXCHANGE TESTNET TRADING BOT');
    logger.info('═'.repeat(60));
    logger.info(`💰 Starting Balance: $${this.currentBalance.toFixed(2)}`);
    logger.info(`📊 Trading Symbols: ${this.config.symbols.join(', ')}`);
    logger.info(`⚡ Max Leverage: BTC=${this.config.maxLeverage.BTCUSD}x, ETH=${this.config.maxLeverage.ETHUSD}x`);
    logger.info(`🎯 Risk Per Trade: ${this.config.riskPerTrade}%`);
    logger.info(`🛡️ Stop Loss: ${this.config.stopLossPercentage}%`);
    logger.info(`🎯 Take Profit Ratio: ${this.config.takeProfitRatio}:1`);
    logger.info(`🔄 Trading Interval: ${this.config.tradingInterval / 1000}s`);
    logger.info(`🏢 Exchange: Delta Exchange India Testnet`);
    logger.info(`🔴 LIVE TRADING: ${this.config.enableLiveTrading ? 'ENABLED' : 'DISABLED'}`);
    logger.info('═'.repeat(60));
  }

  /**
   * Generate trading signal based on simple momentum strategy
   */
  async generateTradingSignal(symbol: string): Promise<{ side: 'buy' | 'sell'; confidence: number } | null> {
    try {
      const marketData = await this.deltaService.getMarketData(symbol);
      const currentPrice = marketData.last_price;

      // Simple momentum-based signal generation
      // This is a basic example - in production you'd use more sophisticated indicators
      
      // For BTC: Buy if price < 105000, Sell if price > 106000
      if (symbol === 'BTCUSD') {
        if (currentPrice < 105000) {
          return { side: 'buy', confidence: 0.8 };
        } else if (currentPrice > 106000) {
          return { side: 'sell', confidence: 0.8 };
        }
      }

      // For ETH: Buy if price < 2600, Sell if price > 2700
      if (symbol === 'ETHUSD') {
        if (currentPrice < 2600) {
          return { side: 'buy', confidence: 0.8 };
        } else if (currentPrice > 2700) {
          return { side: 'sell', confidence: 0.8 };
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
  calculatePositionSize(symbol: string, entryPrice: number, leverage: number): number {
    const riskAmount = this.currentBalance * (this.config.riskPerTrade / 100);
    const stopLossDistance = entryPrice * (this.config.stopLossPercentage / 100);
    
    // Calculate position size in USD value
    const positionValue = (riskAmount / stopLossDistance) * entryPrice * leverage;
    
    // Convert to contract units (Delta Exchange uses integer contract units)
    const contractSize = Math.floor(positionValue);
    
    // Ensure minimum position size
    return Math.max(contractSize, 10);
  }

  /**
   * Place a trade on Delta Exchange
   */
  async placeTrade(symbol: string, side: 'buy' | 'sell', size: number): Promise<boolean> {
    try {
      if (!this.config.enableLiveTrading) {
        logger.info(`📝 [PAPER] Would place ${side} order for ${size} ${symbol}`);
        return true;
      }

      const productId = this.deltaService.getProductId(symbol);
      if (!productId) {
        throw new Error(`Product ID not found for ${symbol}`);
      }

      const orderRequest: DeltaOrderRequest = {
        product_id: productId,
        side: side,
        size: size,
        order_type: 'market_order'
      };

      logger.info(`📤 Placing ${side} order: ${size} ${symbol}`);
      const order = await this.deltaService.placeOrder(orderRequest);
      
      logger.info(`✅ Order placed successfully: ID ${order.id}`);
      this.stats.totalTrades++;
      this.stats.lastTradeTime = Date.now();
      
      return true;
    } catch (error) {
      logger.error(`❌ Failed to place trade:`, error);
      return false;
    }
  }

  /**
   * Main trading loop
   */
  async startTrading(): Promise<void> {
    this.isRunning = true;
    logger.info('🚀 Starting trading loop...');

    let iteration = 0;
    const maxIterations = 200; // Limit for demo

    while (this.isRunning && iteration < maxIterations) {
      try {
        iteration++;
        logger.info(`\n🔄 Trading Iteration ${iteration}`);
        
        // Update balance
        await this.updateBalance();
        
        // Check each symbol for trading opportunities
        for (const symbol of this.config.symbols) {
          // Skip if we already have a position for this symbol
          if (this.positions.has(symbol)) {
            continue;
          }

          // Generate trading signal
          const signal = await this.generateTradingSignal(symbol);
          
          if (signal && signal.confidence > 0.7) {
            const marketData = await this.deltaService.getMarketData(symbol);
            const currentPrice = marketData.last_price;
            const leverage = this.config.maxLeverage[symbol] || 50;
            
            const positionSize = this.calculatePositionSize(symbol, currentPrice, leverage);
            
            if (positionSize > 0) {
              const success = await this.placeTrade(symbol, signal.side, positionSize);
              
              if (success) {
                logger.info(`✅ Trade executed: ${signal.side} ${positionSize} ${symbol} @ $${currentPrice}`);
              }
            }
          }
        }

        // Display current status
        this.displayStatus();
        
        // Wait for next iteration
        logger.info(`⏳ Waiting ${this.config.tradingInterval / 1000}s for next iteration...`);
        await this.sleep(this.config.tradingInterval);
        
      } catch (error) {
        logger.error('❌ Error in trading loop:', error);
        await this.sleep(5000); // Wait 5 seconds before retrying
      }
    }

    logger.info('🏁 Trading loop completed');
    this.generateFinalReport();
  }

  /**
   * Display current trading status
   */
  displayStatus(): void {
    const runtime = (Date.now() - this.stats.startTime) / 1000 / 60; // minutes
    
    logger.info('\n📊 CURRENT STATUS:');
    logger.info(`💰 Balance: $${this.currentBalance.toFixed(2)} (Peak: $${this.peakBalance.toFixed(2)})`);
    logger.info(`📈 Total Trades: ${this.stats.totalTrades}`);
    logger.info(`⏱️ Runtime: ${runtime.toFixed(1)} minutes`);
    logger.info(`🎯 Active Positions: ${this.positions.size}`);
  }

  /**
   * Generate final trading report
   */
  generateFinalReport(): void {
    const runtime = (Date.now() - this.stats.startTime) / 1000 / 60; // minutes
    const totalReturn = ((this.currentBalance - this.config.initialBalance) / this.config.initialBalance) * 100;
    
    logger.info('\n📊 FINAL TRADING REPORT');
    logger.info('═'.repeat(60));
    logger.info(`💰 Starting Balance: $${this.config.initialBalance.toFixed(2)}`);
    logger.info(`💰 Final Balance: $${this.currentBalance.toFixed(2)}`);
    logger.info(`📈 Total Return: ${totalReturn.toFixed(2)}%`);
    logger.info(`📊 Total Trades: ${this.stats.totalTrades}`);
    logger.info(`⏱️ Total Runtime: ${runtime.toFixed(1)} minutes`);
    logger.info(`🏢 Exchange: Delta Exchange India Testnet`);
    logger.info('═'.repeat(60));
  }

  /**
   * Stop the trading bot
   */
  stop(): void {
    this.isRunning = false;
    logger.info('🛑 Trading bot stopped');
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Main execution
async function main() {
  const bot = new DeltaTestnetTradingBot();
  
  try {
    await bot.initialize();
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
      logger.info('\n🛑 Received SIGINT, shutting down gracefully...');
      bot.stop();
      process.exit(0);
    });
    
    process.on('SIGTERM', () => {
      logger.info('\n🛑 Received SIGTERM, shutting down gracefully...');
      bot.stop();
      process.exit(0);
    });
    
    await bot.startTrading();
    
  } catch (error) {
    logger.error('❌ Trading bot failed:', error);
    process.exit(1);
  }
}

// Run the bot
if (require.main === module) {
  main().catch(console.error);
}

export { DeltaTestnetTradingBot };

const EnhancedFibonacciTrader = require('./enhanced-fibonacci-trading');
const { DeltaExchangeUnified } = require('../dist/services/DeltaExchangeUnified');

class EnhancedFibonacciBacktest {
  constructor() {
    this.trader = new EnhancedFibonacciTrader();
    this.backtestResults = {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      totalPnL: 0,
      maxDrawdown: 0,
      winRate: 0,
      profitFactor: 0,
      trades: [],
      dailyPnL: [],
      startBalance: 1000, // $1000 starting capital
      currentBalance: 1000,
      maxBalance: 1000,
      minBalance: 1000
    };
    
    this.backtestConfig = {
      symbols: ['BTCUSD', 'ETHUSD'],
      startDate: new Date('2024-11-01'), // 1 month backtest
      endDate: new Date('2024-12-01'),
      initialBalance: 1000,
      riskPerTrade: 0.02, // 2% risk per trade
      maxPositions: 2,
      leverage: {
        scalp: 25,
        day: 15,
        swing: 10
      }
    };
  }

  /**
   * Initialize backtest environment
   */
  async initialize() {
    try {
      console.log(`🔬 INITIALIZING ENHANCED FIBONACCI BACKTEST`);
      console.log(`════════════════════════════════════════════════════════════════════════════════`);
      console.log(`📅 Period: ${this.backtestConfig.startDate.toDateString()} → ${this.backtestConfig.endDate.toDateString()}`);
      console.log(`💰 Starting Capital: $${this.backtestConfig.initialBalance}`);
      console.log(`📊 Symbols: ${this.backtestConfig.symbols.join(', ')}`);
      console.log(`🎯 Risk Per Trade: ${(this.backtestConfig.riskPerTrade * 100).toFixed(1)}%`);
      console.log(`════════════════════════════════════════════════════════════════════════════════`);

      // Initialize trader with testnet credentials
      this.trader.deltaService = new DeltaExchangeUnified({
        apiKey: process.env.DELTA_EXCHANGE_API_KEY,
        apiSecret: process.env.DELTA_EXCHANGE_API_SECRET,
        testnet: true
      });

      await this.trader.deltaService.initialize();
      
      // Set initial balance
      this.trader.balance.availableBalance = this.backtestConfig.initialBalance;
      
      console.log(`✅ Backtest environment initialized successfully`);
      return true;
    } catch (error) {
      console.error(`❌ Failed to initialize backtest: ${error.message}`);
      return false;
    }
  }

  /**
   * Run comprehensive backtest
   */
  async runBacktest() {
    try {
      console.log(`\n🚀 STARTING ENHANCED FIBONACCI BACKTEST`);
      console.log(`════════════════════════════════════════════════════════════════════════════════`);

      const startTime = Date.now();
      let currentDate = new Date(this.backtestConfig.startDate);
      let dayCount = 0;

      while (currentDate <= this.backtestConfig.endDate) {
        dayCount++;
        console.log(`\n📅 Day ${dayCount}: ${currentDate.toDateString()}`);
        
        // Simulate trading day
        await this.simulateTradingDay(currentDate);
        
        // Move to next day
        currentDate.setDate(currentDate.getDate() + 1);
        
        // Update daily PnL tracking
        this.updateDailyPnL(currentDate);
        
        // Progress indicator
        if (dayCount % 5 === 0) {
          console.log(`📊 Progress: Day ${dayCount}, Balance: $${this.backtestResults.currentBalance.toFixed(2)}`);
        }
      }

      const endTime = Date.now();
      const duration = (endTime - startTime) / 1000;

      console.log(`\n✅ BACKTEST COMPLETED`);
      console.log(`⏱️ Duration: ${duration.toFixed(1)} seconds`);
      console.log(`📊 Days Simulated: ${dayCount}`);
      
      // Calculate final results
      this.calculateFinalResults();
      
      // Display comprehensive results
      this.displayResults();
      
      return this.backtestResults;
    } catch (error) {
      console.error(`❌ Backtest failed: ${error.message}`);
      return null;
    }
  }

  /**
   * Simulate a single trading day
   */
  async simulateTradingDay(date) {
    try {
      // Get historical data for this day
      for (const symbol of this.backtestConfig.symbols) {
        await this.analyzeSymbolForDay(symbol, date);
      }
      
      // Process any open positions
      await this.processOpenPositions(date);
      
    } catch (error) {
      console.error(`❌ Failed to simulate trading day ${date.toDateString()}: ${error.message}`);
    }
  }

  /**
   * Analyze symbol for potential trades on specific day
   */
  async analyzeSymbolForDay(symbol, date) {
    try {
      // Get market structure for the day
      await this.trader.analyzeDailyMarketStructure(symbol);
      
      // Get current price (simulate with historical data)
      const currentPrice = await this.getHistoricalPrice(symbol, date);
      
      if (!currentPrice) return;
      
      // Analyze entry signals
      const entrySignal = await this.trader.analyzeEntrySignals(symbol, currentPrice);
      
      if (entrySignal && entrySignal.isValid) {
        console.log(`🎯 ${symbol} Trade Signal Found:`);
        console.log(`   Confluence: ${(entrySignal.confluence * 100).toFixed(1)}%`);
        console.log(`   Trade Type: ${entrySignal.tradeType.type.toUpperCase()}`);
        console.log(`   Direction: ${entrySignal.signal.direction.toUpperCase()}`);
        
        // Execute trade
        await this.executeTrade(symbol, entrySignal, currentPrice, date);
      }
      
    } catch (error) {
      console.error(`❌ Failed to analyze ${symbol} for ${date.toDateString()}: ${error.message}`);
    }
  }

  /**
   * Get historical price for backtesting
   */
  async getHistoricalPrice(symbol, date) {
    try {
      // For backtesting, we'll use live data as proxy
      // In production, this would use historical data API
      const ticker = await this.trader.deltaService.getMarketData(symbol);
      return parseFloat(ticker.mark_price || ticker.last_price);
    } catch (error) {
      console.error(`❌ Failed to get price for ${symbol}: ${error.message}`);
      return null;
    }
  }

  /**
   * Execute a trade based on signal
   */
  async executeTrade(symbol, signal, price, date) {
    try {
      // Check if we can open new position
      if (this.trader.activePositions.size >= this.backtestConfig.maxPositions) {
        console.log(`⚠️ Maximum positions reached, skipping trade`);
        return;
      }

      // Calculate position size
      const positionSize = this.calculatePositionSize(signal.tradeType.type, price);
      
      if (positionSize <= 0) {
        console.log(`⚠️ Insufficient balance for trade`);
        return;
      }

      // Create trade record
      const trade = {
        id: `${symbol}_${Date.now()}`,
        symbol,
        side: signal.signal.direction,
        entryPrice: price,
        entryTime: date,
        positionSize,
        leverage: this.backtestConfig.leverage[signal.tradeType.type],
        tradeType: signal.tradeType.type,
        fibLevel: signal.signal.level,
        confluence: signal.confluence,
        stopLoss: this.calculateStopLoss(price, signal.signal.direction, signal.tradeType.type),
        takeProfit: this.calculateTakeProfit(price, signal.signal.direction, signal.tradeType.type),
        status: 'open',
        pnl: 0
      };

      // Add to active positions
      this.trader.activePositions.set(trade.id, trade);
      
      console.log(`✅ Trade Executed: ${symbol} ${signal.signal.direction.toUpperCase()} @ $${price.toFixed(2)}`);
      console.log(`   Size: $${positionSize.toFixed(2)}, Leverage: ${trade.leverage}x`);
      console.log(`   Stop Loss: $${trade.stopLoss.toFixed(2)}, Take Profit: $${trade.takeProfit.toFixed(2)}`);
      
    } catch (error) {
      console.error(`❌ Failed to execute trade: ${error.message}`);
    }
  }

  /**
   * Calculate position size based on risk management
   */
  calculatePositionSize(tradeType, price) {
    const riskAmount = this.backtestResults.currentBalance * this.backtestConfig.riskPerTrade;
    const leverage = this.backtestConfig.leverage[tradeType];
    
    // Calculate position size based on risk and leverage
    const positionSize = (riskAmount * leverage) / price;
    
    return Math.min(positionSize, this.backtestResults.currentBalance * 0.15); // Max 15% of balance
  }

  /**
   * Calculate stop loss level
   */
  calculateStopLoss(entryPrice, direction, tradeType) {
    const stopLossPercent = {
      scalp: 0.005,  // 0.5%
      day: 0.01,     // 1%
      swing: 0.02    // 2%
    };

    const stopPercent = stopLossPercent[tradeType];
    
    if (direction === 'buy') {
      return entryPrice * (1 - stopPercent);
    } else {
      return entryPrice * (1 + stopPercent);
    }
  }

  /**
   * Calculate take profit level
   */
  calculateTakeProfit(entryPrice, direction, tradeType) {
    const takeProfitRatio = {
      scalp: 2,   // 1:2 risk/reward
      day: 3,     // 1:3 risk/reward
      swing: 4    // 1:4 risk/reward
    };

    const stopLoss = this.calculateStopLoss(entryPrice, direction, tradeType);
    const riskAmount = Math.abs(entryPrice - stopLoss);
    const rewardAmount = riskAmount * takeProfitRatio[tradeType];
    
    if (direction === 'buy') {
      return entryPrice + rewardAmount;
    } else {
      return entryPrice - rewardAmount;
    }
  }

  /**
   * Process open positions for stop loss/take profit
   */
  async processOpenPositions(date) {
    for (const [tradeId, trade] of this.trader.activePositions) {
      try {
        // Get current price
        const currentPrice = await this.getHistoricalPrice(trade.symbol, date);
        
        if (!currentPrice) continue;
        
        // Check for stop loss or take profit
        let shouldClose = false;
        let closeReason = '';
        
        if (trade.side === 'buy') {
          if (currentPrice <= trade.stopLoss) {
            shouldClose = true;
            closeReason = 'Stop Loss';
          } else if (currentPrice >= trade.takeProfit) {
            shouldClose = true;
            closeReason = 'Take Profit';
          }
        } else {
          if (currentPrice >= trade.stopLoss) {
            shouldClose = true;
            closeReason = 'Stop Loss';
          } else if (currentPrice <= trade.takeProfit) {
            shouldClose = true;
            closeReason = 'Take Profit';
          }
        }
        
        if (shouldClose) {
          await this.closeTrade(trade, currentPrice, date, closeReason);
        }
        
      } catch (error) {
        console.error(`❌ Failed to process position ${tradeId}: ${error.message}`);
      }
    }
  }

  /**
   * Close a trade and calculate PnL
   */
  async closeTrade(trade, exitPrice, exitTime, reason) {
    try {
      // Calculate PnL
      let pnl = 0;
      if (trade.side === 'buy') {
        pnl = (exitPrice - trade.entryPrice) * trade.positionSize;
      } else {
        pnl = (trade.entryPrice - exitPrice) * trade.positionSize;
      }

      // Apply leverage
      pnl *= trade.leverage;

      // Update trade record
      trade.exitPrice = exitPrice;
      trade.exitTime = exitTime;
      trade.pnl = pnl;
      trade.status = 'closed';
      trade.closeReason = reason;

      // Update backtest results
      this.backtestResults.totalTrades++;
      this.backtestResults.totalPnL += pnl;
      this.backtestResults.currentBalance += pnl;

      if (pnl > 0) {
        this.backtestResults.winningTrades++;
      } else {
        this.backtestResults.losingTrades++;
      }

      // Update balance tracking
      this.backtestResults.maxBalance = Math.max(this.backtestResults.maxBalance, this.backtestResults.currentBalance);
      this.backtestResults.minBalance = Math.min(this.backtestResults.minBalance, this.backtestResults.currentBalance);

      // Add to completed trades
      this.backtestResults.trades.push({ ...trade });

      // Remove from active positions
      this.trader.activePositions.delete(trade.id);

      console.log(`🔒 Trade Closed: ${trade.symbol} ${reason}`);
      console.log(`   Entry: $${trade.entryPrice.toFixed(2)} → Exit: $${exitPrice.toFixed(2)}`);
      console.log(`   PnL: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} | Balance: $${this.backtestResults.currentBalance.toFixed(2)}`);

    } catch (error) {
      console.error(`❌ Failed to close trade: ${error.message}`);
    }
  }

  /**
   * Update daily PnL tracking
   */
  updateDailyPnL(date) {
    this.backtestResults.dailyPnL.push({
      date: new Date(date),
      balance: this.backtestResults.currentBalance,
      pnl: this.backtestResults.currentBalance - this.backtestResults.startBalance,
      drawdown: ((this.backtestResults.maxBalance - this.backtestResults.currentBalance) / this.backtestResults.maxBalance) * 100
    });
  }

  /**
   * Calculate final backtest results
   */
  calculateFinalResults() {
    // Win rate
    this.backtestResults.winRate = this.backtestResults.totalTrades > 0 ?
      (this.backtestResults.winningTrades / this.backtestResults.totalTrades) * 100 : 0;

    // Profit factor
    const grossProfit = this.backtestResults.trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0);
    const grossLoss = Math.abs(this.backtestResults.trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));
    this.backtestResults.profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999 : 0;

    // Maximum drawdown
    this.backtestResults.maxDrawdown = ((this.backtestResults.maxBalance - this.backtestResults.minBalance) / this.backtestResults.maxBalance) * 100;

    // Return percentage
    this.backtestResults.totalReturn = ((this.backtestResults.currentBalance - this.backtestResults.startBalance) / this.backtestResults.startBalance) * 100;
  }

  /**
   * Display comprehensive backtest results
   */
  displayResults() {
    console.log(`\n🏆 ENHANCED FIBONACCI BACKTEST RESULTS`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`📊 PERFORMANCE SUMMARY:`);
    console.log(`   Starting Balance: $${this.backtestResults.startBalance.toFixed(2)}`);
    console.log(`   Ending Balance: $${this.backtestResults.currentBalance.toFixed(2)}`);
    console.log(`   Total Return: ${this.backtestResults.totalReturn >= 0 ? '+' : ''}${this.backtestResults.totalReturn.toFixed(2)}%`);
    console.log(`   Total PnL: ${this.backtestResults.totalPnL >= 0 ? '+' : ''}$${this.backtestResults.totalPnL.toFixed(2)}`);
    console.log(`   Maximum Drawdown: ${this.backtestResults.maxDrawdown.toFixed(2)}%`);

    console.log(`\n📈 TRADING STATISTICS:`);
    console.log(`   Total Trades: ${this.backtestResults.totalTrades}`);
    console.log(`   Winning Trades: ${this.backtestResults.winningTrades}`);
    console.log(`   Losing Trades: ${this.backtestResults.losingTrades}`);
    console.log(`   Win Rate: ${this.backtestResults.winRate.toFixed(1)}%`);
    console.log(`   Profit Factor: ${this.backtestResults.profitFactor.toFixed(2)}`);

    console.log(`\n🎯 TRADE TYPE BREAKDOWN:`);
    const tradesByType = this.groupTradesByType();
    for (const [type, trades] of Object.entries(tradesByType)) {
      const winRate = trades.length > 0 ? (trades.filter(t => t.pnl > 0).length / trades.length) * 100 : 0;
      const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
      console.log(`   ${type.toUpperCase()}: ${trades.length} trades, ${winRate.toFixed(1)}% win rate, ${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)} PnL`);
    }

    console.log(`\n📊 FIBONACCI LEVEL PERFORMANCE:`);
    const tradesByFibLevel = this.groupTradesByFibLevel();
    for (const [level, trades] of Object.entries(tradesByFibLevel)) {
      const winRate = trades.length > 0 ? (trades.filter(t => t.pnl > 0).length / trades.length) * 100 : 0;
      const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
      console.log(`   ${level}%: ${trades.length} trades, ${winRate.toFixed(1)}% win rate, ${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)} PnL`);
    }

    console.log(`\n💹 SYMBOL PERFORMANCE:`);
    const tradesBySymbol = this.groupTradesBySymbol();
    for (const [symbol, trades] of Object.entries(tradesBySymbol)) {
      const winRate = trades.length > 0 ? (trades.filter(t => t.pnl > 0).length / trades.length) * 100 : 0;
      const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
      console.log(`   ${symbol}: ${trades.length} trades, ${winRate.toFixed(1)}% win rate, ${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)} PnL`);
    }

    console.log(`════════════════════════════════════════════════════════════════════════════════`);

    // Performance assessment
    this.assessPerformance();
  }

  /**
   * Group trades by type for analysis
   */
  groupTradesByType() {
    const groups = { scalp: [], day: [], swing: [] };
    this.backtestResults.trades.forEach(trade => {
      if (groups[trade.tradeType]) {
        groups[trade.tradeType].push(trade);
      }
    });
    return groups;
  }

  /**
   * Group trades by Fibonacci level
   */
  groupTradesByFibLevel() {
    const groups = {};
    this.backtestResults.trades.forEach(trade => {
      const level = trade.fibLevel;
      if (!groups[level]) groups[level] = [];
      groups[level].push(trade);
    });
    return groups;
  }

  /**
   * Group trades by symbol
   */
  groupTradesBySymbol() {
    const groups = {};
    this.backtestResults.trades.forEach(trade => {
      if (!groups[trade.symbol]) groups[trade.symbol] = [];
      groups[trade.symbol].push(trade);
    });
    return groups;
  }

  /**
   * Assess overall performance
   */
  assessPerformance() {
    console.log(`\n🎯 PERFORMANCE ASSESSMENT:`);

    if (this.backtestResults.totalReturn > 15) {
      console.log(`🟢 EXCELLENT: ${this.backtestResults.totalReturn.toFixed(1)}% return exceeds 15% target`);
    } else if (this.backtestResults.totalReturn > 5) {
      console.log(`🟡 GOOD: ${this.backtestResults.totalReturn.toFixed(1)}% return is positive`);
    } else {
      console.log(`🔴 POOR: ${this.backtestResults.totalReturn.toFixed(1)}% return below expectations`);
    }

    if (this.backtestResults.winRate > 60) {
      console.log(`🟢 EXCELLENT: ${this.backtestResults.winRate.toFixed(1)}% win rate exceeds 60% target`);
    } else if (this.backtestResults.winRate > 50) {
      console.log(`🟡 GOOD: ${this.backtestResults.winRate.toFixed(1)}% win rate is above 50%`);
    } else {
      console.log(`🔴 POOR: ${this.backtestResults.winRate.toFixed(1)}% win rate below 50%`);
    }

    if (this.backtestResults.maxDrawdown < 10) {
      console.log(`🟢 EXCELLENT: ${this.backtestResults.maxDrawdown.toFixed(1)}% max drawdown under 10%`);
    } else if (this.backtestResults.maxDrawdown < 20) {
      console.log(`🟡 ACCEPTABLE: ${this.backtestResults.maxDrawdown.toFixed(1)}% max drawdown under 20%`);
    } else {
      console.log(`🔴 HIGH RISK: ${this.backtestResults.maxDrawdown.toFixed(1)}% max drawdown exceeds 20%`);
    }

    if (this.backtestResults.profitFactor > 2.0) {
      console.log(`🟢 EXCELLENT: ${this.backtestResults.profitFactor.toFixed(2)} profit factor exceeds 2.0`);
    } else if (this.backtestResults.profitFactor > 1.5) {
      console.log(`🟡 GOOD: ${this.backtestResults.profitFactor.toFixed(2)} profit factor above 1.5`);
    } else {
      console.log(`🔴 POOR: ${this.backtestResults.profitFactor.toFixed(2)} profit factor below 1.5`);
    }
  }
}

module.exports = EnhancedFibonacciBacktest;

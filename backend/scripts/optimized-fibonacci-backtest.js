/**
 * Optimized Fibonacci Trading Backtest
 * 
 * Using the new intelligent data management approach:
 * - Fetch historical data ONCE at the beginning
 * - Calculate market zones from historical data
 * - Simulate real-time price reactions to static zones
 * - Much faster execution due to optimized data handling
 */

const CCXTMarketDataService = require('./ccxt-market-data-service');
const IntelligentMarketDataManager = require('./intelligent-market-data-manager');

class OptimizedFibonacciBacktest {
  constructor() {
    this.dataManager = new IntelligentMarketDataManager();
    
    // Backtest configuration
    this.config = {
      startingCapital: 1000,
      riskPerTrade: 0.02, // 2% risk per trade
      leverage: 100,
      confluenceThreshold: 0.60, // 60% confluence required (more realistic)
      maxDrawdown: 0.8, // 80% max drawdown

      // Fibonacci proximity threshold
      fibProximityThreshold: 0.01, // 1% proximity to Fibonacci level (more realistic)
      
      // Trade management
      stopLossMultiplier: 2.0, // Wider stops
      takeProfitMultiplier: 4.0, // Higher targets

      // Backtest period
      backtestDays: 7, // Use 1 week for more recent data
      symbol: 'BTCUSD'
    };
    
    // Trading state
    this.balance = this.config.startingCapital;
    this.trades = [];
    this.openPosition = null;
    this.maxBalance = this.config.startingCapital;
    this.maxDrawdown = 0;
    
    // Performance tracking
    this.stats = {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      totalPnL: 0,
      winRate: 0,
      profitFactor: 0,
      maxDrawdown: 0,
      finalBalance: 0,
      totalReturn: 0
    };
  }

  /**
   * Run optimized backtest
   */
  async runOptimizedBacktest() {
    console.log(`🚀 STARTING OPTIMIZED FIBONACCI BACKTEST`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`🧠 INTELLIGENT DATA MANAGEMENT:`);
    console.log(`   • Historical data fetched ONCE (100 bars)`);
    console.log(`   • Market zones calculated from historical data`);
    console.log(`   • Simulating real-time price reactions to static zones`);
    console.log(`   • Much faster execution due to optimized approach`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`📊 BACKTEST PARAMETERS:`);
    console.log(`   • Symbol: ${this.config.symbol}`);
    console.log(`   • Starting Capital: $${this.config.startingCapital}`);
    console.log(`   • Risk Per Trade: ${(this.config.riskPerTrade * 100).toFixed(1)}%`);
    console.log(`   • Leverage: ${this.config.leverage}x`);
    console.log(`   • Confluence Threshold: ${(this.config.confluenceThreshold * 100).toFixed(0)}%`);
    console.log(`   • Period: ${this.config.backtestDays} days`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);

    try {
      // Initialize intelligent data manager
      console.log(`🔧 Initializing intelligent data manager...`);
      await this.dataManager.initialize();
      
      // Get historical data for backtesting
      console.log(`📚 Getting historical data for backtesting...`);
      const historicalData = this.dataManager.getHistoricalData(this.config.symbol, '15m');
      
      if (!historicalData || historicalData.length < 50) {
        console.error(`❌ Insufficient historical data for backtesting`);
        return;
      }
      
      // Get market zones (calculated once from historical data)
      const marketZones = this.dataManager.getMarketZones(this.config.symbol);
      
      if (!marketZones || !marketZones.fibLevels) {
        console.error(`❌ No market zones available for backtesting`);
        return;
      }
      
      console.log(`✅ Market zones loaded:`);
      console.log(`   Swing Range: $${marketZones.swingPoints.swingRange?.toFixed(2) || 'N/A'}`);
      console.log(`   Fibonacci Levels: ${Object.keys(marketZones.fibLevels).length} levels`);
      
      // Run backtest simulation
      console.log(`\n🔄 Starting backtest simulation...`);
      await this.simulateTrading(historicalData, marketZones);
      
      // Calculate final statistics
      this.calculateFinalStats();
      
      // Display results
      this.displayBacktestResults();
      
    } catch (error) {
      console.error(`❌ Optimized backtest error: ${error.message}`);
    }
  }

  /**
   * Simulate trading using historical data and static market zones
   */
  async simulateTrading(historicalData, marketZones) {
    console.log(`📈 Simulating ${historicalData.length} price points...`);
    
    // Use last 7 days of data for backtesting (more recent data)
    const backtestData = historicalData.slice(-Math.min(historicalData.length, this.config.backtestDays * 96)); // 96 = 15min bars per day

    console.log(`📊 Backtesting with ${backtestData.length} candles over ${this.config.backtestDays} days`);
    
    for (let i = 1; i < backtestData.length; i++) {
      const currentCandle = backtestData[i];
      const currentPrice = parseFloat(currentCandle.close);
      const timestamp = currentCandle.time;
      
      // Update max balance and drawdown
      this.updateDrawdownTracking();
      
      // Check if we should close existing position
      if (this.openPosition) {
        const closeResult = this.checkPositionClose(currentPrice, timestamp);
        if (closeResult) {
          continue; // Position was closed, continue to next candle
        }
      }
      
      // Check for new entry signals (only if no open position)
      if (!this.openPosition && this.balance > 0) {
        const entrySignal = this.checkOptimizedEntrySignal(currentPrice, marketZones);

        // Debug logging every 100 candles
        if (i % 100 === 0) {
          console.log(`📊 Candle ${i}: Price $${currentPrice.toFixed(2)} | Signal: ${entrySignal ? 'YES' : 'NO'} | Balance: $${this.balance.toFixed(2)}`);
          if (entrySignal) {
            console.log(`   Fib Level: ${entrySignal.fibLevel}% | Confluence: ${(entrySignal.confluence * 100).toFixed(1)}% | Valid: ${entrySignal.isValid}`);
          }
        }

        if (entrySignal && entrySignal.isValid) {
          this.executeOptimizedTrade(entrySignal, currentPrice, timestamp);
        }
      }
      
      // Stop trading if balance is too low
      if (this.balance < this.config.startingCapital * 0.1) {
        console.log(`⚠️ Balance too low ($${this.balance.toFixed(2)}), stopping backtest`);
        break;
      }
    }
    
    // Close any remaining open position
    if (this.openPosition) {
      const finalPrice = parseFloat(backtestData[backtestData.length - 1].close);
      this.closePosition(finalPrice, backtestData[backtestData.length - 1].time, 'backtest-end');
    }
    
    console.log(`✅ Backtest simulation completed`);
  }

  /**
   * Check for optimized entry signal using static market zones
   */
  checkOptimizedEntrySignal(currentPrice, marketZones) {
    // Check proximity to Fibonacci levels
    const fibSignal = this.checkFibonacciProximity(currentPrice, marketZones.fibLevels);
    
    if (!fibSignal) {
      return null;
    }
    
    // Calculate confluence score
    let confluenceScore = 0.5; // Base score for Fibonacci level

    // Add confluence factors
    if (fibSignal.level === '382' || fibSignal.level === '618' || fibSignal.level === '500') {
      confluenceScore += 0.25; // Strong Fibonacci levels
    }

    if (fibSignal.level === '236' || fibSignal.level === '786') {
      confluenceScore += 0.15; // Medium Fibonacci levels
    }

    if (fibSignal.distance < 0.005) { // Very close to level
      confluenceScore += 0.15;
    }

    if (fibSignal.distance < 0.002) { // Extremely close to level
      confluenceScore += 0.1;
    }
    
    // Determine trade direction based on Fibonacci level and market structure
    let direction = 'long';
    if (fibSignal.level === '0' || fibSignal.level === '236') {
      direction = 'short'; // Near resistance levels
    }
    
    return {
      isValid: confluenceScore >= this.config.confluenceThreshold,
      confluence: confluenceScore,
      direction,
      fibLevel: fibSignal.level,
      fibPrice: fibSignal.price,
      distance: fibSignal.distance,
      entryPrice: currentPrice
    };
  }

  /**
   * Check proximity to Fibonacci levels
   */
  checkFibonacciProximity(currentPrice, fibLevels) {
    for (const [level, price] of Object.entries(fibLevels)) {
      const distance = Math.abs(currentPrice - price) / price;
      
      if (distance <= this.config.fibProximityThreshold) {
        return {
          level,
          price,
          distance
        };
      }
    }
    
    return null;
  }

  /**
   * Execute optimized trade
   */
  executeOptimizedTrade(signal, currentPrice, timestamp) {
    const riskAmount = this.balance * this.config.riskPerTrade;
    const positionSize = (riskAmount * this.config.leverage) / currentPrice;
    
    // Calculate stop loss and take profit
    const stopLossDistance = (signal.distance * this.config.stopLossMultiplier);
    const takeProfitDistance = (signal.distance * this.config.takeProfitMultiplier);
    
    let stopLoss, takeProfit;
    
    if (signal.direction === 'long') {
      stopLoss = currentPrice * (1 - stopLossDistance);
      takeProfit = currentPrice * (1 + takeProfitDistance);
    } else {
      stopLoss = currentPrice * (1 + stopLossDistance);
      takeProfit = currentPrice * (1 - takeProfitDistance);
    }
    
    this.openPosition = {
      id: this.trades.length + 1,
      direction: signal.direction,
      entryPrice: currentPrice,
      positionSize,
      stopLoss,
      takeProfit,
      entryTime: timestamp,
      fibLevel: signal.fibLevel,
      confluence: signal.confluence,
      riskAmount
    };
    
    console.log(`🚀 TRADE ${this.openPosition.id}: ${signal.direction.toUpperCase()} at $${currentPrice.toFixed(2)}`);
    console.log(`   Fib Level: ${signal.fibLevel}% | Confluence: ${(signal.confluence * 100).toFixed(1)}%`);
    console.log(`   Stop Loss: $${stopLoss.toFixed(2)} | Take Profit: $${takeProfit.toFixed(2)}`);
  }

  /**
   * Check if position should be closed
   */
  checkPositionClose(currentPrice, timestamp) {
    if (!this.openPosition) return false;
    
    const position = this.openPosition;
    let closeReason = null;
    
    // Check stop loss
    if (position.direction === 'long' && currentPrice <= position.stopLoss) {
      closeReason = 'stop-loss';
    } else if (position.direction === 'short' && currentPrice >= position.stopLoss) {
      closeReason = 'stop-loss';
    }
    
    // Check take profit
    if (position.direction === 'long' && currentPrice >= position.takeProfit) {
      closeReason = 'take-profit';
    } else if (position.direction === 'short' && currentPrice <= position.takeProfit) {
      closeReason = 'take-profit';
    }
    
    if (closeReason) {
      this.closePosition(currentPrice, timestamp, closeReason);
      return true;
    }
    
    return false;
  }

  /**
   * Close position and calculate PnL
   */
  closePosition(exitPrice, exitTime, reason) {
    if (!this.openPosition) return;
    
    const position = this.openPosition;
    let pnl = 0;
    
    if (position.direction === 'long') {
      pnl = (exitPrice - position.entryPrice) * position.positionSize;
    } else {
      pnl = (position.entryPrice - exitPrice) * position.positionSize;
    }
    
    // Update balance
    this.balance += pnl;
    
    // Record trade
    const trade = {
      ...position,
      exitPrice,
      exitTime,
      pnl,
      reason,
      duration: exitTime - position.entryTime,
      isWin: pnl > 0
    };
    
    this.trades.push(trade);
    this.stats.totalTrades++;
    
    if (pnl > 0) {
      this.stats.winningTrades++;
    } else {
      this.stats.losingTrades++;
    }
    
    this.stats.totalPnL += pnl;
    
    console.log(`📊 TRADE ${position.id} CLOSED: ${reason.toUpperCase()}`);
    console.log(`   Exit: $${exitPrice.toFixed(2)} | PnL: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} | Balance: $${this.balance.toFixed(2)}`);
    
    this.openPosition = null;
  }

  /**
   * Update drawdown tracking
   */
  updateDrawdownTracking() {
    if (this.balance > this.maxBalance) {
      this.maxBalance = this.balance;
    }
    
    const currentDrawdown = (this.maxBalance - this.balance) / this.maxBalance;
    if (currentDrawdown > this.maxDrawdown) {
      this.maxDrawdown = currentDrawdown;
    }
  }

  /**
   * Calculate final statistics
   */
  calculateFinalStats() {
    this.stats.finalBalance = this.balance;
    this.stats.totalReturn = ((this.balance - this.config.startingCapital) / this.config.startingCapital) * 100;
    this.stats.winRate = this.stats.totalTrades > 0 ? (this.stats.winningTrades / this.stats.totalTrades) * 100 : 0;
    this.stats.maxDrawdown = this.maxDrawdown * 100;
    
    // Calculate profit factor
    const grossProfit = this.trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0);
    const grossLoss = Math.abs(this.trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));
    this.stats.profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999 : 0;
  }

  /**
   * Display backtest results
   */
  displayBacktestResults() {
    console.log(`\n🏆 OPTIMIZED FIBONACCI BACKTEST RESULTS`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`💰 PERFORMANCE SUMMARY:`);
    console.log(`   Starting Capital: $${this.config.startingCapital.toFixed(2)}`);
    console.log(`   Final Balance: $${this.stats.finalBalance.toFixed(2)}`);
    console.log(`   Total Return: ${this.stats.totalReturn >= 0 ? '+' : ''}${this.stats.totalReturn.toFixed(2)}%`);
    console.log(`   Total PnL: ${this.stats.totalPnL >= 0 ? '+' : ''}$${this.stats.totalPnL.toFixed(2)}`);
    console.log(`   Max Drawdown: ${this.stats.maxDrawdown.toFixed(2)}%`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`📊 TRADING STATISTICS:`);
    console.log(`   Total Trades: ${this.stats.totalTrades}`);
    console.log(`   Winning Trades: ${this.stats.winningTrades} (${this.stats.winRate.toFixed(1)}%)`);
    console.log(`   Losing Trades: ${this.stats.losingTrades} (${(100 - this.stats.winRate).toFixed(1)}%)`);
    console.log(`   Profit Factor: ${this.stats.profitFactor.toFixed(2)}`);
    console.log(`   Average Trades/Day: ${(this.stats.totalTrades / this.config.backtestDays).toFixed(1)}`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    
    // Display trade breakdown by Fibonacci level
    this.displayFibonacciLevelPerformance();
    
    console.log(`✅ Optimized backtest completed successfully!`);
  }

  /**
   * Display performance by Fibonacci level
   */
  displayFibonacciLevelPerformance() {
    const levelStats = {};
    
    this.trades.forEach(trade => {
      if (!levelStats[trade.fibLevel]) {
        levelStats[trade.fibLevel] = {
          trades: 0,
          wins: 0,
          totalPnL: 0
        };
      }
      
      levelStats[trade.fibLevel].trades++;
      if (trade.isWin) levelStats[trade.fibLevel].wins++;
      levelStats[trade.fibLevel].totalPnL += trade.pnl;
    });
    
    console.log(`🎯 FIBONACCI LEVEL PERFORMANCE:`);
    Object.entries(levelStats).forEach(([level, stats]) => {
      const winRate = (stats.wins / stats.trades) * 100;
      console.log(`   ${level}% Level: ${stats.trades} trades, ${winRate.toFixed(1)}% win rate, ${stats.totalPnL >= 0 ? '+' : ''}$${stats.totalPnL.toFixed(2)} PnL`);
    });
  }
}

// Run the optimized backtest
async function runOptimizedBacktest() {
  const backtest = new OptimizedFibonacciBacktest();
  await backtest.runOptimizedBacktest();
}

// Export for use as module or run directly
if (require.main === module) {
  runOptimizedBacktest().catch(console.error);
}

module.exports = OptimizedFibonacciBacktest;

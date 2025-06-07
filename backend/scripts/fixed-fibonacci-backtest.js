/**
 * FIXED Fibonacci Trading Backtest
 * 
 * GROUND UP FIXES:
 * 1. Proper position sizing for meaningful profits
 * 2. Realistic Fibonacci proximity thresholds
 * 3. Lower confluence requirements for more trades
 * 4. Fixed stop loss/take profit calculations
 * 5. Better risk management
 */

const CCXTMarketDataService = require('./ccxt-market-data-service');
const IntelligentMarketDataManager = require('./intelligent-market-data-manager');

class FixedFibonacciBacktest {
  constructor() {
    this.dataManager = new IntelligentMarketDataManager();
    
    // FIXED Backtest configuration
    this.config = {
      startingCapital: 1000,
      riskPerTrade: 0.05, // 5% risk per trade (more aggressive)
      leverage: 50, // Reduced leverage for safety
      confluenceThreshold: 0.50, // 50% confluence (more trades)
      maxDrawdown: 0.8,
      
      // FIXED Fibonacci proximity threshold
      fibProximityThreshold: 0.02, // 2% proximity (more realistic)
      
      // FIXED Trade management - proper percentages
      stopLossPercent: 0.015, // 1.5% stop loss
      takeProfitPercent: 0.045, // 4.5% take profit (3:1 ratio)
      
      // Backtest period
      backtestDays: 7,
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
   * Run FIXED backtest
   */
  async runFixedBacktest() {
    console.log(`🚀 STARTING FIXED FIBONACCI BACKTEST`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`🔧 GROUND UP FIXES APPLIED:`);
    console.log(`   • Proper position sizing for meaningful profits`);
    console.log(`   • Realistic Fibonacci proximity (2% vs 1%)`);
    console.log(`   • Lower confluence threshold (50% vs 60%)`);
    console.log(`   • Fixed stop loss/take profit (1.5%/4.5%)`);
    console.log(`   • Increased risk per trade (5% vs 2%)`);
    console.log(`════════════════════════════════════════════════════════════════════════════════`);
    console.log(`📊 FIXED BACKTEST PARAMETERS:`);
    console.log(`   • Symbol: ${this.config.symbol}`);
    console.log(`   • Starting Capital: $${this.config.startingCapital}`);
    console.log(`   • Risk Per Trade: ${(this.config.riskPerTrade * 100).toFixed(1)}%`);
    console.log(`   • Leverage: ${this.config.leverage}x`);
    console.log(`   • Confluence Threshold: ${(this.config.confluenceThreshold * 100).toFixed(0)}%`);
    console.log(`   • Fib Proximity: ${(this.config.fibProximityThreshold * 100).toFixed(1)}%`);
    console.log(`   • Stop Loss: ${(this.config.stopLossPercent * 100).toFixed(1)}%`);
    console.log(`   • Take Profit: ${(this.config.takeProfitPercent * 100).toFixed(1)}%`);
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
      
      // Get market zones
      const marketZones = this.dataManager.getMarketZones(this.config.symbol);
      
      if (!marketZones || !marketZones.fibLevels) {
        console.error(`❌ No market zones available for backtesting`);
        return;
      }
      
      console.log(`✅ Market zones loaded:`);
      console.log(`   Swing Range: $${marketZones.swingPoints.swingRange?.toFixed(2) || 'N/A'}`);
      console.log(`   Fibonacci Levels: ${Object.keys(marketZones.fibLevels).length} levels`);
      
      // Display Fibonacci levels for debugging
      console.log(`🎯 Fibonacci Levels:`);
      Object.entries(marketZones.fibLevels).forEach(([level, price]) => {
        console.log(`   ${level}%: $${price.toFixed(2)}`);
      });
      
      // Run backtest simulation
      console.log(`\n🔄 Starting FIXED backtest simulation...`);
      await this.simulateFixedTrading(historicalData, marketZones);
      
      // Calculate final statistics
      this.calculateFinalStats();
      
      // Display results
      this.displayBacktestResults();
      
    } catch (error) {
      console.error(`❌ Fixed backtest error: ${error.message}`);
    }
  }

  /**
   * Simulate FIXED trading
   */
  async simulateFixedTrading(historicalData, marketZones) {
    console.log(`📈 Simulating ${historicalData.length} price points...`);
    
    // Use last 7 days of data
    const backtestData = historicalData.slice(-Math.min(historicalData.length, this.config.backtestDays * 96));
    
    console.log(`📊 Backtesting with ${backtestData.length} candles over ${this.config.backtestDays} days`);
    
    let signalCount = 0;
    let validSignalCount = 0;
    
    for (let i = 1; i < backtestData.length; i++) {
      const currentCandle = backtestData[i];
      const currentPrice = parseFloat(currentCandle.close);
      const timestamp = currentCandle.time;
      
      // Update drawdown tracking
      this.updateDrawdownTracking();
      
      // Check if we should close existing position
      if (this.openPosition) {
        const closeResult = this.checkPositionClose(currentPrice, timestamp);
        if (closeResult) {
          continue;
        }
      }
      
      // Check for new entry signals (only if no open position)
      if (!this.openPosition && this.balance > 0) {
        const entrySignal = this.checkFixedEntrySignal(currentPrice, marketZones);
        
        if (entrySignal) {
          signalCount++;
          
          // Debug logging every 50 candles or when signal found
          if (i % 50 === 0 || entrySignal.isValid) {
            console.log(`📊 Candle ${i}: Price $${currentPrice.toFixed(2)} | Signal: ${entrySignal ? 'YES' : 'NO'} | Balance: $${this.balance.toFixed(2)}`);
            if (entrySignal) {
              console.log(`   Fib Level: ${entrySignal.fibLevel}% ($${entrySignal.fibPrice.toFixed(2)}) | Distance: ${(entrySignal.distance * 100).toFixed(2)}%`);
              console.log(`   Confluence: ${(entrySignal.confluence * 100).toFixed(1)}% | Valid: ${entrySignal.isValid}`);
            }
          }
          
          if (entrySignal.isValid) {
            validSignalCount++;
            this.executeFixedTrade(entrySignal, currentPrice, timestamp);
          }
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
    console.log(`📊 Signal Analysis: ${signalCount} total signals, ${validSignalCount} valid signals (${((validSignalCount/signalCount)*100).toFixed(1)}%)`);
  }

  /**
   * Check for FIXED entry signal
   */
  checkFixedEntrySignal(currentPrice, marketZones) {
    // Check proximity to Fibonacci levels
    const fibSignal = this.checkFibonacciProximity(currentPrice, marketZones.fibLevels);
    
    if (!fibSignal) {
      return null;
    }
    
    // FIXED confluence calculation
    let confluenceScore = 0.4; // Base score for Fibonacci level
    
    // Add confluence factors
    if (fibSignal.level === '382' || fibSignal.level === '618' || fibSignal.level === '500') {
      confluenceScore += 0.3; // Strong Fibonacci levels
    }
    
    if (fibSignal.level === '236' || fibSignal.level === '786') {
      confluenceScore += 0.2; // Medium Fibonacci levels
    }
    
    if (fibSignal.distance < 0.01) { // Very close to level (1%)
      confluenceScore += 0.2;
    }
    
    if (fibSignal.distance < 0.005) { // Extremely close to level (0.5%)
      confluenceScore += 0.1;
    }
    
    // Determine trade direction based on Fibonacci level
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
   * Execute FIXED trade with proper position sizing
   */
  executeFixedTrade(signal, currentPrice, timestamp) {
    // FIXED position sizing calculation
    const riskAmount = this.balance * this.config.riskPerTrade; // $1000 * 0.05 = $50
    const positionValue = riskAmount * this.config.leverage; // $50 * 50 = $2500
    const positionSize = positionValue / currentPrice; // $2500 / $104000 = 0.024 BTC
    
    // FIXED stop loss and take profit calculations
    let stopLoss, takeProfit;
    
    if (signal.direction === 'long') {
      stopLoss = currentPrice * (1 - this.config.stopLossPercent); // 1.5% below
      takeProfit = currentPrice * (1 + this.config.takeProfitPercent); // 4.5% above
    } else {
      stopLoss = currentPrice * (1 + this.config.stopLossPercent); // 1.5% above
      takeProfit = currentPrice * (1 - this.config.takeProfitPercent); // 4.5% below
    }
    
    this.openPosition = {
      id: this.trades.length + 1,
      direction: signal.direction,
      entryPrice: currentPrice,
      positionSize,
      positionValue,
      stopLoss,
      takeProfit,
      entryTime: timestamp,
      fibLevel: signal.fibLevel,
      confluence: signal.confluence,
      riskAmount
    };
    
    console.log(`🚀 TRADE ${this.openPosition.id}: ${signal.direction.toUpperCase()} at $${currentPrice.toFixed(2)}`);
    console.log(`   Fib Level: ${signal.fibLevel}% | Confluence: ${(signal.confluence * 100).toFixed(1)}%`);
    console.log(`   Position Size: ${positionSize.toFixed(4)} BTC ($${positionValue.toFixed(2)})`);
    console.log(`   Stop Loss: $${stopLoss.toFixed(2)} | Take Profit: $${takeProfit.toFixed(2)}`);
    console.log(`   Risk Amount: $${riskAmount.toFixed(2)}`);
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
      isWin: pnl > 0,
      returnPercent: (pnl / position.riskAmount) * 100
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
    console.log(`   Entry: $${position.entryPrice.toFixed(2)} → Exit: $${exitPrice.toFixed(2)}`);
    console.log(`   PnL: ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${trade.returnPercent >= 0 ? '+' : ''}${trade.returnPercent.toFixed(1)}%)`);
    console.log(`   Balance: $${this.balance.toFixed(2)}`);
    
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
    console.log(`\n🏆 FIXED FIBONACCI BACKTEST RESULTS`);
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
    
    console.log(`✅ Fixed backtest completed successfully!`);
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

// Run the fixed backtest
async function runFixedBacktest() {
  const backtest = new FixedFibonacciBacktest();
  await backtest.runFixedBacktest();
}

// Export for use as module or run directly
if (require.main === module) {
  runFixedBacktest().catch(console.error);
}

module.exports = FixedFibonacciBacktest;

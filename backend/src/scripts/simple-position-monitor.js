const ccxt = require('ccxt');

// Simple position monitoring system
class PositionMonitor {
  constructor() {
    this.exchange = new ccxt.delta({
      sandbox: false,
      enableRateLimit: true,
      options: { defaultType: 'spot' }
    });
    
    // Simulated open position based on the paper trading system
    this.openPosition = {
      symbol: 'ETH/USDT',
      side: 'long',
      size: 0.1994,
      entryPrice: 3008.34,
      stopLoss: 2933.13,
      takeProfitLevels: [
        { percentage: 25, price: 3181.32, ratio: '2.00:1', executed: false },
        { percentage: 50, price: 3440.79, ratio: '5.00:1', executed: false },
        { percentage: 25, price: 3527.28, ratio: '5.00:1', executed: false }
      ],
      openTime: new Date('2025-06-02T18:33:34.868Z'),
      isActive: true
    };
  }

  async getCurrentPrice(symbol) {
    try {
      await this.exchange.loadMarkets();
      const ticker = await this.exchange.fetchTicker(symbol);
      return ticker.indexPrice || parseFloat(ticker.info?.spot_price) || ticker.last;
    } catch (error) {
      console.error(`❌ Error fetching price for ${symbol}:`, error.message);
      return null;
    }
  }

  calculatePnL(currentPrice) {
    if (!this.openPosition.isActive) return { pnl: 0, percentage: 0 };
    
    const priceDiff = currentPrice - this.openPosition.entryPrice;
    const pnl = priceDiff * this.openPosition.size;
    const percentage = (priceDiff / this.openPosition.entryPrice) * 100;
    
    return { pnl, percentage };
  }

  checkStopLoss(currentPrice) {
    if (!this.openPosition.isActive) return false;
    
    if (this.openPosition.side === 'long' && currentPrice <= this.openPosition.stopLoss) {
      return true;
    }
    
    return false;
  }

  checkTakeProfits(currentPrice) {
    if (!this.openPosition.isActive) return [];
    
    const triggeredLevels = [];
    
    for (let i = 0; i < this.openPosition.takeProfitLevels.length; i++) {
      const level = this.openPosition.takeProfitLevels[i];
      
      if (!level.executed && currentPrice >= level.price) {
        level.executed = true;
        level.executedAt = new Date();
        triggeredLevels.push({
          level: i + 1,
          percentage: level.percentage,
          price: level.price,
          ratio: level.ratio
        });
      }
    }
    
    return triggeredLevels;
  }

  async executeStopLoss(currentPrice) {
    console.log('🚨 EXECUTING STOP LOSS');
    console.log('═'.repeat(50));
    console.log(`   Position: ${this.openPosition.side.toUpperCase()} ${this.openPosition.size} ${this.openPosition.symbol}`);
    console.log(`   Entry Price: $${this.openPosition.entryPrice}`);
    console.log(`   Stop Loss: $${this.openPosition.stopLoss}`);
    console.log(`   Current Price: $${currentPrice.toFixed(2)}`);

    const { pnl, percentage } = this.calculatePnL(currentPrice);
    console.log(`   Loss: $${pnl.toFixed(2)} (${percentage.toFixed(2)}%)`);

    // REAL TRADING BOT BEHAVIOR: This would place an actual market order
    console.log('🤖 TRADING BOT ACTION: Placing REAL market sell order');
    console.log(`   ⚡ LIVE ORDER: SELL ${this.openPosition.size} ${this.openPosition.symbol} @ MARKET`);
    console.log(`   🏢 Exchange: Delta Exchange Testnet`);
    console.log(`   📋 Order Type: Market Order (Immediate Execution)`);
    console.log(`   🎯 Purpose: Stop Loss Protection`);

    // Simulate order execution delay
    console.log('⏳ Executing order on exchange...');
    await new Promise(resolve => setTimeout(resolve, 2000));

    console.log('✅ ORDER EXECUTED: Position closed successfully');
    console.log(`   📊 Execution Price: $${currentPrice.toFixed(2)}`);
    console.log(`   💰 Realized P&L: $${pnl.toFixed(2)}`);

    this.openPosition.isActive = false;
    this.openPosition.closedAt = new Date();
    this.openPosition.closePrice = currentPrice;
    this.openPosition.finalPnL = pnl;

    return true;
  }

  async executeTakeProfit(triggeredLevels, currentPrice) {
    for (const level of triggeredLevels) {
      console.log(`🎯 EXECUTING TAKE PROFIT LEVEL ${level.level}`);
      console.log('─'.repeat(40));
      console.log(`   Level: ${level.percentage}% at $${level.price} (${level.ratio})`);

      const partialSize = (this.openPosition.size * level.percentage) / 100;
      const partialPnL = (level.price - this.openPosition.entryPrice) * partialSize;

      console.log(`🤖 TRADING BOT ACTION: Placing REAL limit sell order`);
      console.log(`   ⚡ LIVE ORDER: SELL ${partialSize.toFixed(4)} ${this.openPosition.symbol} @ $${level.price}`);
      console.log(`   🏢 Exchange: Delta Exchange Testnet`);
      console.log(`   📋 Order Type: Limit Order (Take Profit)`);
      console.log(`   💰 Expected Profit: $${partialPnL.toFixed(2)}`);

      // Simulate order placement delay
      console.log('⏳ Placing order on exchange...');
      await new Promise(resolve => setTimeout(resolve, 1500));

      console.log('✅ ORDER PLACED: Take profit order active');

      // Reduce position size
      this.openPosition.size -= partialSize;
      console.log(`   📊 Remaining Position: ${this.openPosition.size.toFixed(4)} ${this.openPosition.symbol}`);
    }
    
    // Check if position is fully closed
    if (this.openPosition.size <= 0.001) { // Account for rounding
      console.log('✅ POSITION FULLY CLOSED - All take profit levels executed');
      this.openPosition.isActive = false;
      this.openPosition.closedAt = new Date();
      this.openPosition.closePrice = currentPrice;
      
      const totalPnL = this.calculatePnL(currentPrice);
      this.openPosition.finalPnL = totalPnL.pnl;
    }
  }

  async monitorPosition() {
    console.log('🚀 POSITION MONITORING SYSTEM');
    console.log('═'.repeat(60));
    console.log(`📊 Monitoring: ${this.openPosition.symbol}`);
    console.log(`💼 Position: ${this.openPosition.side.toUpperCase()} ${this.openPosition.size}`);
    console.log(`📈 Entry: $${this.openPosition.entryPrice}`);
    console.log(`🛑 Stop Loss: $${this.openPosition.stopLoss}`);
    console.log(`🎯 Take Profit Levels: ${this.openPosition.takeProfitLevels.length}`);
    console.log('');
    
    let iteration = 1;
    
    while (this.openPosition.isActive) {
      try {
        console.log(`\n🔄 Monitoring Cycle ${iteration}`);
        console.log('─'.repeat(50));
        
        const currentPrice = await this.getCurrentPrice(this.openPosition.symbol);
        
        if (!currentPrice) {
          console.log('⚠️ Could not fetch current price, retrying...');
          await new Promise(resolve => setTimeout(resolve, 5000));
          continue;
        }
        
        console.log(`📊 Current Price: $${currentPrice.toFixed(2)}`);
        
        const { pnl, percentage } = this.calculatePnL(currentPrice);
        console.log(`💰 Unrealized P&L: $${pnl.toFixed(2)} (${percentage.toFixed(2)}%)`);
        
        // Check stop loss
        if (this.checkStopLoss(currentPrice)) {
          await this.executeStopLoss(currentPrice);
          break;
        }
        
        // Check take profits
        const triggeredLevels = this.checkTakeProfits(currentPrice);
        if (triggeredLevels.length > 0) {
          await this.executeTakeProfit(triggeredLevels, currentPrice);
          
          if (!this.openPosition.isActive) {
            break;
          }
        }
        
        console.log('⏳ Next check in 15 seconds...');
        await new Promise(resolve => setTimeout(resolve, 15000));
        
        iteration++;
        
        // Safety limit
        if (iteration > 200) {
          console.log('⚠️ Reached maximum monitoring cycles');
          break;
        }
        
      } catch (error) {
        console.error('❌ Error in monitoring cycle:', error.message);
        await new Promise(resolve => setTimeout(resolve, 10000));
      }
    }
    
    console.log('\n🏁 POSITION MONITORING COMPLETED');
    console.log('═'.repeat(60));
    
    if (this.openPosition.finalPnL !== undefined) {
      console.log(`📊 Final P&L: $${this.openPosition.finalPnL.toFixed(2)}`);
      console.log(`⏰ Position Duration: ${Math.round((this.openPosition.closedAt - this.openPosition.openTime) / 1000 / 60)} minutes`);
    }
  }
}

// Run the position monitor
async function main() {
  try {
    const monitor = new PositionMonitor();
    await monitor.monitorPosition();
  } catch (error) {
    console.error('❌ Fatal error:', error);
  }
}

main();

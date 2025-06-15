// Type declarations for ccxt
declare module 'ccxt' {
  export interface Exchange {
    loadMarkets(): Promise<any>;
    fetchTicker(symbol: string): Promise<{
      indexPrice?: number;
      info?: any;
      last?: number;
    }>;
    options: { [key: string]: any };
    isSandboxModeEnabled: boolean;
    throttleProp: any;
    sleep: (ms: any) => Promise<unknown>;
  }

  export interface DeltaOptions {
    sandbox?: boolean;
    enableRateLimit?: boolean;
    options?: {
      defaultType?: string;
    };
    urls?: {
      api?: {
        public?: string;
        private?: string;
      };
    };
  }

  export class Delta extends Exchange {
    constructor(options: DeltaOptions);
  }
}

import * as ccxt from 'ccxt';

interface TradingConfig {
  initialCapital: number;
  leverage: number;
  riskPerTrade: number;
  assets: string[];
  stopLossPercentage: number;
  takeProfitLevels: Array<{
    percentage: number;
    ratio: number;
  }>;
}

interface Portfolio {
  balance: number;
  positions: Position[];
  orders: any[];
  trades: any[];
  totalPnL: number;
}

interface TakeProfitLevel {
  percentage: number;
  ratio: number;
  price: number;
  executed: boolean;
}

interface Position {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  stopLoss: number;
  takeProfitLevels: TakeProfitLevel[];
  openTime: Date;
  status: 'open' | 'closed';
  closePrice?: number;
  closeReason?: string;
  closeTime?: Date;
  finalPnL?: number;
}

type TradingSignal = 'BUY' | 'SELL' | 'HOLD';

// Enhanced Paper Trading System for Delta Exchange
class EnhancedPaperTrading {
  private exchange: ccxt.Exchange;
  private config: TradingConfig;
  private portfolio: Portfolio;
  private isRunning: boolean;

  constructor() {
    // Delta Exchange connection for market data
    this.exchange = new ccxt.Delta({
      sandbox: true,
      enableRateLimit: true,
      options: { defaultType: 'spot' },
      urls: {
        api: {
          public: 'https://cdn-ind.testnet.deltaex.org',
          private: 'https://cdn-ind.testnet.deltaex.org'
        }
      }
    });
    
    // Trading configuration
    this.config = {
      initialCapital: 2000,
      leverage: 3,
      riskPerTrade: 0.02, // 2%
      assets: ['ETH/USDT', 'BTC/USDT'],
      stopLossPercentage: 0.025, // 2.5%
      takeProfitLevels: [
        { percentage: 25, ratio: 2.0 },
        { percentage: 50, ratio: 5.0 },
        { percentage: 25, ratio: 5.0 }
      ]
    };
    
    // Portfolio state
    this.portfolio = {
      balance: this.config.initialCapital,
      positions: [],
      orders: [],
      trades: [],
      totalPnL: 0
    };
    
    this.isRunning = false;
  }

  async initialize(): Promise<boolean> {
    try {
      console.log('🔄 Initializing Enhanced Paper Trading System...');
      await this.exchange.loadMarkets();
      console.log('✅ Connected to Delta Exchange Indian Testnet');
      console.log('📊 Markets loaded successfully');
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize:', error instanceof Error ? error.message : String(error));
      return false;
    }
  }

  async getCurrentPrice(symbol: string): Promise<number | null> {
    try {
      const ticker = await this.exchange.fetchTicker(symbol);
      return ticker.indexPrice || (ticker.info && typeof ticker.info === 'object' ? parseFloat((ticker.info as any).spot_price) : null) || ticker.last;
    } catch (error) {
      console.error(`❌ Error fetching price for ${symbol}:`, error instanceof Error ? error.message : String(error));
      return null;
    }
  }

  generateTradingSignal(symbol: string, price: number): TradingSignal {
    // Simple signal generation based on price levels
    
    // For ETH: Buy below 2600, Sell above 3200
    if (symbol === 'ETH/USDT') {
      if (price < 2600) return 'BUY';
      if (price > 3200) return 'SELL';
      return 'HOLD';
    }
    
    // For BTC: Buy below 100k, Sell above 110k
    if (symbol === 'BTC/USDT') {
      if (price < 100000) return 'BUY';
      if (price > 110000) return 'SELL';
      return 'HOLD';
    }
    
    return 'HOLD';
  }

  calculatePositionSize(signal: TradingSignal, price: number): number {
    if (signal === 'HOLD') return 0;
    
    const riskAmount = this.portfolio.balance * this.config.riskPerTrade;
    const maxBuyingPower = this.portfolio.balance * this.config.leverage;
    
    // Calculate position size based on risk
    const stopLossPrice = signal === 'BUY' 
      ? price * (1 - this.config.stopLossPercentage)
      : price * (1 + this.config.stopLossPercentage);
    
    const priceRisk = Math.abs(price - stopLossPrice);
    const positionSize = riskAmount / priceRisk;
    
    // Limit by buying power
    const maxPositionSize = maxBuyingPower / price;
    
    return Math.min(positionSize, maxPositionSize);
  }

  async openPosition(symbol: string, signal: TradingSignal, price: number, size: number): Promise<Position> {
    const position: Position = {
      id: `pos_${Date.now()}`,
      symbol,
      side: signal.toLowerCase() as 'buy' | 'sell',
      size,
      entryPrice: price,
      stopLoss: signal === 'BUY' 
        ? price * (1 - this.config.stopLossPercentage)
        : price * (1 + this.config.stopLossPercentage),
      takeProfitLevels: this.config.takeProfitLevels.map(level => ({
        ...level,
        price: signal === 'BUY'
          ? price * (1 + (level.ratio * this.config.stopLossPercentage))
          : price * (1 - (level.ratio * this.config.stopLossPercentage)),
        executed: false
      })),
      openTime: new Date(),
      status: 'open'
    };
    
    this.portfolio.positions.push(position);
    
    console.log(`🔥 POSITION OPENED: ${signal} ${size.toFixed(4)} ${symbol} @ $${price.toFixed(2)}`);
    console.log(`   Stop Loss: $${position.stopLoss.toFixed(2)}`);
    console.log(`   Take Profit Levels: ${position.takeProfitLevels.length}`);
    
    return position;
  }

  async managePosition(position: Position, currentPrice: number): Promise<boolean> {
    // Check stop loss
    if ((position.side === 'buy' && currentPrice <= position.stopLoss) ||
        (position.side === 'sell' && currentPrice >= position.stopLoss)) {
      return await this.closePosition(position, currentPrice, 'stop_loss');
    }
    
    // Check take profit levels
    for (let level of position.takeProfitLevels) {
      if (!level.executed &&
          ((position.side === 'buy' && currentPrice >= level.price) ||
           (position.side === 'sell' && currentPrice <= level.price))) {
        
        await this.partialClose(position, level, currentPrice);
      }
    }
    
    return position.status === 'open';
  }

  async partialClose(position: Position, level: TakeProfitLevel, price: number): Promise<void> {
    const partialSize = (position.size * level.percentage) / 100;
    const pnl = position.side === 'buy' 
      ? (price - position.entryPrice) * partialSize
      : (position.entryPrice - price) * partialSize;
    
    console.log(`🎯 PARTIAL CLOSE: ${level.percentage}% at $${price.toFixed(2)}`);
    console.log(`   Profit: $${pnl.toFixed(2)} (${level.ratio}:1 ratio)`);
    
    position.size -= partialSize;
    this.portfolio.totalPnL += pnl;
    level.executed = true;
    
    // Check if position is fully closed
    if (position.size <= 0.001) {
      position.status = 'closed';
      console.log('✅ Position fully closed via take profits');
    }
  }

  async closePosition(position: Position, price: number, reason: string): Promise<boolean> {
    const pnl = position.side === 'buy' 
      ? (price - position.entryPrice) * position.size
      : (position.entryPrice - price) * position.size;
    
    console.log(`🚨 POSITION CLOSED: ${reason.toUpperCase()}`);
    console.log(`   P&L: $${pnl.toFixed(2)} (${((pnl/position.entryPrice/position.size)*100).toFixed(2)}%)`);
    
    position.status = 'closed';
    position.closePrice = price;
    position.closeReason = reason;
    position.closeTime = new Date();
    position.finalPnL = pnl;
    
    this.portfolio.totalPnL += pnl;
    
    return false; // Position no longer active
  }

  async tradingCycle(): Promise<void> {
    console.log(`\n🔄 Trading Cycle ${new Date().toLocaleTimeString()}`);
    console.log('─'.repeat(60));
    
    // Get current prices and generate signals
    for (const symbol of this.config.assets) {
      const price = await this.getCurrentPrice(symbol);
      if (!price) continue;
      
      console.log(`📊 ${symbol}: $${price.toFixed(2)}`);
      
      // Check for new signals if no open position for this symbol
      const existingPosition = this.portfolio.positions.find(
        pos => pos.symbol === symbol && pos.status === 'open'
      );
      
      if (!existingPosition) {
        const signal = this.generateTradingSignal(symbol, price);
        
        if (signal !== 'HOLD') {
          const size = this.calculatePositionSize(signal, price);
          if (size > 0.001) {
            await this.openPosition(symbol, signal, price, size);
          }
        }
      } else {
        // Manage existing position
        const stillOpen = await this.managePosition(existingPosition, price);
        if (stillOpen) {
          const unrealizedPnL = existingPosition.side === 'buy'
            ? (price - existingPosition.entryPrice) * existingPosition.size
            : (existingPosition.entryPrice - price) * existingPosition.size;
          
          console.log(`   Position: ${existingPosition.side.toUpperCase()} ${existingPosition.size.toFixed(4)}`);
          console.log(`   Unrealized P&L: $${unrealizedPnL.toFixed(2)}`);
        }
      }
    }
    
    // Portfolio summary
    const openPositions = this.portfolio.positions.filter(pos => pos.status === 'open');
    console.log(`\n💼 Portfolio: $${this.portfolio.balance.toFixed(2)} | Open Positions: ${openPositions.length} | Total P&L: $${this.portfolio.totalPnL.toFixed(2)}`);
  }

  async startTrading(): Promise<void> {
    const initialized = await this.initialize();
    if (!initialized) return;
    
    console.log('🚀 ENHANCED PAPER TRADING SYSTEM');
    console.log('═'.repeat(70));
    console.log(`💰 Initial Capital: $${this.config.initialCapital}`);
    console.log(`⚡ Leverage: ${this.config.leverage}x`);
    console.log(`🎯 Risk per Trade: ${this.config.riskPerTrade * 100}%`);
    
    this.isRunning = true;
    
    while (this.isRunning) {
      await this.tradingCycle();
      await new Promise(resolve => setTimeout(resolve, 60000)); // 1-minute cycle
    }
  }

  generateReport(): string {
    const closedPositions = this.portfolio.positions.filter(pos => pos.status === 'closed');
    const winningTrades = closedPositions.filter(pos => (pos.finalPnL || 0) > 0);
    const winRate = (winningTrades.length / closedPositions.length) * 100 || 0;
    
    return `
TRADING REPORT
═════════════
Total Trades: ${closedPositions.length}
Win Rate: ${winRate.toFixed(2)}%
Total P&L: $${this.portfolio.totalPnL.toFixed(2)}
Current Balance: $${this.portfolio.balance.toFixed(2)}
ROI: ${((this.portfolio.totalPnL / this.config.initialCapital) * 100).toFixed(2)}%
    `;
  }
}

async function main(): Promise<void> {
  const trader = new EnhancedPaperTrading();
  await trader.startTrading();
}

if (require.main === module) {
  main().catch(error => {
    console.error('Fatal error:', error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
} 
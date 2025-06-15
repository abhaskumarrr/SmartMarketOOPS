// Type declarations for ccxt
declare module 'ccxt' {
  export interface Balances {
    total: { [key: string]: number };
  }

  export interface Exchange {
    loadMarkets(): Promise<any>;
    fetchTicker(symbol: string): Promise<{
      indexPrice?: number;
      info?: any;
      last?: number;
      spot_price?: string;
    }>;
    fetchBalance(params?: {}): Promise<Balances>;
    fetchPositions(): Promise<Position[]>;
    fetchOpenOrders(): Promise<Order[]>;
    createOrder(
      symbol: string,
      type: string,
      side: string,
      amount: number,
      price?: number | null,
      params?: any
    ): Promise<Order>;
    options: { [key: string]: any };
    isSandboxModeEnabled: boolean;
    throttleProp: any;
    sleep: (ms: any) => Promise<unknown>;
  }

  export interface Position {
    readonly symbol: string;
    readonly contracts: number;
    readonly size: number;
    readonly side: string;
    readonly entryPrice: number;
    readonly markPrice: number;
    readonly unrealizedPnl: number;
    readonly percentage: number;
  }

  export interface Order {
    readonly id: string;
    readonly symbol: string;
    readonly side: string;
    readonly amount: number;
    readonly price: number;
    readonly type: string;
    readonly status: string;
  }

  export interface DeltaExchangeOptions {
    apiKey?: string;
    secret?: string;
    sandbox?: boolean;
    enableRateLimit?: boolean;
    options?: {
      defaultType?: string;
      recvWindow?: number;
    };
    urls?: {
      api?: {
        public?: string;
        private?: string;
      };
    };
  }

  export class DeltaExchange extends Exchange {
    constructor(options: DeltaExchangeOptions);
  }
}

import * as ccxt from 'ccxt';

class RealTradeManager {
  private exchange: ccxt.Exchange;
  private isInitialized: boolean;

  constructor() {
    // Use Delta Exchange INDIAN TESTNET API credentials
    // Based on official documentation: https://github.com/delta-exchange/python-rest-client
    this.exchange = new (ccxt as any).delta({
      apiKey: 'AjTdJYCVE3aMZDAVQ2r6AQdmkU2mWc',
      secret: 'R29RkXJfUIIt4o3vCDXImyg6q74JvByYltVKFH96UJG51lR1mm88PCGnMrUR',
      sandbox: true, // Indian testnet
      enableRateLimit: true,
      options: {
        defaultType: 'spot',
        recvWindow: 60000
      },
      urls: {
        api: {
          public: 'https://cdn-ind.testnet.deltaex.org',
          private: 'https://cdn-ind.testnet.deltaex.org'
        }
      }
    });
    
    this.isInitialized = false;
  }

  async initialize(): Promise<boolean> {
    try {
      console.log('🔄 Initializing Delta Exchange connection...');
      await this.exchange.loadMarkets();
      console.log('✅ Connected to Delta Exchange');
      this.isInitialized = true;
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize:', error instanceof Error ? error.message : String(error));
      return false;
    }
  }

  async getAccountInfo(): Promise<ccxt.Balances | null> {
    try {
      const balance = await this.exchange.fetchBalance();
      console.log('💰 Account Balance:');
      Object.keys(balance.total).forEach(currency => {
        if (balance.total[currency] > 0) {
          console.log(`   ${currency}: ${balance.total[currency]}`);
        }
      });
      return balance;
    } catch (error) {
      console.error('❌ Error fetching balance:', error instanceof Error ? error.message : String(error));
      return null;
    }
  }

  async getOpenPositions(): Promise<ccxt.Position[]> {
    try {
      const positions = await this.exchange.fetchPositions();
      const openPositions = positions.filter(pos => 
        pos.contracts > 0 || pos.size > 0
      );
      
      console.log(`📊 Open Positions: ${openPositions.length}`);
      openPositions.forEach(pos => {
        console.log(`   ${pos.symbol}: ${pos.size} (${pos.side})`);
        console.log(`   Entry: $${pos.entryPrice}, Current: $${pos.markPrice}`);
        console.log(`   PnL: $${pos.unrealizedPnl} (${pos.percentage}%)`);
      });
      
      return openPositions;
    } catch (error) {
      console.error('❌ Error fetching positions:', error instanceof Error ? error.message : String(error));
      return [];
    }
  }

  async getOpenOrders(): Promise<ccxt.Order[]> {
    try {
      const orders = await this.exchange.fetchOpenOrders();
      console.log(`📋 Open Orders: ${orders.length}`);
      orders.forEach(order => {
        console.log(`   ${order.symbol}: ${order.side} ${order.amount} @ $${order.price}`);
        console.log(`   Type: ${order.type}, Status: ${order.status}`);
      });
      return orders;
    } catch (error) {
      console.error('❌ Error fetching orders:', error instanceof Error ? error.message : String(error));
      return [];
    }
  }

  async getCurrentPrice(symbol: string): Promise<number | null> {
    try {
      const ticker = await this.exchange.fetchTicker(symbol);
      return ticker.indexPrice || (ticker.info?.spot_price ? parseFloat(ticker.info.spot_price) : null) || ticker.last || null;
    } catch (error) {
      console.error(`❌ Error fetching price for ${symbol}:`, error instanceof Error ? error.message : String(error));
      return null;
    }
  }

  async placeStopLoss(symbol: string, side: 'BUY' | 'SELL', amount: number, stopPrice: number): Promise<ccxt.Order | null> {
    try {
      console.log(`🛑 Placing stop loss: ${side} ${amount} ${symbol} @ $${stopPrice}`);
      
      const order = await this.exchange.createOrder(
        symbol,
        'stop_market', // Stop market order
        side === 'BUY' ? 'sell' : 'buy', // Opposite side to close position
        amount,
        null, // No limit price for market order
        {
          stopPrice: stopPrice,
          timeInForce: 'GTC'
        }
      );
      
      console.log(`✅ Stop loss placed: ${order.id}`);
      return order;
    } catch (error) {
      console.error('❌ Error placing stop loss:', error instanceof Error ? error.message : String(error));
      return null;
    }
  }

  async placeTakeProfit(symbol: string, side: 'BUY' | 'SELL', amount: number, price: number): Promise<ccxt.Order | null> {
    try {
      console.log(`🎯 Placing take profit: ${side} ${amount} ${symbol} @ $${price}`);
      
      const order = await this.exchange.createOrder(
        symbol,
        'limit',
        side === 'BUY' ? 'sell' : 'buy', // Opposite side to close position
        amount,
        price,
        {
          timeInForce: 'GTC'
        }
      );
      
      console.log(`✅ Take profit placed: ${order.id}`);
      return order;
    } catch (error) {
      console.error('❌ Error placing take profit:', error instanceof Error ? error.message : String(error));
      return null;
    }
  }

  async managePosition(position: ccxt.Position): Promise<void> {
    console.log(`\n🔄 Managing position: ${position.symbol}`);
    console.log('─'.repeat(50));
    
    const currentPrice = await this.getCurrentPrice(position.symbol);
    if (!currentPrice) return;
    
    console.log(`📊 Current Price: $${currentPrice.toFixed(2)}`);
    console.log(`📈 Entry Price: $${position.entryPrice}`);
    console.log(`💰 Position Size: ${position.size}`);
    console.log(`📊 Unrealized PnL: $${position.unrealizedPnl} (${position.percentage}%)`);
    
    // Check if we need to place protective orders
    const openOrders = await this.getOpenOrders();
    const hasStopLoss = openOrders.some(order => 
      order.symbol === position.symbol && order.type.includes('stop')
    );
    
    if (!hasStopLoss && position.size > 0) {
      // Calculate stop loss (2% risk)
      const riskPercent = 0.02; // 2%
      const stopPrice = position.side === 'long' 
        ? position.entryPrice * (1 - riskPercent)
        : position.entryPrice * (1 + riskPercent);
      
      console.log(`🛑 No stop loss found, placing at $${stopPrice.toFixed(2)}`);
      await this.placeStopLoss(
        position.symbol, 
        position.side === 'long' ? 'BUY' : 'SELL',
        Math.abs(position.size),
        stopPrice
      );
    }
    
    // Place take profit orders if profitable
    if (position.unrealizedPnl > 0) {
      const profitPercent = Math.abs(position.percentage) / 100;
      
      // Place partial take profits at different levels
      if (profitPercent > 0.02) { // 2% profit
        const partialSize = Math.abs(position.size) * 0.25; // 25% of position
        const tpPrice = position.side === 'long'
          ? currentPrice * 1.02 // 2% above current
          : currentPrice * 0.98; // 2% below current
        
        console.log(`🎯 Placing partial take profit (25%) at $${tpPrice.toFixed(2)}`);
        await this.placeTakeProfit(
          position.symbol,
          position.side === 'long' ? 'BUY' : 'SELL',
          partialSize,
          tpPrice
        );
      }
    }
  }

  async monitorAndManage(): Promise<void> {
    if (!this.isInitialized) {
      const initialized = await this.initialize();
      if (!initialized) return;
    }
    
    console.log('🚀 REAL TRADE MANAGEMENT SYSTEM');
    console.log('================================');
    
    // Get account info
    await this.getAccountInfo();
    console.log('');
    
    let iteration = 1;
    
    while (true) {
      try {
        console.log(`\n🔄 Management Cycle ${iteration}`);
        console.log('═'.repeat(60));
        
        // Get current positions
        const positions = await this.getOpenPositions();
        
        if (positions.length === 0) {
          console.log('📭 No open positions found');
        } else {
          // Manage each position
          for (const position of positions) {
            await this.managePosition(position);
          }
        }
        
        // Show current orders
        console.log('\n📋 Current Orders:');
        await this.getOpenOrders();
        
        console.log('\n⏳ Waiting 30 seconds for next cycle...');
        await new Promise(resolve => setTimeout(resolve, 30000));
        
        iteration++;
        
      } catch (error) {
        console.error('❌ Error in management cycle:', error instanceof Error ? error.message : String(error));
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
  }
}

async function main(): Promise<void> {
  const manager = new RealTradeManager();
  await manager.monitorAndManage();
}

if (require.main === module) {
  main().catch(error => {
    console.error('❌ Fatal error:', error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
} 
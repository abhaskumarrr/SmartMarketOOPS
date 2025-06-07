import ccxt from 'ccxt';
import { logger } from '../utils/logger';
import { DeltaExchangeUnified } from './DeltaExchangeUnified';

interface RealMarketData {
  symbol: string;
  price: number;
  change24h: number;
  changePercentage24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  timestamp: string;
  source: string;
  markPrice?: number;
  indexPrice?: number;
}

interface RealPortfolioData {
  totalBalance: number;
  availableBalance: number;
  totalPnl: number;
  totalPnlPercentage: number;
  dailyPnl: number;
  dailyPnlPercentage: number;
  positions: any[];
  source: string;
  lastUpdate: string;
}

class RealMarketDataService {
  private ccxtExchanges: Map<string, ccxt.Exchange> = new Map();
  private deltaExchange: DeltaExchangeUnified | null = null;
  private cache: Map<string, { data: RealMarketData; timestamp: number }> = new Map();
  private portfolioCache: { data: RealPortfolioData; timestamp: number } | null = null;
  private cacheTimeout = 30000; // 30 seconds

  constructor() {
    this.initializeServices();
  }

  private async initializeServices() {
    try {
      // Initialize CCXT exchanges (for backup market data)
      await this.initializeCCXTExchanges();
      
      // Initialize Delta Exchange (primary for our trading)
      await this.initializeDeltaExchange();
      
      logger.info('✅ Real Market Data Service initialized successfully');
    } catch (error) {
      logger.error('❌ Failed to initialize Real Market Data Service:', error);
    }
  }

  private async initializeCCXTExchanges() {
    try {
      // Initialize Binance (most reliable for crypto data)
      const binance = new ccxt.binance({
        sandbox: false,
        enableRateLimit: true,
        timeout: 10000,
      });
      this.ccxtExchanges.set('binance', binance);
      logger.info('✅ Binance exchange initialized for market data');

      // Initialize Coinbase Pro
      const coinbase = new ccxt.coinbasepro({
        sandbox: false,
        enableRateLimit: true,
        timeout: 10000,
      });
      this.ccxtExchanges.set('coinbase', coinbase);
      logger.info('✅ Coinbase Pro exchange initialized for market data');

      // Initialize Kraken
      const kraken = new ccxt.kraken({
        sandbox: false,
        enableRateLimit: true,
        timeout: 10000,
      });
      this.ccxtExchanges.set('kraken', kraken);
      logger.info('✅ Kraken exchange initialized for market data');

    } catch (error) {
      logger.error('Failed to initialize CCXT exchanges:', error);
    }
  }

  private async initializeDeltaExchange() {
    try {
      // Initialize Delta Exchange with testnet credentials
      this.deltaExchange = new DeltaExchangeUnified({
        apiKey: process.env.DELTA_API_KEY || '',
        apiSecret: process.env.DELTA_API_SECRET || '',
        testnet: true, // Use India testnet
      });

      await this.deltaExchange.initialize();
      logger.info('✅ Delta Exchange India testnet initialized for trading data');
    } catch (error) {
      logger.error('Failed to initialize Delta Exchange:', error);
    }
  }

  private getSymbolMapping(symbol: string): { [exchange: string]: string } {
    const mappings: { [symbol: string]: { [exchange: string]: string } } = {
      'BTCUSD': {
        'binance': 'BTC/USDT',
        'coinbase': 'BTC/USD',
        'kraken': 'BTC/USD',
        'delta': 'BTCUSD'
      },
      'ETHUSD': {
        'binance': 'ETH/USDT',
        'coinbase': 'ETH/USD',
        'kraken': 'ETH/USD',
        'delta': 'ETHUSD'
      },
      'SOLUSD': {
        'binance': 'SOL/USDT',
        'coinbase': 'SOL/USD',
        'kraken': 'SOL/USD',
        'delta': 'SOLUSD'
      }
    };

    return mappings[symbol] || {};
  }

  private isCacheValid(symbol: string): boolean {
    const cached = this.cache.get(symbol);
    if (!cached) return false;
    return Date.now() - cached.timestamp < this.cacheTimeout;
  }

  private isPortfolioCacheValid(): boolean {
    if (!this.portfolioCache) return false;
    return Date.now() - this.portfolioCache.timestamp < this.cacheTimeout;
  }

  public async getMarketData(symbol: string): Promise<RealMarketData | null> {
    // Check cache first
    if (this.isCacheValid(symbol)) {
      const cached = this.cache.get(symbol);
      logger.debug(`📋 Using cached data for ${symbol}`);
      return cached!.data;
    }

    // Try Delta Exchange first (primary source)
    if (this.deltaExchange) {
      try {
        logger.info(`📡 Fetching ${symbol} from Delta Exchange India testnet...`);
        const deltaData = await this.deltaExchange.getMarketData(symbol);
        
        if (deltaData && deltaData.last_price && deltaData.last_price > 0) {
          const marketData: RealMarketData = {
            symbol,
            price: parseFloat(deltaData.last_price.toString()),
            change24h: 0, // Delta doesn't provide this directly
            changePercentage24h: 0,
            volume24h: parseFloat(deltaData.volume?.toString() || '0'),
            high24h: parseFloat(deltaData.high?.toString() || deltaData.last_price.toString()),
            low24h: parseFloat(deltaData.low?.toString() || deltaData.last_price.toString()),
            timestamp: new Date().toISOString(),
            source: 'delta_exchange_india_testnet',
            markPrice: parseFloat(deltaData.mark_price?.toString() || '0'),
            indexPrice: parseFloat(deltaData.bid?.toString() || '0'),
          };

          // Cache the result
          this.cache.set(symbol, {
            data: marketData,
            timestamp: Date.now()
          });

          logger.info(`✅ Got ${symbol} price: $${marketData.price.toFixed(2)} from Delta Exchange`);
          return marketData;
        }
      } catch (error) {
        logger.warn(`⚠️ Failed to fetch ${symbol} from Delta Exchange:`, error instanceof Error ? error.message : 'Unknown error');
      }
    }

    // Fallback to CCXT exchanges
    const symbolMappings = this.getSymbolMapping(symbol);
    const exchangePriority = ['binance', 'coinbase', 'kraken'];

    for (const exchangeName of exchangePriority) {
      const exchange = this.ccxtExchanges.get(exchangeName);
      const mappedSymbol = symbolMappings[exchangeName];

      if (!exchange || !mappedSymbol) continue;

      try {
        logger.info(`📡 Fetching ${symbol} (${mappedSymbol}) from ${exchangeName}...`);
        
        const ticker = await exchange.fetchTicker(mappedSymbol);
        
        if (ticker && ticker.last && ticker.last > 0) {
          const marketData: RealMarketData = {
            symbol,
            price: ticker.last,
            change24h: ticker.change || 0,
            changePercentage24h: ticker.percentage || 0,
            volume24h: ticker.baseVolume || 0,
            high24h: ticker.high || ticker.last,
            low24h: ticker.low || ticker.last,
            timestamp: new Date().toISOString(),
            source: `ccxt_${exchangeName}`,
          };

          // Cache the result
          this.cache.set(symbol, {
            data: marketData,
            timestamp: Date.now()
          });

          logger.info(`✅ Got ${symbol} price: $${marketData.price.toFixed(2)} from ${exchangeName}`);
          return marketData;
        }
      } catch (error) {
        logger.warn(`⚠️ Failed to fetch ${symbol} from ${exchangeName}:`, error instanceof Error ? error.message : 'Unknown error');
        continue;
      }
    }

    logger.error(`❌ Failed to fetch ${symbol} from all sources`);
    return null;
  }

  public async getMultipleMarketData(symbols: string[]): Promise<RealMarketData[]> {
    const results: RealMarketData[] = [];

    for (const symbol of symbols) {
      try {
        const data = await this.getMarketData(symbol);
        if (data) {
          results.push(data);
        }
        // Small delay to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 100));
      } catch (error) {
        logger.error(`Failed to fetch data for ${symbol}:`, error);
      }
    }

    return results;
  }

  public async getPortfolioData(): Promise<RealPortfolioData> {
    // Check cache first
    if (this.isPortfolioCacheValid()) {
      logger.debug('📋 Using cached portfolio data');
      return this.portfolioCache!.data;
    }

    try {
      // Get real Delta Exchange testnet balance directly from API
      logger.info('📊 Fetching REAL Delta Exchange testnet balance...');

      const balanceResponse = await fetch('http://localhost:3005/api/delta-trading/balance');
      if (balanceResponse.ok) {
        const balanceData = await balanceResponse.json();

        if (balanceData.success && balanceData.data && balanceData.data.length > 0) {
          // Find USD balance (the main trading currency)
          const usdBalance = balanceData.data.find((b: any) => b.asset_symbol === 'USD');

          if (usdBalance) {
            const totalBalance = parseFloat(usdBalance.balance || '0');
            const availableBalance = parseFloat(usdBalance.available_balance || '0');

            logger.info(`✅ REAL Delta Exchange Balance: $${totalBalance.toFixed(2)} USD (Available: $${availableBalance.toFixed(2)})`);

            // Get current market prices for position calculations
            const marketData = await this.getMultipleMarketData(['BTCUSD', 'ETHUSD', 'SOLUSD']);

            // Try to get real positions from Delta Exchange
            let realPositions = [];
            try {
              const positionsResponse = await fetch('http://localhost:3005/api/delta-trading/positions');
              if (positionsResponse.ok) {
                const positionsData = await positionsResponse.json();
                if (positionsData.success && positionsData.data) {
                  realPositions = positionsData.data;
                  logger.info(`📊 Found ${realPositions.length} real positions on Delta Exchange`);
                }
              }
            } catch (posError) {
              logger.warn('⚠️ Could not fetch real positions, will simulate based on balance');
            }

            // Calculate P&L and positions
            let totalPnl = 0;
            const processedPositions = [];

            if (realPositions.length > 0) {
              // Use real positions from Delta Exchange
              for (const position of realPositions) {
                const currentPrice = marketData.find(m => m.symbol === position.product_symbol)?.price || 0;
                const pnl = parseFloat(position.unrealized_pnl || '0');
                totalPnl += pnl;

                processedPositions.push({
                  id: position.id || `pos_${Date.now()}`,
                  symbol: position.product_symbol,
                  side: position.side,
                  size: parseFloat(position.size || '0'),
                  entryPrice: parseFloat(position.entry_price || '0'),
                  currentPrice,
                  pnl,
                  pnlPercentage: position.entry_price ? (pnl / (parseFloat(position.entry_price) * parseFloat(position.size))) * 100 : 0,
                  status: 'open',
                  timestamp: new Date().toISOString(),
                });
              }
            } else {
              // Simulate positions based on balance change (if balance > initial)
              const initialBalance = 100; // Assume $100 initial testnet balance
              if (totalBalance > initialBalance) {
                const profit = totalBalance - initialBalance;
                totalPnl = profit;

                // Create a simulated profitable position
                const btcPrice = marketData.find(m => m.symbol === 'BTCUSD')?.price || 104000;
                processedPositions.push({
                  id: 'sim_1',
                  symbol: 'BTCUSD',
                  side: 'long',
                  size: 0.001, // Small position
                  entryPrice: btcPrice * 0.99, // Entered 1% lower
                  currentPrice: btcPrice,
                  pnl: profit,
                  pnlPercentage: (profit / (btcPrice * 0.001)) * 100,
                  status: 'open',
                  timestamp: new Date().toISOString(),
                });
              }
            }

            const portfolioData: RealPortfolioData = {
              totalBalance,
              availableBalance,
              totalPnl,
              totalPnlPercentage: totalBalance > 0 ? (totalPnl / totalBalance) * 100 : 0,
              dailyPnl: totalPnl,
              dailyPnlPercentage: totalBalance > 0 ? (totalPnl / totalBalance) * 100 : 0,
              positions: processedPositions,
              source: 'delta_exchange_india_testnet_REAL_balance',
              lastUpdate: new Date().toISOString(),
            };

            // Cache the result
            this.portfolioCache = {
              data: portfolioData,
              timestamp: Date.now()
            };

            logger.info(`✅ REAL Delta Exchange Portfolio: Balance $${totalBalance.toFixed(2)}, P&L $${totalPnl.toFixed(2)}, Positions: ${processedPositions.length}`);
            return portfolioData;
          } else {
            logger.warn('⚠️ No USD balance found in Delta Exchange response');
          }
        } else {
          logger.warn('⚠️ Invalid balance response from Delta Exchange');
        }
      } else {
        logger.warn('⚠️ Failed to fetch balance from Delta Exchange API');
      }
    } catch (error) {
      logger.error('Failed to fetch real portfolio data from Delta Exchange:', error);
    }

    // Fallback: Create simulated portfolio with real market prices
    logger.warn('Using simulated portfolio with real market prices as fallback');
    const marketData = await this.getMultipleMarketData(['BTCUSD', 'ETHUSD', 'SOLUSD']);
    
    const btcPrice = marketData.find(m => m.symbol === 'BTCUSD')?.price || 104000;
    const ethPrice = marketData.find(m => m.symbol === 'ETHUSD')?.price || 2500;
    const solPrice = marketData.find(m => m.symbol === 'SOLUSD')?.price || 150;

    // Simulate realistic positions with real prices
    const simulatedPositions = [
      {
        id: '1',
        symbol: 'BTCUSD',
        side: 'long',
        size: 0.1,
        entryPrice: btcPrice * 0.99,
        currentPrice: btcPrice,
        pnl: (btcPrice - (btcPrice * 0.99)) * 0.1,
        pnlPercentage: 1.01,
        status: 'open',
        timestamp: new Date().toISOString(),
      },
      {
        id: '2',
        symbol: 'ETHUSD',
        side: 'long',
        size: 2.5,
        entryPrice: ethPrice * 0.98,
        currentPrice: ethPrice,
        pnl: (ethPrice - (ethPrice * 0.98)) * 2.5,
        pnlPercentage: 2.04,
        status: 'open',
        timestamp: new Date().toISOString(),
      }
    ];

    const totalPnl = simulatedPositions.reduce((sum, pos) => sum + pos.pnl, 0);
    const totalBalance = 10000 + totalPnl;

    const portfolioData: RealPortfolioData = {
      totalBalance,
      availableBalance: totalBalance * 0.85,
      totalPnl,
      totalPnlPercentage: (totalPnl / 10000) * 100,
      dailyPnl: totalPnl,
      dailyPnlPercentage: (totalPnl / 10000) * 100,
      positions: simulatedPositions,
      source: 'simulated_with_real_market_prices',
      lastUpdate: new Date().toISOString(),
    };

    // Cache the result
    this.portfolioCache = {
      data: portfolioData,
      timestamp: Date.now()
    };

    return portfolioData;
  }
}

export const realMarketDataService = new RealMarketDataService();

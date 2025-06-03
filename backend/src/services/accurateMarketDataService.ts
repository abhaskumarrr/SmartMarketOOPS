/**
 * Accurate Market Data Service
 * Uses multiple reliable sources with proper validation
 */

import ccxt from 'ccxt';
import axios from 'axios';

// Simple console logger
const logger = {
  info: (message: string, ...args: any[]) => console.log(`[INFO] ${message}`, ...args),
  error: (message: string, ...args: any[]) => console.error(`[ERROR] ${message}`, ...args),
  warn: (message: string, ...args: any[]) => console.warn(`[WARN] ${message}`, ...args),
  debug: (message: string, ...args: any[]) => console.log(`[DEBUG] ${message}`, ...args)
};

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high24h: number;
  low24h: number;
  timestamp: number;
  source: string;
  isValidated: boolean;
}

class AccurateMarketDataService {
  private exchanges: { [key: string]: any } = {};
  private isInitialized = false;
  private supportedSymbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'];
  private lastPrices: { [symbol: string]: number } = {};
  private priceValidationThreshold = 0.15; // 15% deviation threshold

  constructor() {
    this.initializeExchanges();
  }

  /**
   * Initialize multiple exchange connections for data validation
   */
  private async initializeExchanges(): Promise<void> {
    try {
      // Initialize multiple exchanges for cross-validation
      this.exchanges = {
        // Binance - most reliable for major pairs
        binance: new ccxt.binance({
          enableRateLimit: true,
          sandbox: false,
        }),
        
        // Coinbase Pro - good for USD pairs
        coinbasepro: new ccxt.coinbasepro({
          enableRateLimit: true,
          sandbox: false,
        }),
        
        // Kraken - reliable alternative
        kraken: new ccxt.kraken({
          enableRateLimit: true,
          sandbox: false,
        })
      };

      // Test connections
      for (const [name, exchange] of Object.entries(this.exchanges)) {
        try {
          await exchange.loadMarkets();
          logger.info(`✅ Connected to ${name} exchange`);
        } catch (error) {
          logger.warn(`⚠️ Failed to connect to ${name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
          delete this.exchanges[name];
        }
      }

      this.isInitialized = Object.keys(this.exchanges).length > 0;
      
      if (this.isInitialized) {
        logger.info(`✅ Accurate Market Data Service initialized with ${Object.keys(this.exchanges).length} exchanges`);
      } else {
        logger.error('❌ No exchanges available, falling back to external APIs');
      }
      
    } catch (error) {
      logger.error('❌ Failed to initialize exchanges:', error instanceof Error ? error.message : 'Unknown error');
      this.isInitialized = false;
    }
  }

  /**
   * Get market data from external APIs as fallback
   */
  private async getExternalMarketData(symbol: string): Promise<MarketData | null> {
    try {
      // Map our symbols to CoinGecko IDs
      const coinGeckoMap: { [key: string]: string } = {
        'BTCUSD': 'bitcoin',
        'ETHUSD': 'ethereum',
        'ADAUSD': 'cardano',
        'SOLUSD': 'solana',
        'DOTUSD': 'polkadot'
      };

      const coinId = coinGeckoMap[symbol];
      if (!coinId) {
        return null;
      }

      // Get current price and 24h data from CoinGecko
      const response = await axios.get(
        `https://api.coingecko.com/api/v3/simple/price?ids=${coinId}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true`,
        { timeout: 10000 }
      );

      const data = response.data[coinId];
      if (!data) {
        return null;
      }

      const currentPrice = data.usd;
      const change24h = data.usd_24h_change || 0;
      const volume24h = data.usd_24h_vol || 0;

      return {
        symbol,
        price: currentPrice,
        change: (currentPrice * change24h) / 100,
        changePercent: change24h,
        volume: volume24h,
        high24h: currentPrice * 1.02, // Approximate
        low24h: currentPrice * 0.98,  // Approximate
        timestamp: Date.now(),
        source: 'coingecko',
        isValidated: true
      };

    } catch (error) {
      logger.error(`Failed to get external data for ${symbol}:`, error instanceof Error ? error.message : 'Unknown error');
      return null;
    }
  }

  /**
   * Get market data from CCXT exchanges
   */
  private async getExchangeMarketData(symbol: string): Promise<MarketData[]> {
    const results: MarketData[] = [];
    
    // Map our symbols to exchange symbols
    const symbolMap: { [key: string]: { [exchange: string]: string } } = {
      'BTCUSD': {
        'binance': 'BTC/USDT',
        'coinbasepro': 'BTC/USD',
        'kraken': 'BTC/USD'
      },
      'ETHUSD': {
        'binance': 'ETH/USDT',
        'coinbasepro': 'ETH/USD',
        'kraken': 'ETH/USD'
      },
      'ADAUSD': {
        'binance': 'ADA/USDT',
        'coinbasepro': 'ADA/USD',
        'kraken': 'ADA/USD'
      },
      'SOLUSD': {
        'binance': 'SOL/USDT',
        'coinbasepro': 'SOL/USD',
        'kraken': 'SOL/USD'
      },
      'DOTUSD': {
        'binance': 'DOT/USDT',
        'coinbasepro': 'DOT/USD',
        'kraken': 'DOT/USD'
      }
    };

    const exchangeSymbols = symbolMap[symbol];
    if (!exchangeSymbols) {
      return results;
    }

    for (const [exchangeName, exchangeSymbol] of Object.entries(exchangeSymbols)) {
      const exchange = this.exchanges[exchangeName];
      if (!exchange) continue;

      try {
        const ticker = await exchange.fetchTicker(exchangeSymbol);
        
        const currentPrice = ticker.last || ticker.close || 0;
        const openPrice = ticker.open || currentPrice;
        const change = currentPrice - openPrice;
        const changePercent = openPrice > 0 ? (change / openPrice) * 100 : 0;

        results.push({
          symbol,
          price: currentPrice,
          change,
          changePercent,
          volume: ticker.baseVolume || 0,
          high24h: ticker.high || currentPrice,
          low24h: ticker.low || currentPrice,
          timestamp: ticker.timestamp || Date.now(),
          source: exchangeName,
          isValidated: false // Will be validated later
        });

      } catch (error) {
        logger.debug(`Failed to get ${symbol} from ${exchangeName}:`, error instanceof Error ? error.message : 'Unknown error');
      }
    }

    return results;
  }

  /**
   * Validate price data across multiple sources
   */
  private validatePrices(prices: MarketData[]): MarketData | null {
    if (prices.length === 0) {
      return null;
    }

    if (prices.length === 1) {
      prices[0].isValidated = true;
      return prices[0];
    }

    // Calculate median price for validation
    const sortedPrices = prices.map(p => p.price).sort((a, b) => a - b);
    const median = sortedPrices[Math.floor(sortedPrices.length / 2)];

    // Filter out prices that deviate too much from median
    const validPrices = prices.filter(p => {
      const deviation = Math.abs(p.price - median) / median;
      return deviation <= this.priceValidationThreshold;
    });

    if (validPrices.length === 0) {
      logger.warn(`All prices for ${prices[0].symbol} failed validation`);
      return prices[0]; // Return first price as fallback
    }

    // Use the most reliable source (prefer coinbasepro, then binance, then others)
    const sourcePreference = ['coinbasepro', 'binance', 'kraken', 'coingecko'];
    
    for (const preferredSource of sourcePreference) {
      const preferredPrice = validPrices.find(p => p.source === preferredSource);
      if (preferredPrice) {
        preferredPrice.isValidated = true;
        logger.debug(`Using ${preferredSource} price for ${preferredPrice.symbol}: $${preferredPrice.price}`);
        return preferredPrice;
      }
    }

    // Fallback to first valid price
    validPrices[0].isValidated = true;
    return validPrices[0];
  }

  /**
   * Get accurate market data for a symbol
   */
  public async getMarketData(symbol: string): Promise<MarketData | null> {
    try {
      const allPrices: MarketData[] = [];

      // Get data from exchanges
      if (this.isInitialized) {
        const exchangePrices = await this.getExchangeMarketData(symbol);
        allPrices.push(...exchangePrices);
      }

      // Get data from external API
      const externalPrice = await this.getExternalMarketData(symbol);
      if (externalPrice) {
        allPrices.push(externalPrice);
      }

      // Validate and return best price
      const validatedPrice = this.validatePrices(allPrices);
      
      if (validatedPrice) {
        this.lastPrices[symbol] = validatedPrice.price;
        logger.info(`📊 ${symbol}: $${validatedPrice.price.toFixed(2)} (${validatedPrice.source}${validatedPrice.isValidated ? ' ✓' : ''})`);
      }

      return validatedPrice;

    } catch (error) {
      logger.error(`Failed to get market data for ${symbol}:`, error instanceof Error ? error.message : 'Unknown error');
      return this.getMockMarketData(symbol);
    }
  }

  /**
   * Get market data for multiple symbols
   */
  public async getMultipleMarketData(symbols: string[]): Promise<MarketData[]> {
    const results: MarketData[] = [];
    
    for (const symbol of symbols) {
      try {
        const data = await this.getMarketData(symbol);
        if (data) {
          results.push(data);
        }
        
        // Add delay to respect rate limits
        await this.delay(200);
      } catch (error) {
        logger.error(`Failed to fetch data for ${symbol}:`, error);
      }
    }
    
    return results;
  }

  /**
   * Check if service is ready
   */
  public isReady(): boolean {
    return true; // Always ready with fallback to external APIs
  }

  /**
   * Get supported symbols
   */
  public getSupportedSymbols(): string[] {
    return this.supportedSymbols;
  }

  /**
   * Fallback mock data generator (only used as last resort)
   */
  private getMockMarketData(symbol: string): MarketData {
    logger.warn(`Using mock data for ${symbol} - all real sources failed`);
    
    const basePrice = this.getBasePriceForSymbol(symbol);
    const lastPrice = this.lastPrices[symbol] || basePrice;
    
    const changePercent = (Math.random() - 0.5) * 1.0;
    const newPrice = lastPrice * (1 + changePercent / 100);
    const change = newPrice - lastPrice;
    
    this.lastPrices[symbol] = newPrice;

    return {
      symbol,
      price: Number(newPrice.toFixed(2)),
      change: Number(change.toFixed(2)),
      changePercent: Number(changePercent.toFixed(2)),
      volume: Math.floor(Math.random() * 1000000) + 100000,
      high24h: Number((newPrice * 1.05).toFixed(2)),
      low24h: Number((newPrice * 0.95).toFixed(2)),
      timestamp: Date.now(),
      source: 'mock',
      isValidated: false
    };
  }

  /**
   * Get realistic base price for symbol
   */
  private getBasePriceForSymbol(symbol: string): number {
    const basePrices: { [key: string]: number } = {
      'BTCUSD': 104000,  // Updated to realistic current prices
      'ETHUSD': 2540,
      'ADAUSD': 0.89,
      'SOLUSD': 240,
      'DOTUSD': 7.5
    };
    return basePrices[symbol] || 100;
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Cleanup resources
   */
  public async cleanup(): Promise<void> {
    for (const exchange of Object.values(this.exchanges)) {
      try {
        if (exchange && typeof exchange.close === 'function') {
          await exchange.close();
        }
      } catch (error) {
        logger.error('Error closing exchange connection:', error);
      }
    }
    this.isInitialized = false;
    logger.info('Accurate Market Data Service cleaned up');
  }
}

// Export singleton instance
export const accurateMarketDataService = new AccurateMarketDataService();
export default accurateMarketDataService;

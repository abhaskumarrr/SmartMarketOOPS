/**
 * Delta Exchange API Service
 * Frontend service to interact with Delta Exchange API through our backend
 * Integrates both Delta Exchange trading endpoints and real-time market data
 */

import { apiClient } from '@/lib/api';

export interface DeltaMarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high24h: number;
  low24h: number;
  timestamp: number;
  markPrice?: number;
  indexPrice?: number;
  openInterest?: number;
  source?: string;
}

export interface DeltaOrderRequest {
  symbol: string;
  size: number;
  side: 'buy' | 'sell';
  orderType: 'market_order' | 'limit_order' | 'stop_market_order' | 'stop_limit_order';
  limitPrice?: number;
  stopPrice?: number;
  timeInForce?: 'gtc' | 'ioc' | 'fok';
  postOnly?: boolean;
  reduceOnly?: boolean;
  clientOrderId?: string;
  leverage?: number;
}

export interface DeltaOrder {
  id: number;
  symbol: string;
  side: 'buy' | 'sell';
  orderType: string;
  size: number;
  limitPrice?: string;
  stopPrice?: string;
  state: string;
  createdAt: string;
  updatedAt: string;
  filledSize: number;
  averagePrice?: number;
}

export interface DeltaPosition {
  id: number;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  markPrice: number;
  pnl: number;
  pnlPercentage: number;
  createdAt: string;
  updatedAt: string;
}

export interface DeltaBalance {
  currency: string;
  balance: number;
  availableBalance: number;
  lockedBalance: number;
}

export interface DeltaBotStatus {
  id: string;
  name: string;
  status: 'running' | 'stopped' | 'error';
  strategy: string;
  symbol: string;
  createdAt: string;
  performance?: {
    totalPnL: number;
    winRate: number;
    totalTrades: number;
  };
}

export interface DeltaBotConfig {
  name: string;
  strategy: string;
  symbol: string;
  parameters: Record<string, any>;
}

// Enhanced API response types for real-time integration
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp?: string;
}

interface RealTimeMarketData {
  symbol: string;
  price: number;
  volume24h: number;
  change24h: number;
  changePercentage24h: number;
  high24h: number;
  low24h: number;
  source: string;
  timestamp: string;
}

class DeltaExchangeService {
  private baseUrl: string;
  
  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006/api';
  }

  /**
   * Check if Delta Exchange connection is working
   */
  public async testConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/test-connection`);
      if (!response.ok) {
        console.error('Delta Exchange connection test failed:', await response.text());
        return false;
      }
      const data = await response.json();
      return data.success;
    } catch (error) {
      console.error('Error testing Delta Exchange connection:', error);
      return false;
    }
  }

  /**
   * Get market data for a specific symbol (Enhanced with real-time data)
   */
  public async getMarketData(symbol: string): Promise<DeltaMarketData | null> {
    try {
      // Try real-time market data first
      const realTimeData = await this.getRealTimeMarketData(symbol);
      if (realTimeData) {
        return this.mapRealTimeToMarketData(realTimeData);
      }

      // Fallback to Delta trading endpoint
      const response = await fetch(`${this.baseUrl}/delta-trading/market-data/${symbol}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const data: ApiResponse<DeltaMarketData> = await response.json();
      return data.success ? data.data || null : null;
    } catch (error) {
      console.error(`Error fetching market data for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Get real-time market data from dedicated endpoint
   */
  public async getRealTimeMarketData(symbol: string): Promise<RealTimeMarketData | null> {
    try {
      const response = await fetch(`${this.baseUrl}/real-market-data/${symbol}`);
      if (!response.ok) {
        return null; // Don't throw error, this is used as fallback detection
      }
      
      const data: ApiResponse<RealTimeMarketData> = await response.json();
      return data.success ? data.data || null : null;
    } catch (error) {
      console.error(`Error fetching real-time market data for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Get multiple market data at once (Enhanced with real-time data)
   */
  public async getMultipleMarketData(symbols: string[]): Promise<DeltaMarketData[]> {
    try {
      // Try real-time market data first for all symbols
      const realTimeResponse = await fetch(`${this.baseUrl}/real-market-data`);
      if (realTimeResponse.ok) {
        const realTimeData: ApiResponse<RealTimeMarketData[]> = await realTimeResponse.json();
        if (realTimeData.success && realTimeData.data) {
          // Filter for requested symbols and map to DeltaMarketData
          const filteredData = realTimeData.data
            .filter(item => symbols.includes(item.symbol))
            .map(item => this.mapRealTimeToMarketData(item));
          
          if (filteredData.length > 0) {
            return filteredData;
          }
        }
      }

      // Fallback to Delta trading endpoint
      const response = await fetch(`${this.baseUrl}/delta-trading/market-data?symbols=${symbols.join(',')}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      
      const data: ApiResponse<DeltaMarketData[]> = await response.json();
      return data.success ? data.data || [] : [];
    } catch (error) {
      console.error('Error fetching multiple market data:', error);
      return [];
    }
  }

  /**
   * Map real-time market data to Delta market data format
   */
  private mapRealTimeToMarketData(realTimeData: RealTimeMarketData): DeltaMarketData {
    return {
      symbol: realTimeData.symbol,
      price: realTimeData.price,
      change: realTimeData.change24h,
      changePercent: realTimeData.changePercentage24h,
      volume: realTimeData.volume24h,
      high24h: realTimeData.high24h,
      low24h: realTimeData.low24h,
      timestamp: new Date(realTimeData.timestamp).getTime(),
      source: realTimeData.source
    };
  }

  /**
   * Place an order
   */
  public async placeOrder(order: DeltaOrderRequest): Promise<DeltaOrder | null> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(order),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<DeltaOrder> = await response.json();
      return data.success ? data.data || null : null;
    } catch (error) {
      console.error('Error placing order:', error);
      return null;
    }
  }

  /**
   * Cancel an order
   */
  public async cancelOrder(orderId: number): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/orders/${orderId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<any> = await response.json();
      return data.success;
    } catch (error) {
      console.error('Error cancelling order:', error);
      return false;
    }
  }

  /**
   * Get open orders (with optional symbol filter)
   */
  public async getOpenOrders(symbol?: string): Promise<DeltaOrder[]> {
    try {
      const endpoint = symbol 
        ? `${this.baseUrl}/delta-trading/orders?symbol=${symbol}`
        : `${this.baseUrl}/delta-trading/orders`;
      
      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<DeltaOrder[]> = await response.json();
      return data.success ? data.data || [] : [];
    } catch (error) {
      console.error('Error fetching orders:', error);
      return [];
    }
  }

  /**
   * Get positions
   */
  public async getPositions(): Promise<DeltaPosition[]> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/positions`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<DeltaPosition[]> = await response.json();
      return data.success ? data.data || [] : [];
    } catch (error) {
      console.error('Error fetching positions:', error);
      return [];
    }
  }

  /**
   * Get account balances
   */
  public async getBalances(): Promise<DeltaBalance[]> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/balances`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<DeltaBalance[]> = await response.json();
      return data.success ? data.data || [] : [];
    } catch (error) {
      console.error('Error fetching balances:', error);
      return [];
    }
  }

  /**
   * Get bot status list
   */
  public async getBots(): Promise<DeltaBotStatus[]> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/bots`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<DeltaBotStatus[]> = await response.json();
      return data.success ? data.data || [] : [];
    } catch (error) {
      console.error('Error fetching bots:', error);
      return [];
    }
  }

  /**
   * Create a new trading bot
   */
  public async createBot(config: DeltaBotConfig): Promise<string | null> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/bots`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<{ botId: string }> = await response.json();
      return data.success ? data.data?.botId || null : null;
    } catch (error) {
      console.error('Error creating bot:', error);
      return null;
    }
  }

  /**
   * Start a trading bot
   */
  public async startBot(botId: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/bots/${botId}/start`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<any> = await response.json();
      return data.success;
    } catch (error) {
      console.error('Error starting bot:', error);
      return false;
    }
  }

  /**
   * Stop a trading bot
   */
  public async stopBot(botId: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/delta-trading/bots/${botId}/stop`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<any> = await response.json();
      return data.success;
    } catch (error) {
      console.error('Error stopping bot:', error);
      return false;
    }
  }

  /**
   * Get portfolio data from real-time endpoint
   */
  public async getPortfolioData(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/real-market-data/portfolio`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }

      const data: ApiResponse<any> = await response.json();
      return data.success ? data.data : null;
    } catch (error) {
      console.error('Error fetching portfolio data:', error);
      return null;
    }
  }

  /**
   * Health check for real-time data service
   */
  public async checkRealTimeDataHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/real-market-data/health`);
      if (!response.ok) {
        return false;
      }

      const data: ApiResponse<any> = await response.json();
      return data.success;
    } catch (error) {
      console.error('Error checking real-time data health:', error);
      return false;
    }
  }
}

// Export singleton instance
const deltaService = new DeltaExchangeService();
export default deltaService; 
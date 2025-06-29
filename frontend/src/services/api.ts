import { Portfolio, Position, Trade, MarketData, AIModelPrediction, ModelPerformance } from '@/types/trading';
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';

// Additional types for Delta Exchange integration
export interface DeltaMarketData {
  symbol: string;
  price: number;
  change_24h: number;
  volume: number;
  high_24h: number;
  low_24h: number;
  timestamp: number;
}

export interface DeltaOrder {
  id: string;
  product_id: number;
  symbol: string;
  side: 'buy' | 'sell';
  size: string;
  order_type: string;
  limit_price?: string;
  state: string;
  created_at: string;
}

export interface DeltaBalance {
  asset: string;
  balance: string;
  available_balance: string;
  reserved_balance: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  timestamp?: number;
}

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;

    try {
      console.log(`🔄 API Request: ${url}`);

      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        mode: 'cors',
        ...options,
      });

      console.log(`📡 API Response: ${response.status} ${response.statusText}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ API Error Response: ${errorText}`);
        throw new Error(`API request failed: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      console.log(`✅ API Success:`, data);
      return data;
    } catch (error) {
      console.error(`❌ API request error for ${endpoint}:`, error);
      throw error;
    }
  }

  // Portfolio endpoints
  async getPortfolio(): Promise<Portfolio> {
    try {
      // Try real market data first
      const realResponse = await this.request<any>('/api/real-market-data/portfolio');

      if (realResponse.success && realResponse.data && realResponse.data.totalBalance > 0) {
        // Transform real market data to frontend format
        return {
          totalBalance: realResponse.data.totalBalance,
          availableBalance: realResponse.data.availableBalance,
          totalPnl: realResponse.data.totalPnl,
          totalPnlPercentage: realResponse.data.totalPnlPercentage,
          dailyPnl: realResponse.data.dailyPnl,
          dailyPnlPercentage: realResponse.data.dailyPnlPercentage,
          positions: realResponse.data.positions || [],
        };
      }

      // Fallback to regular portfolio endpoint
      const response = await this.request<any>('/api/portfolio');

      if (response.success && response.data && response.data.totalValue > 0) {
        return {
          totalBalance: response.data.totalValue || response.data.simulatedBalance || 10000,
          availableBalance: response.data.availableBalance || response.data.simulatedBalance * 0.85 || 8500,
          totalPnl: response.data.totalPnL || response.data.dailyPnL || 250,
          totalPnlPercentage: response.data.totalPnLPercentage || 2.5,
          dailyPnl: response.data.dailyPnL || 250,
          dailyPnlPercentage: response.data.dailyPnLPercentage || 2.5,
          positions: response.data.positions || [],
        };
      }

      // Return minimal fallback data if all endpoints fail
      console.warn('⚠️ All portfolio endpoints failed, using minimal fallback');
      return {
        totalBalance: 0,
        availableBalance: 0,
        totalPnl: 0,
        totalPnlPercentage: 0,
        dailyPnl: 0,
        dailyPnlPercentage: 0,
        positions: [],
      };
    } catch (error) {
      console.error('Failed to fetch portfolio data:', error);
      // Return minimal fallback data instead of throwing
      return {
        totalBalance: 0,
        availableBalance: 0,
        totalPnl: 0,
        totalPnlPercentage: 0,
        dailyPnl: 0,
        dailyPnlPercentage: 0,
        positions: [],
      };
    }
  }

  async getPositions(): Promise<Position[]> {
    return this.request<Position[]>('/api/positions');
  }

  async getTrades(): Promise<Trade[]> {
    return this.request<Trade[]>('/api/trades');
  }

  // Market data endpoints
  async getMarketData(symbol?: string): Promise<MarketData[]> {
    try {
      // Try real market data first
      const realEndpoint = symbol ? `/api/real-market-data/${symbol}` : '/api/real-market-data';
      const realResponse = await this.request<any>(realEndpoint);

      if (realResponse.success) {
        const data = Array.isArray(realResponse.data) ? realResponse.data : [realResponse.data];
        const validData = data.filter((item: any) => item && item.price > 0);

        if (validData.length > 0) {
          return validData.map((item: any) => ({
            symbol: item.symbol,
            price: item.price,
            change24h: item.change24h || 0,
            changePercentage24h: item.changePercentage24h || 0,
            volume24h: item.volume24h || 0,
            high24h: item.high24h || item.price,
            low24h: item.low24h || item.price,
            timestamp: item.timestamp || new Date().toISOString(),
          }));
        }
      }

      // Fallback to regular market data endpoint
      const endpoint = symbol ? `/api/market-data/${symbol}` : '/api/market-data';
      const response = await this.request<any>(endpoint);

      if (response.success && response.data && Array.isArray(response.data)) {
        const validData = response.data.filter((item: any) => item.price > 0);

        if (validData.length > 0) {
          return validData.map((item: any) => ({
            symbol: item.symbol,
            price: item.price,
            change24h: item.change || item.change24h || 0,
            changePercentage24h: item.changePercent || item.changePercentage24h || 0,
            volume24h: item.volume || item.volume24h || 0,
            high24h: item.high24h || item.price,
            low24h: item.low24h || item.price,
            timestamp: item.timestamp || new Date().toISOString(),
          }));
        }
      }

      // Return minimal fallback data if all endpoints fail
      console.warn('⚠️ All market data endpoints failed, using minimal fallback');
      return [
        {
          symbol: 'BTCUSD',
          price: 0,
          change24h: 0,
          changePercentage24h: 0,
          volume24h: 0,
          high24h: 0,
          low24h: 0,
          timestamp: new Date().toISOString(),
        },
        {
          symbol: 'ETHUSD',
          price: 0,
          change24h: 0,
          changePercentage24h: 0,
          volume24h: 0,
          high24h: 0,
          low24h: 0,
          timestamp: new Date().toISOString(),
        },
        {
          symbol: 'SOLUSD',
          price: 0,
          change24h: 0,
          changePercentage24h: 0,
          volume24h: 0,
          high24h: 0,
          low24h: 0,
          timestamp: new Date().toISOString(),
        }
      ];
    } catch (error) {
      console.error('Failed to fetch market data:', error);
      // Return minimal fallback data instead of throwing
      return [
        {
          symbol: 'BTCUSD',
          price: 0,
          change24h: 0,
          changePercentage24h: 0,
          volume24h: 0,
          high24h: 0,
          low24h: 0,
          timestamp: new Date().toISOString(),
        },
        {
          symbol: 'ETHUSD',
          price: 0,
          change24h: 0,
          changePercentage24h: 0,
          volume24h: 0,
          high24h: 0,
          low24h: 0,
          timestamp: new Date().toISOString(),
        },
        {
          symbol: 'SOLUSD',
          price: 0,
          change24h: 0,
          changePercentage24h: 0,
          volume24h: 0,
          high24h: 0,
          low24h: 0,
          timestamp: new Date().toISOString(),
        }
      ];
    }
  }

  async getCandlestickData(symbol: string, timeframe: string = '1h'): Promise<any[]> {
    return this.request<any[]>(`/api/candlestick/${symbol}?timeframe=${timeframe}`);
  }

  // AI Model endpoints
  async getModelPredictions(): Promise<AIModelPrediction[]> {
    return this.request<AIModelPrediction[]>('/api/ai/predictions');
  }

  async getModelPerformance(): Promise<ModelPerformance> {
    return this.request<ModelPerformance>('/api/ai/performance');
  }

  // Trading endpoints
  async placeTrade(trade: Partial<Trade>): Promise<Trade> {
    return this.request<Trade>('/api/trades', {
      method: 'POST',
      body: JSON.stringify(trade),
    });
  }

  async closePosition(positionId: string): Promise<Position> {
    return this.request<Position>(`/api/positions/${positionId}/close`, {
      method: 'POST',
    });
  }

  // Delta Exchange Trading Methods
  async getTradingStatus(): Promise<ApiResponse> {
    try {
      return await this.request<ApiResponse>('/api/trading/status');
    } catch (error) {
      console.error('Failed to get trading status:', error);
      return {
        success: false,
        error: 'Failed to get trading status',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async getDeltaMarketData(symbol: string): Promise<ApiResponse<DeltaMarketData>> {
    try {
      return await this.request<ApiResponse<DeltaMarketData>>(`/api/trading/market-data/${symbol}`);
    } catch (error) {
      console.error(`Failed to get market data for ${symbol}:`, error);
      return {
        success: false,
        error: 'Failed to get market data',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async getDeltaPositions(): Promise<ApiResponse<Position[]>> {
    try {
      return await this.request<ApiResponse<Position[]>>('/api/trading/positions');
    } catch (error) {
      console.error('Failed to get positions:', error);
      return {
        success: false,
        data: [],
        error: 'Failed to get positions',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async getDeltaOrders(): Promise<ApiResponse<DeltaOrder[]>> {
    try {
      return await this.request<ApiResponse<DeltaOrder[]>>('/api/trading/orders');
    } catch (error) {
      console.error('Failed to get orders:', error);
      return {
        success: false,
        data: [],
        error: 'Failed to get orders',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async getDeltaBalances(): Promise<ApiResponse<DeltaBalance[]>> {
    try {
      return await this.request<ApiResponse<DeltaBalance[]>>('/api/trading/balances');
    } catch (error) {
      console.error('Failed to get balances:', error);
      return {
        success: false,
        data: [],
        error: 'Failed to get balances',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async placeDeltaOrder(orderData: {
    product_id: number;
    side: 'buy' | 'sell';
    size: number;
    order_type: 'market_order' | 'limit_order';
    limit_price?: number;
    leverage?: number;
    reduce_only?: boolean;
    post_only?: boolean;
  }): Promise<ApiResponse> {
    try {
      return await this.request<ApiResponse>('/api/trading/orders', {
        method: 'POST',
        body: JSON.stringify(orderData),
      });
    } catch (error) {
      console.error('Failed to place order:', error);
      return {
        success: false,
        error: 'Failed to place order',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async placeTradeWithTPSL(tradeData: {
    symbol?: string;
    side?: 'buy' | 'sell';
    size?: string;
    order_type?: 'market_order' | 'limit_order';
    take_profit_percentage?: number;
    stop_loss_percentage?: number;
  }): Promise<ApiResponse> {
    try {
      return await this.request<ApiResponse>('/api/trading/place-trade-with-tpsl', {
        method: 'POST',
        body: JSON.stringify(tradeData),
      });
    } catch (error) {
      console.error('Failed to place trade with TP/SL:', error);
      return {
        success: false,
        error: 'Failed to place trade with TP/SL',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async activateTradingBot(botConfig: {
    name?: string;
    strategy?: string;
    symbols?: string[];
    risk_per_trade?: number;
    take_profit?: number;
    stop_loss?: number;
    max_positions?: number;
    enabled?: boolean;
  }): Promise<ApiResponse> {
    try {
      return await this.request<ApiResponse>('/api/trading/activate-bot', {
        method: 'POST',
        body: JSON.stringify(botConfig),
      });
    } catch (error) {
      console.error('Failed to activate trading bot:', error);
      return {
        success: false,
        error: 'Failed to activate trading bot',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async getAvailableMarkets(): Promise<ApiResponse> {
    try {
      return await this.request<ApiResponse>('/api/markets');
    } catch (error) {
      console.error('Failed to get available markets:', error);
      // Return mock data for development
      return {
        success: true,
        data: [
          { id: 27, symbol: 'BTCUSD', name: 'Bitcoin', minSize: 0.001, tickSize: 0.5, status: 'active' },
          { id: 3136, symbol: 'ETHUSD', name: 'Ethereum', minSize: 0.01, tickSize: 0.05, status: 'active' },
          { id: 139, symbol: 'SOLUSD', name: 'Solana', minSize: 0.1, tickSize: 0.01, status: 'active' },
          { id: 4444, symbol: 'ADAUSD', name: 'Cardano', minSize: 1, tickSize: 0.001, status: 'active' },
          { id: 5555, symbol: 'BNBUSD', name: 'Binance Coin', minSize: 0.01, tickSize: 0.1, status: 'active' }
        ],
        message: 'Using mock market data for development'
      };
    }
  }
}

export const apiService = new ApiService();

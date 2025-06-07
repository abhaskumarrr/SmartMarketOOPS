/**
 * API Integration Layer
 * Centralized API client for all backend communications
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3005/api'
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001'

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
  timestamp?: number
}

export interface MarketData {
  symbol: string
  price: number
  changePercentage24h: number
  volume24h: number
  high24h: number
  low24h: number
  timestamp: number
}

export interface CandleData {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Portfolio {
  totalBalance: number
  availableBalance: number
  totalPnL: number
  totalPnLPercentage: number
  dayPnL: number
  dayPnLPercentage: number
  positions: Position[]
  recentTrades: Trade[]
}

export interface Position {
  id: string
  symbol: string
  side: 'long' | 'short'
  size: number
  entryPrice: number
  currentPrice: number
  pnl: number
  pnlPercentage: number
  leverage: number
  margin: number
  liquidationPrice: number
  stopLoss?: number
  takeProfit?: number
  timestamp: string
}

export interface Trade {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  size: number
  price: number
  timestamp: string
  status: 'filled' | 'pending' | 'cancelled'
  fee?: number
}

export interface Order {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop'
  size: number
  price?: number
  status: 'pending' | 'filled' | 'cancelled' | 'rejected'
  timestamp: string
  stopLoss?: number
  takeProfit?: number
}

export interface OrderRequest {
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop'
  size: number
  price?: number
  stopLoss?: number
  takeProfit?: number
  leverage?: number
}

// API Client Class
class ApiClient {
  private baseUrl: string
  private wsUrl: string

  constructor() {
    this.baseUrl = API_BASE_URL
    this.wsUrl = WS_BASE_URL
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseUrl}${endpoint}`
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  // Market Data APIs
  async getMarketData(symbols?: string[]): Promise<ApiResponse<MarketData[]>> {
    const symbolsParam = symbols ? `?symbols=${symbols.join(',')}` : ''
    return this.request<MarketData[]>(`/market-data${symbolsParam}`)
  }

  async getSymbolMarketData(symbol: string): Promise<ApiResponse<MarketData>> {
    return this.request<MarketData>(`/trading/market-data/${symbol}`)
  }

  async getCandleData(
    symbol: string,
    timeframe: string = '1h',
    limit: number = 100
  ): Promise<ApiResponse<{ candles: CandleData[] }>> {
    return this.request<{ candles: CandleData[] }>(
      `/trading/market-data/${symbol}?timeframe=${timeframe}&limit=${limit}`
    )
  }

  // Portfolio APIs
  async getPortfolio(): Promise<ApiResponse<Portfolio>> {
    return this.request<Portfolio>('/real-market-data/portfolio')
  }

  async getPositions(): Promise<ApiResponse<Position[]>> {
    return this.request<Position[]>('/positions')
  }

  async closePosition(positionId: string): Promise<ApiResponse<any>> {
    return this.request(`/positions/${positionId}/close`, {
      method: 'POST',
    })
  }

  // Trading APIs
  async placeOrder(orderRequest: OrderRequest): Promise<ApiResponse<Order>> {
    return this.request<Order>('/orders', {
      method: 'POST',
      body: JSON.stringify(orderRequest),
    })
  }

  async cancelOrder(orderId: string): Promise<ApiResponse<any>> {
    return this.request(`/orders/${orderId}`, {
      method: 'DELETE',
    })
  }

  async getOrders(): Promise<ApiResponse<Order[]>> {
    return this.request<Order[]>('/orders')
  }

  async getTrades(): Promise<ApiResponse<Trade[]>> {
    return this.request<Trade[]>('/trades/history')
  }

  // AI/ML APIs
  async getAIPredictions(): Promise<ApiResponse<any>> {
    return this.request('/ml/predictions')
  }

  async getModelPerformance(): Promise<ApiResponse<any>> {
    return this.request('/ml/models/performance')
  }

  // WebSocket Connection
  createWebSocketConnection(
    onMessage: (data: any) => void,
    onError?: (error: Event) => void,
    onClose?: (event: CloseEvent) => void
  ): WebSocket | null {
    try {
      const ws = new WebSocket(this.wsUrl)

      ws.onopen = () => {
        console.log('✅ WebSocket connected')
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage(data)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        onError?.(error)
      }

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        onClose?.(event)
      }

      return ws
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      return null
    }
  }

  // Health Check
  async healthCheck(): Promise<ApiResponse<any>> {
    return this.request('/health')
  }
}

// Export singleton instance
export const apiClient = new ApiClient()

// Utility functions for error handling
export const handleApiError = (response: ApiResponse<any>): string => {
  if (response.error) {
    return response.error
  }
  if (response.message) {
    return response.message
  }
  return 'An unknown error occurred'
}

export const isApiSuccess = <T>(response: ApiResponse<T>): response is ApiResponse<T> & { data: T } => {
  return response.success && response.data !== undefined
}

// Mock data generators for development
export const generateMockMarketData = (): MarketData[] => {
  const symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'DOTUSD', 'LINKUSD']
  return symbols.map(symbol => ({
    symbol,
    price: Math.random() * 50000 + 1000,
    changePercentage24h: (Math.random() - 0.5) * 10,
    volume24h: Math.random() * 1000000000,
    high24h: Math.random() * 55000 + 1000,
    low24h: Math.random() * 45000 + 1000,
    timestamp: Date.now(),
  }))
}

export const generateMockCandleData = (count: number = 100): CandleData[] => {
  const data: CandleData[] = []
  let price = 50000
  const now = Date.now()

  for (let i = count; i >= 0; i--) {
    const timestamp = now - i * 60 * 60 * 1000 // 1 hour intervals
    const open = price
    const change = (Math.random() - 0.5) * 1000
    const close = open + change
    const high = Math.max(open, close) + Math.random() * 500
    const low = Math.min(open, close) - Math.random() * 500

    data.push({
      timestamp: Math.floor(timestamp / 1000),
      open,
      high,
      low,
      close,
      volume: Math.random() * 1000000,
    })

    price = close
  }

  return data
}

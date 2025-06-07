/**
 * WebSocket Hook for Real-time Data
 * Manages WebSocket connections with automatic reconnection
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { apiClient } from '@/lib/api'

export interface WebSocketMessage {
  type: string
  data: any
  timestamp: number
}

export interface UseWebSocketOptions {
  reconnectInterval?: number
  maxReconnectAttempts?: number
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  autoConnect?: boolean
}

export interface UseWebSocketReturn {
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  lastMessage: WebSocketMessage | null
  sendMessage: (message: any) => void
  connect: () => void
  disconnect: () => void
  reconnect: () => void
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onConnect,
    onDisconnect,
    onError,
    autoConnect = true,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const shouldReconnectRef = useRef(true)

  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }, [])

  const handleMessage = useCallback((data: any) => {
    const message: WebSocketMessage = {
      type: data.type || 'unknown',
      data: data.data || data,
      timestamp: data.timestamp || Date.now(),
    }
    setLastMessage(message)
  }, [])

  const handleError = useCallback((error: Event) => {
    console.error('WebSocket error:', error)
    setError('WebSocket connection error')
    onError?.(error)
  }, [onError])

  const handleClose = useCallback((event: CloseEvent) => {
    console.log('WebSocket closed:', event.code, event.reason)
    setIsConnected(false)
    setIsConnecting(false)
    wsRef.current = null
    onDisconnect?.()

    // Attempt to reconnect if enabled and not manually disconnected
    if (shouldReconnectRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
      reconnectAttemptsRef.current++
      console.log(`Attempting to reconnect... (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`)
      
      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, reconnectInterval)
    } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      setError('Maximum reconnection attempts reached')
    }
  }, [maxReconnectAttempts, reconnectInterval, onDisconnect])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return // Already connected
    }

    if (isConnecting) {
      return // Already connecting
    }

    setIsConnecting(true)
    setError(null)
    clearReconnectTimeout()

    try {
      const ws = apiClient.createWebSocketConnection(
        handleMessage,
        handleError,
        handleClose
      )

      if (!ws) {
        throw new Error('Failed to create WebSocket connection')
      }

      wsRef.current = ws

      ws.onopen = () => {
        console.log('✅ WebSocket connected successfully')
        setIsConnected(true)
        setIsConnecting(false)
        setError(null)
        reconnectAttemptsRef.current = 0
        onConnect?.()
      }

    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      setError(error instanceof Error ? error.message : 'Connection failed')
      setIsConnecting(false)
    }
  }, [isConnecting, handleMessage, handleError, handleClose, onConnect, clearReconnectTimeout])

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false
    clearReconnectTimeout()

    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect')
      wsRef.current = null
    }

    setIsConnected(false)
    setIsConnecting(false)
    setError(null)
  }, [clearReconnectTimeout])

  const reconnect = useCallback(() => {
    disconnect()
    shouldReconnectRef.current = true
    reconnectAttemptsRef.current = 0
    setTimeout(connect, 100)
  }, [disconnect, connect])

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message))
      } catch (error) {
        console.error('Failed to send WebSocket message:', error)
        setError('Failed to send message')
      }
    } else {
      console.warn('WebSocket is not connected. Cannot send message.')
      setError('WebSocket not connected')
    }
  }, [])

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      shouldReconnectRef.current = true
      connect()
    }

    return () => {
      shouldReconnectRef.current = false
      clearReconnectTimeout()
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmount')
      }
    }
  }, [autoConnect, connect, clearReconnectTimeout])

  return {
    isConnected,
    isConnecting,
    error,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    reconnect,
  }
}

// Specialized hooks for different data types
export function useMarketDataWebSocket() {
  const [marketData, setMarketData] = useState<any[]>([])
  const [lastUpdate, setLastUpdate] = useState<number>(0)

  const { isConnected, error, lastMessage, ...rest } = useWebSocket({
    onConnect: () => console.log('📊 Market data WebSocket connected'),
    onDisconnect: () => console.log('📊 Market data WebSocket disconnected'),
  })

  useEffect(() => {
    if (lastMessage?.type === 'market_data') {
      setMarketData(lastMessage.data)
      setLastUpdate(lastMessage.timestamp)
    }
  }, [lastMessage])

  return {
    marketData,
    lastUpdate,
    isConnected,
    error,
    ...rest,
  }
}

export function useTradingSignalsWebSocket() {
  const [signals, setSignals] = useState<any[]>([])
  const [lastSignal, setLastSignal] = useState<any>(null)

  const { isConnected, error, lastMessage, ...rest } = useWebSocket({
    onConnect: () => console.log('🎯 Trading signals WebSocket connected'),
    onDisconnect: () => console.log('🎯 Trading signals WebSocket disconnected'),
  })

  useEffect(() => {
    if (lastMessage?.type === 'trading_signal') {
      const signal = lastMessage.data
      setLastSignal(signal)
      setSignals(prev => [signal, ...prev.slice(0, 9)]) // Keep last 10 signals
    }
  }, [lastMessage])

  return {
    signals,
    lastSignal,
    isConnected,
    error,
    ...rest,
  }
}

export function usePortfolioWebSocket() {
  const [portfolio, setPortfolio] = useState<any>(null)
  const [positions, setPositions] = useState<any[]>([])

  const { isConnected, error, lastMessage, ...rest } = useWebSocket({
    onConnect: () => console.log('💼 Portfolio WebSocket connected'),
    onDisconnect: () => console.log('💼 Portfolio WebSocket disconnected'),
  })

  useEffect(() => {
    if (lastMessage?.type === 'portfolio_update') {
      setPortfolio(lastMessage.data)
    } else if (lastMessage?.type === 'positions_update') {
      setPositions(lastMessage.data)
    }
  }, [lastMessage])

  return {
    portfolio,
    positions,
    isConnected,
    error,
    ...rest,
  }
}

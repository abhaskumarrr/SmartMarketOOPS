"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TrendingUp, TrendingDown, DollarSign, Target, AlertCircle, RefreshCw } from 'lucide-react'
import { apiClient, isApiSuccess, Order, OrderRequest } from '@/lib/api'
import { useMarketDataWebSocket } from '@/hooks/useWebSocket'

interface OrderFormData {
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop'
  size: string
  price: string
  stopLoss: string
  takeProfit: string
  leverage: string
}

interface Order {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop'
  size: number
  price: number
  status: 'pending' | 'filled' | 'cancelled' | 'rejected'
  timestamp: string
  stopLoss?: number
  takeProfit?: number
}

export function OrderManagement() {
  const [orderForm, setOrderForm] = useState<OrderFormData>({
    symbol: 'BTCUSD',
    side: 'buy',
    type: 'market',
    size: '',
    price: '',
    stopLoss: '',
    takeProfit: '',
    leverage: '10',
  })

  const [orders, setOrders] = useState<Order[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isLoadingOrders, setIsLoadingOrders] = useState(true)
  const [currentPrice, setCurrentPrice] = useState<Record<string, number>>({
    BTCUSD: 50000,
    ETHUSD: 3100,
    SOLUSD: 100,
  })

  // WebSocket for real-time market data
  const { marketData, isConnected } = useMarketDataWebSocket()

  // Load orders on component mount
  useEffect(() => {
    loadOrders()
  }, [])

  // Update current prices from WebSocket
  useEffect(() => {
    if (marketData && Array.isArray(marketData)) {
      const priceMap: Record<string, number> = {}
      marketData.forEach(data => {
        if (data.symbol && data.price) {
          priceMap[data.symbol] = data.price
        }
      })
      setCurrentPrice(prev => ({ ...prev, ...priceMap }))
    }
  }, [marketData])

  const loadOrders = async () => {
    try {
      setIsLoadingOrders(true)
      const response = await apiClient.getOrders()

      if (isApiSuccess(response)) {
        setOrders(response.data)
      } else {
        console.error('Failed to load orders:', response.error)
        // Load mock data as fallback
        loadMockOrders()
      }
    } catch (error) {
      console.error('Error loading orders:', error)
      loadMockOrders()
    } finally {
      setIsLoadingOrders(false)
    }
  }

  const loadMockOrders = () => {
    const mockOrders: Order[] = [
      {
        id: '1',
        symbol: 'BTCUSD',
        side: 'buy',
        type: 'limit',
        size: 0.1,
        price: 49000,
        status: 'pending',
        timestamp: new Date().toISOString(),
        stopLoss: 47000,
        takeProfit: 52000,
      },
      {
        id: '2',
        symbol: 'ETHUSD',
        side: 'sell',
        type: 'market',
        size: 1,
        price: 3100,
        status: 'filled',
        timestamp: new Date(Date.now() - 300000).toISOString(),
      },
    ]
    setOrders(mockOrders)
  }

  const symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD']
  const leverageOptions = ['1', '5', '10', '20', '50', '100']

  const handleInputChange = (field: keyof OrderFormData, value: string) => {
    setOrderForm(prev => ({ ...prev, [field]: value }))
  }

  const calculateOrderValue = () => {
    const size = parseFloat(orderForm.size) || 0
    const price = orderForm.type === 'market' 
      ? currentPrice[orderForm.symbol] || 0 
      : parseFloat(orderForm.price) || 0
    return size * price
  }

  const calculateMargin = () => {
    const orderValue = calculateOrderValue()
    const leverage = parseFloat(orderForm.leverage) || 1
    return orderValue / leverage
  }

  const handleSubmitOrder = async () => {
    setIsSubmitting(true)

    try {
      const orderRequest: OrderRequest = {
        symbol: orderForm.symbol,
        side: orderForm.side,
        type: orderForm.type,
        size: parseFloat(orderForm.size),
        price: orderForm.type === 'market' ? undefined : parseFloat(orderForm.price),
        stopLoss: orderForm.stopLoss ? parseFloat(orderForm.stopLoss) : undefined,
        takeProfit: orderForm.takeProfit ? parseFloat(orderForm.takeProfit) : undefined,
        leverage: parseFloat(orderForm.leverage),
      }

      const response = await apiClient.placeOrder(orderRequest)

      if (isApiSuccess(response)) {
        setOrders(prev => [response.data, ...prev])

        // Reset form
        setOrderForm(prev => ({
          ...prev,
          size: '',
          price: '',
          stopLoss: '',
          takeProfit: '',
        }))

        alert('Order placed successfully!')
      } else {
        throw new Error(response.error || 'Failed to place order')
      }
    } catch (error) {
      console.error('Error placing order:', error)
      alert(`Failed to place order: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleCancelOrder = async (orderId: string) => {
    try {
      const response = await apiClient.cancelOrder(orderId)

      if (isApiSuccess(response)) {
        setOrders(prev => prev.map(order =>
          order.id === orderId
            ? { ...order, status: 'cancelled' as const }
            : order
        ))
        alert('Order cancelled successfully!')
      } else {
        throw new Error(response.error || 'Failed to cancel order')
      }
    } catch (error) {
      console.error('Error cancelling order:', error)
      alert(`Failed to cancel order: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const getStatusColor = (status: Order['status']) => {
    switch (status) {
      case 'pending': return 'default'
      case 'filled': return 'default'
      case 'cancelled': return 'secondary'
      case 'rejected': return 'destructive'
      default: return 'secondary'
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Order Form */}
      <Card>
        <CardHeader>
          <CardTitle>Place Order</CardTitle>
          <CardDescription>Create a new trading order</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Tabs value={orderForm.side} onValueChange={(value) => handleInputChange('side', value)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="buy" className="text-green-600">Buy</TabsTrigger>
              <TabsTrigger value="sell" className="text-red-600">Sell</TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="symbol">Symbol</Label>
              <Select value={orderForm.symbol} onValueChange={(value) => handleInputChange('symbol', value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {symbols.map((symbol) => (
                    <SelectItem key={symbol} value={symbol}>
                      {symbol}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="type">Order Type</Label>
              <Select value={orderForm.type} onValueChange={(value) => handleInputChange('type', value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="market">Market</SelectItem>
                  <SelectItem value="limit">Limit</SelectItem>
                  <SelectItem value="stop">Stop</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="size">Size</Label>
              <Input
                id="size"
                type="number"
                step="0.001"
                placeholder="0.1"
                value={orderForm.size}
                onChange={(e) => handleInputChange('size', e.target.value)}
              />
            </div>

            {orderForm.type !== 'market' && (
              <div className="space-y-2">
                <Label htmlFor="price">Price</Label>
                <Input
                  id="price"
                  type="number"
                  step="0.01"
                  placeholder="50000"
                  value={orderForm.price}
                  onChange={(e) => handleInputChange('price', e.target.value)}
                />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="leverage">Leverage</Label>
              <Select value={orderForm.leverage} onValueChange={(value) => handleInputChange('leverage', value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {leverageOptions.map((lev) => (
                    <SelectItem key={lev} value={lev}>
                      {lev}x
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="stopLoss">Stop Loss (Optional)</Label>
              <Input
                id="stopLoss"
                type="number"
                step="0.01"
                placeholder="47000"
                value={orderForm.stopLoss}
                onChange={(e) => handleInputChange('stopLoss', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="takeProfit">Take Profit (Optional)</Label>
              <Input
                id="takeProfit"
                type="number"
                step="0.01"
                placeholder="55000"
                value={orderForm.takeProfit}
                onChange={(e) => handleInputChange('takeProfit', e.target.value)}
              />
            </div>
          </div>

          {/* Order Summary */}
          <div className="p-4 bg-muted rounded-lg space-y-2">
            <div className="flex justify-between text-sm">
              <span>Order Value:</span>
              <span>${calculateOrderValue().toLocaleString()}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span>Required Margin:</span>
              <span>${calculateMargin().toLocaleString()}</span>
            </div>
            <div className="flex justify-between text-sm font-medium">
              <span>Current Price:</span>
              <span>${(currentPrice[orderForm.symbol] || 0).toLocaleString()}</span>
            </div>
          </div>

          <Button 
            onClick={handleSubmitOrder} 
            disabled={isSubmitting || !orderForm.size}
            className={`w-full ${orderForm.side === 'buy' ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'}`}
          >
            {isSubmitting ? 'Placing Order...' : `${orderForm.side.toUpperCase()} ${orderForm.symbol}`}
          </Button>
        </CardContent>
      </Card>

      {/* Order History */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle>Order History</CardTitle>
            <CardDescription>Your recent and pending orders</CardDescription>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant={isConnected ? 'default' : 'secondary'}>
              {isConnected ? '🟢 Live' : '🔴 Offline'}
            </Badge>
            <Button
              variant="outline"
              size="sm"
              onClick={loadOrders}
              disabled={isLoadingOrders}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoadingOrders ? 'animate-spin' : ''}`} />
              {isLoadingOrders ? 'Loading...' : 'Refresh'}
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoadingOrders ? (
            <div className="text-center py-8">
              <RefreshCw className="h-8 w-8 text-muted-foreground mx-auto mb-4 animate-spin" />
              <p className="text-muted-foreground">Loading orders...</p>
            </div>
          ) : orders.length === 0 ? (
            <div className="text-center py-8">
              <Target className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No orders yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {orders.map((order) => (
                <div key={order.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    {order.side === 'buy' ? (
                      <TrendingUp className="h-4 w-4 text-green-500" />
                    ) : (
                      <TrendingDown className="h-4 w-4 text-red-500" />
                    )}
                    <div>
                      <div className="font-medium">{order.symbol}</div>
                      <div className="text-sm text-muted-foreground">
                        {order.type.toUpperCase()} • {order.size} @ ${order.price?.toLocaleString() || 'Market'}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant={getStatusColor(order.status)}>
                      {order.status}
                    </Badge>
                    {order.status === 'pending' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleCancelOrder(order.id)}
                      >
                        Cancel
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

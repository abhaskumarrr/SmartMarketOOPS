"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TrendingUp, TrendingDown, DollarSign, Target, AlertCircle, RefreshCw, Wifi, WifiOff, Zap } from 'lucide-react'
import { useDeltaExchange } from '@/hooks/useDeltaExchange'
import { DeltaOrderRequest, DeltaOrder } from '@/services/deltaService'

interface OrderFormData {
  symbol: string
  side: 'buy' | 'sell'
  orderType: 'market_order' | 'limit_order' | 'stop_market_order' | 'stop_limit_order'
  size: string
  limitPrice: string
  stopPrice: string
  leverage: string
  reduceOnly: boolean
  postOnly: boolean
}

interface OrderManagementProps {
  symbols?: string[]
}

export default function OrderManagement({ symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'BNBUSD'] }: OrderManagementProps) {
  const [activeTab, setActiveTab] = useState('place-order')
  const [orderForm, setOrderForm] = useState<OrderFormData>({
    symbol: symbols[0],
    side: 'buy',
    orderType: 'market_order',
    size: '',
    limitPrice: '',
    stopPrice: '',
    leverage: '10',
    reduceOnly: false,
    postOnly: false
  })

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [lastOrderResult, setLastOrderResult] = useState<string | null>(null)

  // Enhanced Delta Exchange hook with real-time updates
  const {
    isConnected,
    isLoading,
    error,
    marketData,
    orders,
    positions,
    placeOrder,
    cancelOrder,
    refreshData,
    getRealTimePrice,
    hasRealTimeData,
    lastTrade,
    lastPortfolioUpdate
  } = useDeltaExchange({ 
    symbols, 
    autoRefresh: true, 
    refreshInterval: 5000 // Reduced interval since we have real-time data
  })

  // Auto-update limit price when symbol changes (using real-time price)
  useEffect(() => {
    const currentPrice = getRealTimePrice(orderForm.symbol)
    if (currentPrice > 0 && orderForm.orderType === 'limit_order') {
      setOrderForm(prev => ({
        ...prev,
        limitPrice: currentPrice.toString()
      }))
    }
  }, [orderForm.symbol, orderForm.orderType, getRealTimePrice])

  // Show notification when new trade signals are received
  useEffect(() => {
    if (lastTrade && lastTrade.symbol === orderForm.symbol) {
      setLastOrderResult(`🎯 New ${lastTrade.side} signal for ${lastTrade.symbol} at $${lastTrade.price}`)
      
      // Clear notification after 5 seconds
      setTimeout(() => setLastOrderResult(null), 5000)
    }
  }, [lastTrade, orderForm.symbol])

  const handleInputChange = (field: keyof OrderFormData, value: string) => {
    setOrderForm(prev => ({ ...prev, [field]: value }))
    
    // Clear any previous order result when form changes
    if (lastOrderResult) {
      setLastOrderResult(null)
    }
  }

  const getCurrentPrice = (symbol: string): number => {
    return marketData[symbol]?.price || 0
  }

  const calculateOrderValue = () => {
    const size = parseFloat(orderForm.size) || 0
    const price = orderForm.orderType === 'market_order' 
      ? getCurrentPrice(orderForm.symbol) 
      : parseFloat(orderForm.limitPrice) || 0
    return size * price
  }

  const calculateMargin = () => {
    const orderValue = calculateOrderValue()
    const leverage = parseFloat(orderForm.leverage) || 1
    return orderValue / leverage
  }

  const handleSubmitOrder = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!isConnected) {
      setLastOrderResult('❌ Not connected to Delta Exchange API')
      return
    }

    if (!orderForm.size || parseFloat(orderForm.size) <= 0) {
      setLastOrderResult('❌ Please enter a valid order size')
      return
    }

    if (orderForm.orderType === 'limit_order' && (!orderForm.limitPrice || parseFloat(orderForm.limitPrice) <= 0)) {
      setLastOrderResult('❌ Please enter a valid limit price')
      return
    }

    setIsSubmitting(true)
    setLastOrderResult(null)

    try {
      const orderRequest: DeltaOrderRequest = {
        symbol: orderForm.symbol,
        side: orderForm.side,
        orderType: orderForm.orderType,
        size: parseFloat(orderForm.size),
        leverage: parseInt(orderForm.leverage)
      }

      if (orderForm.orderType === 'limit_order') {
        orderRequest.limitPrice = parseFloat(orderForm.limitPrice)
      }

      const result = await placeOrder(orderRequest)
      
      if (result) {
        setLastOrderResult(`✅ Order placed successfully! ID: ${result.id}`)
        
        // Reset form for new order
        setOrderForm(prev => ({
          ...prev,
          size: '',
          limitPrice: getRealTimePrice(prev.symbol).toString()
        }))
      } else {
        setLastOrderResult('❌ Failed to place order. Please try again.')
      }
    } catch (error) {
      setLastOrderResult('❌ Error placing order: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleCancelOrder = async (orderId: number) => {
    try {
      setIsSubmitting(true)
      const success = await cancelOrder(orderId)
      
      if (success) {
        setLastOrderResult(`✅ Order ${orderId} cancelled successfully`)
      } else {
        setLastOrderResult(`❌ Failed to cancel order ${orderId}`)
      }
    } catch (error) {
      setLastOrderResult('❌ Error cancelling order: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      setIsSubmitting(false)
    }
  }

  const getStatusColor = (state: string) => {
    switch (state.toLowerCase()) {
      case 'open':
      case 'pending':
        return 'bg-blue-100 text-blue-800'
      case 'filled':
      case 'done':
        return 'bg-green-100 text-green-800'
      case 'cancelled':
        return 'bg-gray-100 text-gray-800'
      case 'rejected':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-yellow-100 text-yellow-800'
    }
  }

  // Determine if we can show limit price input
  const showLimitPrice = orderForm.orderType === 'limit_order' || orderForm.orderType === 'stop_limit_order'
  
  // Determine if we can show stop price input
  const showStopPrice = orderForm.orderType === 'stop_market_order' || orderForm.orderType === 'stop_limit_order'

  const formatPrice = (price: number | string) => {
    const numPrice = typeof price === 'string' ? parseFloat(price) : price
    return isNaN(numPrice) ? '0.00' : numPrice.toLocaleString('en-US', { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 8 
    })
  }

  const currentSymbolData = marketData[orderForm.symbol]
  const currentPrice = getRealTimePrice(orderForm.symbol)
  const isRealTimeData = hasRealTimeData(orderForm.symbol)

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Trading Status</CardTitle>
            <div className="flex items-center gap-2">
              {isConnected ? (
                <>
                  <Wifi className="w-4 h-4 text-green-500" />
                  <span className="text-sm text-green-600">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-4 h-4 text-red-500" />
                  <span className="text-sm text-red-600">Disconnected</span>
                </>
              )}
            </div>
          </div>
        </CardHeader>
        {error && (
          <CardContent className="pt-0">
            <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-md">
              <AlertCircle className="w-4 h-4 text-red-500" />
              <span className="text-sm text-red-700">{error}</span>
            </div>
          </CardContent>
        )}
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="place-order">Place Order</TabsTrigger>
          <TabsTrigger value="manage-orders">Manage Orders</TabsTrigger>
        </TabsList>

        <TabsContent value="place-order" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5" />
                Place New Order
              </CardTitle>
              <CardDescription>
                Create a new trading order with real-time market data
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Market Data Display */}
              {currentSymbolData && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold">{orderForm.symbol}</h3>
                      {isRealTimeData && (
                        <div className="flex items-center gap-1">
                          <Zap className="w-3 h-3 text-green-500" />
                          <span className="text-xs text-green-600">Live</span>
                        </div>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold">${formatPrice(currentPrice)}</div>
                      <div className={`text-sm ${currentSymbolData.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        24h: {currentSymbolData.change >= 0 ? '+' : ''}{currentSymbolData.change.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                    <div>Volume: {formatPrice(currentSymbolData.volume)}</div>
                    <div>Updated: {new Date(currentSymbolData.timestamp).toLocaleTimeString()}</div>
                  </div>
                </div>
              )}

              <form onSubmit={handleSubmitOrder} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="symbol">Symbol</Label>
                    <Select 
                      value={orderForm.symbol} 
                      onValueChange={(value) => handleInputChange('symbol', value)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select symbol" />
                      </SelectTrigger>
                      <SelectContent>
                        {symbols.map((symbol) => (
                          <SelectItem key={symbol} value={symbol}>
                            <div className="flex items-center justify-between w-full">
                              <span>{symbol}</span>
                              {marketData[symbol] && (
                                <span className="ml-2 text-sm text-gray-500">
                                  ${formatPrice(marketData[symbol].price)}
                                </span>
                              )}
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="side">Side</Label>
                    <Select 
                      value={orderForm.side} 
                      onValueChange={(value) => handleInputChange('side', value as 'buy' | 'sell')}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="buy">
                          <div className="flex items-center gap-2">
                            <TrendingUp className="w-4 h-4 text-green-500" />
                            Buy
                          </div>
                        </SelectItem>
                        <SelectItem value="sell">
                          <div className="flex items-center gap-2">
                            <TrendingDown className="w-4 h-4 text-red-500" />
                            Sell
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="orderType">Order Type</Label>
                    <Select 
                      value={orderForm.orderType} 
                      onValueChange={(value) => handleInputChange('orderType', value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="market_order">Market Order</SelectItem>
                        <SelectItem value="limit_order">Limit Order</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="leverage">Leverage</Label>
                    <Select 
                      value={orderForm.leverage} 
                      onValueChange={(value) => handleInputChange('leverage', value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">1x</SelectItem>
                        <SelectItem value="5">5x</SelectItem>
                        <SelectItem value="10">10x</SelectItem>
                        <SelectItem value="20">20x</SelectItem>
                        <SelectItem value="50">50x</SelectItem>
                        <SelectItem value="100">100x</SelectItem>
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
                      placeholder="Enter order size"
                      value={orderForm.size}
                      onChange={(e) => handleInputChange('size', e.target.value)}
                      required
                    />
                  </div>

                  {orderForm.orderType === 'limit_order' && (
                    <div className="space-y-2">
                      <Label htmlFor="limitPrice">Limit Price</Label>
                      <Input
                        id="limitPrice"
                        type="number"
                        step="0.01"
                        placeholder="Enter limit price"
                        value={orderForm.limitPrice}
                        onChange={(e) => handleInputChange('limitPrice', e.target.value)}
                        required
                      />
                    </div>
                  )}
                </div>

                <Button 
                  type="submit" 
                  disabled={isSubmitting || !isConnected}
                  className="w-full"
                >
                  {isSubmitting ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Placing Order...
                    </>
                  ) : (
                    <>
                      <DollarSign className="w-4 h-4 mr-2" />
                      Place {orderForm.side === 'buy' ? 'Buy' : 'Sell'} Order
                    </>
                  )}
                </Button>
              </form>

              {lastOrderResult && (
                <div className={`p-3 rounded-md border ${
                  lastOrderResult.includes('✅') ? 'bg-green-50 border-green-200 text-green-700' : 'bg-red-50 border-red-200 text-red-700'
                }`}>
                  {lastOrderResult}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="manage-orders" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Open Orders</CardTitle>
                  <CardDescription>
                    Manage your active trading orders with real-time updates
                  </CardDescription>
                </div>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={refreshData}
                  disabled={isLoading}
                >
                  <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {orders.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  No open orders found
                </div>
              ) : (
                <div className="space-y-4">
                  {orders.map((order) => (
                    <div 
                      key={order.id} 
                      className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold">{order.symbol}</span>
                            <Badge className={getStatusColor(order.state)}>
                              {order.state}
                            </Badge>
                            <span className={`text-sm font-medium ${
                              order.side === 'buy' ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {order.side.toUpperCase()}
                            </span>
                          </div>
                          <div className="text-sm text-gray-600">
                            Size: {order.size} | Type: {order.orderType}
                            {order.limitPrice && ` | Limit: $${formatPrice(order.limitPrice)}`}
                          </div>
                          <div className="text-xs text-gray-500">
                            Created: {new Date(order.createdAt).toLocaleString()}
                            {order.filledSize > 0 && (
                              <span className="ml-2">
                                Filled: {order.filledSize}/{order.size}
                              </span>
                            )}
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          {marketData[order.symbol] && (
                            <div className="text-right text-sm">
                              <div className="font-medium">
                                ${formatPrice(marketData[order.symbol].price)}
                              </div>
                              {hasRealTimeData(order.symbol) && (
                                <div className="flex items-center gap-1 text-green-600">
                                  <Zap className="w-3 h-3" />
                                  <span className="text-xs">Live</span>
                                </div>
                              )}
                            </div>
                          )}
                          
                          {order.state === 'open' && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleCancelOrder(order.id)}
                              disabled={isSubmitting}
                            >
                              Cancel
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Real-time Updates Info */}
      {lastPortfolioUpdate && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Latest Portfolio Update</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Total Balance:</span>
                <div className="font-semibold">${formatPrice(lastPortfolioUpdate.totalBalance)}</div>
              </div>
              <div>
                <span className="text-gray-600">Total PnL:</span>
                <div className={`font-semibold ${lastPortfolioUpdate.totalPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${formatPrice(lastPortfolioUpdate.totalPnL)} ({lastPortfolioUpdate.totalPnLPercentage.toFixed(2)}%)
                </div>
              </div>
            </div>
            <div className="text-xs text-gray-500 mt-2">
              Updated: {new Date(lastPortfolioUpdate.timestamp).toLocaleString()}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

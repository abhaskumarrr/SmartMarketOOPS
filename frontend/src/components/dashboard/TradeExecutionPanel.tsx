'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Switch } from '../ui/switch';
import { Badge } from '../ui/badge';
import { Slider } from '../ui/slider';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle,
  RefreshCw,
  DollarSign,
  Loader2
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useRealTimeData } from '@/hooks/useRealTimeData';
import { toast } from '../ui/use-toast';
import axios from 'axios';

interface TradeExecutionPanelProps {
  className?: string;
  symbol?: string; // Optional symbol to pre-select
  compact?: boolean; // Optional compact mode for smaller screens
}

interface Market {
  id: number;
  symbol: string;
  name: string;
  minSize: number;
  tickSize: number;
  status: string;
}

interface OrderFormState {
  symbol: string;
  side: 'buy' | 'sell';
  orderType: 'market' | 'limit';
  size: string;
  price: string;
  leverage: number;
  reduceOnly: boolean;
  postOnly: boolean;
}

export default function TradeExecutionPanel({ className, symbol, compact = false }: TradeExecutionPanelProps) {
  const { isConnected, lastMarketData } = useRealTimeData();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [markets, setMarkets] = useState<Market[]>([]);
  const [loadingMarkets, setLoadingMarkets] = useState(true);
  const [orderForm, setOrderForm] = useState<OrderFormState>({
    symbol: symbol || '',
    side: 'buy',
    orderType: 'market',
    size: '',
    price: '',
    leverage: 5,
    reduceOnly: false,
    postOnly: false
  });

  // Fetch available markets
  useEffect(() => {
    const fetchMarkets = async () => {
      try {
        setLoadingMarkets(true);
        const response = await axios.get('/api/markets');
        
        if (response.data && response.data.success) {
          setMarkets(response.data.data);
          
          // Set default symbol if available and not provided as prop
          if (response.data.data.length > 0 && !orderForm.symbol) {
            const defaultSymbol = symbol || response.data.data[0].symbol;
            setOrderForm(prev => ({ ...prev, symbol: defaultSymbol }));
          }
        } else {
          console.error('Failed to fetch markets:', response.data?.message || 'Unknown error');
          // Use mock markets data for development
          const mockMarkets = getMockMarkets();
          setMarkets(mockMarkets);
          setOrderForm(prev => ({ ...prev, symbol: symbol || mockMarkets[0].symbol }));
        }
      } catch (error) {
        console.error('Error fetching markets:', error);
        // Use mock markets data for development
        const mockMarkets = getMockMarkets();
        setMarkets(mockMarkets);
        setOrderForm(prev => ({ ...prev, symbol: symbol || mockMarkets[0].symbol }));
      } finally {
        setLoadingMarkets(false);
      }
    };

    fetchMarkets();
  }, [symbol]);

  // Update when symbol prop changes
  useEffect(() => {
    if (symbol && symbol !== orderForm.symbol) {
      setOrderForm(prev => ({ ...prev, symbol }));
    }
  }, [symbol]);

  // Update price field when market data changes
  useEffect(() => {
    if (!lastMarketData || !orderForm.symbol || Object.keys(lastMarketData).length === 0) {
      return;
    }

    const symbolData = lastMarketData[orderForm.symbol];
    if (symbolData) {
      // Only update price if using market order or price not manually set
      if (orderForm.orderType === 'market' || !orderForm.price) {
        setOrderForm(prev => ({
          ...prev,
          price: symbolData.price.toFixed(2)
        }));
      }
    }
  }, [lastMarketData, orderForm.symbol, orderForm.orderType]);

  // Generate mock markets for development
  const getMockMarkets = (): Market[] => {
    return [
      { id: 1, symbol: 'BTCUSD', name: 'Bitcoin', minSize: 0.001, tickSize: 0.5, status: 'active' },
      { id: 2, symbol: 'ETHUSD', name: 'Ethereum', minSize: 0.01, tickSize: 0.05, status: 'active' },
      { id: 3, symbol: 'SOLUSD', name: 'Solana', minSize: 0.1, tickSize: 0.01, status: 'active' },
      { id: 4, symbol: 'AVAXUSD', name: 'Avalanche', minSize: 0.1, tickSize: 0.01, status: 'active' },
      { id: 5, symbol: 'BNBUSD', name: 'Binance Coin', minSize: 0.01, tickSize: 0.1, status: 'active' }
    ];
  };

  // Get current market price for selected symbol
  const getCurrentPrice = (): number => {
    if (lastMarketData && orderForm.symbol && lastMarketData[orderForm.symbol]) {
      return lastMarketData[orderForm.symbol].price;
    }
    
    // Fallback prices
    const fallbackPrices: Record<string, number> = {
      'BTCUSD': 48250.45,
      'ETHUSD': 2870.12,
      'SOLUSD': 106.78,
      'AVAXUSD': 34.56,
      'BNBUSD': 570.23
    };
    
    return fallbackPrices[orderForm.symbol] || 0;
  };

  // Handle form input changes
  const handleInputChange = (field: keyof OrderFormState, value: any) => {
    setOrderForm(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // Calculate order value
  const calculateOrderValue = (): number => {
    if (!orderForm.size) return 0;
    
    const size = parseFloat(orderForm.size);
    const price = orderForm.orderType === 'limit' && orderForm.price 
      ? parseFloat(orderForm.price) 
      : getCurrentPrice();
    
    return size * price;
  };

  // Handle order submission
  const handleSubmitOrder = async () => {
    try {
      // Validate order parameters
      if (!orderForm.symbol) {
        toast({
          title: 'Missing symbol',
          description: 'Please select a trading pair',
          variant: 'destructive',
        });
        return;
      }

      if (!orderForm.size || parseFloat(orderForm.size) <= 0) {
        toast({
          title: 'Invalid size',
          description: 'Please enter a valid order size',
          variant: 'destructive',
        });
        return;
      }

      if (orderForm.orderType === 'limit' && (!orderForm.price || parseFloat(orderForm.price) <= 0)) {
        toast({
          title: 'Invalid price',
          description: 'Please enter a valid limit price',
          variant: 'destructive',
        });
        return;
      }

      setIsSubmitting(true);

      // Get selected market
      const selectedMarket = markets.find(m => m.symbol === orderForm.symbol);
      if (!selectedMarket) {
        throw new Error('Selected market not found');
      }

      // Prepare order request
      const orderRequest = {
        product_id: selectedMarket.id,
        side: orderForm.side,
        size: parseFloat(orderForm.size),
        order_type: orderForm.orderType === 'market' ? 'market_order' : 'limit_order',
        leverage: orderForm.leverage,
        reduce_only: orderForm.reduceOnly,
        post_only: orderForm.postOnly && orderForm.orderType === 'limit'
      };

      // Add limit price if applicable
      if (orderForm.orderType === 'limit') {
        orderRequest.limit_price = orderForm.price;
      }

      // Send order request to API
      const response = await axios.post('/api/trading/orders', orderRequest);

      if (response.data && response.data.success) {
        toast({
          title: 'Order placed successfully',
          description: `${orderForm.side.toUpperCase()} ${orderForm.size} ${orderForm.symbol}`,
          variant: 'default',
        });

        // Reset form fields except symbol and side
        setOrderForm(prev => ({
          ...prev,
          size: '',
          price: orderForm.orderType === 'market' ? prev.price : '',
        }));
      } else {
        throw new Error(response.data?.message || 'Failed to place order');
      }
    } catch (error) {
      console.error('Error placing order:', error);
      
      toast({
        title: 'Order Failed',
        description: error instanceof Error ? error.message : 'Failed to place order',
        variant: 'destructive',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={cn("flex flex-col", className)}>
      <div className={cn(
        "flex items-center justify-between",
        compact ? "px-3 py-2" : "px-4 py-3"
      )}>
        <div className="flex items-center">
          <Badge variant={isConnected ? "outline" : "destructive"} className="mr-2">
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
          {!compact && (
            <Badge variant="secondary">
              Mode: {process.env.NODE_ENV === 'production' ? 'Live' : 'Test'}
            </Badge>
          )}
        </div>
        <Button variant="ghost" size="sm" onClick={() => window.location.reload()}>
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      <Tabs defaultValue="market" className={compact ? "px-3 pb-3" : "px-4 pb-4"}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="market" className={compact ? "text-xs py-1" : ""}>Market</TabsTrigger>
          <TabsTrigger value="limit" className={compact ? "text-xs py-1" : ""}>Limit</TabsTrigger>
        </TabsList>

        <div className={cn("space-y-3", compact ? "mt-2" : "mt-4")}>
          {/* Symbol Selector */}
          {(!symbol || compact === false) && (
            <div className="space-y-1">
              <Label htmlFor="symbol" className={cn("text-xs", compact && "text-xs")}>Symbol</Label>
              <Select 
                disabled={loadingMarkets} 
                value={orderForm.symbol} 
                onValueChange={(value) => handleInputChange('symbol', value)}
              >
                <SelectTrigger id="symbol" className={cn(compact ? "h-8 text-xs" : "text-sm")}>
                  <SelectValue placeholder="Select Symbol" />
                </SelectTrigger>
                <SelectContent>
                  {markets.map((market) => (
                    <SelectItem key={market.id} value={market.symbol}>
                      {market.symbol} - {market.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Size and Price Inputs */}
          <div className={compact ? "grid grid-cols-2 gap-2" : "space-y-3"}>
            {/* Size Input */}
            <div className="space-y-1">
              <Label htmlFor="size" className="text-xs">Size</Label>
              <div className="relative">
                <Input
                  id="size"
                  type="number"
                  value={orderForm.size}
                  onChange={(e) => handleInputChange('size', e.target.value)}
                  placeholder="Enter size"
                  step="0.001"
                  min="0"
                  className={cn(compact ? "h-8 text-xs py-1" : "text-sm")}
                />
              </div>
            </div>

            {/* Price Input (for Limit Orders) */}
            {orderForm.orderType === 'limit' && (
              <div className="space-y-1">
                <Label htmlFor="price" className="text-xs">Price</Label>
                <Input
                  id="price"
                  type="number"
                  value={orderForm.price}
                  onChange={(e) => handleInputChange('price', e.target.value)}
                  placeholder="Enter price"
                  step="0.01"
                  min="0"
                  className={cn(compact ? "h-8 text-xs py-1" : "text-sm")}
                />
              </div>
            )}
          </div>

          {/* Leverage Slider */}
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <Label htmlFor="leverage" className="text-xs">Leverage: {orderForm.leverage}x</Label>
            </div>
            <Slider
              id="leverage"
              defaultValue={[5]}
              max={20}
              min={1}
              step={1}
              value={[orderForm.leverage]}
              onValueChange={(value) => handleInputChange('leverage', value[0])}
              className={compact ? "py-1" : "py-2"}
            />
          </div>

          {/* Order Value */}
          <div className="border rounded p-2 flex justify-between items-center">
            <span className={cn("text-muted-foreground", compact ? "text-xs" : "text-sm")}>Order Value</span>
            <span className={compact ? "text-sm font-medium" : "text-base font-medium"}>
              ${calculateOrderValue().toFixed(2)}
            </span>
          </div>

          {/* Additional Options - Condensed on compact mode */}
          {compact ? (
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center gap-1">
                <Switch
                  id="reduce-only-compact"
                  checked={orderForm.reduceOnly}
                  onCheckedChange={(checked) => handleInputChange('reduceOnly', checked)}
                  className="scale-75"
                />
                <Label htmlFor="reduce-only-compact" className="text-xs">Reduce Only</Label>
              </div>
              {orderForm.orderType === 'limit' && (
                <div className="flex items-center gap-1">
                  <Switch
                    id="post-only-compact"
                    checked={orderForm.postOnly}
                    onCheckedChange={(checked) => handleInputChange('postOnly', checked)}
                    disabled={orderForm.orderType !== 'limit'}
                    className="scale-75"
                  />
                  <Label htmlFor="post-only-compact" className="text-xs">Post Only</Label>
                </div>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              <div className="flex items-center space-x-2">
                <Switch
                  id="reduce-only"
                  checked={orderForm.reduceOnly}
                  onCheckedChange={(checked) => handleInputChange('reduceOnly', checked)}
                />
                <Label htmlFor="reduce-only" className="text-xs">Reduce Only</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="post-only"
                  checked={orderForm.postOnly}
                  onCheckedChange={(checked) => handleInputChange('postOnly', checked)}
                  disabled={orderForm.orderType !== 'limit'}
                />
                <Label htmlFor="post-only" className="text-xs">Post Only</Label>
              </div>
            </div>
          )}

          {/* Buy/Sell Buttons */}
          <div className="grid grid-cols-2 gap-2 pt-2">
            <Button 
              variant="default" 
              className="bg-green-500 hover:bg-green-600 text-white w-full"
              size={compact ? "sm" : "default"}
              onClick={() => {
                handleInputChange('side', 'buy');
                handleSubmitOrder();
              }}
              disabled={isSubmitting}
            >
              {isSubmitting && orderForm.side === 'buy' ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <TrendingUp className="h-4 w-4 mr-2" />
              )}
              <span className={compact ? "text-xs" : ""}>Buy</span>
            </Button>
            <Button 
              variant="default" 
              className="bg-red-500 hover:bg-red-600 text-white w-full"
              size={compact ? "sm" : "default"}
              onClick={() => {
                handleInputChange('side', 'sell');
                handleSubmitOrder();
              }}
              disabled={isSubmitting}
            >
              {isSubmitting && orderForm.side === 'sell' ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <TrendingDown className="h-4 w-4 mr-2" />
              )}
              <span className={compact ? "text-xs" : ""}>Sell</span>
            </Button>
          </div>
        </div>
      </Tabs>
    </div>
  );
} 
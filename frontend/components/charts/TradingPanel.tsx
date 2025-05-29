import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  InputAdornment,
  InputLabel,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Select,
  SelectChangeEvent,
  Slider,
  Tab,
  Tabs,
  TextField,
  Typography,
  useTheme
} from '@mui/material';
import { useSnackbar } from 'notistack';
import { useTradingContext } from '../../lib/contexts/TradingContext';
import deltaExchangeApi from '../../lib/api/deltaExchangeApi';

interface Product {
  id: number;
  symbol: string;
  description: string;
  base_currency: string;
  quote_currency: string;
  last_price: number;
  min_price_increment: number;
  min_size: number;
  size_increment: number;
  max_leverage: number;
}

interface TradingPanelProps {
  defaultSymbol?: string;
}

const TradingPanel: React.FC<TradingPanelProps> = ({ defaultSymbol = 'BTCUSDT' }) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  const { marketData } = useTradingContext();
  
  // State
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [tabValue, setTabValue] = useState<string>('limit');
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [price, setPrice] = useState<string>('');
  const [amount, setAmount] = useState<string>('');
  const [leverage, setLeverage] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [marketPrice, setMarketPrice] = useState<number | null>(null);
  const [orderBookData, setOrderBookData] = useState<{
    bids: [number, number][];
    asks: [number, number][];
  }>({ bids: [], asks: [] });
  
  // Fetch product details (orderbook, etc)
  const fetchProductDetails = useCallback(async (symbol: string) => {
    try {
      const marketData = await deltaExchangeApi.getMarketData(symbol);
      if (marketData?.success && marketData.data) {
        const { ticker, orderbook } = marketData.data;
        
        if (ticker) {
          setMarketPrice(ticker.last_price);
        }
        
        if (orderbook) {
          setOrderBookData({
            bids: orderbook.bids.slice(0, 5),
            asks: orderbook.asks.slice(0, 5).reverse()
          });
        }
      }
    } catch (error) {
      console.error('Error fetching product details:', error);
    }
  }, []);
  
  // Load products and set default
  useEffect(() => {
    const fetchProducts = async () => {
      try {
        setLoading(true);
        const response = await deltaExchangeApi.getProducts();
        if (response?.success && response.data) {
          const availableProducts = response.data.filter(
            (product: any) => product.status === 'live'
          );
          setProducts(availableProducts);
          
          // Set default product
          const defaultProduct = availableProducts.find(
            (p: any) => p.symbol === defaultSymbol
          );
          
          if (defaultProduct) {
            setSelectedProduct(defaultProduct);
            setPrice(defaultProduct.last_price.toString());
            fetchProductDetails(defaultProduct.symbol);
          }
        }
      } catch (error) {
        console.error('Error fetching products:', error);
        enqueueSnackbar('Failed to load trading products', { variant: 'error' });
      } finally {
        setLoading(false);
      }
    };
    
    fetchProducts();
  }, [defaultSymbol, enqueueSnackbar, fetchProductDetails]);
  
  // Update leverage
  const updateLeverage = async (newLeverage: number) => {
    if (!selectedProduct) return;
    
    try {
      const response = await deltaExchangeApi.setLeverage(
        selectedProduct.symbol,
        newLeverage
      );
      
      if (response?.success) {
        setLeverage(newLeverage);
        enqueueSnackbar(`Leverage set to ${newLeverage}x`, { variant: 'success' });
      }
    } catch (error) {
      console.error('Error setting leverage:', error);
      enqueueSnackbar('Failed to set leverage', { variant: 'error' });
    }
  };
  
  // Handle tab change
  const handleTabChange = (_event: React.SyntheticEvent, newValue: string) => {
    setTabValue(newValue);
    
    // If market order, clear price
    if (newValue === 'market') {
      setPrice('');
    } else if (newValue === 'limit' && marketPrice) {
      setPrice(marketPrice.toString());
    }
  };
  
  // Handle product change
  const handleProductChange = (event: SelectChangeEvent<string>) => {
    const productId = event.target.value;
    const product = products.find(p => p.id.toString() === productId);
    
    if (product) {
      setSelectedProduct(product);
      setPrice(product.last_price.toString());
      fetchProductDetails(product.symbol);
    }
  };
  
  // Handle price change
  const handlePriceChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPrice(event.target.value);
  };
  
  // Handle amount change
  const handleAmountChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setAmount(event.target.value);
  };
  
  // Handle order side change
  const handleOrderSideChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setOrderSide(event.target.value as 'buy' | 'sell');
  };
  
  // Handle leverage change
  const handleLeverageChange = (_event: Event, value: number | number[]) => {
    setLeverage(value as number);
  };
  
  // Handle leverage change committed
  const handleLeverageChangeCommitted = (_event: React.SyntheticEvent | Event, value: number | number[]) => {
    updateLeverage(value as number);
  };
  
  // Handle order creation
  const handleCreateOrder = async () => {
    if (!selectedProduct) return;
    
    try {
      setSubmitting(true);
      
      const orderData: any = {
        product_id: selectedProduct.id,
        side: orderSide,
        size: amount
      };
      
      if (tabValue === 'limit') {
        orderData.order_type = 'limit_order';
        orderData.limit_price = price;
      } else {
        orderData.order_type = 'market_order';
      }
      
      const response = await deltaExchangeApi.createOrder(orderData);
      
      if (response?.success) {
        enqueueSnackbar(
          `Successfully placed ${orderSide} ${tabValue} order for ${amount} ${selectedProduct.base_currency}`,
          { variant: 'success' }
        );
        
        // Reset form
        setAmount('');
      }
    } catch (error) {
      console.error('Error creating order:', error);
      enqueueSnackbar('Failed to create order', { variant: 'error' });
    } finally {
      setSubmitting(false);
    }
  };
  
  // Calculate total
  const calculateTotal = () => {
    if (!amount || (!price && tabValue === 'limit')) return 'N/A';
    const calculatedTotal = parseFloat(amount) * (tabValue === 'limit' ? parseFloat(price) : (marketPrice || 0));
    return calculatedTotal.toFixed(2);
  };
  
  // Render loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  return (
    <Card>
      <CardHeader 
        title="Trade" 
        subheader={selectedProduct ? `${selectedProduct.base_currency}/${selectedProduct.quote_currency}` : 'Select a product'} 
      />
      <CardContent>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', mx: -1 }}>
          {/* Left column - Order form */}
          <Box sx={{ width: { xs: '100%', md: '58.33%' }, p: 1 }}>
            <Box mb={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Select Product</InputLabel>
                <Select
                  value={selectedProduct?.id.toString() || ''}
                  onChange={handleProductChange}
                  label="Select Product"
                >
                  {products.map(product => (
                    <MenuItem key={product.id} value={product.id.toString()}>
                      {product.symbol} ({product.base_currency}/{product.quote_currency})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
            
            <Box mb={2}>
              <Paper sx={{ mb: 2 }}>
                <Tabs
                  value={tabValue}
                  onChange={handleTabChange}
                  indicatorColor="primary"
                  textColor="primary"
                  variant="fullWidth"
                >
                  <Tab label="Limit" value="limit" />
                  <Tab label="Market" value="market" />
                </Tabs>
              </Paper>
              
              <Box mb={2}>
                <RadioGroup
                  row
                  value={orderSide}
                  onChange={handleOrderSideChange}
                  sx={{ mb: 2 }}
                >
                  <FormControlLabel 
                    value="buy" 
                    control={<Radio color="success" />} 
                    label="Buy" 
                  />
                  <FormControlLabel 
                    value="sell" 
                    control={<Radio color="error" />} 
                    label="Sell" 
                  />
                </RadioGroup>
              </Box>
              
              {tabValue === 'limit' && (
                <TextField
                  label="Price"
                  value={price}
                  onChange={handlePriceChange}
                  fullWidth
                  margin="normal"
                  type="number"
                  InputProps={{
                    startAdornment: <InputAdornment position="start">$</InputAdornment>,
                  }}
                  size="small"
                />
              )}
              
              <TextField
                label="Amount"
                value={amount}
                onChange={handleAmountChange}
                fullWidth
                margin="normal"
                type="number"
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      {selectedProduct?.base_currency || 'BTC'}
                    </InputAdornment>
                  ),
                }}
                size="small"
              />
              
              <Box mt={2}>
                <Typography variant="body2" gutterBottom>
                  Total: ${calculateTotal()} {selectedProduct?.quote_currency}
                </Typography>
              </Box>
              
              {selectedProduct && selectedProduct.max_leverage > 1 && (
                <Box mt={3}>
                  <Typography variant="body2" gutterBottom>
                    Leverage: {leverage}x
                  </Typography>
                  <Slider
                    value={leverage}
                    min={1}
                    max={selectedProduct?.max_leverage || 1}
                    step={1}
                    marks
                    onChange={handleLeverageChange}
                    onChangeCommitted={handleLeverageChangeCommitted}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(x) => `${x}x`}
                  />
                </Box>
              )}
              
              <Box mt={3}>
                <Button
                  variant="contained"
                  fullWidth
                  color={orderSide === 'buy' ? 'success' : 'error'}
                  onClick={handleCreateOrder}
                  disabled={submitting || !selectedProduct || !amount || (tabValue === 'limit' && !price)}
                >
                  {submitting ? (
                    <CircularProgress size={24} color="inherit" />
                  ) : (
                    `${orderSide === 'buy' ? 'Buy' : 'Sell'} ${selectedProduct?.base_currency || 'BTC'}`
                  )}
                </Button>
              </Box>
            </Box>
          </Box>
          
          {/* Right column - Order book */}
          <Box sx={{ width: { xs: '100%', md: '41.67%' }, p: 1 }}>
            <Paper variant="outlined" sx={{ height: '100%', p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Order Book
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" color="textSecondary">
                  Market Price: ${marketPrice?.toFixed(2) || 'N/A'}
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {/* Asks (Sell orders) */}
                {orderBookData.asks.map(([price, size], index) => (
                  <Box 
                    key={`ask-${index}`}
                    sx={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      bgcolor: 'error.main',
                      color: 'white',
                      opacity: 0.7 + (index * 0.05),
                      p: 0.5,
                      borderRadius: 1
                    }}
                  >
                    <Typography variant="caption">{size.toFixed(4)}</Typography>
                    <Typography variant="caption">${price.toFixed(2)}</Typography>
                  </Box>
                ))}
                
                <Divider sx={{ my: 1 }} />
                
                {/* Bids (Buy orders) */}
                {orderBookData.bids.map(([price, size], index) => (
                  <Box 
                    key={`bid-${index}`}
                    sx={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      bgcolor: 'success.main',
                      color: 'white',
                      opacity: 0.7 + (index * 0.05),
                      p: 0.5,
                      borderRadius: 1
                    }}
                  >
                    <Typography variant="caption">{size.toFixed(4)}</Typography>
                    <Typography variant="caption">${price.toFixed(2)}</Typography>
                  </Box>
                ))}
              </Box>
            </Paper>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default TradingPanel; 
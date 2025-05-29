import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  useTheme,
} from '@mui/material';
import { useSnackbar } from 'notistack';
import deltaExchangeApi from '../../lib/api/deltaExchangeApi';

interface Product {
  id: number;
  symbol: string;
  description: string;
  underlying: string;
  base_currency: string;
  quote_currency: string;
  last_price: number;
  mark_price: number;
  index_price: number;
  bid: number;
  ask: number;
  high: number;
  low: number;
  volume_24h: number;
  price_change_percent_24h: number;
  funding_rate: number;
  open_interest: number;
}

const MarketDataTable: React.FC = () => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        setLoading(true);
        const response = await deltaExchangeApi.getProducts();
        if (response?.success && response.data) {
          // Sort by volume
          const sortedProducts = response.data
            .filter((product: any) => product.status === 'live')
            .sort((a: any, b: any) => b.volume_24h - a.volume_24h)
            .slice(0, 20); // Top 20 by volume
          
          setProducts(sortedProducts);
        } else {
          throw new Error('Failed to fetch market data');
        }
      } catch (err) {
        console.error('Error fetching market data:', err);
        setError('Failed to load market data. Please try again later.');
        enqueueSnackbar('Failed to load market data', { variant: 'error' });
      } finally {
        setLoading(false);
      }
    };

    fetchProducts();
    
    // Set up refresh interval
    const intervalId = setInterval(fetchProducts, 30000); // Refresh every 30 seconds
    
    return () => clearInterval(intervalId);
  }, [enqueueSnackbar]);

  const formatNumber = (num: number, decimals = 2) => {
    if (num === undefined || num === null) return 'N/A';
    return num.toLocaleString(undefined, { 
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals 
    });
  };

  const formatVolume = (volume: number) => {
    if (volume === undefined || volume === null) return 'N/A';
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(2)}M`;
    if (volume >= 1000) return `$${(volume / 1000).toFixed(2)}K`;
    return `$${volume.toFixed(2)}`;
  };

  const formatPercentage = (percentage: number) => {
    if (percentage === undefined || percentage === null) return 'N/A';
    const formatted = percentage.toFixed(2) + '%';
    if (percentage > 0) return `+${formatted}`;
    return formatted;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Card>
      <CardHeader 
        title="Market Data" 
        subheader="Top trading products by volume" 
      />
      <CardContent>
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell>Price</TableCell>
                <TableCell align="right">24h Change</TableCell>
                <TableCell align="right">24h Volume</TableCell>
                <TableCell align="right">Funding Rate</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {products.map((product) => (
                <TableRow key={product.id}>
                  <TableCell component="th" scope="row">
                    <Typography variant="body2" fontWeight="bold">
                      {product.symbol}
                    </Typography>
                    <Typography variant="caption" display="block" color="textSecondary">
                      {product.base_currency}/{product.quote_currency}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      ${formatNumber(product.last_price)}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography 
                      variant="body2"
                      color={product.price_change_percent_24h >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(product.price_change_percent_24h)}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2">
                      {formatVolume(product.volume_24h)}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography 
                      variant="body2"
                      color={product.funding_rate >= 0 ? 'success.main' : 'error.main'}
                    >
                      {formatPercentage(product.funding_rate)}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
              {products.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    <Typography variant="body2">No market data available</Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default MarketDataTable; 
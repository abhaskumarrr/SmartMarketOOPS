import React, { useState } from 'react';
import { 
  Container, 
  Box,
  Typography, 
  Paper, 
  Card, 
  CardContent, 
  CardHeader, 
  Divider,
  Tabs,
  Tab,
  Chip,
  Button,
  Stack,
  IconButton,
  Avatar,
  Skeleton,
  useTheme,
  alpha
} from '@mui/material';
import { 
  ArrowUpward as ArrowUpIcon, 
  ArrowDownward as ArrowDownIcon,
  MoreVert as MoreIcon,
  TrendingUp as TrendingUpIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  ShowChart as ShowChartIcon
} from '@mui/icons-material';
import Head from 'next/head';
import TradingPanel from '../components/charts/TradingPanel';
import { useTradingContext } from '../lib/contexts/TradingContext';

// Tab panel component for switching between views
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`market-tabpanel-${index}`}
      aria-labelledby={`market-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 2 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

// Create tab props
function a11yProps(index: number) {
  return {
    id: `market-tab-${index}`,
    'aria-controls': `market-tabpanel-${index}`,
  };
}

const DashboardPage: React.FC = () => {
  const theme = useTheme();
  const { 
    marketData, 
    selectedSymbol, 
    setSelectedSymbol, 
    tradingSignals, 
    isMarketDataLoading,
    positions
  } = useTradingContext();
  
  // State for tab selection
  const [tabValue, setTabValue] = useState(0);
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  // Format price with appropriate decimals
  const formatPrice = (price: number) => {
    if (price >= 1000) {
      return price.toLocaleString(undefined, { maximumFractionDigits: 2 });
    } else if (price >= 1) {
      return price.toLocaleString(undefined, { maximumFractionDigits: 4 });
    } else {
      return price.toLocaleString(undefined, { maximumFractionDigits: 8 });
    }
  };
  
  return (
    <>
      <Head>
        <title>Dashboard - SmartMarketOOPS</title>
        <meta name="description" content="Trading dashboard with real-time market data" />
      </Head>
      
      <Container maxWidth="xl" sx={{ mt: 1, mb: 4 }}>
        {/* Overview Cards */}
        <Box sx={{ display: 'flex', flexWrap: 'wrap', mb: 4, mx: -1.5 }}>
          <Box sx={{ width: { xs: '100%', sm: '50%', md: '33.33%' }, p: 1.5 }}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Portfolio Value
                  </Typography>
                  <Chip 
                    label="+2.5%" 
                    size="small" 
                    color="success"
                    sx={{ fontWeight: 'bold' }}
                  />
                </Box>
                {isMarketDataLoading ? (
                  <Skeleton width="60%" height={40} />
                ) : (
                  <Typography variant="h4" component="div" fontWeight="bold">
                    $124,586.24
                  </Typography>
                )}
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  +$3,052.42 today
                </Typography>
                <Divider sx={{ my: 1 }} />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Button size="small" color="primary" startIcon={<RefreshIcon />}>
                    Refresh
                  </Button>
                  <Typography variant="caption" color="text.secondary">
                    Updated just now
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Box>
          
          <Box sx={{ width: { xs: '100%', sm: '50%', md: '33.33%' }, p: 1.5 }}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Active Positions
                  </Typography>
                  <IconButton size="small">
                    <FilterIcon fontSize="small" />
                  </IconButton>
                </Box>
                {isMarketDataLoading ? (
                  <Skeleton width="40%" height={40} />
                ) : (
                  <Typography variant="h4" component="div" fontWeight="bold">
                    {positions.length}
                  </Typography>
                )}
                <Stack direction="row" spacing={1} sx={{ mt: 1, mb: 2 }}>
                  <Chip 
                    label={`${positions.filter(p => p.type === 'long').length} Long`} 
                    size="small" 
                    color="success"
                    variant="outlined"
                  />
                  <Chip 
                    label={`${positions.filter(p => p.type === 'short').length} Short`} 
                    size="small" 
                    color="error"
                    variant="outlined"
                  />
                </Stack>
                <Divider sx={{ my: 1 }} />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Button size="small" color="primary">
                    View All
                  </Button>
                  <Typography variant="caption" color="text.secondary">
                    Total PnL: +$1,324.56
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Box>
          
          <Box sx={{ width: { xs: '100%', sm: '50%', md: '33.33%' }, p: 1.5 }}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Trading Signals
                  </Typography>
                  <Chip 
                    label="3 New" 
                    size="small" 
                    color="primary"
                    sx={{ fontWeight: 'bold' }}
                  />
                </Box>
                {isMarketDataLoading ? (
                  <Skeleton width="50%" height={40} />
                ) : (
                  <Typography variant="h4" component="div" fontWeight="bold">
                    {tradingSignals.length}
                  </Typography>
                )}
                <Stack direction="row" spacing={1} sx={{ mt: 1, mb: 2 }}>
                  <Chip 
                    label={`${tradingSignals.filter(s => s.action === 'Buy').length} Buy`} 
                    size="small" 
                    color="success"
                    variant="outlined"
                  />
                  <Chip 
                    label={`${tradingSignals.filter(s => s.action === 'Sell').length} Sell`} 
                    size="small" 
                    color="error"
                    variant="outlined"
                  />
                  <Chip 
                    label={`${tradingSignals.filter(s => s.action === 'Hold').length} Hold`} 
                    size="small" 
                    color="warning"
                    variant="outlined"
                  />
                </Stack>
                <Divider sx={{ my: 1 }} />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Button size="small" color="primary">
                    View Signals
                  </Button>
                  <Typography variant="caption" color="text.secondary">
                    Updated 5m ago
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Box>
        </Box>
        
        {/* Market Data Section */}
        <Card sx={{ mb: 4 }}>
          <CardHeader 
            title="Market Overview" 
            subheader="Real-time prices and market data"
            action={
              <IconButton>
                <MoreIcon />
              </IconButton>
            }
          />
          <Divider />
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
              value={tabValue} 
              onChange={handleTabChange} 
              aria-label="market data tabs"
              variant="scrollable"
              scrollButtons="auto"
            >
              <Tab label="Favorites" {...a11yProps(0)} />
              <Tab label="Crypto" {...a11yProps(1)} />
              <Tab label="Gainers" {...a11yProps(2)} />
              <Tab label="Losers" {...a11yProps(3)} />
              <Tab label="Volume" {...a11yProps(4)} />
            </Tabs>
          </Box>
          
          <TabPanel value={tabValue} index={0}>
            <Box sx={{ overflow: 'auto' }}>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', minWidth: 650 }}>
                {/* Table header */}
                <Box sx={{ width: '25%' }}>
                  <Typography variant="subtitle2" fontWeight="bold">Asset</Typography>
                </Box>
                <Box sx={{ width: '25%' }}>
                  <Typography variant="subtitle2" fontWeight="bold">Price</Typography>
                </Box>
                <Box sx={{ width: '25%' }}>
                  <Typography variant="subtitle2" fontWeight="bold">24h Change</Typography>
                </Box>
                <Box sx={{ width: '25%' }}>
                  <Typography variant="subtitle2" fontWeight="bold">24h Volume</Typography>
                </Box>
                
                <Box sx={{ width: '100%' }}>
                  <Divider />
                </Box>
                
                {/* Market Data Rows */}
                {isMarketDataLoading ? (
                  // Loading skeletons
                  Array.from(new Array(5)).map((_, index) => (
                    <React.Fragment key={`skeleton-${index}`}>
                      <Box sx={{ width: '25%' }}>
                        <Skeleton height={40} />
                      </Box>
                      <Box sx={{ width: '25%' }}>
                        <Skeleton height={40} />
                      </Box>
                      <Box sx={{ width: '25%' }}>
                        <Skeleton height={40} />
                      </Box>
                      <Box sx={{ width: '25%' }}>
                        <Skeleton height={40} />
                      </Box>
                    </React.Fragment>
                  ))
                ) : (
                  // Actual market data
                  Object.values(marketData).map((coin) => (
                    <React.Fragment key={coin.symbol}>
                      <Box sx={{ width: '25%' }}>
                        <Box 
                          sx={{ 
                            display: 'flex', 
                            alignItems: 'center',
                            cursor: 'pointer',
                            '&:hover': { color: 'primary.main' },
                          }}
                          onClick={() => setSelectedSymbol(coin.symbol)}
                        >
                          <Avatar 
                            sx={{ 
                              width: 28, 
                              height: 28, 
                              mr: 1,
                              bgcolor: alpha(theme.palette.primary.main, 0.1),
                              color: 'primary.main'
                            }}
                          >
                            <ShowChartIcon fontSize="small" />
                          </Avatar>
                          <Box>
                            <Typography 
                              variant="body1" 
                              fontWeight={selectedSymbol === coin.symbol ? 'bold' : 'normal'}
                              color={selectedSymbol === coin.symbol ? 'primary.main' : 'inherit'}
                            >
                              {coin.symbol.replace('USDT', '')}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {coin.symbol}
                            </Typography>
                          </Box>
                        </Box>
                      </Box>
                      <Box sx={{ width: '25%' }}>
                        <Typography variant="body1" fontWeight="medium">
                          ${formatPrice(coin.price)}
                        </Typography>
                      </Box>
                      <Box sx={{ width: '25%' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          {coin.changePercent > 0 ? (
                            <ArrowUpIcon fontSize="small" color="success" sx={{ mr: 0.5 }} />
                          ) : (
                            <ArrowDownIcon fontSize="small" color="error" sx={{ mr: 0.5 }} />
                          )}
                          <Typography 
                            variant="body1" 
                            color={coin.changePercent > 0 ? 'success.main' : 'error.main'}
                          >
                            {coin.changePercent.toFixed(2)}%
                          </Typography>
                        </Box>
                      </Box>
                      <Box sx={{ width: '25%' }}>
                        <Typography variant="body1">
                          ${(coin.volume / 1000000).toFixed(2)}M
                        </Typography>
                      </Box>
                      <Box sx={{ width: '100%' }}>
                        <Divider />
                      </Box>
                    </React.Fragment>
                  ))
                )}
              </Box>
            </Box>
          </TabPanel>
          
          <TabPanel value={tabValue} index={1}>
            <Typography>Cryptocurrency markets data...</Typography>
          </TabPanel>
          
          <TabPanel value={tabValue} index={2}>
            <Typography>Top gainers data...</Typography>
          </TabPanel>
          
          <TabPanel value={tabValue} index={3}>
            <Typography>Top losers data...</Typography>
          </TabPanel>
          
          <TabPanel value={tabValue} index={4}>
            <Typography>Highest volume assets...</Typography>
          </TabPanel>
        </Card>
        
        {/* Trading Panel */}
        <Card>
          <CardHeader 
            title="Trading Chart" 
            subheader={`${selectedSymbol.replace('USDT', '')}/USDT`}
            action={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Chip 
                  icon={<TrendingUpIcon />}
                  label="ML Signals Active" 
                  color="primary" 
                  size="small"
                  sx={{ mr: 1 }}
                />
                <IconButton>
                  <MoreIcon />
                </IconButton>
              </Box>
            }
          />
          <Divider />
          <CardContent>
            <TradingPanel defaultSymbol={selectedSymbol} />
          </CardContent>
        </Card>
      </Container>
    </>
  );
};

export default DashboardPage; 
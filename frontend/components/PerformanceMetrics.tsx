import React, { useState, useEffect } from 'react';
import { Box, Paper, Typography,  Card, CardContent, Divider, CircularProgress } from '@mui/material';
import {
  TrendingUp as ProfitIcon,
  TrendingDown as LossIcon,
  Loop as TurnoverIcon,
  Timeline as WinRateIcon,
  Schedule as TimeIcon,
  BarChart as VolumeIcon,
} from '@mui/icons-material';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Types
interface DailyPerformanceData {
  date: string;
  profit: number;
  trades: number;
  winRate: number;
}

interface MetricsData {
  totalProfit: number;
  totalTrades: number;
  winRate: number;
  avgProfit: number;
  avgLoss: number;
  profitFactor: number;
  maxDrawdown: number;
  bestTrade: number;
  worstTrade: number;
  avgTradeTime: string;
  dailyVolume: number;
}

interface AssetAllocationData {
  name: string;
  value: number;
}

interface PerformanceMetricsProps {
  darkMode: boolean;
}

export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ darkMode }) => {
  // State for API data
  const [dailyPerfData, setDailyPerfData] = useState<DailyPerformanceData[]>([]);
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [assetAllocationData, setAssetAllocationData] = useState<AssetAllocationData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch metrics data from API
  useEffect(() => {
    const fetchMetricsData = async () => {
      try {
        setIsLoading(true);
        
        // Fetch daily performance data
        const dailyPerfResponse = await fetch('/api/metrics/daily');
        const dailyPerfResult = await dailyPerfResponse.json();
        
        // Fetch overall metrics
        const metricsResponse = await fetch('/api/metrics/summary');
        const metricsResult = await metricsResponse.json();
        
        // Fetch asset allocation
        const allocationResponse = await fetch('/api/metrics/allocation');
        const allocationResult = await allocationResponse.json();
        
        if (dailyPerfResult.success) {
          setDailyPerfData(dailyPerfResult.data || []);
        }
        
        if (metricsResult.success) {
          setMetricsData(metricsResult.data || null);
        }
        
        if (allocationResult.success) {
          setAssetAllocationData(allocationResult.data || []);
        }
      } catch (error) {
        console.error('Error fetching metrics data:', error);
        setError('Failed to load performance metrics');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchMetricsData();
  }, []);
  
  // Get styles based on dark mode
  const getStyles = () => {
    return {
      paper: {
        backgroundColor: darkMode ? '#1E1E2D' : '#FFFFFF',
        boxShadow: darkMode ? '0 4px 20px rgba(0,0,0,0.25)' : '0 4px 20px rgba(0,0,0,0.1)',
      },
      cardBg: {
        backgroundColor: darkMode ? '#2A2A3C' : '#F5F5F5',
      },
      title: {
        color: darkMode ? '#E0E0E0' : '#121212',
      },
      text: {
        color: darkMode ? '#E0E0E0' : '#121212',
      },
      subtitle: {
        color: darkMode ? '#B0B0C0' : '#5F5F7A',
      },
      profit: {
        color: darkMode ? '#4CAF50' : '#1E8E3E',
      },
      loss: {
        color: darkMode ? '#FF5252' : '#D32F2F',
      },
      neutral: {
        color: darkMode ? '#90CAF9' : '#1976D2',
      },
      divider: {
        backgroundColor: darkMode ? '#3A3A50' : '#E0E0E0',
      },
      chart: {
        backgroundColor: darkMode ? '#2A2A3C' : '#F5F5F5',
        borderRadius: '8px',
      },
      chartText: {
        fill: darkMode ? '#E0E0E0' : '#121212',
      },
      chartGrid: {
        stroke: darkMode ? '#3A3A50' : '#E0E0E0',
      },
    };
  };
  
  const styles = getStyles();

  // Custom tooltip for charts
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            backgroundColor: darkMode ? '#242635' : '#FFFFFF',
            border: `1px solid ${darkMode ? '#3A3A50' : '#E0E0E0'}`,
            padding: '10px',
            borderRadius: '4px',
            boxShadow: darkMode ? '0 4px 12px rgba(0,0,0,0.4)' : '0 4px 12px rgba(0,0,0,0.1)',
          }}
        >
          <Typography variant="body2" style={{ color: darkMode ? '#E0E0E0' : '#121212' }}>
            {label}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography 
              key={`tooltip-${index}`} 
              variant="body2" 
              style={{ 
                color: entry.name === 'profit' && entry.value >= 0
                  ? styles.profit.color
                  : entry.name === 'profit' && entry.value < 0
                    ? styles.loss.color
                    : entry.color
              }}
            >
              {entry.name === 'profit' ? 'Profit: ' : entry.name === 'trades' ? 'Trades: ' : 'Win Rate: '}
              {entry.name === 'profit' ? `$${entry.value}` : entry.name === 'winRate' ? `${entry.value}%` : entry.value}
            </Typography>
          ))}
        </Box>
      );
    }
    return null;
  };

  // Loading state
  if (isLoading) {
    return (
      <Box sx={{ mb: 4 }}>
        <Paper elevation={3} style={styles.paper} sx={{ borderRadius: '8px', overflow: 'hidden', p: 3 }}>
          <Typography variant="h6" sx={{ mb: 3 }} style={styles.title}>
            Performance Metrics
          </Typography>
          <Box display="flex" justifyContent="center" alignItems="center" sx={{ p: 5 }}>
            <CircularProgress />
          </Box>
        </Paper>
      </Box>
    );
  }

  // Error state
  if (error || !metricsData) {
    return (
      <Box sx={{ mb: 4 }}>
        <Paper elevation={3} style={styles.paper} sx={{ borderRadius: '8px', overflow: 'hidden', p: 3 }}>
          <Typography variant="h6" sx={{ mb: 3 }} style={styles.title}>
            Performance Metrics
          </Typography>
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body1" color="error">
              {error || 'Unable to load metrics data'}
            </Typography>
          </Box>
        </Paper>
      </Box>
    );
  }

  return (
    <Box sx={{ mb: 4 }}>
      <Paper elevation={3} style={styles.paper} sx={{ borderRadius: '8px', overflow: 'hidden', p: 3 }}>
        <Typography variant="h6" sx={{ mb: 3 }} style={styles.title}>
          Performance Metrics
        </Typography>
        
        {/* Key Metrics Cards */}
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' }, gap: 3, mb: 4 }}>
          {/* Total Profit */}
          <Box>
            <Card elevation={0} sx={{ borderRadius: '8px' }} style={styles.cardBg}>
              <CardContent>
                <Box display="flex" alignItems="center" sx={{ mb: 1 }}>
                  <ProfitIcon sx={{ mr: 1 }} style={{ color: metricsData.totalProfit >= 0 ? styles.profit.color : styles.loss.color }} />
                  <Typography variant="body2" style={styles.subtitle}>Total Profit/Loss</Typography>
                </Box>
                <Typography variant="h5" style={{ 
                  color: metricsData.totalProfit >= 0 ? styles.profit.color : styles.loss.color 
                }}>
                  {metricsData.totalProfit >= 0 ? '+' : ''}${metricsData.totalProfit.toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Box>
          
          {/* Win Rate */}
          <Box>
            <Card elevation={0} sx={{ borderRadius: '8px' }} style={styles.cardBg}>
              <CardContent>
                <Box display="flex" alignItems="center" sx={{ mb: 1 }}>
                  <WinRateIcon sx={{ mr: 1 }} style={styles.neutral} />
                  <Typography variant="body2" style={styles.subtitle}>Win Rate</Typography>
                </Box>
                <Typography variant="h5" style={styles.text}>
                  {metricsData.winRate}%
                </Typography>
              </CardContent>
            </Card>
          </Box>
          
          {/* Total Trades */}
          <Box>
            <Card elevation={0} sx={{ borderRadius: '8px' }} style={styles.cardBg}>
              <CardContent>
                <Box display="flex" alignItems="center" sx={{ mb: 1 }}>
                  <TurnoverIcon sx={{ mr: 1 }} style={styles.neutral} />
                  <Typography variant="body2" style={styles.subtitle}>Total Trades</Typography>
                </Box>
                <Typography variant="h5" style={styles.text}>
                  {metricsData.totalTrades}
                </Typography>
              </CardContent>
            </Card>
          </Box>
          
          {/* Profit Factor */}
          <Box>
            <Card elevation={0} sx={{ borderRadius: '8px' }} style={styles.cardBg}>
              <CardContent>
                <Box display="flex" alignItems="center" sx={{ mb: 1 }}>
                  <ProfitIcon sx={{ mr: 1 }} style={styles.neutral} />
                  <Typography variant="body2" style={styles.subtitle}>Profit Factor</Typography>
                </Box>
                <Typography variant="h5" style={styles.text}>
                  {metricsData.profitFactor.toFixed(2)}
                </Typography>
              </CardContent>
            </Card>
          </Box>
          
          {/* Max Drawdown */}
          <Box>
            <Card elevation={0} sx={{ borderRadius: '8px' }} style={styles.cardBg}>
              <CardContent>
                <Box display="flex" alignItems="center" sx={{ mb: 1 }}>
                  <LossIcon sx={{ mr: 1 }} style={styles.loss} />
                  <Typography variant="body2" style={styles.subtitle}>Max Drawdown</Typography>
                </Box>
                <Typography variant="h5" style={styles.loss}>
                  -${metricsData.maxDrawdown.toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Box>
          
          {/* Daily Volume */}
          <Box>
            <Card elevation={0} sx={{ borderRadius: '8px' }} style={styles.cardBg}>
              <CardContent>
                <Box display="flex" alignItems="center" sx={{ mb: 1 }}>
                  <VolumeIcon sx={{ mr: 1 }} style={styles.neutral} />
                  <Typography variant="body2" style={styles.subtitle}>Daily Volume</Typography>
                </Box>
                <Typography variant="h5" style={styles.text}>
                  ${metricsData.dailyVolume.toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Box>
        </Box>
        
        {/* Daily Performance Chart */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="subtitle1" sx={{ mb: 2 }} style={styles.title}>
            Daily Performance
          </Typography>
          <Box style={styles.chart} sx={{ p: 2 }}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={dailyPerfData}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid stroke={styles.chartGrid.stroke} strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  style={styles.chartText}
                  tick={{ fill: styles.chartText.fill }}
                />
                <YAxis 
                  yAxisId="left"
                  style={styles.chartText}
                  tick={{ fill: styles.chartText.fill }}
                />
                <YAxis 
                  yAxisId="right" 
                  orientation="right" 
                  style={styles.chartText}
                  tick={{ fill: styles.chartText.fill }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ color: styles.chartText.fill }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="profit"
                  stroke={styles.profit.color}
                  activeDot={{ r: 8 }}
                  strokeWidth={2}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="winRate"
                  stroke="#8884d8"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </Box>
        
        {/* Trade Metrics & Asset Allocation */}
        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 3 }}>
          {/* Detailed Metrics */}
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 2 }} style={styles.title}>
              Detailed Metrics
            </Typography>
            <Box style={styles.cardBg} sx={{ p: 2, borderRadius: '8px' }}>
              <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 2 }}>
                <Box>
                  <Typography variant="body2" style={styles.subtitle}>Avg. Profit per Trade:</Typography>
                  <Typography variant="body1" style={styles.profit}>
                    ${metricsData.avgProfit.toFixed(2)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" style={styles.subtitle}>Avg. Loss per Trade:</Typography>
                  <Typography variant="body1" style={styles.loss}>
                    ${metricsData.avgLoss.toFixed(2)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" style={styles.subtitle}>Best Trade:</Typography>
                  <Typography variant="body1" style={styles.profit}>
                    +${metricsData.bestTrade.toLocaleString()}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" style={styles.subtitle}>Worst Trade:</Typography>
                  <Typography variant="body1" style={styles.loss}>
                    ${metricsData.worstTrade.toLocaleString()}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="body2" style={styles.subtitle}>Avg. Trade Time:</Typography>
                  <Typography variant="body1" style={styles.text}>
                    {metricsData.avgTradeTime}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Box>
          
          {/* Asset Allocation */}
          <Box>
            <Typography variant="subtitle1" sx={{ mb: 2 }} style={styles.title}>
              Asset Allocation
            </Typography>
            <Box style={styles.cardBg} sx={{ p: 2, borderRadius: '8px', height: '100%' }}>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart
                  layout="vertical"
                  data={assetAllocationData}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid stroke={styles.chartGrid.stroke} strokeDasharray="3 3" />
                  <XAxis 
                    type="number" 
                    style={styles.chartText}
                    tick={{ fill: styles.chartText.fill }}
                  />
                  <YAxis 
                    dataKey="name" 
                    type="category" 
                    style={styles.chartText}
                    tick={{ fill: styles.chartText.fill }}
                  />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default PerformanceMetrics; 
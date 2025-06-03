import React, { useState, useEffect } from 'react';
import {
  Box,
  
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Speed,
  Security,
  Psychology,
  Analytics,
  Refresh,
  Warning,
  CheckCircle,
  Error
} from '@mui/icons-material';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  ArcElement,
  BarElement
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  ArcElement,
  BarElement
);

interface PerformanceMetrics {
  timestamp: string;
  accuracy_rate: number;
  win_rate: number;
  confidence_score: number;
  quality_score: number;
  total_trades: number;
  active_positions: number;
  portfolio_balance: number;
  total_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
}

interface SystemHealth {
  ml_service_status: 'online' | 'offline' | 'degraded';
  market_data_status: 'connected' | 'disconnected' | 'delayed';
  trading_engine_status: 'active' | 'paused' | 'error';
  risk_manager_status: 'operational' | 'warning' | 'critical';
  last_update: string;
  uptime: string;
}

interface SignalQuality {
  symbol: string;
  prediction: number;
  confidence: number;
  quality_score: number;
  signal_valid: boolean;
  market_regime: string;
  recommendation: string;
  timestamp: string;
}

const EnhancedMonitoringDashboard: React.FC = () => {
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [signalQuality, setSignalQuality] = useState<SignalQuality[]>([]);
  const [historicalData, setHistoricalData] = useState<PerformanceMetrics[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5 seconds

  // Fetch data from APIs
  const fetchPerformanceMetrics = async () => {
    try {
      const response = await fetch('/api/monitoring/performance');
      const data = await response.json();
      setPerformanceMetrics(data);
      
      // Add to historical data
      setHistoricalData(prev => [...prev.slice(-49), data]); // Keep last 50 points
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
    }
  };

  const fetchSystemHealth = async () => {
    try {
      const response = await fetch('/api/monitoring/health');
      const data = await response.json();
      setSystemHealth(data);
    } catch (error) {
      console.error('Error fetching system health:', error);
    }
  };

  const fetchSignalQuality = async () => {
    try {
      const response = await fetch('/api/monitoring/signals');
      const data = await response.json();
      setSignalQuality(data);
    } catch (error) {
      console.error('Error fetching signal quality:', error);
    }
  };

  const fetchAllData = () => {
    fetchPerformanceMetrics();
    fetchSystemHealth();
    fetchSignalQuality();
  };

  useEffect(() => {
    fetchAllData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchAllData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  // Chart configurations
  const performanceChartData = {
    labels: historicalData.map(d => new Date(d.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'Win Rate (%)',
        data: historicalData.map(d => d.win_rate * 100),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      },
      {
        label: 'Accuracy Rate (%)',
        data: historicalData.map(d => d.accuracy_rate * 100),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1
      }
    ]
  };

  const confidenceChartData = {
    labels: ['High Confidence', 'Medium Confidence', 'Low Confidence'],
    datasets: [{
      data: [
        signalQuality.filter(s => s.confidence > 0.8).length,
        signalQuality.filter(s => s.confidence > 0.6 && s.confidence <= 0.8).length,
        signalQuality.filter(s => s.confidence <= 0.6).length
      ],
      backgroundColor: ['#4CAF50', '#FF9800', '#F44336'],
      borderWidth: 1
    }]
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
      case 'connected':
      case 'active':
      case 'operational':
        return 'success';
      case 'degraded':
      case 'delayed':
      case 'warning':
        return 'warning';
      case 'offline':
      case 'disconnected':
      case 'error':
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
      case 'connected':
      case 'active':
      case 'operational':
        return <CheckCircle />;
      case 'degraded':
      case 'delayed':
      case 'warning':
        return <Warning />;
      case 'offline':
      case 'disconnected':
      case 'error':
      case 'critical':
        return <Error />;
      default:
        return <CheckCircle />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Enhanced Trading System Monitor
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          <IconButton onClick={fetchAllData} color="primary">
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* System Health Status */}
      {systemHealth && (
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 3, mb: 3 }}>
          <Box>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {getStatusIcon(systemHealth.ml_service_status)}
                  <Typography variant="h6">ML Service</Typography>
                </Box>
                <Chip
                  label={systemHealth.ml_service_status.toUpperCase()}
                  color={getStatusColor(systemHealth.ml_service_status) as any}
                  size="small"
                />
              </CardContent>
            </Card>
          </Box>
          <Box>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {getStatusIcon(systemHealth.market_data_status)}
                  <Typography variant="h6">Market Data</Typography>
                </Box>
                <Chip
                  label={systemHealth.market_data_status.toUpperCase()}
                  color={getStatusColor(systemHealth.market_data_status) as any}
                  size="small"
                />
              </CardContent>
            </Card>
          </Box>
          <Box>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {getStatusIcon(systemHealth.trading_engine_status)}
                  <Typography variant="h6">Trading Engine</Typography>
                </Box>
                <Chip
                  label={systemHealth.trading_engine_status.toUpperCase()}
                  color={getStatusColor(systemHealth.trading_engine_status) as any}
                  size="small"
                />
              </CardContent>
            </Card>
          </Box>
          <Box>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {getStatusIcon(systemHealth.risk_manager_status)}
                  <Typography variant="h6">Risk Manager</Typography>
                </Box>
                <Chip
                  label={systemHealth.risk_manager_status.toUpperCase()}
                  color={getStatusColor(systemHealth.risk_manager_status) as any}
                  size="small"
                />
              </CardContent>
            </Card>
          </Box>
        </Box>
      )}

      {/* Performance Metrics */}
      {performanceMetrics && (
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 3, mb: 3 }}>
          <Box>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <TrendingUp color="primary" />
                  <Typography variant="h6">Win Rate</Typography>
                </Box>
                <Typography variant="h4" color="primary">
                  {(performanceMetrics.win_rate * 100).toFixed(1)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={performanceMetrics.win_rate * 100}
                  sx={{ mt: 1 }}
                />
              </CardContent>
            </Card>
          </Box>
          <Box>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Analytics color="secondary" />
                  <Typography variant="h6">Accuracy</Typography>
                </Box>
                <Typography variant="h4" color="secondary">
                  {(performanceMetrics.accuracy_rate * 100).toFixed(1)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={performanceMetrics.accuracy_rate * 100}
                  color="secondary"
                  sx={{ mt: 1 }}
                />
              </CardContent>
            </Card>
          </Box>
          <Box>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Psychology color="info" />
                  <Typography variant="h6">Confidence</Typography>
                </Box>
                <Typography variant="h4" color="info.main">
                  {(performanceMetrics.confidence_score * 100).toFixed(1)}%
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={performanceMetrics.confidence_score * 100}
                  color="info"
                  sx={{ mt: 1 }}
                />
              </CardContent>
            </Card>
          </Box>
          <Box>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Security color="success" />
                  <Typography variant="h6">Portfolio</Typography>
                </Box>
                <Typography variant="h4" color="success.main">
                  ${performanceMetrics.portfolio_balance.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  P&L: {performanceMetrics.total_pnl > 0 ? '+' : ''}
                  {performanceMetrics.total_pnl.toFixed(2)}%
                </Typography>
              </CardContent>
            </Card>
          </Box>
        </Box>
      )}

      {/* Charts */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3, mb: 3 }}>
        <Box>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Trends
              </Typography>
              <Box sx={{ height: 300 }}>
                <Line
                  data={performanceChartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 100
                      }
                    }
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Box>
        <Box>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Signal Confidence Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <Doughnut
                  data={confidenceChartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>

      {/* Signal Quality Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Signal Quality
          </Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Prediction</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Quality Score</TableCell>
                  <TableCell>Market Regime</TableCell>
                  <TableCell>Recommendation</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Time</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {signalQuality.slice(0, 10).map((signal, index) => (
                  <TableRow key={index}>
                    <TableCell>{signal.symbol}</TableCell>
                    <TableCell>{signal.prediction.toFixed(3)}</TableCell>
                    <TableCell>
                      <Chip
                        label={`${(signal.confidence * 100).toFixed(1)}%`}
                        color={signal.confidence > 0.8 ? 'success' : signal.confidence > 0.6 ? 'warning' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <LinearProgress
                        variant="determinate"
                        value={signal.quality_score * 100}
                        sx={{ width: 100 }}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip label={signal.market_regime} size="small" />
                    </TableCell>
                    <TableCell>{signal.recommendation}</TableCell>
                    <TableCell>
                      <Chip
                        label={signal.signal_valid ? 'Valid' : 'Invalid'}
                        color={signal.signal_valid ? 'success' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default EnhancedMonitoringDashboard;
